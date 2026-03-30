#include <lua.h>
#include <lauxlib.h>
#include <santoku/lua/utils.h>
#include <santoku/fvec.h>
#include <llama.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define TK_LLAMA_MT "tk_llama_t"
#define TK_LLAMA_DEFAULT_N_SEQ 32

static int tk_llama_refs = 0;

static void tk_llama_log_noop (enum ggml_log_level level, const char *text, void *user_data) {
  (void)level; (void)text; (void)user_data;
}

typedef struct {
  struct llama_model *model;
  struct llama_context *ctx;
  int32_t n_embd;
  int32_t n_ctx;
  int32_t n_seq;
  bool has_encoder;
  bool pooled;
  bool destroyed;
} tk_llama_t;

static inline tk_llama_t *tk_llama_peek (lua_State *L, int i) {
  return (tk_llama_t *)luaL_checkudata(L, i, TK_LLAMA_MT);
}

static inline int tk_llama_gc (lua_State *L) {
  tk_llama_t *ll = tk_llama_peek(L, 1);
  if (!ll->destroyed) {
    if (ll->ctx) llama_free(ll->ctx);
    if (ll->model) llama_model_free(ll->model);
    ll->destroyed = true;
    tk_llama_refs--;
    if (tk_llama_refs <= 0)
      llama_backend_free();
  }
  return 0;
}

static inline void tk_llama_l2_normalize (float *v, int32_t n) {
  float sum = 0.0f;
  for (int32_t i = 0; i < n; i++)
    sum += v[i] * v[i];
  if (sum > 0.0f) {
    float inv = 1.0f / sqrtf(sum);
    for (int32_t i = 0; i < n; i++)
      v[i] *= inv;
  }
}

static inline int tk_llama_encode_lua (lua_State *L) {
  tk_llama_t *ll = tk_llama_peek(L, 1);
  if (ll->destroyed)
    return luaL_error(L, "encode: encoder destroyed");
  luaL_checktype(L, 2, LUA_TTABLE);
  int n = (int)lua_objlen(L, 2);
  int32_t dim = ll->n_embd;
  int do_norm = lua_isnoneornil(L, 3) ? 1 : lua_toboolean(L, 3);

  int has_output = !lua_isnoneornil(L, 4);
  tk_fvec_t *out;
  if (has_output) {
    out = tk_fvec_peek(L, 4);
  } else {
    out = tk_fvec_create(L, (uint64_t)n * (uint64_t)dim);
    out->n = (uint64_t)n * (uint64_t)dim;
  }

  const struct llama_vocab *vocab = llama_model_get_vocab(ll->model);
  int32_t max_tok = ll->n_ctx;
  int32_t max_seq = ll->n_seq;
  int32_t batch_tokens = max_tok * max_seq;

  int32_t *tok_lens = NULL;
  llama_token *tok_all = NULL;
  llama_token *tmp_buf = NULL;
  struct llama_batch batch = { 0 };
  int rc = 0;

  tok_lens = (int32_t *)malloc((uint64_t)n * sizeof(int32_t));
  tok_all = (llama_token *)malloc((uint64_t)n * (uint64_t)max_tok * sizeof(llama_token));
  if (!tok_lens || !tok_all) {
    rc = luaL_error(L, "encode: out of memory");
    goto done;
  }

  int32_t tmp_cap = max_tok * 2;
  tmp_buf = (llama_token *)malloc((uint64_t)tmp_cap * sizeof(llama_token));
  if (!tmp_buf) {
    rc = luaL_error(L, "encode: out of memory");
    goto done;
  }

  for (int i = 0; i < n; i++) {
    lua_rawgeti(L, 2, i + 1);
    size_t len;
    const char *text = lua_tolstring(L, -1, &len);
    lua_pop(L, 1);
    int32_t nt = llama_tokenize(vocab, text, (int32_t)len,
                                tmp_buf, tmp_cap, true, true);
    if (nt < 0) {
      int32_t need = -nt;
      if (need > tmp_cap) {
        tmp_cap = need;
        llama_token *t = (llama_token *)realloc(tmp_buf, (uint64_t)tmp_cap * sizeof(llama_token));
        if (!t) {
          rc = luaL_error(L, "encode: out of memory");
          goto done;
        }
        tmp_buf = t;
      }
      nt = llama_tokenize(vocab, text, (int32_t)len,
                          tmp_buf, tmp_cap, true, true);
      if (nt < 0) {
        rc = luaL_error(L, "encode: tokenization failed for text %d", i + 1);
        goto done;
      }
    }
    if (nt > max_tok)
      nt = max_tok;
    tok_lens[i] = nt;
    memcpy(tok_all + (uint64_t)i * (uint64_t)max_tok, tmp_buf,
           (uint64_t)nt * sizeof(llama_token));
  }
  free(tmp_buf);
  tmp_buf = NULL;

  batch = llama_batch_init(batch_tokens, 0, 1);

  int cur = 0;
  while (cur < n) {
    batch.n_tokens = 0;
    int32_t total = 0;
    int start = cur;
    int count = 0;

    while (cur < n && count < max_seq) {
      int32_t nt = tok_lens[cur];
      if (total + nt > batch_tokens && count > 0)
        break;
      llama_token *src = tok_all + (uint64_t)cur * (uint64_t)max_tok;
      for (int32_t j = 0; j < nt; j++) {
        int32_t idx = batch.n_tokens++;
        batch.token[idx] = src[j];
        batch.pos[idx] = j;
        batch.n_seq_id[idx] = 1;
        batch.seq_id[idx][0] = (llama_seq_id)count;
        batch.logits[idx] = 0;
      }
      if (!ll->pooled)
        batch.logits[batch.n_tokens - 1] = 1;
      total += nt;
      count++;
      cur++;
    }

    int err = ll->has_encoder
      ? llama_encode(ll->ctx, batch)
      : llama_decode(ll->ctx, batch);
    if (err != 0) {
      rc = luaL_error(L, "encode: forward pass failed (batch at text %d)", start + 1);
      goto done;
    }

    if (ll->pooled) {
      for (int s = 0; s < count; s++) {
        float *emb = llama_get_embeddings_seq(ll->ctx, (llama_seq_id)s);
        if (!emb) {
          rc = luaL_error(L, "encode: no embeddings for text %d", start + s + 1);
          goto done;
        }
        memcpy(out->a + (uint64_t)(start + s) * (uint64_t)dim, emb,
               (uint64_t)dim * sizeof(float));
        if (do_norm)
          tk_llama_l2_normalize(out->a + (uint64_t)(start + s) * (uint64_t)dim, dim);
      }
    } else {
      int32_t tok_off = 0;
      for (int s = 0; s < count; s++) {
        tok_off += tok_lens[start + s];
        float *emb = llama_get_embeddings_ith(ll->ctx, tok_off - 1);
        if (!emb) {
          rc = luaL_error(L, "encode: no embeddings for text %d", start + s + 1);
          goto done;
        }
        memcpy(out->a + (uint64_t)(start + s) * (uint64_t)dim, emb,
               (uint64_t)dim * sizeof(float));
        if (do_norm)
          tk_llama_l2_normalize(out->a + (uint64_t)(start + s) * (uint64_t)dim, dim);
      }
    }

    if (!ll->has_encoder)
      llama_memory_clear(llama_get_memory(ll->ctx), true);
  }

  rc = 0;

done:
  if (batch.token) llama_batch_free(batch);
  free(tmp_buf);
  free(tok_all);
  free(tok_lens);
  if (rc != 0)
    return rc;
  if (has_output) {
    return 0;
  }
  lua_pushinteger(L, dim);
  return 2;
}

static inline int tk_llama_dims_lua (lua_State *L) {
  tk_llama_t *ll = tk_llama_peek(L, 1);
  lua_pushinteger(L, ll->n_embd);
  return 1;
}

static luaL_Reg tk_llama_mt_fns[] = {
  { "encode", tk_llama_encode_lua },
  { "dims", tk_llama_dims_lua },
  { NULL, NULL }
};

static int tk_llama_create_lua (lua_State *L) {
  const char *path = luaL_checkstring(L, 1);
  int32_t n_seq = luaL_optinteger(L, 2, TK_LLAMA_DEFAULT_N_SEQ);
  int32_t n_threads = omp_get_max_threads();
  if (tk_llama_refs == 0) {
    llama_backend_init();
    llama_log_set(tk_llama_log_noop, NULL);
  }
  struct llama_model_params mp = llama_model_default_params();
  struct llama_model *model = llama_model_load_from_file(path, mp);
  if (!model) {
    if (tk_llama_refs == 0)
      llama_backend_free();
    return luaL_error(L, "create: failed to load model '%s'", path);
  }
  int32_t n_embd = llama_model_n_embd(model);
  int32_t n_ctx = llama_model_n_ctx_train(model);
  int32_t n_total = n_ctx * n_seq;
  struct llama_context_params cp = llama_context_default_params();
  cp.embeddings = true;
  cp.n_ctx = (uint32_t)n_total;
  cp.n_batch = (uint32_t)n_total;
  cp.n_ubatch = (uint32_t)n_total;
  cp.n_seq_max = (uint32_t)n_seq;
  cp.n_threads = n_threads;
  cp.n_threads_batch = n_threads;
  struct llama_context *ctx = llama_init_from_model(model, cp);
  if (!ctx) {
    llama_model_free(model);
    if (tk_llama_refs == 0)
      llama_backend_free();
    return luaL_error(L, "create: failed to create context (n_seq=%d)", n_seq);
  }
  tk_llama_t *ll = tk_lua_newuserdata(L, tk_llama_t,
    TK_LLAMA_MT, tk_llama_mt_fns, tk_llama_gc);
  ll->model = model;
  ll->ctx = ctx;
  ll->n_embd = n_embd;
  ll->n_ctx = n_ctx;
  ll->n_seq = n_seq;
  ll->has_encoder = llama_model_has_encoder(model);
  ll->pooled = llama_pooling_type(ctx) != LLAMA_POOLING_TYPE_NONE;
  ll->destroyed = false;
  tk_llama_refs++;
  return 1;
}

static luaL_Reg tk_llama_fns[] = {
  { "create", tk_llama_create_lua },
  { NULL, NULL }
};

int luaopen_santoku_learn_llama (lua_State *L) {
  lua_newtable(L);
  tk_lua_register(L, tk_llama_fns, 0);
  return 1;
}
