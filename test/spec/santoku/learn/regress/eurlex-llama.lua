local env = require("santoku.env")
local ds = require("santoku.learn.dataset")
local eval = require("santoku.learn.evaluator")
local optimize = require("santoku.learn.optimize")
local str = require("santoku.string")
local test = require("santoku.test")
local util = require("santoku.learn.util")
local utc = require("santoku.utc")

io.stdout:setvbuf("line")

local model_path = env.var("LLAMA_MODEL", nil)
if not model_path then
  print("LLAMA_MODEL not set. Skipping.")
  return
end

local llama = require("santoku.learn.llama")

local cfg = {
  data = { max = nil },
  emb = { k = 256 },
  ridge = {
    lambda = { def = 2.73e-02 },
    propensity_a = { def = 0.04 },
    propensity_b = { def = 6.31 },
    search_trials = 400,
  },
}

test("eurlex classifier (llama)", function ()

  local stopwatch = utc.stopwatch()
  local function sw()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  local function fmt_metrics(m)
    return str.format("miP=%.4f miR=%.4f miF1=%.4f",
      m.micro_precision, m.micro_recall, m.micro_f1)
  end

  str.printf("[Data] Loading\n")
  local train, dev, test_set = ds.read_eurlex57k("test/res/eurlex57k", cfg.data.max)
  local n_labels = train.n_labels
  local k = cfg.emb.k or n_labels
  str.printf("[Data] train=%d dev=%d test=%d labels=%d %s\n", train.n, dev.n, test_set.n, n_labels, sw())

  local train_label_off, train_label_nbr = train.sol_offsets, train.sol_neighbors
  local dev_label_off, dev_label_nbr = dev.sol_offsets, dev.sol_neighbors
  local test_label_off, test_label_nbr = test_set.sol_offsets, test_set.sol_neighbors

  local function collect_texts(text_iter_fn, n)
    local texts = {}
    local iter = text_iter_fn()
    for i = 1, n do texts[i] = iter() end
    return texts
  end

  str.printf("[Llama] Loading model\n")
  local enc = llama.create(model_path)
  local n_dims = enc:dims()
  str.printf("[Llama] n_embd=%d %s\n", n_dims, sw())

  str.printf("[Llama] Encoding train (%d texts)\n", train.n)
  local train_texts = collect_texts(train.text_iter, train.n)
  local train_codes = enc:encode(train_texts)
  train_texts = nil -- luacheck: ignore
  str.printf("[Llama] Train encoded %s\n", sw())

  str.printf("[Llama] Encoding dev (%d texts)\n", dev.n)
  local dev_texts = collect_texts(dev.text_iter, dev.n)
  local dev_codes = enc:encode(dev_texts)
  dev_texts = nil -- luacheck: ignore
  str.printf("[Llama] Dev encoded %s\n", sw())

  str.printf("[Ridge] Fitting\n")
  local ridge_obj, best_params, gfm_obj = optimize.ridge({
    train_codes = train_codes,
    n_samples = train.n,
    n_dims = n_dims,
    label_offsets = train_label_off,
    label_neighbors = train_label_nbr,
    n_labels = n_labels,
    val_codes = dev_codes,
    val_n_samples = dev.n,
    val_expected_offsets = dev_label_off,
    val_expected_neighbors = dev_label_nbr,
    lambda = cfg.ridge.lambda,
    propensity_a = cfg.ridge.propensity_a,
    propensity_b = cfg.ridge.propensity_b,
    k = k,
    search_trials = cfg.ridge.search_trials,
    gfm = true,
    each = util.make_ridge_log(stopwatch, function (m)
      if m.gfm_f1 then return "oracle: " .. fmt_metrics(m.oracle) end
      if m.oracle then return fmt_metrics(m.oracle) end
      return ""
    end),
  })
  train_codes = nil -- luacheck: ignore
  collectgarbage("collect")
  str.printf("[Ridge] lambda=%.4e pa=%.4f pb=%.4f %s\n",
    best_params.lambda, best_params.propensity_a, best_params.propensity_b, sw())

  local dv_off, dv_nbr, _ = ridge_obj:label(dev_codes, dev.n, k)
  local _, dv_oracle = eval.retrieval_ks({
    pred_offsets = dv_off, pred_neighbors = dv_nbr,
    expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr,
  })
  str.printf("[Dv Oracle] %s %s\n", fmt_metrics(dv_oracle), sw())

  dev_codes = nil -- luacheck: ignore
  collectgarbage("collect")

  str.printf("[Llama] Encoding test (%d texts)\n", test_set.n)
  local test_texts = collect_texts(test_set.text_iter, test_set.n)
  local test_codes = enc:encode(test_texts)
  test_texts = nil -- luacheck: ignore
  str.printf("[Llama] Test encoded %s\n", sw())

  local ts_off, ts_nbr, ts_sco = ridge_obj:label(test_codes, test_set.n, k)
  test_codes = nil; ridge_obj:shrink() -- luacheck: ignore
  collectgarbage("collect")

  local _, ts_oracle = eval.retrieval_ks({
    pred_offsets = ts_off, pred_neighbors = ts_nbr,
    expected_offsets = test_label_off, expected_neighbors = test_label_nbr,
  })
  str.printf("[Ts Oracle] %s %s\n", fmt_metrics(ts_oracle), sw())

  local ts_ks = gfm_obj:predict({ offsets = ts_off, neighbors = ts_nbr, scores = ts_sco, n_samples = test_set.n })
  local _, ts_pred_m = eval.retrieval_ks({
    pred_offsets = ts_off, pred_neighbors = ts_nbr,
    expected_offsets = test_label_off, expected_neighbors = test_label_nbr,
    ks = ts_ks,
  })
  str.printf("[Ts Pred] %s %s\n", fmt_metrics(ts_pred_m), sw())

  local dv_rp = eval.rp_at_k({
    pred_offsets = dv_off, pred_neighbors = dv_nbr,
    expected_offsets = dev_label_off, expected_neighbors = dev_label_nbr,
    max_k = 8,
  })
  local ts_rp = eval.rp_at_k({
    pred_offsets = ts_off, pred_neighbors = ts_nbr,
    expected_offsets = test_label_off, expected_neighbors = test_label_nbr,
    max_k = 8,
  })

  str.printf("\nSummary\n")
  str.printf("  %-10s lambda=%.4e pa=%.4f pb=%.4f\n",
    "Params", best_params.lambda, best_params.propensity_a, best_params.propensity_b)
  str.printf("  %-10s %s\n", "Dv Oracle", fmt_metrics(dv_oracle))
  str.printf("  %-10s %s\n", "Ts Oracle", fmt_metrics(ts_oracle))
  str.printf("  %-10s %s\n", "Ts Pred", fmt_metrics(ts_pred_m))
  str.printf("\n  RP@K       ")
  for i = 1, 8 do str.printf("%-7d", i) end
  str.printf("\n  %-10s ", "Dev")
  for i = 1, 8 do str.printf("%-7.4f", dv_rp[i]) end
  str.printf("\n  %-10s ", "Test")
  for i = 1, 8 do str.printf("%-7.4f", ts_rp[i]) end
  str.printf("\n")
  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
