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

local snli_dir = env.var("SNLI_DIR", nil)
if not snli_dir then
  print("SNLI_DIR not set. Skipping.")
  return
end

local llama = require("santoku.learn.llama")

local cfg = {
  data = { max = nil, tvr = 0.1 },
  ridge = {
    lambda = { def = 1.9316e-02 },
    propensity_a = { def = 3.5412 },
    propensity_b = { def = 6.6018 },
    search_trials = 400,
    k = 1,
  },
}

test("snli classifier (llama)", function ()

  local stopwatch = utc.stopwatch()
  local n_classes = 3
  local function sw()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  local function build_pair_texts (split)
    local texts = {}
    for i = 1, split.n do
      local a_idx = split.idx1:get(i - 1)
      local b_idx = split.idx2:get(i - 1)
      texts[i] = split.unique_texts[a_idx + 1] .. "\n" .. split.unique_texts[b_idx + 1]
    end
    return texts
  end

  str.printf("[Data] Loading\n")
  local train, dev, test_set, validate = ds.read_snli(snli_dir, cfg.data.max, cfg.data.tvr)
  str.printf("[Data] train=%d val=%d dev=%d test=%d %s\n",
    train.n, validate.n, dev.n, test_set.n, sw())

  str.printf("[Llama] Loading model\n")
  local enc = llama.create(model_path)
  local n_dims = enc:dims()
  str.printf("[Llama] n_embd=%d %s\n", n_dims, sw())

  str.printf("[Llama] Encoding train (%d pairs)\n", train.n)
  local train_codes = enc:encode(build_pair_texts(train))
  str.printf("[Llama] Train encoded %s\n", sw())

  str.printf("[Llama] Encoding val (%d pairs)\n", validate.n)
  local val_codes = enc:encode(build_pair_texts(validate))
  str.printf("[Llama] Val encoded %s\n", sw())

  str.printf("[Ridge] Fitting\n")
  local ridge_obj, best_params = optimize.ridge({
    train_codes = train_codes,
    n_samples = train.n,
    n_dims = n_dims,
    label_offsets = train.sol_offsets,
    label_neighbors = train.sol_neighbors,
    n_labels = n_classes,
    val_codes = val_codes,
    val_n_samples = validate.n,
    val_expected_offsets = validate.sol_offsets,
    val_expected_neighbors = validate.sol_neighbors,
    lambda = cfg.ridge.lambda,
    propensity_a = cfg.ridge.propensity_a,
    propensity_b = cfg.ridge.propensity_b,
    k = cfg.ridge.k,
    search_trials = cfg.ridge.search_trials,
    each = util.make_ridge_log(stopwatch),
  })
  train_codes = nil -- luacheck: ignore
  collectgarbage("collect")
  str.printf("[Ridge] lambda=%.4e pa=%.4f pb=%.4f %s\n",
    best_params.lambda, best_params.propensity_a, best_params.propensity_b, sw())

  str.printf("[Eval] Labeling splits\n")
  local _, val_labels = ridge_obj:label(val_codes, validate.n, 1)
  val_codes = nil -- luacheck: ignore

  local val_stats = eval.class_accuracy(val_labels, validate.sol_offsets, validate.sol_neighbors, validate.n, n_classes)
  str.printf("[Val] F1=%.4f precision=%.4f recall=%.4f %s\n",
    val_stats.f1, val_stats.precision, val_stats.recall, sw())

  local function eval_split (name, split)
    str.printf("[Llama] Encoding %s (%d pairs)\n", name, split.n)
    local codes = enc:encode(build_pair_texts(split))
    str.printf("[Llama] %s encoded %s\n", name, sw())
    local _, labels = ridge_obj:label(codes, split.n, 1)
    local stats = eval.class_accuracy(labels, split.sol_offsets, split.sol_neighbors, split.n, n_classes)
    str.printf("[%s] F1=%.4f precision=%.4f recall=%.4f %s\n",
      name, stats.f1, stats.precision, stats.recall, sw())
    return stats
  end

  eval_split("Dev", dev)
  eval_split("Test", test_set)

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
