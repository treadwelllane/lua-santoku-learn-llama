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
  data = { max = nil, ttr = 0.5, tvr = 0.1 },
  ridge = {
    lambda = { def = 4.20e-01 },
    propensity_a = { def = 0.62 },
    propensity_b = { def = 2.15 },
    classes = 2,
    search_trials = 400,
    k = 1,
  },
}

test("imdb classifier (llama)", function ()

  local stopwatch = utc.stopwatch()
  local function sw()
    local d, dd = stopwatch()
    return str.format("(%.1fs +%.1fs)", d, dd)
  end

  str.printf("[Data] Loading\n")
  local dataset = ds.read_imdb("test/res/imdb.50k", cfg.data.max)
  local train, test_set, validate = ds.split_imdb(dataset, cfg.data.ttr, cfg.data.tvr)
  local n_classes = cfg.ridge.classes
  local label_off, label_nbr = train.sol_offsets, train.sol_neighbors
  local val_label_off, val_label_nbr = validate.sol_offsets, validate.sol_neighbors
  str.printf("[Data] train=%d val=%d test=%d classes=%d %s\n",
    train.n, validate.n, test_set.n, n_classes, sw())

  str.printf("[Llama] Loading model\n")
  local enc = llama.create(model_path)
  local n_dims = enc:dims()
  str.printf("[Llama] n_embd=%d %s\n", n_dims, sw())

  str.printf("[Llama] Encoding train (%d texts)\n", train.n)
  local train_codes = enc:encode(train.problems)
  train.problems = nil
  str.printf("[Llama] Train encoded %s\n", sw())

  str.printf("[Llama] Encoding val (%d texts)\n", validate.n)
  local val_codes = enc:encode(validate.problems)
  validate.problems = nil
  str.printf("[Llama] Val encoded %s\n", sw())

  str.printf("[Ridge] Fitting\n")
  local ridge_obj, best_params = optimize.ridge({
    train_codes = train_codes,
    n_samples = train.n,
    n_dims = n_dims,
    label_offsets = label_off,
    label_neighbors = label_nbr,
    n_labels = n_classes,
    val_codes = val_codes,
    val_n_samples = validate.n,
    val_expected_offsets = val_label_off,
    val_expected_neighbors = val_label_nbr,
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

  str.printf("[Llama] Encoding test (%d texts)\n", test_set.n)
  local test_codes = enc:encode(test_set.problems)
  test_set.problems = nil
  local _, test_labels = ridge_obj:label(test_codes, test_set.n, 1)
  test_codes = nil -- luacheck: ignore
  str.printf("[Eval] Labels done %s\n", sw())

  local val_stats = eval.class_accuracy(val_labels, validate.sol_offsets, validate.sol_neighbors, validate.n, n_classes)
  local test_stats = eval.class_accuracy(test_labels, test_set.sol_offsets, test_set.sol_neighbors, test_set.n, n_classes)
  str.printf("[Class] F1: val=%.2f test=%.2f %s\n",
    val_stats.f1, test_stats.f1, sw())

  local _, total = stopwatch()
  str.printf("\nTotal: %.1fs\n", total)

end)
