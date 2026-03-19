local env = {
  name = "santoku-learn-llama",
  version = "0.0.1-1",
  license = "MIT",
  public = true,
  dependencies = {
    "lua >= 5.1",
    "santoku >= 0.0.321-1",
    "santoku-matrix >= 0.0.293-1",
  },
  cflags = {
    "-std=gnu11", "-D_GNU_SOURCE", "-Wall", "-Wextra",
    "-Wno-unused-parameter", "-fopenmp",
    "-I$(shell luarocks show santoku --rock-dir)/include/",
    "-I$(shell luarocks show santoku-matrix --rock-dir)/include/",
    "-I$(PWD)/deps/llama/llama.cpp/include",
    "-I$(PWD)/deps/llama/llama.cpp/ggml/include",
  },
  ldflags = {
    "-Wl,--start-group",
    "$(shell find $(PWD)/deps/llama/llama.cpp/build -name '*.a' | tr '\\n' ' ')",
    "-Wl,--end-group",
    "-lstdc++", "-lm", "-fopenmp",
    "$(shell pkg-config --libs blas lapack lapacke)",
  },
  test = {
    dependencies = {
      "santoku-learn >= 0.0.9-1",
      "santoku-fs >= 0.0.41-1",
      "lua-cjson >= 2.1.0.10-1",
    }
  }
}

env.homepage = "https://github.com/treadwelllane/lua-" .. env.name
env.tarball = env.name .. "-" .. env.version .. ".tar.gz"
env.download = env.homepage .. "/releases/download/" .. env.version .. "/" .. env.tarball

return { env = env }
