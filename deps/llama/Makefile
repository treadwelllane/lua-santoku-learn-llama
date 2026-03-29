STUB := $(CURDIR)/stub.pl

results.mk:
	[ ! -d llama.cpp ] && \
		git clone --depth 1 --filter=blob:none --sparse https://github.com/ggml-org/llama.cpp && \
		cd llama.cpp && \
		git sparse-checkout set --no-cone /ggml /src /include /cmake /CMakeLists.txt
	cd llama.cpp/src/models && \
		for f in *.cpp; do \
			case "$$f" in bert.cpp) ;; \
				*) perl $(STUB) "$$f" ;; \
			esac; \
		done
	cd llama.cpp/src && \
		for f in llama-chat.cpp llama-grammar.cpp llama-sampler.cpp llama-quant.cpp llama-model-saver.cpp llama-adapter.cpp; do \
			perl $(STUB) "$$f"; \
		done
	cd llama.cpp && cmake -B build \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DBUILD_SHARED_LIBS=OFF \
		-DLLAMA_BUILD_TESTS=OFF \
		-DLLAMA_BUILD_EXAMPLES=OFF \
		-DLLAMA_BUILD_SERVER=OFF \
		-DLLAMA_BUILD_COMMON=OFF \
		-DLLAMA_BUILD_TOOLS=OFF \
		-DGGML_NATIVE=ON \
		-DGGML_OPENMP=ON \
		-DGGML_BLAS=ON \
		-DGGML_BLAS_VENDOR=OpenBLAS \
		-DGGML_CUDA=OFF \
		-DGGML_METAL=OFF \
		-DGGML_VULKAN=OFF \
		-DGGML_OPENCL=OFF \
		-DGGML_SYCL=OFF \
		-DGGML_RPC=OFF \
		-DGGML_LLAMAFILE=OFF
	cd llama.cpp && cmake --build build -j$$(nproc)
	touch results.mk
