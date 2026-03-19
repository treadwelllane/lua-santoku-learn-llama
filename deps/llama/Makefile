results.mk:
	[ ! -d llama.cpp ] && \
		git clone --depth 1 --filter=blob:none --sparse https://github.com/ggml-org/llama.cpp && \
		cd llama.cpp && \
		git sparse-checkout set --no-cone /ggml /src /include /cmake /common /vendor /tools /CMakeLists.txt
	cd llama.cpp && cmake -B build \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_POSITION_INDEPENDENT_CODE=ON \
		-DBUILD_SHARED_LIBS=OFF \
		-DLLAMA_BUILD_TESTS=OFF \
		-DLLAMA_BUILD_EXAMPLES=OFF \
		-DLLAMA_BUILD_SERVER=OFF \
		-DGGML_BLAS=ON \
		-DGGML_BLAS_VENDOR=OpenBLAS
	cd llama.cpp && cmake --build build -j$$(nproc)
	touch results.mk
