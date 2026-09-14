ARG UBUNTU_VERSION=24.04
# This needs to generally match the container host's environment.
ARG CUDA_VERSION=12.8.1
ARG GCC_VERSION=14
# Target the CUDA build image
ARG BASE_CUDA_DEV_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

ARG BASE_CUDA_RUN_CONTAINER=docker.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A

ARG NODE_VERSION=24

FROM docker.io/node:$NODE_VERSION AS web

ARG APP_VERSION

WORKDIR /app/tools/ui

COPY tools/ui/package.json tools/ui/package-lock.json ./
RUN npm ci

COPY tools/ui/ ./
RUN LLAMA_BUILD_NUMBER="$APP_VERSION" npm run build

FROM ${BASE_CUDA_DEV_CONTAINER} AS build

ARG GCC_VERSION
# CUDA architecture to build for (defaults to all supported archs)
ARG CUDA_DOCKER_ARCH=default
# Explicit ON requires NCCL; default keeps upstream's optional discovery.
ARG GGML_CUDA_NCCL=default
ARG LLAMA_SERVER_ONLY=OFF
ARG LLAMA_SERVER_FEATURE_CHECK=OFF
ARG BUILD_JOBS=0

RUN apt-get update && \
    apt-get install -y gcc-${GCC_VERSION} g++-${GCC_VERSION} build-essential cmake python3 python3-pip git libssl-dev libgomp1

ENV CC=gcc-${GCC_VERSION} CXX=g++-${GCC_VERSION} CUDAHOSTCXX=g++-${GCC_VERSION}

WORKDIR /app

COPY . .

COPY --from=web /app/tools/ui/dist tools/ui/dist

RUN set -eu; \
    mkdir -p /app/validation; \
    CMAKE_ARGS=""; \
    if [ "${GGML_CUDA_NCCL}" = "ON" ]; then sh .devops/cuda-nccl.sh prepare; fi && \
    if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
    export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    case "${GGML_CUDA_NCCL}" in \
      ON) CMAKE_ARGS="${CMAKE_ARGS} -DGGML_CUDA_NCCL=ON -DCMAKE_REQUIRE_FIND_PACKAGE_NCCL=ON" ;; \
      OFF) CMAKE_ARGS="${CMAKE_ARGS} -DGGML_CUDA_NCCL=OFF" ;; \
      default) ;; \
      *) echo "GGML_CUDA_NCCL must be default, ON or OFF" >&2; exit 1 ;; \
    esac && \
    BUILD_TARGET=all && \
    if [ "${LLAMA_SERVER_ONLY}" = "ON" ]; then \
      CMAKE_ARGS="${CMAKE_ARGS} -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_EXAMPLES=OFF"; \
      BUILD_TARGET=llama-server; \
    fi && \
    if cmake -B build -DGGML_NATIVE=OFF -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DLLAMA_BUILD_TESTS=OFF ${CMAKE_ARGS} -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined . > /app/validation/configure.log 2>&1; then \
      cat /app/validation/configure.log; \
    else \
      cat /app/validation/configure.log; exit 1; \
    fi && \
    if [ "${BUILD_JOBS}" = "0" ]; then BUILD_JOBS=$(nproc); fi && \
    cmake --build build --config Release --target "$BUILD_TARGET" -j"${BUILD_JOBS}" && \
    if [ "${LLAMA_SERVER_FEATURE_CHECK}" = "ON" ]; then \
      cmake -DSERVER=/app/build/bin/llama-server -DOUTPUT_DIR=/app/validation -P .devops/check-server-features.cmake; \
    fi

RUN mkdir -p /app/lib && \
    find build -name "*.so*" -exec cp -P {} /app/lib \;

RUN if [ "${GGML_CUDA_NCCL}" = "ON" ]; then \
      sh .devops/cuda-nccl.sh collect /app/build /app/lib /app/validation && \
      sh .devops/check-cuda-runtime.sh /app/lib /app/validation/build /usr/local/cuda/lib64/stubs/libcuda.so; \
    fi

RUN mkdir -p /app/server && cp build/bin/llama-server /app/server/ && \
    if [ "${LLAMA_SERVER_ONLY}" != "ON" ]; then cp build/bin/llama /app/server/; fi

RUN mkdir -p /app/full \
    && cp build/bin/* /app/full \
    && cp *.py /app/full \
    && cp -r conversion /app/full \
    && cp -r gguf-py /app/full \
    && cp -r requirements /app/full \
    && cp requirements.txt /app/full \
    && cp .devops/tools.sh /app/full/tools.sh

## Base image
FROM ${BASE_CUDA_RUN_CONTAINER} AS base

ARG BUILD_DATE=N/A
ARG APP_VERSION=N/A
ARG APP_REVISION=N/A
ARG IMAGE_URL=https://github.com/ggml-org/llama.cpp
ARG IMAGE_SOURCE=https://github.com/ggml-org/llama.cpp
LABEL org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.revision=$APP_REVISION \
      org.opencontainers.image.title="llama.cpp" \
      org.opencontainers.image.description="LLM inference in C/C++" \
      org.opencontainers.image.url=$IMAGE_URL \
      org.opencontainers.image.source=$IMAGE_SOURCE

RUN apt-get update \
    && apt-get install -y libgomp1 curl ffmpeg \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete

COPY --from=build /app/lib/ /app

### Full
FROM base AS full

COPY --from=build /app/full /app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-wheel \
    && pip install --break-system-packages --upgrade setuptools \
    && pip install --break-system-packages -r requirements.txt \
    && apt autoremove -y \
    && apt clean -y \
    && rm -rf /tmp/* /var/tmp/* \
    && find /var/cache/apt/archives /var/lib/apt/lists -not -name lock -type f -delete \
    && find /var/cache -type f -delete


ENTRYPOINT ["/app/tools.sh"]

### Light, CLI only
FROM base AS light

COPY --from=build /app/full/llama /app/full/llama-cli /app/full/llama-completion /app

WORKDIR /app

ENTRYPOINT [ "/app/llama-cli" ]

### Server, Server only
FROM base AS server

ARG GGML_CUDA_NCCL=default
ARG LLAMA_SERVER_FEATURE_CHECK=OFF
ARG CUDA_VERSION
ARG CUDA_DOCKER_ARCH=default
LABEL ai.llama.cpp.cuda.version=$CUDA_VERSION \
      ai.llama.cpp.cuda.nccl=$GGML_CUDA_NCCL \
      ai.llama.cpp.cuda.architectures=$CUDA_DOCKER_ARCH

ENV LLAMA_ARG_HOST=0.0.0.0

COPY --from=build /app/server/ /app
COPY --from=build /app/validation/ /app/validation/

RUN --mount=type=bind,from=build,source=/app/.devops,target=/tmp/checks \
    --mount=type=bind,from=build,source=/usr/local/cuda/lib64/stubs,target=/tmp/cuda-stubs \
    if [ "${GGML_CUDA_NCCL}" = "ON" ]; then \
      sh /tmp/checks/check-cuda-runtime.sh /app /app/validation/runtime /tmp/cuda-stubs/libcuda.so; \
    elif [ "${LLAMA_SERVER_FEATURE_CHECK}" = "ON" ]; then \
      /app/llama-server --help > /app/validation/server-help.txt 2>&1 && \
      for feature in --lazy-mode on-direct draft-mtp ngram-mod; do \
        grep -F -- "$feature" /app/validation/server-help.txt || exit 1; \
      done; \
    fi

WORKDIR /app

HEALTHCHECK CMD [ "curl", "-f", "http://localhost:8080/health" ]

ENTRYPOINT [ "/app/llama-server" ]
