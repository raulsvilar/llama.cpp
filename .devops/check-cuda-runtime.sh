#!/bin/sh
set -eu

app_dir=${1:-/app}
evidence_dir=${2:-/tmp/cuda-validation}
driver_stub=${3:-}
mkdir -p "$evidence_dir"

# The driver is supplied by the GPU host. Mount only its SDK stub on GPU-less builders.
if test -n "$driver_stub"; then
    test -f "$driver_stub"
    driver_dir=$(mktemp -d)
    trap 'rm -rf "$driver_dir"' EXIT HUP INT TERM
    ln -s "$driver_stub" "$driver_dir/libcuda.so.1"
    export LD_LIBRARY_PATH="$app_dir:$driver_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo 'Link inspection uses a temporary CUDA driver stub; NCCL must resolve from the image.'
else
    export LD_LIBRARY_PATH="$app_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

backend="$app_dir/libggml-cuda.so"
test -f "$backend"
ldd "$backend" > "$evidence_dir/cuda-ldd.txt" 2>&1
cat "$evidence_dir/cuda-ldd.txt"
grep -E 'libnccl\.so[^ ]* => /' "$evidence_dir/cuda-ldd.txt"
if grep -F 'not found' "$evidence_dir/cuda-ldd.txt"; then
    exit 1
fi

if test -x "$app_dir/llama-server"; then
    "$app_dir/llama-server" --version > "$evidence_dir/server-version.txt" 2>&1
    "$app_dir/llama-server" --help > "$evidence_dir/server-help.txt" 2>&1
    cat "$evidence_dir/server-version.txt"
    if test "${LLAMA_SERVER_FEATURE_CHECK:-OFF}" = ON; then
        for feature in --lazy-mode on-direct draft-mtp ngram-mod; do
            grep -F -- "$feature" "$evidence_dir/server-help.txt"
        done
        for mode in auto off on on-direct; do
            "$app_dir/llama-server" --lazy-mode "$mode" --spec-type draft-mtp,ngram-mod --parallel 1 --help > /dev/null
        done
    fi
fi
