#!/bin/sh
set -eu

nccl_files_present() {
    header=$(find /usr/include /usr/local/cuda/include -name nccl.h -print -quit)
    library=$(find /usr/lib /usr/local/cuda/lib64 -name libnccl.so -print -quit)
    test -n "$header" && test -f "$header" && test -n "$library" && test -f "$library"
}

installed_version() {
    dpkg-query -W -f='${db:Status-Status} ${Version}\n' "$1" 2>/dev/null | awk '$1 == "installed" { print $2 }'
}

case "${1:-}" in
    prepare)
        if nccl_files_present; then
            printf 'Using existing NCCL: %s, %s\n' "$header" "$library"
            exit 0
        fi

        runtime_version=$(installed_version libnccl2)
        dev_version=$(installed_version libnccl-dev)
        version=${runtime_version:-${dev_version:-${NV_NCCL_PACKAGE_VERSION:-}}}
        test -n "$version" || { echo 'No NCCL version supplied by the CUDA image' >&2; exit 1; }
        if test -n "$runtime_version" && test -n "$dev_version" && test "$runtime_version" != "$dev_version"; then
            echo 'Installed NCCL runtime/development versions differ' >&2
            exit 1
        fi

        set --
        test -n "$runtime_version" || set -- "$@" "libnccl2=$version"
        test -n "$dev_version" || set -- "$@" "libnccl-dev=$version"
        if test "$#" -gt 0; then
            apt-get update
            apt-get install -y --no-upgrade --no-install-recommends "$@"
            ldconfig
        fi
        nccl_files_present || { echo 'NCCL packages do not provide nccl.h and libnccl.so' >&2; exit 1; }
        printf 'Using NCCL: %s, %s\n' "$header" "$library"
        ;;
    collect)
        build_dir=$2
        lib_dir=$3
        evidence_dir=$4
        cache="$build_dir/CMakeCache.txt"
        grep -Fx 'GGML_CUDA:BOOL=ON' "$cache"
        grep -Fx 'GGML_CUDA_NCCL:BOOL=ON' "$cache"
        grep -E '^CMAKE_REQUIRE_FIND_PACKAGE_NCCL:[^=]+=ON$' "$cache"
        grep -E 'Found NCCL:' "$evidence_dir/configure.log"
        if grep -Ei 'NCCL.*not found|Could NOT find NCCL' "$evidence_dir/configure.log"; then
            exit 1
        fi
        include_dir=$(sed -n 's/^NCCL_INCLUDE_DIR:PATH=//p' "$cache")
        library=$(sed -n 's/^NCCL_LIBRARY:FILEPATH=//p' "$cache")
        test -f "$include_dir/nccl.h"
        test -f "$library"
        backend="$lib_dir/libggml-cuda.so"
        test -f "$backend"
        readelf -d "$backend" > "$evidence_dir/cuda-needed.txt"
        grep -E '\(NEEDED\).*\[libnccl\.so(\.[0-9]+)*\]' "$evidence_dir/cuda-needed.txt"
        soname=$(readelf -d "$library" | sed -n 's/.*(SONAME).*\[\(libnccl\.so[^]]*\)\].*/\1/p')
        test -n "$soname"
        cp -L "$library" "$lib_dir/$soname"
        grep -E '^(GGML_CUDA|GGML_CUDA_NCCL|CMAKE_CUDA_ARCHITECTURES|CMAKE_REQUIRE_FIND_PACKAGE_NCCL|NCCL_INCLUDE_DIR|NCCL_LIBRARY):' "$cache" > "$evidence_dir/cmake-nccl.txt"
        (cd "$lib_dir" && sha256sum "$soname") > "$evidence_dir/nccl-sha256.txt"
        ;;
    *)
        echo "Usage: $0 prepare | collect BUILD_DIR LIB_DIR EVIDENCE_DIR" >&2
        exit 1
        ;;
esac
