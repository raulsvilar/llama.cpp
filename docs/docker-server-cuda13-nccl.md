# Qwen3.8 MTP + direct PLE: CUDA 13 / NCCL integration

This personal-fork branch targets linux/amd64 and Ampere RTX 3060 GPUs. All three published tags, including `server-cuda13`, contain CUDA code for **sm_86 only** (`CMAKE_CUDA_ARCHITECTURES=86-real`, no PTX for other architectures). Use `--parallel 1` for the initial Qwen3.8 MTP deployment.

## Source provenance

Snapshot inspected on 2026-09-13:

| Source | Ref | Full SHA |
| --- | --- | --- |
| Upstream master | `ggml-org/llama.cpp:master` | `ad6c66839af3c5646fba8c6c2e2087a1e4e38948` |
| [MTP PR #28243](https://github.com/ggml-org/llama.cpp/pull/28243) | `refs/pull/28243/head`, `danielhanchen:qwen4exp/mtp` | `d1a92352cbd417fd840b4e765c0b82f5fe3d1d89` |
| [Direct PLE PR #28136](https://github.com/ggml-org/llama.cpp/pull/28136) | `refs/pull/28136/head`, `coder543:master` | `c6a9e5c9ae6d6a551217f75c9a04b2e8b1aa62dd` |

The initial working tree was clean on `master`. `origin` was and remains `https://github.com/raulsvilar/llama.cpp.git`. The added `upstream` remote points to `https://github.com/ggml-org/llama.cpp.git`. After fetching both, local master, origin/master and upstream/master were identical. Updating master was therefore a no-op; no reset, stash, history rewrite or force-push was needed.

Branch `qwen4exp-mtp-ondirect` starts at that master. The merge bases were `95ef7fc16054e63b427a3ef00188e055ef7586d8` for MTP and `67a17c17caa95742186f8b1ecadd1b5abd6d5ebb` for on-direct. Master had 159 and 175 newer commits respectively. Directly replacing master files with either PR's tree would lose those changes.

`git cherry master <pr-ref>` marked all 11 MTP commits and both on-direct commits as new patches. Graph, commit and feature-diff inspection found no already-absorbed feature delta to omit. The existing generic `draft-mtp`, lazy mmap and `ngram-mod` support was retained. PR #27836 was not applied separately: #28243 already carries its Qwen4Exp NextN tensor registration, model graph and converter work.

MTP commits retained, in order:

| Commit | Scope |
| --- | --- |
| `c8c3a5bfe` | GGUF NextN tensor registration |
| `57672d973` | Qwen4Exp NextN/MTP graph and memory routing |
| `84f955885` | MTP converter/export |
| `86d321a29` | Initial shared embedding/head support |
| `30b65375a` | Detached draft loading |
| `44c79602c` | Comment cleanup |
| `eb65412fc` | Shared-embedding mixin declaration |
| `44ff80323` | Reject detached draft used as a target |
| `cc2c59a74` | Replace initial sharing API with `ctx_other` |
| `2c967293c` | Comment cleanup |
| `d1a92352c` | Only Gemma4 assistant shares target KV memory |

On-direct commits retained: `90fde1f7f` introduces direct PLE reads; `c6a9e5c9a` shares the reader and adds Gemma4 support. Merge commits `98b8008c4` and `36eeb8a52` preserve the two complete histories without squash.

## Merge resolution

MTP merged without textual conflicts. The only on-direct conflict was in `tools/llama-bench/llama-bench.cpp`, in `print_usage()`. Master commit `14a9d09f7` removed obsolete `--mmap` and `--direct-io` help entries. The resolved version keeps their removal and adds `on-direct` to the current `--lazy-mode` list.

The automatically combined code was also reviewed:

- Qwen4Exp retains master's `build_gdn_l2_norm` correction, QSA and recurrent rollback infrastructure.
- Speculative decoding retains master's `pos0` position handling and draft device placement changes. MTP borrowing through `ctx_other` does not make Qwen4Exp share its target's KV cache.
- The MTP graph supports embedded, detached and shared embedding/head exports. The normal trunk graph remains selected outside MTP mode.
- PLE remains `TENSOR_READ_LAZY`. An available reader stages F32 rows under `on-direct`; otherwise the graph uses the existing `ggml_get_rows` path. `on`, `off` and `auto` retain their meanings, including master's AUTO fallback on devices without mmap support.
- Gemma4 retains master's fused QKV support, per-layer expert metadata and SWA non-causal behavior.
- No CUDA kernels, `ngram-mod` implementation or server scheduler/concurrency code were changed by this integration.

The direct reader uses buffered `pread()` with sorted and deduplicated row requests. `on-direct` does not mean Linux `O_DIRECT` or complete bypass of the OS page cache. Windows uses the PR's mmap fallback; actual direct-read validation requires Linux.

## Docker and NCCL checks

The existing `.devops/cuda.Dockerfile` is extended. Its default build still builds the upstream targets and performs optional NCCL discovery. New opt-in build arguments are:

| Argument | Default | This workflow |
| --- | --- | --- |
| `CUDA_VERSION` | `12.8.1` | `13.3.0`, matching upstream's CUDA13 workflow |
| `CUDA_DOCKER_ARCH` | `default` | `86-real` |
| `GGML_CUDA_NCCL` | `default` | `ON`, with required discovery |
| `GGML_CUDA_FA_QUANTS` | `default` (upstream selection) | `all`, every supported FlashAttention K/V type combination |
| `LLAMA_SERVER_ONLY` | `OFF` | `ON`, build only `llama-server` and its dependencies |
| `LLAMA_SERVER_FEATURE_CHECK` | `OFF` | `ON` |
| `BUILD_JOBS` | `0` (nproc) | `2`, to bound runner memory use |

`GGML_NATIVE=OFF`, `GGML_CUDA=ON` and dynamic backends are retained. Building only sm_86 avoids generating all upstream CUDA13 architectures. No build-time speedup measurement is claimed.

The Action passes `-DGGML_CUDA_FA_QUANTS=all`, the current replacement for the deprecated `GGML_CUDA_FA_ALL_QUANTS=ON`. This compiles all FlashAttention K/V combinations of f16, bf16, q4_0, q4_1, q5_0, q5_1 and q8_0, at the cost of a longer build. Configure records `GGML_CUDA_FA=ON` and the exact selection in `/app/validation/cmake-fa-quants.txt`; the final-image smoke test requires both settings. This option selects FlashAttention kernels, not the GGUF model weight format or the runtime KV cache types.

When NCCL is explicitly ON:

1. `.devops/cuda-nccl.sh prepare` looks for existing `nccl.h` and a usable `libnccl.so` before considering apt. Missing packages are installed at the version of the already installed partner, or `NV_NCCL_PACKAGE_VERSION` supplied by the NVIDIA image. Already installed packages are not requested for upgrade. Inconsistent/incomplete packages fail explicitly; held packages are not forced.
2. Configure passes `-DGGML_CUDA_NCCL=ON -DCMAKE_REQUIRE_FIND_PACKAGE_NCCL=ON`. The latter makes the existing `find_package(NCCL)` call required without changing ggml's global default. The upstream finder must locate both its include directory and library. See [CMake's required-package option](https://cmake.org/cmake/help/latest/variable/CMAKE_REQUIRE_FIND_PACKAGE_PackageName.html).
3. The build checks `Found NCCL:`, rejects missing-NCCL messages, validates CMakeCache paths and uses `readelf -d /app/lib/libggml-cuda.so` to require a real `DT_NEEDED` entry for `libnccl.so.*`. A static archive or unused flag cannot pass this check.
4. The exact discovered shared library is copied under its ELF SONAME into `/app/lib` for packaging. The final image receives it at `/app/libnccl.so.*`. Its SHA256 is recorded and verified again in the loaded final image. No runtime NCCL upgrade is required.
5. `.devops/check-cuda-runtime.sh` runs `ldd` on the collected build backend and again in the final `server` filesystem, requiring a resolved NCCL dependency and rejecting every `not found` entry.
6. The server version, help and parsing of `draft-mtp,ngram-mod` with each lazy mode are checked. These tests need neither a GPU nor model weights.

The driver library `libcuda.so.1` is normally injected by NVIDIA Container Toolkit on the GPU host. A GPU-less builder has no driver. For the structural `ldd` checks, BuildKit mounts only the SDK driver stubs temporarily, and the script creates a temporary SONAME symlink. **The driver stub is not copied into the runtime image. NCCL always resolves from the real library packaged in the image.** This preserves CUDA VMM support rather than disabling it to make a GPU-less test pass.

The loaded image is then smoke-tested again without that mount. Version/help commands can complete without a usable CUDA backend. Actual inference, driver compatibility, peer access, collective communication and MTP acceptance still require the target machine.

## Action and tags

`.github/workflows/server-cuda13-nccl.yml` contains one job, one Buildx build, target `server`, `linux/amd64`, CUDA 13.3.0 and NCCL ON. No CPU/ARM/full/light matrix, upstream workflow dependency or manifest-merging job is used. The CPU backend libraries remain available inside the CUDA image for normal offloading/fallback behavior.

The image is loaded into the runner, tested, and then that same job pushes it directly to `ghcr.io/raulsvilar/llama.cpp` using `GITHUB_TOKEN` with `packages: write`. Tags are:

- `server-cuda13-sm86-<12-char-sha>-<run-id>-<attempt>`: unique to this source commit and build execution; retries never overwrite a previous execution's tag.
- `server-cuda13-sm86`: latest successful experimental image.
- `server-cuda13`: alias of that same sm_86-only image on this fork.

OCI labels include source repository, full revision and build date. Additional labels `ai.llama.cpp.cuda.version`, `ai.llama.cpp.cuda.nccl` and `ai.llama.cpp.cuda.architectures` identify CUDA/NCCL/architectures; the OCI description explicitly says sm_86 only. Use the published image digest for content-addressed reproducibility. NVIDIA base tags can change over time, so a later build of the same source need not have the same digest.

Push the integration branch to trigger the initial build:

```sh
git push -u origin qwen4exp-mtp-ondirect
gh run list --repo raulsvilar/llama.cpp --branch qwen4exp-mtp-ondirect --workflow server-cuda13-nccl.yml --limit 5
gh run watch RUN_ID --repo raulsvilar/llama.cpp --exit-status
```

For manual dispatch, once the workflow is available on the fork's default branch:

```sh
gh workflow run server-cuda13-nccl.yml --repo raulsvilar/llama.cpp --ref qwen4exp-mtp-ondirect
```

GitHub requires the workflow to exist on the default branch for `workflow_dispatch`; the push trigger is the bootstrap path without changing master. See [manual workflow requirements](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow). A failed run can also be retried with `gh run rerun RUN_ID --repo raulsvilar/llama.cpp --failed`.

Fetch the actual validation evidence after a successful run (replace the placeholders):

```sh
gh run view RUN_ID --repo raulsvilar/llama.cpp --log
gh run download RUN_ID --repo raulsvilar/llama.cpp --name server-cuda13-sm86-SHORT_SHA-validation --dir validation
```

The artifact and `/app/validation` contain `configure.log`, `cmake-nccl.txt`, `cuda-needed.txt`, `nccl-sha256.txt`, `build/cuda-ldd.txt`, `runtime/cuda-ldd.txt` and server help/version outputs. A passing run must contain `Found NCCL:`, an ELF `NEEDED` entry for NCCL and a runtime resolution such as `libnccl.so.2 => /app/libnccl.so.2`, with no `not found`. These are acceptance criteria, not results measured locally.

## Pull and smoke-test

```sh
docker pull ghcr.io/raulsvilar/llama.cpp:server-cuda13-sm86
docker run --rm ghcr.io/raulsvilar/llama.cpp:server-cuda13-sm86 --version
docker run --rm --entrypoint /bin/sh ghcr.io/raulsvilar/llama.cpp:server-cuda13-sm86 -ec '
  /app/llama-server --help > /tmp/help.txt 2>&1
  for feature in on-direct draft-mtp ngram-mod; do
    grep -F -- "$feature" /tmp/help.txt
  done
'
```

For a newly published private GHCR package, authenticate with `docker login ghcr.io` or configure the package visibility/access in GitHub. See [GHCR access and first publication](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).

On the NVIDIA host, verify all dynamic dependencies with the real driver:

```sh
docker run --rm --gpus all --entrypoint /bin/sh ghcr.io/raulsvilar/llama.cpp:server-cuda13-sm86 -ec '
  export LD_LIBRARY_PATH=/app${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  ldd /app/libggml-cuda.so > /tmp/ldd.txt
  cat /tmp/ldd.txt
  grep -E "libnccl\.so[^ ]* => /app/" /tmp/ldd.txt
  if grep -F "not found" /tmp/ldd.txt; then exit 1; fi
'
```

Example initial model launch, with GGUF files on the host NVMe:

```sh
docker run --rm --gpus all -p 8080:8080 -v /nvme/models:/models:ro \
  ghcr.io/raulsvilar/llama.cpp:server-cuda13-sm86 \
  -m /models/target.gguf -md /models/mtp-shared.gguf \
  --lazy-mode on-direct --spec-type draft-mtp,ngram-mod --parallel 1
```

## Validation scope

Only source/history inspection, diff whitespace checks, YAML parsing and shell syntax checks were completed locally. A Windows CPU build had been attempted before the instruction to use Actions only; it did not produce a validated server. Local compilation was then stopped and is not an acceptance result.

The CUDA build, NCCL discovery, ELF dependency checks and final-image smoke tests must pass in the Action before treating the image as validated. No model weights or GPU inference tests were run locally. Existing MTP limitations, including multiple-slot behavior, are outside this integration's scope. The CI help/parser checks establish option availability, not numerical equivalence of model output.
