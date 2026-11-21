# Inkling 2 Mini: faster single-GPU inference

This branch (`inkling-relfa`) speeds up [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
in llama.cpp for one user on one consumer GPU. It builds on two llama.cpp pull requests that were
not yet merged when it was written:

- [#32695](https://github.com/ggml-org/llama.cpp/pull/32695): the looped Inkling 2 architecture (Unsloth GGUFs)
- [#32218](https://github.com/ggml-org/llama.cpp/pull/32218): DSpark speculative decoding (Inkling's MTP head)

Credit for those goes to their authors. The commits added here are the five on top of them:

| Commit | Change |
|---|---|
| `inkling2: evaluate the relative-position bias inside flash attention` | `ggml_flash_attn_ext_add_rel`: the bias table is looked up inside the CUDA vec/MMA FA kernels, so the full bias matrix is never built. Removes the `-ub 128` limit and the VRAM it cost. |
| `inkling2: fuse clip, depth mix and loop control into GGML_OP_INKLING` | One op with CPU reference and CUDA kernels replaces ~30 small ops per layer pass. |
| `inkling2: fused short convolution (GGML_INKLING_SCONV)` | Fused shortconv and dynamic taps. |
| `inkling2: fused rel table and SiTU; topk-moe fusion ...` | Fused relative-bias table, expert SiTU activation, and MoE routing. |
| `dspark: cap the inkling2 markov loop (LLAMA_DSPARK_STEPS) ...` | The draft head no longer re-runs the 201K-vocab output layer for draft positions that are never used. |

The decode graph drops from 6,830 to about 2,700 kernels per token. All new ops have
`test-backend-ops` cases (CUDA vs CPU).

## Results

RTX 4090 (24 GB), `Inkling-2-Mini-UD-Q4_K_XL.gguf`, q8_0 KV cache, one request at a time.

| | Before | This branch |
|---|---:|---:|
| Generation, no draft | 72 tok/s | **106 tok/s** |
| With DSpark draft (depth 3): code / bash | ~90 / ~85 | **~135 / ~130** |
| With DSpark draft: prose | ~56 | **~88** |
| Prompt processing, 2K-token prompt | 1,494 tok/s | **5,430 tok/s** |
| Context that fits in 24 GB | 98K | **131K** |

Quality vs the unfused graph: mean KL divergence ~0.003, same top token ~98%. On prose the
draft is still slower than plain generation, so turn it off for prose-heavy work.

## Build

```sh
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server llama-bench test-backend-ops
```

## Run

Models: [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (UD-Q4_K_XL) and a
DSpark draft GGUF converted from the official `mtp.safetensors`.

```sh
export LLAMA_DSPARK_STEPS=3
build/bin/llama-server -m Inkling-2-Mini-UD-Q4_K_XL.gguf \
  -ngl 999 -fa 1 -c 131072 -ctk q8_0 -ctv q8_0 -ub 512 -np 1 --jinja --no-mmproj \
  -md dspark-Inkling-2-Mini-Q8_0.gguf --spec-draft-ngl all \
  --spec-type draft-dspark --spec-draft-n-max 3 --spec-draft-n-min 1
```

Drop the four draft lines (`-md` onward) for prose.

## Switches

| Variable | Effect |
|---|---|
| `LLAMA_INKLING_UNFUSED=1` | Use the original unfused graph (for comparison). |
| `LLAMA_DSPARK_STEPS=n` | Run the draft head for the first `n` positions only; set it to `--spec-draft-n-max`. |
| `LLAMA_SPEC_ADAPTIVE=x` | Pause drafting while accepted tokens per draft average below `x`. Off by default; on Inkling it costs more on code than it gains on prose. |
| `GGML_CUDA_OPPROF=1` | Per-op CUDA event profiler for the eager path (`=2` prints one graph's op sequence). |

The new ops are implemented for CUDA plus a CPU reference only, and tested only on an RTX 4090.
Other backends (Metal, Vulkan, ROCm) are untested; try `LLAMA_INKLING_UNFUSED=1` there.
