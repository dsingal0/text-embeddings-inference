# T5Gemma2 Encoder Support — Implementation Notes

> **Repository:** `git@github.com:dsingal0/text-embeddings-inference.git`  
> **Branch:** `main` (forked from `huggingface/text-embeddings-inference`)  
> **Model tested:** `PhatcatDK/t5gemma-2-270m-270m-encoder-only`  
> **Target hardware:** NVIDIA H100 (sm90) and Blackwell B100/B200 (sm120)

---

## 1. What Was Implemented

### 1.1 T5Gemma2 Encoder Model

We added full inference support for the **T5Gemma2 text encoder** (the encoder-only portion of `T5Gemma2ForConditionalGeneration`). The implementation lives in:

- `backends/candle/src/models/t5gemma2.rs` — the model itself
- `backends/candle/src/models/mod.rs` — module registration
- `backends/candle/src/lib.rs` — backend dispatch and `Config` enum wiring
- `backends/src/dtype.rs` — un-gated `bfloat16` for the `candle` feature
- `router/src/lib.rs` — router-side model-type detection (`t5gemma2`) and BF16 default

### 1.2 Architecture Details

T5Gemma2's encoder is structurally identical to **Gemma3** with the following key differences that we accounted for:

| Feature | How we handled it |
|---|---|
| **Per-layer attention types** (`layer_types` array) | Full layer-type array with alternating `sliding_attention` and `full_attention` layers. |
| **Per-layer-type RoPE parameters** (`rope_parameters`) | Separate `cos`/`sin` caches for `full_attention` (linear scaling, factor=8.0, theta=1M) and `sliding_attention` (default, theta=10k). |
| **Bidirectional attention** | Forced `use_bidirectional_attention = true` in code regardless of the config JSON value. |
| **EOI token embedding** | Added a learnable EOI embedding (index 256000) fused into the final hidden states, matching the original encoder behavior. |
| **Separate Q/K/V weights** | Unlike Gemma3's fused `qkv_proj`, T5Gemma2 uses separate `q_proj`, `k_proj`, `v_proj` weights. |
| **Head dimension mismatch** | `head_dim` is configurable (256 in the tested model) and may differ from `hidden_size / num_attention_heads`. |
| **Sliding window** | Configurable `sliding_window` (512 in tested model) enforced per-layer based on `layer_types`. |
| **Weight key prefix** | Checkpoint uses `encoder.text_model.*` prefix, mapped via `vb.pp("encoder.text_model")`. |
| **MLP with SwiGLU** | `hidden_activation` is `gelu_pytorch_tanh` but the MLP is a SwiGLU gate (same as Gemma3). |
| **Attention logit softcapping** | Parsed from config but only applied when `attn_logit_softcapping` is non-null. |

### 1.3 Verified Capabilities

- **BF16 inference** on CUDA (H100)
- **Mean pooling** (tested; CLS and LastToken also supported by the architecture)
- **32K context window** (`max_position_embeddings = 32768`)
- **640-dimension embeddings**
- **Semantic similarity** works correctly (e.g., "cat/mat" vs "feline/rug" cosine = 0.96)

### 1.4 Docker Image

A working image was built and tested:

```bash
DOCKER_BUILDKIT=1 docker build \
  --build-arg CUDA_COMPUTE_CAP=90 \
  -f Dockerfile-cuda \
  -t tei-t5gemma2:h100 \
  .
```

Run:
```bash
docker run -d --name tei-t5gemma2 \
  --gpus all \
  -e CUDA_VISIBLE_DEVICES=1 \
  -p 8080:80 \
  -v /path/to/t5gemma-2-270m-270m-encoder-only:/data/model \
  tei-t5gemma2:h100 \
  --model-id /data/model \
  --dtype bfloat16 \
  --pooling mean
```

---

## 2. What Is Still Left

### 2.1 Flash Attention Variant (`flash_t5gemma2.rs`)

We did **not** implement a `FlashT5Gemma2Model` variant. The current model uses the standard `candle` attention path (naive scaled dot-product attention + RoPE applied manually).

**Why it matters:** Flash Attention would significantly improve throughput and memory efficiency for long sequences (up to 32K tokens).

**What would be needed:**
1. Create `backends/candle/src/models/flash_t5gemma2.rs`
2. Copy the `T5Gemma2EncoderModel` structure but replace the manual attention loop with `FlashAttention` from `candle-flash-attn`
3. Handle per-layer RoPE parameters and sliding-window masks in the Flash Attention call
4. Register it in the backend dispatch (`backends/candle/src/lib.rs`) under the `cuda` + `flash-attn` feature path

The model architecture is encoder-only and bidirectional, so the Flash Attention call would need `causal=false` and a custom attention mask for sliding windows.

---

## 3. Blockers Encountered

### 3.1 CUDA 13.1 Incompatibility with `candle-flash-attn`

The primary blocker for the Flash Attention variant is the **CUDA version on the build node**.

- **Build node CUDA:** 13.1 (host driver)
- **`candle-flash-attn` pinned CUTLASS version:** Compatible with CUDA 12.x but **not** CUDA 13.1
- **Symptom:** `nvcc` compilation fails inside `candle-flash-attn/build.rs` with CUTLASS-related template errors when compiling the `sm80` flash-forward kernels.
- **Impact:** The `candle-cuda` feature (which enables Flash Attention) cannot be compiled on this node. We had to fall back to `candle-cuda-volta` (non-Flash CUDA path).

**Workaround used:** We built inside the `nvidia/cuda:12.9.1-devel-ubuntu24.04` Docker image, which has CUDA 12.9.1. This environment successfully compiles the non-Flash CUDA path. The Flash Attention path would also compile successfully inside this Docker image because it uses the container's `nvcc` (12.9.1), not the host's.

### 3.2 Missing Top-Level Config Fields

The model's `config.json` has `max_position_embeddings` nested inside `encoder.text_config`, but TEI's router (`router/src/lib.rs`) expects it at the top level. We had to add:

```json
"max_position_embeddings": 32768,
"pad_token_id": 0
```

to the root of `config.json` for the router to parse it.

### 3.3 Build Timeouts

The `cargo build` with CUDA features (`candle-cuda-volta` + `static-linking`) times out when run via automated tooling because `candle-layer-norm` and `candle-kernels` CUDA kernel compilation exceeds 2 minutes. The build must be run manually or in CI with longer timeouts.

---

## 4. How to Build for Blackwell (sm120)

The image we built (`tei-t5gemma2:h100`) targets **sm90** and will **NOT** run on Blackwell (sm120) because:
- `libcublas_static.a` was `nvprune`-d to only sm90 code
- `candle-layer-norm` kernels were compiled for sm90
- The binary is statically linked, so no fallback to the runtime shared library

### Build Command for sm120

```bash
cd text-embeddings-inference

DOCKER_BUILDKIT=1 docker build \
  --build-arg CUDA_COMPUTE_CAP=120 \
  -f Dockerfile-cuda \
  -t tei-t5gemma2:b120 \
  .
```

### Why this works

1. **`Dockerfile-cuda` handles sm120 explicitly** (lines 71–73):
   ```dockerfile
   elif [ ${CUDA_COMPUTE_CAP} -eq 120 ]; then
     nvprune --generate-code code=sm_120 /usr/local/cuda/lib64/libcublas_static.a ...
   ```

2. **CUDA 12.9.1 supports Blackwell.** The base image `nvidia/cuda:12.9.1-devel-ubuntu24.04` ships with CUDA 12.9.1, which added sm120 support (CUDA 12.8+).

3. **`bindgen_cuda` respects `CUDA_COMPUTE_CAP`.** The `candle-layer-norm` and `candle-kernels` build scripts read `CUDA_COMPUTE_CAP` from the environment and compile kernels for `sm_120`.

4. **Flash Attention kernels compile for sm120 too.** `candle-flash-attn` uses `bindgen_cuda` which targets the same compute cap. Even though the kernel files are named `*_sm80.cu`, `nvcc` recompiles them for `sm_120` at build time.

5. **Our T5Gemma2 model does not depend on Flash Attention.** It uses standard `candle` attention ops and `cublasLt` matmuls. The Flash Attention kernels are compiled but never loaded, so even if they had runtime issues on Blackwell, they would not affect this model.

### Run on Blackwell

```bash
docker run -d --name tei-t5gemma2-b120 \
  --gpus all \
  -p 8080:80 \
  -v /path/to/t5gemma-2-270m-270m-encoder-only:/data/model \
  tei-t5gemma2:b120 \
  --model-id /data/model \
  --dtype bfloat16 \
  --pooling mean
```

---

## 5. Files Changed

| File | Change |
|---|---|
| `backends/candle/src/models/t5gemma2.rs` | **New** — T5Gemma2 encoder model implementation |
| `backends/candle/src/models/mod.rs` | Added `pub mod t5gemma2;` and `pub use t5gemma2::*;` |
| `backends/candle/src/lib.rs` | Added `T5Gemma2(T5Gemma2TopLevelConfig)` to `Config` enum; added BF16 dispatch for `T5Gemma2`; added `"bfloat16"` dtype parsing |
| `backends/src/dtype.rs` | Removed `#[cfg(not(feature = "candle"))]` gate on `DType::Bfloat16` |
| `router/src/lib.rs` | Added `t5gemma2` model-type detection with BF16 default and `max_position_embeddings` lookup |

---

## 6. Testing Checklist

- [x] `cargo check` (non-CUDA) compiles successfully
- [x] Docker image builds with `CUDA_COMPUTE_CAP=90`
- [x] Model loads on H100 GPU in BF16
- [x] HTTP `/embed` endpoint returns 640-dimension embeddings
- [x] Semantic similarity produces plausible cosine scores
- [x] `/info` endpoint reports correct dtype, pooling, and max input length
- [ ] Flash Attention variant (`flash_t5gemma2.rs`) — **not implemented**
- [ ] Test on Blackwell (sm120) — **requires rebuild with `CUDA_COMPUTE_CAP=120`**

---

## 7. References

- Original TEI repo: https://github.com/huggingface/text-embeddings-inference
- Target model: https://huggingface.co/PhatcatDK/t5gemma-2-270m-270m-encoder-only
- T5Gemma2 architecture: Google DeepMind T5Gemma2 (encoder-decoder variant of Gemma3)
- TEI CI matrix supports sm120: `.github/workflows/matrix.json` (`blackwell-120` entry)
