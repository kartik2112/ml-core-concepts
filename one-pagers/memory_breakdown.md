# Memory Breakdowns: Params, Grads, Activations (One-Pager)

## Overview
Understanding memory usage is critical for:
- Choosing batch size
- Deciding parallelism strategy
- Debugging OOM errors
- Optimizing training throughput

---

## 1. Model Parameters

### Storage Requirements
```
FP32: 4 bytes per parameter
FP16/BF16: 2 bytes per parameter
INT8: 1 byte per parameter

Example (GPT-3, 175B params):
  FP32: 175B × 4 = 700 GB
  FP16: 175B × 2 = 350 GB
```

### Typical Model Sizes
```
BERT-Base: 110M params → 440 MB (FP32), 220 MB (FP16)
BERT-Large: 340M params → 1.36 GB (FP32), 680 MB (FP16)
GPT-2: 1.5B params → 6 GB (FP32), 3 GB (FP16)
LLaMA-7B: 7B params → 28 GB (FP32), 14 GB (FP16)
LLaMA-70B: 70B params → 280 GB (FP32), 140 GB (FP16)
```

### Parameter Distribution (Transformer)
```
Embedding: V × D (V=vocab size, D=model dim)
  GPT-3: 50,257 × 12,288 = 617M params

Per Layer (12D² for standard transformer):
  MHA: 4D² (QKV + output projection)
  FFN: 8D² (two linear layers with D_ff=4D)
  LayerNorm: ~4D (negligible)

Output Head: V × D (language modeling)

Total: Embedding + L×12D² + V×D
```

---

## 2. Gradients

### Storage Requirements
```
Gradients: Same size as parameters
FP32: M × 4 bytes
FP16: M × 2 bytes (mixed precision)

Example (7B params, FP16):
  Gradients: 7B × 2 = 14 GB
```

### Gradient Accumulation
```
Without: Clear grads after each step
With N accumulation steps:
  Memory: Same (M × dtype_size)
  Accumulate in place: grad += new_grad / N
  
No extra memory for accumulation!
```

### Gradient Checkpointing
```
Normal: Store all activations for backward
Checkpointed: Recompute activations during backward

Memory trade: Activations ↓, Gradient memory unchanged
Compute trade: +33% FLOPs (recomputation)
```

---

## 3. Optimizer States

### Adam/AdamW (Most Common)
```
Per parameter:
  - Parameter: 1× (FP16 or FP32)
  - Gradient: 1× (FP16 or FP32)
  - Master parameter: 1× (FP32, for mixed precision)
  - Momentum (m): 1× (FP32)
  - Variance (v): 1× (FP32)

Mixed Precision Training:
  Model params (FP16): M × 2 bytes
  Gradients (FP16): M × 2 bytes
  Master params (FP32): M × 4 bytes
  Momentum (FP32): M × 4 bytes
  Variance (FP32): M × 4 bytes
  
Total: M × (2 + 2 + 4 + 4 + 4) = 16×M bytes
```

### SGD with Momentum
```
Per parameter:
  - Parameter: 1× (FP16)
  - Gradient: 1× (FP16)
  - Momentum: 1× (FP32)
  - Master param: 1× (FP32, mixed precision)

Total: M × (2 + 2 + 4 + 4) = 12×M bytes
```

### Memory Comparison (7B params)
```
FP32 Training (Adam):
  Params: 28 GB
  Grads: 28 GB
  Momentum: 28 GB
  Variance: 28 GB
  Total: 112 GB (4× model size)

FP16 Mixed Precision (Adam):
  Params (FP16): 14 GB
  Grads (FP16): 14 GB
  Master (FP32): 28 GB
  Momentum (FP32): 28 GB
  Variance (FP32): 28 GB
  Total: 112 GB (still 4×, but 2× from optimizer)
```

### 8-bit Optimizers (bitsandbytes)
```
Quantize optimizer states to INT8:
  Momentum/variance: 1 byte instead of 4
  
Total: M × (2 + 2 + 4 + 1 + 1) = 10×M bytes
Savings: ~37% compared to standard mixed precision
```

---

## 4. Activations (Forward Pass)

### What Are Activations?
Intermediate outputs stored during forward pass, needed for backward pass.

### Per-Layer Activations (Transformer)

```
Input to layer: B×S×D bytes

Multi-Head Attention:
  - QKV: 3×B×S×D bytes
  - Attention scores: B×H×S×S bytes ← QUADRATIC!
  - Context: B×S×D bytes
  Subtotal: 4×B×S×D + B×H×S×S bytes

FFN:
  - Intermediate: B×S×D_ff bytes (D_ff=4D typically)
  Subtotal: B×S×4D bytes

LayerNorm: Negligible (means, variances)

Total per layer: ~8×B×S×D + B×H×S×S bytes
```

### Full Model Activations
```
Embeddings: B×S×D bytes
L Transformer Layers: L × (8×B×S×D + B×H×S×S) bytes
Output: B×S×V bytes (cross-entropy logits, V=vocab)

Total: B×S×(D + 8LD + V) + L×B×H×S×S bytes

For long sequences (S large), dominated by attention scores!
```

### Example: GPT-3 (L=96, D=12,288, H=96)
```
Batch B=1, Sequence S=2048:
  Per layer: 8×1×2048×12288 + 1×96×2048×2048
           = 201 MB + 805 MB = 1 GB per layer
  96 layers: 96 GB just for activations!

Batch B=8, Sequence S=2048:
  96 layers: 768 GB (won't fit even on A100!)
```

### Activation Memory vs Sequence Length
```
Linear terms: 8×B×S×D per layer → O(S)
Quadratic term: B×H×S×S per layer → O(S²)

For S=512: Linear ~200 MB, Quadratic ~100 MB → Linear dominant
For S=4096: Linear ~1.6 GB, Quadratic ~6.4 GB → Quadratic dominant!
```

---

## 5. Memory Reduction Techniques

### Gradient Checkpointing
```
Store only K checkpoints (e.g., every √L layers)
Recompute intermediate activations in backward

Memory reduction: O(√L) instead of O(L)
Compute increase: +33% to +50%

Example (24 layers):
  Normal: 24 layer activations
  Checkpointed: √24 ≈ 5 checkpoints → 5× less memory
```

### Flash Attention
```
Standard: Materialize attention scores B×H×S×S in HBM
Flash: Compute attention in tiles, never materialize full matrix

Memory saved: B×H×S×S bytes per layer
Example (B=8, H=96, S=2048): 3.2 GB per layer → 0 GB
```

### Activation Recomputation (Selective)
```
Recompute cheap operations (ReLU, LayerNorm)
Store expensive operations (MatMul outputs)

Fine-grained control over memory-compute tradeoff
```

### Mixed Precision Activations
```
FP32 → FP16: 2× memory reduction
BF16: Better numerical stability than FP16

Caveat: Some layers need FP32 (LayerNorm, Softmax)
```

---

## 6. Total Memory Formula

### Training (Single Device)
```
Total = Params + Grads + Optimizer + Activations

Mixed Precision (Adam):
  = 2M + 2M + 12M + A
  = 16M + A

Where:
  M = model size in bytes (FP16)
  A = activation memory
```

### Inference (No Training)
```
Total = Params + Activations (single batch)

Batch size 1, no gradients needed:
  = 2M (FP16) + B×S×D×(L+overhead)
  
Can also quantize to INT8/INT4 for inference:
  = M/2 or M/4 + activations
```

---

## 7. Breakdown by Model Size

### LLaMA-7B Training (Mixed Precision, Adam)
```
Parameters (FP16): 14 GB
Gradients (FP16): 14 GB
Master params (FP32): 28 GB
Momentum (FP32): 28 GB
Variance (FP32): 28 GB
Subtotal: 112 GB

Activations (B=4, S=2048, 32 layers):
  Per layer: ~500 MB
  Total: ~16 GB

Grand Total: ~128 GB
Fits on: 2× A100 80GB with FSDP
```

### LLaMA-70B Training
```
Parameters + Optimizer: ~1.1 TB
Activations: ~100 GB

Total: ~1.2 TB
Requires: 16+ A100 80GB with FSDP
```

---

## 8. Practical Guidelines

### Choosing Batch Size
```
Available Memory: M_avail
Model footprint: M_model (params + grads + optimizer)
Activation per sample: A_sample

Max batch size: (M_avail - M_model) / A_sample
```

### OOM Debugging
1. **Check param count**: M params × 16 bytes (mixed precision Adam)
2. **Check activation memory**: Estimate B×S×D×L×factor
3. **Profile**: Use `torch.cuda.memory_summary()`
4. **Gradual reduction**: Halve batch size until it fits

### Memory Optimization Priority
1. **Mixed precision**: 2× savings for free
2. **Gradient checkpointing**: √L savings, 33% compute cost
3. **Flash Attention**: Saves S² term
4. **FSDP/ZeRO**: Linear scaling with devices
5. **8-bit optimizer**: 37% savings on optimizer

---

## 9. Memory Hierarchy Summary

```
Component          | Storage        | FP32    | FP16 Mixed
-------------------|----------------|---------|-------------
Parameters         | GPU HBM        | 4M      | 2M
Gradients          | GPU HBM        | 4M      | 2M
Master Params      | GPU HBM        | -       | 4M
Momentum (Adam)    | GPU HBM        | 4M      | 4M
Variance (Adam)    | GPU HBM        | 4M      | 4M
Activations        | GPU HBM        | 4A      | 2A
KV Cache (Infer)   | GPU HBM        | varies  | varies
Temp Buffers       | GPU HBM        | varies  | varies
-------------------|----------------|---------|-------------
Total Training     |                | 20M+4A  | 16M+2A
Total Inference    |                | 4M+A    | 2M+A
```

---

## Interview Quick Facts
- **Adam optimizer**: Adds 8× parameter size (4× from params/grads, 4× from states)
- **Activation memory**: Often dominates for large batch/sequence
- **Gradient checkpointing**: Standard for large models (GPT-3, LLaMA)
- **Flash Attention**: Eliminates S² term in attention
- **FSDP/ZeRO-3**: Reduces per-device memory by N× (N=devices)
- **Rule of thumb**: 16× model size for training (mixed precision Adam)
- **Peak memory**: During backward pass (all activations + gradients)
