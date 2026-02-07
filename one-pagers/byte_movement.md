# Byte-Movement Formulas (One-Pager)

## Overview
Memory bandwidth, not compute, often limits modern deep learning. Understanding byte movement is critical for optimization.

## Key Concepts

### Arithmetic Intensity
```
AI = FLOPs / Bytes Moved
```
- **Compute-bound**: AI > Peak_FLOPs/Peak_Bandwidth
- **Memory-bound**: AI < Peak_FLOPs/Peak_Bandwidth

**Example (A100)**:
- Peak FLOPs: 312 TFLOPS (FP16)
- Peak Bandwidth: 2 TB/s (HBM)
- Threshold: AI > 156 FLOPs/byte

---

## 1. Matrix Multiplication (GEMM)

### Basic GEMM: C = A @ B
```
A: (M, K), B: (K, N) → C: (M, N)
FLOPs: 2×M×N×K
Bytes (naive): (M×K + K×N + M×N) × sizeof(dtype)
Arithmetic Intensity: 2×M×N×K / ((M×K + K×N + M×N) × bytes_per_elem)
```

### Optimal Case (Large Square Matrix, M=N=K)
```
AI = 2×N³ / (3×N² × bytes) = (2×N) / (3×bytes)
```
For N=1024, FP16 (2 bytes): AI = 341 FLOPs/byte → Compute-bound ✅

### Small Matrix (M=N=K=256)
```
AI = 512 / (3×2) = 85 FLOPs/byte → Memory-bound on A100 ❌
```

### With Reuse (Batched GEMM)
```
Batch of B matrices: A @ B_i for i in [1..batch]
Bytes saved: A loaded once, reused B times
Effective AI increases by ~factor of batch size
```

---

## 2. Layer-Specific Byte Movement

### Linear/Dense Layer
```
Forward:
  Input: B×D_in bytes
  Weight: D_in×D_out bytes
  Output: B×D_out bytes (write)
  Total Read: B×D_in + D_in×D_out bytes
  Total Write: B×D_out bytes

Backward (data):
  grad_output: B×D_out bytes
  Weight: D_in×D_out bytes (reuse from forward)
  grad_input: B×D_in bytes (write)
  Total Read: B×D_out + D_in×D_out bytes
  Total Write: B×D_in bytes

Backward (weight):
  Input: B×D_in bytes
  grad_output: B×D_out bytes
  grad_weight: D_in×D_out bytes (write)
  Total Read: B×D_in + B×D_out bytes
  Total Write: D_in×D_out bytes
```

### Total for Linear Layer (Forward + Backward)
```
Read: 2×B×D_in + 2×B×D_out + 3×D_in×D_out bytes
Write: B×D_out + B×D_in + D_in×D_out bytes
```

### Attention Layer (Self-Attention)
```
QKV Projection:
  Read: B×S×D + 3×D×D bytes
  Write: 3×B×S×D bytes

Attention Matrix:
  Read: B×S×D + B×S×D bytes (Q, K)
  Write: B×S×S bytes (attention scores)

Context:
  Read: B×S×S + B×S×D bytes (attn, V)
  Write: B×S×D bytes

Output Projection:
  Read: B×S×D + D×D bytes
  Write: B×S×D bytes

Total Attention Block:
  ~4×B×S×D + 4×D² + 2×B×S² bytes
```

**Key insight**: S² term dominates for long sequences!

---

## 3. Activation Memory

### Storage Requirements
```
Forward: Store activations for backward pass
Backward: Load activations, compute gradients

Transformer Layer:
  Input: B×S×D bytes
  QKV: 3×B×S×D bytes
  Attention scores: B×H×S×S bytes (quadratic!)
  Context: B×S×D bytes
  FFN intermediate: B×S×D_ff bytes
  
Total per layer: ~5×B×S×D + B×H×S×S bytes
```

### Gradient Checkpointing
```
Without: Store all activations → O(L×memory_per_layer)
With: Store only subset, recompute in backward → O(√L×memory_per_layer)

Tradeoff:
  Memory saved: ~√L factor
  Compute added: +33% FLOPs (recomputation)
  Byte movement: More reads (recompute activation)
```

---

## 4. Gradient All-Reduce (Data Parallel)

### Communication Volume
```
Parameters: M bytes (total model size)
All-Reduce using Ring algorithm:
  Steps: 2×(N-1) where N = number of devices
  Bytes per device: 2×(N-1)/N × M bytes

Example (4 GPUs, M=1GB):
  Each GPU: 2×(4-1)/4 × 1GB = 1.5GB transmitted
```

### Bandwidth Calculation
```
Time = Bytes / Bandwidth + Latency × Steps
     = (2×(N-1)/N × M) / BW + α×(N-1)

For large M, dominated by bandwidth:
  Time ≈ 2×M / BW
```

### Overlap with Computation
```
Communication while computing backward:
  Bucket gradients (e.g., 25MB buckets)
  Start all-reduce as soon as bucket ready
  Overlap reduces total time
```

---

## 5. Attention Optimization: Flash Attention

### Standard Attention
```
S = Q @ K^T / √d      # B×S×S matrix materialized → Memory write
P = softmax(S)        # B×S×S matrix → Memory read/write  
O = P @ V             # B×S×S matrix → Memory read

Total Bytes: 4×B×H×S×S (read/write attention matrix multiple times)
```

### Flash Attention (Tiled)
```
Tile Q, K, V into blocks
Compute attention on-chip (SRAM)
Never materialize full S matrix in HBM

Total Bytes: B×S×D (just inputs/outputs, no S matrix I/O)
Speedup: 2-4× (memory bandwidth limited)
```

**Key**: Memory I/O reduced from O(S²) to O(S)

---

## 6. Mixed Precision Training

### Memory & Bandwidth Savings
```
FP32: 4 bytes per param
FP16/BF16: 2 bytes per param

Forward/Backward (FP16):
  Activations: 2× less memory
  Gradients: 2× less memory
  Bandwidth: 2× less

Master Weights (FP32):
  Keep for numerical stability
  Optimizer states: Still FP32
  
Net savings: ~1.5-2× memory for large models
```

### Byte Movement Impact
```
Forward:
  Read weights: FP16 (2 bytes) instead of FP32 (4 bytes)
  50% bandwidth reduction

Backward:
  Gradient all-reduce: FP16 → 50% less communication
  
Optimizer:
  Convert FP16 grads → FP32: 2→4 bytes (small overhead)
  Update in FP32
  Convert FP32 params → FP16: 4→2 bytes
```

---

## 7. Hardware Memory Hierarchy

### A100 GPU
```
Level           | Size      | Bandwidth     | Latency
----------------|-----------|---------------|----------
Registers       | ~256 KB   | ~20 TB/s      | 1 cycle
L1/Shared Mem   | 192 KB    | ~10 TB/s      | ~10 cycles
L2 Cache        | 40 MB     | ~5 TB/s       | ~100 cycles
HBM (Main)      | 80 GB     | 2 TB/s        | ~300 cycles
NVLink          | -         | 600 GB/s      | ~1μs
PCIe 4.0        | -         | 64 GB/s       | ~10μs
CPU Memory      | ~1 TB     | ~100 GB/s     | ~100μs
```

### Optimization Strategy
1. **Kernel fusion**: Reduce HBM round-trips
2. **Tiling**: Maximize L2/shared memory reuse
3. **Persistent kernels**: Keep data on-chip
4. **Async transfers**: Overlap CPU↔GPU, GPU↔GPU

---

## 8. Common Patterns & Optimizations

### Element-wise Operations (ReLU, LayerNorm, etc.)
```
Read: B×S×D bytes
Compute: B×S×D FLOPs
Write: B×S×D bytes
AI = 1 FLOP/byte → Extremely memory-bound!

Optimization: Fuse with adjacent operations
```

### Gradient Accumulation
```
Without: All-reduce after each micro-batch
With: Accumulate M micro-batches, then all-reduce

Communication saved: M× fewer all-reduces
Tradeoff: Higher memory (store accumulated grads)
```

### ZeRO Optimizer (FSDP)
```
Stage 1 (Optimizer sharding):
  All-reduce grads: 2×M bytes
  Gather optimizer: (N-1)/N × O bytes (O=optimizer state size)
  
Stage 2 (+ Gradient sharding):
  Reduce-scatter grads: M bytes
  No all-reduce needed!
  
Stage 3 (+ Param sharding):
  All-gather params per layer: M/N bytes × L times
  Reduce-scatter grads: M bytes
  More communication, but memory linear in N
```

---

## Roofline Model

### Concept
```
Performance (FLOPs/s) = min(Peak_FLOPs, AI × Bandwidth)
```

### Plotting
```
Log-log plot:
  X-axis: Arithmetic Intensity (FLOPs/byte)
  Y-axis: Performance (TFLOPS)
  
Horizontal line: Peak compute
Diagonal line: Bandwidth limit (slope = BW)
Intersection: Ridge point
```

### Application
- Operations left of ridge: Memory-bound → Optimize bandwidth
- Operations right of ridge: Compute-bound → Optimize FLOPs

---

## Interview Quick Facts
- **A100 HBM**: 2 TB/s, 80GB
- **H100 HBM**: 3.35 TB/s, 80GB
- **NVLink**: 600 GB/s (bidirectional)
- **Flash Attention**: Reduces memory I/O from O(S²) to O(S)
- **Typical AI**: MatMul ~100-500, LayerNorm ~1, Attention ~10-50
- **Rule of thumb**: For transformers, memory bandwidth often limits training speed more than compute
