# Parallelism Modes & Collectives (One-Pager)

## Overview of Parallelism Strategies

### Why Parallelize?
- **Model Size**: Models exceed single GPU memory (e.g., 175B params × 2 bytes = 350GB)
- **Training Speed**: Distribute computation across devices
- **Batch Size**: Process more samples simultaneously

## 1. Data Parallelism (DP)

### Concept
Each device holds **full model copy**, processes different data batches.

### How It Works
```
GPU 0: Model (copy 1) → Batch [0:4]
GPU 1: Model (copy 2) → Batch [4:8]
GPU 2: Model (copy 3) → Batch [8:12]
GPU 3: Model (copy 4) → Batch [12:16]
    ↓
Forward & Backward (independent)
    ↓
All-Reduce Gradients
    ↓
Update model (same updates on all GPUs)
```

### Collectives Used
- **AllReduce**: Average gradients across all devices
  - Operation: Each device gets sum/average of all gradients
  - Volume: `2 × (N-1)/N × param_bytes` per device
  - Example (4 GPUs): Each sends/receives ~3× parameter size

### Memory Per Device
```
Parameters: M bytes
Gradients: M bytes
Optimizer States: α×M bytes (α=2 for Adam: momentum + variance)
Activations: A bytes
Total: M(1 + 1 + α) + A
```

### Pros & Cons
✅ Simple, good scaling for small models  
✅ No code changes (PyTorch DDP)  
❌ Memory replicated (wasteful for large models)  
❌ Communication scales with model size

### Scaling Efficiency
- **Ideal**: Linear with N devices
- **Reality**: Communication overhead grows
- **Sweet spot**: Models < 10B params, up to 8-16 GPUs

---

## 2. Fully Sharded Data Parallelism (FSDP)

### Concept
Shards parameters, gradients, and optimizer states across devices. Zero Redundancy Optimizer (ZeRO).

### ZeRO Stages
**Stage 1**: Shard optimizer states only → 4× memory reduction  
**Stage 2**: Shard gradients too → 8× memory reduction  
**Stage 3**: Shard parameters too → N× memory reduction (N=devices)

### How It Works (ZeRO-3)
```
Forward Pass:
  GPU 0: Owns params[0:25%], all-gathers params[25:100%]
  GPU 1: Owns params[25:50%], all-gathers others
  ... compute ...
  
Backward Pass:
  Compute gradients
  Reduce-scatter gradients (each GPU keeps its shard)
  Free other params
  
Optimizer Step:
  Each GPU updates only its parameter shard
```

### Collectives Used
- **AllGather**: Fetch full parameters for computation
  - Volume: M bytes per layer per device
- **ReduceScatter**: Aggregate and distribute gradient shards
  - Volume: M bytes per layer per device

### Memory Per Device (ZeRO-3)
```
Parameters: M/N bytes (sharded)
Gradients: M/N bytes (sharded)
Optimizer States: α×M/N bytes
Activations: A bytes (not sharded)
Temp (AllGather): M bytes during layer computation
Total Working: M/N(1 + 1 + α) + A + M (temporary)
```

### Pros & Cons
✅ Massive memory savings (linear with N)  
✅ Can train 100B+ models  
✅ Good scaling efficiency  
❌ More communication than DP (all-gather per layer)  
❌ Complexity in implementation

---

## 3. Tensor Parallelism (TP)

### Concept
Split **individual layers** across devices. Intra-layer parallelism.

### Column-Parallel Linear
```
Input: X (B, D_in)
Weight: W (D_in, D_out) → Split columns

GPU 0: W[:, 0:D_out/2]   → Y_0 (B, D_out/2)
GPU 1: W[:, D_out/2:end] → Y_1 (B, D_out/2)

Concat outputs: Y = [Y_0, Y_1] (B, D_out)
No communication needed!
```

### Row-Parallel Linear
```
Input: X (B, D_in) → Split across devices
Weight: W (D_in, D_out) → Split rows

GPU 0: X_0 @ W[0:D_in/2, :]   → Y_0
GPU 1: X_1 @ W[D_in/2:end, :] → Y_1

All-Reduce: Y = Y_0 + Y_1 (B, D_out)
```

### Transformer Block Sharding
```
MHA:
  QKV projection: Column-parallel → (B, S, D/N) per device
  Attention: Local computation (no comm)
  Output: Row-parallel → AllReduce

FFN:
  Linear1: Column-parallel
  GELU: Local
  Linear2: Row-parallel → AllReduce
```

### Collectives Used
- **AllReduce**: After row-parallel layers (2× per block)
  - Volume: `B × S × D` bytes per all-reduce
- **Identity** (forward), **AllReduce** (backward) for column-parallel

### Memory Per Device
```
Parameters: M/N bytes (sharded)
Activations: ~A bytes (mostly sharded)
Communication during forward/backward
```

### Pros & Cons
✅ Reduces both memory and computation per device  
✅ Low latency (within node, e.g., NVLink)  
✅ Exact same results as single GPU  
❌ Requires fast interconnect (NVLink/IB)  
❌ Communication on critical path (limits scaling)

### Scaling
- **Best**: 2-8 GPUs per node (NVLink)
- **Typical**: TP=4 or TP=8
- **Beyond**: Diminishing returns (communication cost)

---

## 4. Pipeline Parallelism (PP)

### Concept
Split model **vertically** (layers) across devices. Temporal parallelism.

### How It Works
```
GPU 0: Layers [0-6]
GPU 1: Layers [7-13]
GPU 2: Layers [14-20]
GPU 3: Layers [21-27]

Micro-batches: Split batch into chunks
Timeline:
  t0: GPU0(μB0)
  t1: GPU0(μB1), GPU1(μB0)
  t2: GPU0(μB2), GPU1(μB1), GPU2(μB0)
  t3: GPU0(μB3), GPU1(μB2), GPU2(μB1), GPU3(μB0)
```

### Schedules
**GPipe**: Fill-drain pipeline  
**PipeDream**: 1F1B (One Forward One Backward) - more efficient  
**Interleaved**: Multiple layer chunks per device

### Collectives Used
- **Point-to-Point** (P2P): Send/Recv between adjacent stages
  - Volume: `B × S × D` bytes per micro-batch
- **No AllReduce** in forward/backward

### Memory Per Device
```
Parameters: M/N bytes (layer-sharded)
Activations: A/num_microbatches (less memory per micro-batch)
Pipeline bubble: Idle time overhead
```

### Pros & Cons
✅ Scales to many devices (loose coupling)  
✅ Reduces activation memory (micro-batching)  
❌ Pipeline bubble (idle time)  
❌ Point-to-point communication inefficiency  
❌ Load balancing challenges

### Efficiency
- **Bubble overhead**: `(N-1) / (N + M - 1)` where M=micro-batches
- **Example**: N=4, M=16 → ~16% bubble
- **Rule**: M ≥ 4N for good efficiency

---

## 5. 3D Parallelism (DP + TP + PP)

### Concept
Combine all three strategies for massive models.

### Typical Configuration
```
GPT-3 (175B):
  DP=8, TP=8, PP=16
  Total: 1024 GPUs

Cluster layout:
  - TP within node (8 GPUs, NVLink)
  - PP across nodes (fast IB)
  - DP for throughput (replicated groups)
```

### Memory Calculation
```
Per device memory = M / (TP × PP) × (overhead) + activations
Communication = TP collectives + PP p2p + DP all-reduce
```

### Pros & Cons
✅ Handles any model size  
✅ Flexible resource utilization  
❌ Complex to configure and debug  
❌ Requires expert tuning

---

## Communication Collectives Summary

| Collective | Operation | Volume (per device) | Use Case |
|------------|-----------|---------------------|----------|
| **AllReduce** | Sum/avg all → broadcast | 2×(N-1)/N × data | DP gradients, TP output |
| **AllGather** | Gather all → broadcast | (N-1) × data | FSDP param fetch |
| **ReduceScatter** | Sum all → scatter shards | (N-1)/N × data | FSDP gradient reduce |
| **Broadcast** | One → all | data | Unused in modern training |
| **SendRecv (P2P)** | Device i → Device j | data | PP stage transfer |

---

## Choosing Parallelism Strategy

### Decision Tree
1. **Model fits on 1 GPU?** → No parallelism or DP
2. **Model < 10B params?** → DP or FSDP
3. **Model 10-100B?** → FSDP or TP+PP
4. **Model > 100B?** → 3D (DP+TP+PP)

### Hardware Considerations
- **Within node (NVLink)**: TP preferred
- **Cross node (InfiniBand)**: PP or FSDP
- **Many nodes**: DP for outer dimension

---

## Interview Quick Facts
- **Megatron-LM**: TP+PP implementation by NVIDIA
- **DeepSpeed**: FSDP (ZeRO) by Microsoft
- **FSDP PyTorch**: Native FSDP in PyTorch 2.0+
- **Communication bandwidth**: NVLink ~600 GB/s, PCIe ~64 GB/s, IB ~200 Gb/s
- **Ring AllReduce**: O(N) steps, bandwidth-optimal
