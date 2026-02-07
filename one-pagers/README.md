# ML Core Concepts - One-Pagers

Dense, focused reference sheets for intermediate to advanced research engineers. Designed for quick review during interviews and practical work.

## 📑 Available One-Pagers

### 1. [Per-Layer FLOPs](per_layer_flops.md)
Computational complexity analysis for deep learning layers.

**Topics covered:**
- Linear/Dense layers
- Convolutional layers
- Self-attention and Multi-Head Attention
- Feed-forward networks
- Layer normalization and activations
- Transformer block totals
- Optimization insights and hardware utilization

**Key formulas:**
- Linear: `2 × B × D_in × D_out`
- Attention: `2 × B × S × D × (4D + 2S × H)`
- FFN: `4 × B × S × D × D_ff`

---

### 2. [The Entire Attention Block](attention_block.md)
Complete architecture and mathematics of attention mechanisms.

**Topics covered:**
- Multi-head attention architecture
- Scaled dot-product attention mathematics
- Masked attention for causal models
- Pre-LN vs Post-LN variants
- Flash Attention and optimizations
- Parameter counts and memory requirements
- Attention patterns (sparse, linear, sliding window)

**Key insights:**
- Complexity: O(BSD² + BS²D)
- Memory bottleneck: O(BS²H) for attention matrices
- Flash Attention: Reduces memory from O(S²) to O(S)

---

### 3. [Parallelism Modes & Collectives](parallelism_modes.md)
Comprehensive guide to distributed training strategies.

**Modes covered:**
1. **Data Parallelism (DP)**: Full model replication
2. **Fully Sharded Data Parallelism (FSDP/ZeRO)**: Shard params, grads, optimizer
3. **Tensor Parallelism (TP)**: Intra-layer sharding
4. **Pipeline Parallelism (PP)**: Layer-wise sharding
5. **3D Parallelism**: Combined DP+TP+PP

**Collectives:**
- AllReduce, AllGather, ReduceScatter
- Point-to-point (P2P) for pipeline
- Communication volumes and patterns

**Decision tree:** When to use which strategy based on model size and hardware.

---

### 4. [Byte-Movement Formulas](byte_movement.md)
Memory bandwidth analysis and optimization strategies.

**Topics covered:**
- Arithmetic intensity and roofline model
- Layer-specific byte movement (GEMM, attention, FFN)
- Activation memory and gradient checkpointing
- Communication costs (all-reduce, all-gather)
- Flash Attention memory optimization
- Mixed precision training bandwidth
- Hardware memory hierarchy (registers → HBM → NVLink)

**Key concepts:**
- AI = FLOPs / Bytes Moved
- Memory-bound vs compute-bound operations
- A100: 312 TFLOPS, 2 TB/s HBM bandwidth

---

### 5. [Memory Breakdowns](memory_breakdown.md)
Detailed analysis of memory consumption during training and inference.

**Topics covered:**
- Model parameters (FP32, FP16, INT8)
- Gradient storage requirements
- Optimizer states (Adam, SGD, 8-bit optimizers)
- Activation memory (per-layer breakdown)
- Memory reduction techniques
- Total memory formulas for training
- OOM debugging strategies

**Key formulas:**
- Training (mixed precision Adam): `16M + A`
- M = model size, A = activation memory
- Activations scale with: `B×S×D×L + B×H×S²×L`

---

## 🎯 Usage Guide

### For Interviews
- **Quick review**: Scan formulas and key insights before technical rounds
- **Deep dives**: Understand tradeoffs between different approaches
- **Practical examples**: Reference real model sizes (GPT-3, LLaMA, BERT)

### For Work
- **Performance debugging**: Identify compute vs memory bottlenecks
- **Architecture decisions**: Choose appropriate parallelism strategies
- **Memory optimization**: Calculate and reduce memory footprint
- **Hardware selection**: Match workload to hardware capabilities

### Study Path
Recommended reading order:
1. **Memory Breakdowns** → Understand what consumes memory
2. **Per-Layer FLOPs** → Understand computational costs
3. **Attention Block** → Deep dive into key architecture
4. **Byte-Movement** → Optimize memory bandwidth
5. **Parallelism Modes** → Scale to multiple devices

---

## 🔑 Key Abbreviations

- **B**: Batch size
- **S**: Sequence length
- **D**: Model dimension
- **L**: Number of layers
- **H**: Number of attention heads
- **M**: Model size (bytes)
- **A**: Activation memory (bytes)
- **N**: Number of devices
- **DP**: Data Parallelism
- **TP**: Tensor Parallelism
- **PP**: Pipeline Parallelism
- **FSDP**: Fully Sharded Data Parallelism
- **MHA**: Multi-Head Attention
- **FFN**: Feed-Forward Network
- **AI**: Arithmetic Intensity

---

## 📚 Related Resources

### In This Repository
- **Notebooks**: Hands-on implementations (`../notebooks/`)
  - Multi-Head Attention (`04_multi_head_attention.ipynb`)
  - Parallelism Strategies (`05_pure_numpy_mlp_parallelism.ipynb`)
  - Transformer Architecture (`07_transformer_with_moe.ipynb`)

### External References
- **Papers**:
  - "Attention Is All You Need" (Vaswani et al., 2017)
  - "FlashAttention" (Dao et al., 2022)
  - "Megatron-LM" (Shoeybi et al., 2019)
  - "ZeRO" (Rajbhandari et al., 2020)

- **Frameworks**:
  - PyTorch FSDP
  - DeepSpeed (ZeRO)
  - Megatron-LM (TP/PP)
  - HuggingFace Accelerate

---

## 💡 Tips for Learning

1. **Start with examples**: Plug in real numbers (e.g., BERT-Base dimensions)
2. **Cross-reference**: Connect formulas across one-pagers
3. **Hands-on**: Use notebooks to validate theoretical understanding
4. **Scale thinking**: Consider how patterns change with model/sequence size
5. **Hardware awareness**: Map operations to GPU/interconnect capabilities

---

## 🤝 Contributing

These one-pagers are living documents. Contributions welcome:
- Corrections and clarifications
- Additional formulas or patterns
- Real-world examples and benchmarks
- New one-pagers for related topics

---

## 📧 Feedback

For questions, corrections, or suggestions, please open an issue on GitHub.

---

**Dense knowledge for efficient learning! 🚀**
