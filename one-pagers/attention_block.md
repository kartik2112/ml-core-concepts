# The Entire Attention Block (One-Pager)

## Architecture Overview
```
Input (B, S, D)
    ↓
[LayerNorm] → Pre-LN or Post-LN variant
    ↓
[Multi-Head Attention]
    ├─ QKV Projection: 3 × Linear(D → D)
    ├─ Split into H heads: (B, S, D) → (B, H, S, D_h)
    ├─ Scaled Dot-Product: softmax(QK^T/√D_h) × V
    ├─ Concatenate heads: (B, H, S, D_h) → (B, S, D)
    └─ Output Projection: Linear(D → D)
    ↓
[Residual Connection] → Add input
    ↓
[LayerNorm] → If pre-LN, otherwise before
    ↓
[Feed-Forward Network]
    ├─ Linear(D → D_ff), typically D_ff = 4D
    ├─ Activation (GELU/ReLU)
    └─ Linear(D_ff → D)
    ↓
[Residual Connection] → Add previous
    ↓
Output (B, S, D)
```

## Mathematical Formulation

### 1. Multi-Head Attention (Detailed)
```python
# Input: X ∈ ℝ^(B×S×D)
Q = X @ W_q  # (B, S, D) @ (D, D) → (B, S, D)
K = X @ W_k  # Same
V = X @ W_v  # Same

# Split heads: (B, S, D) → (B, S, H, D_h) → (B, H, S, D_h)
# where D_h = D / H

# Per-head attention
scores = Q @ K^T / √D_h  # (B, H, S, S)
attn = softmax(scores, dim=-1)  # (B, H, S, S)
context = attn @ V  # (B, H, S, D_h)

# Concatenate and project
output = concat(context) @ W_o  # (B, S, D) @ (D, D) → (B, S, D)
```

### 2. Attention Score Computation
```
Attention(Q, K, V) = softmax(QK^T / √D_h) V

Score_ij = exp(q_i · k_j / √D_h) / Σ_j exp(q_i · k_j / √D_h)
```
- **Scaling factor √D_h**: Prevents saturation in softmax

### 3. Masked Attention (Causal/Decoder)
```python
mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
scores = scores.masked_fill(mask, float('-inf'))
attn = softmax(scores, dim=-1)
```

## Complexity Analysis

### Computational
- **QKV Projection**: O(BSD²)
- **Attention Matrix**: O(BS²D_h × H) = O(BS²D)
- **Output Projection**: O(BSD²)
- **Total**: O(BSD² + BS²D)

### Memory
- **Attention Matrix**: O(BS²H) - the bottleneck!
- **Activations**: O(BSD)
- **Gradients**: Same as activations

## Variants & Optimizations

### 1. Pre-LN vs Post-LN
**Pre-LN** (Modern, GPT-2+):
```
x = x + MHA(LN(x))
x = x + FFN(LN(x))
```
- More stable training
- No LN on final output

**Post-LN** (Original Transformer):
```
x = LN(x + MHA(x))
x = LN(x + FFN(x))
```
- Requires careful init/warmup

### 2. Flash Attention
- **Standard**: O(S²) memory for attention matrix
- **Flash**: Fused kernel, tiled computation
- **Result**: O(S) memory, 2-4× faster
- Same FLOPs, optimized memory I/O

### 3. Attention Patterns
**Full Attention**: All-to-all, O(S²)
**Sparse**: Fixed patterns (local, strided)
**Linear**: Kernel trick approximation, O(S)
**Sliding Window**: Local context, O(S×w)

## Parameter Count

### Standard Transformer Block
```
MHA Parameters:
  - QKV: 3 × D × D = 3D²
  - Output: D × D = D²
  - Total: 4D²

FFN Parameters:
  - Up: D × D_ff = 4D²
  - Down: D_ff × D = 4D²
  - Total: 8D²

LayerNorm: ~4D (γ, β for 2 layers)

Block Total: 12D² + 4D ≈ 12D²
```

## Key Design Choices

### Why Multi-Head?
- Different heads learn different patterns
- Computational efficiency vs single large head
- Typical: H=12-32, D_h=64-128

### Why FFN after MHA?
- MHA: token mixing (cross-position)
- FFN: feature mixing (per-position)
- Complementary operations

### Residual Connections
- Enable gradient flow (prevent vanishing)
- Allow identity mapping
- Critical for deep models (>12 layers)

## Practical Considerations

### Memory Bottleneck
For S=2048, D=1024, H=16:
- Attention: 2048² × 16 × 2 bytes = 128MB per sample
- Long sequences dominate memory usage

### Optimization Strategies
1. **Gradient Checkpointing**: Recompute activations
2. **Flash Attention**: Kernel fusion
3. **Mixed Precision**: FP16/BF16 for speed
4. **Sequence Packing**: Batch efficiency

## Interview Quick Facts
- **GPT-3**: 96 layers, D=12288, H=96, D_h=128
- **BERT-Base**: 12 layers, D=768, H=12, D_h=64
- **Attention complexity**: Main limit for context length
- **KV Cache**: For inference, store past K,V (O(BSd) memory)
- **Positional Encoding**: Added before first layer (sinusoidal or learned)
