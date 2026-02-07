# Per-Layer FLOPs (One-Pager)

## Overview
FLOPs (Floating Point Operations) quantify computational cost. Critical for model efficiency analysis, hardware selection, and optimization.

## Key Formulas

### 1. Linear/Dense Layer
```
FLOPs = 2 × B × D_in × D_out
```
- **B**: batch size
- **D_in**: input dimension
- **D_out**: output dimension
- Factor of 2: one multiply + one add per MAC (Multiply-Accumulate)

**With bias**: Add `B × D_out` FLOPs

### 2. Convolutional Layer (Conv2D)
```
FLOPs = 2 × B × H_out × W_out × C_out × (K_h × K_w × C_in)
```
- **B**: batch size
- **H_out, W_out**: output height/width
- **C_out**: output channels
- **C_in**: input channels
- **K_h, K_w**: kernel height/width

**Simplified**: `2 × B × H_out × W_out × C_out × C_in × K²` (for square kernels)

### 3. Self-Attention (Single Head)
```
QKV Projection: 6 × B × S × D × D_head
Attention Scores: 2 × B × S² × D_head
Score × V: 2 × B × S² × D_head
Output Projection: 2 × B × S × D × D
Total ≈ 2 × B × S × D × (4D + 2S)
```
- **S**: sequence length
- **D**: model dimension
- **D_head**: head dimension (usually D/num_heads)

### 4. Multi-Head Attention (MHA)
```
FLOPs = 2 × B × S × D × (4D + 2S × H)
```
- **H**: number of heads
- For large D, dominated by: `8BSD²` (QKV + output projections)

### 5. Feed-Forward Network (FFN/MLP)
```
FLOPs = 2 × B × S × D × D_ff × 2
      = 4 × B × S × D × D_ff
```
- **D_ff**: hidden dimension (typically 4D)
- Two linear layers: D→D_ff→D

### 6. Layer Normalization
```
FLOPs ≈ 5 × B × S × D
```
- Mean: B×S×D, Variance: B×S×D, Normalize: 3×B×S×D

### 7. GELU/SiLU Activation
```
FLOPs ≈ 8 × B × S × D_ff
```
- More expensive than ReLU due to exponentials

## Transformer Block Total
```
Total FLOPs ≈ B × S × D × (24D + 4D_ff + 4SH)
```

For typical config (D_ff=4D, H=D/64):
```
≈ B × S × D × (40D + 4S·D/64)
```

## Practical Tips

### Comparing Models
- **Total FLOPs**: Sum across all layers
- **Per-token FLOPs**: Divide by (B × S)
- **MACs vs FLOPs**: 1 MAC ≈ 2 FLOPs

### Optimization Insights
1. **Bottlenecks**: MHA quadratic in S, FFN linear but large D_ff
2. **Flash Attention**: Same FLOPs, reduces memory I/O
3. **Sparse Attention**: Reduces S² term to S·S_local
4. **MoE**: Reduces effective D_ff via sparse routing

### Hardware Utilization
- **Peak FLOPs**: Theoretical hardware max (e.g., A100 = 312 TFLOPS)
- **Achieved FLOPs**: Actual throughput
- **MFU** (Model FLOPs Utilization): Achieved/Peak
- Target: 40-60% MFU for well-optimized training

## Common Pitfall
Don't forget:
- Batch size in all calculations
- Backward pass ≈ 2× forward FLOPs
- Gradient checkpointing trades FLOPs for memory (recomputation)

## Interview Quick Facts
- GPT-3 forward pass: ~3.14×10¹⁴ FLOPs per token
- LLaMA-7B: ~1.4×10¹³ FLOPs per token
- ViT-Base: ~17.6×10⁹ FLOPs per image (224×224)
