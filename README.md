# RoPE: Rotary Positional Embeddings

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean, efficient implementation of Rotary Positional Embeddings (RoPE) for transformers, with support for advanced variants like YARN scaling.

## What is RoPE?

Rotary Positional Embedding (RoPE) is a modern technique for injecting positional information into transformer models. Unlike traditional absolute or relative positional embeddings, RoPE encodes position using **rotations in a complex plane**, providing several key advantages:

- **Extrapolation**: Works well beyond training sequence lengths
- **Efficiency**: No additional parameters to learn
- **Long-range modeling**: Better at capturing long-distance dependencies
- **Multi-head compatibility**: Naturally supports different attention patterns per head

## How RoPE Works

### The Core Idea

RoPE treats each attention head's embedding dimension as a **complex vector** in 2D space. Position information is encoded by **rotating** these vectors by an angle proportional to their position.

For a query/key vector pair at positions `m` and `n`, RoPE applies a rotation by angles `θᵢ = m * ωᵢ` and `θᵢ = n * ωᵢ` respectively, where `ωᵢ` are learned frequencies.

### Mathematical Foundation

Consider a 2D vector `[x₁, x₂]` representing real and imaginary parts of a complex number `x₁ + i x₂`.

**Rotation by angle θ:**
```
[x₁', x₂'] = [x₁ cosθ - x₂ sinθ, x₁ sinθ + x₂ cosθ]
```

**In complex notation:**
```
(x₁ + i x₂) * e^(iθ) = (x₁ cosθ - x₂ sinθ) + i (x₁ sinθ + x₂ cosθ)
```

**RoPE applies this to attention:**
- Query at position m: `qᵢ ↗ e^(i m ωᵢ)`
- Key at position n: `kᵢ ↗ e^(i n ωᵢ)`
- Attention becomes: `⟨q, k⟩ = Σᵢ (qᵢ kᵢ*) e^(i(m-n)ωᵢ)`

The key insight: **relative position encoding emerges naturally** from the complex exponential!

### Frequency Design

RoPE uses **log-spaced frequencies** to capture different granularity levels:

```python
# Base frequencies (θ_base = 10000 is common)
ωᵢ = θ_base^(-2i/d) for i in 0, 1, ..., d/2-1
```

Where `d` is the head dimension. This creates a geometric progression of frequencies, allowing the model to attend to both fine-grained local patterns and coarse-grained global structure.

## Variants

### Default RoPE
Standard implementation as described in the original paper. Frequencies are fixed and work well for sequences up to ~2x training length.

### YARN (YAml-based Rotary scaling)
An advanced variant that enables **much longer context windows** through:
- **Extrapolation**: Smooth frequency scaling for positions beyond training
- **Interpolation**: Fine-tuned frequencies for trained positions
- **Dynamic scaling**: Context-dependent frequency adjustment

YARN introduces parameters like `β_fast`, `β_slow`, and `rope_factor` to control the scaling behavior.

## Installation

```bash
pip install torch  # Only dependency
```

Or clone and use locally:
```bash
git clone https://github.com/shaheennabi/ROPE--Rotary-Positional-Embeddings
cd rope-rotary-positional-embeddings
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
import torch
from rope import compute_rope_parameters, apply_rope

# Model configuration
head_dim = 64
seq_len = 1024
batch_size = 2
num_heads = 8

# 1. Precompute RoPE parameters (do this once)
cos, sin = compute_rope_parameters(
    head_dim=head_dim,
    context_length=seq_len,
    theta_base=10000.0
)

# 2. Create sample input (query/key vectors)
x = torch.randn(batch_size, num_heads, seq_len, head_dim)

# 3. Apply RoPE
x_rotated = apply_rope(x, sin, cos)
```

### YARN Scaling for Long Contexts

```python
# Enable YARN for 128K context (trained on 8K)
cos, sin = compute_rope_parameters(
    head_dim=head_dim,
    context_length=128000,  # Much longer than training
    rope_type="yarn",
    rope_factor=8.0,        # Scaling factor
    rope_orig_max=8192,     # Original training length
    beta_fast=32.0,
    beta_slow=1.0
)
```

### Streaming/Inference with KV Cache

```python
# For generation, apply RoPE with offset
current_pos = 50  # Current generation position
x_rotated = apply_rope(x, sin, cos, offset=current_pos)
```

## API Reference

### `compute_rope_parameters()`

Computes the cosine and sine rotation matrices for RoPE.

**Parameters:**
- `head_dim` (int): Embedding dimension per head (must be even)
- `theta_base` (float): Base frequency (default: 10000)
- `context_length` (int): Maximum sequence length
- `attention_factor` (float): Scaling factor for attention (default: 1.0)
- `rope_type` (str): "default" or "yarn"
- `rope_factor` (float): YARN scaling factor
- `rope_orig_max` (int): YARN original training length
- `beta_fast` (float): YARN fast decay parameter
- `beta_slow` (float): YARN slow decay parameter
- `dtype`: PyTorch data type

**Returns:** `(cos, sin)` tensors of shape `[context_length, head_dim]`

### `apply_rope()`

Applies RoPE rotation to input tensors.

**Parameters:**
- `x`: Input tensor `[batch, num_heads, seq_len, head_dim]`
- `sin`: Precomputed sine values
- `cos`: Precomputed cosine values
- `offset` (int): Position offset for KV-cache

**Returns:** Rotated tensor with same shape as input

## Performance Notes

- **Memory efficient**: Parameters computed once, reused for all sequences
- **GPU optimized**: All operations vectorized with PyTorch
- **Mixed precision ready**: Supports float16/bfloat16
- **KV-cache friendly**: Offset parameter enables efficient generation

## Why RoPE vs Other Methods?

| Method | Parameters | Extrapolation | Efficiency |
|--------|------------|---------------|------------|
| Absolute PE | O(seq_len × d) | ❌ Poor | ❌ High memory |
| Relative PE | O(seq_len² × d) | ❌ Poor | ❌ Quadratic |
| **RoPE** | **O(d)** | ✅ Excellent | ✅ Minimal |

## References

1. **RoPE Paper**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
2. **YARN**: [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071)
3. **Llama 2**: Uses RoPE for long context modeling
4. **GPT-J**: Early adopter of rotary embeddings

## Contributing

Contributions welcome! Please feel free to submit issues and pull requests.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

*Built with ❤️ for better transformer positional encoding*
