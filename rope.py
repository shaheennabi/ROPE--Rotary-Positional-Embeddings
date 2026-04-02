import math
import torch

def compute_rope_parameters(
    head_dim,
    theta_base=10000,
    context_length=4096,
    attention_factor=1.0,
    rope_type="default",
    rope_factor=1.0,
    rope_orig_max=8192,
    beta_fast=32.0,
    beta_slow=1.0,
    dtype=torch.float32,
):
    """
    Compute RoPE (Rotary Positional Embedding) parameters.

    Returns:
        cos, sin: [context_length, head_dim]
    """

    assert head_dim % 2 == 0, "head_dim must be even"
    num_freqs = head_dim // 2  # number of frequency pairs

    if rope_type == "yarn":

        def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
            """
            Map a rotation threshold -> frequency index.

            Solves:
                rotations ≈ pos / freq_i

            Returns dimension index where rotation crosses threshold.
            """
            return (
                dim
                * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings):
            """
            Compute frequency index range [low, high] where we transition
            from extrapolation → interpolation.
            """
            low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
            high = find_correction_dim(high_rot, dim, base, max_position_embeddings)

            low = math.floor(low)
            high = math.ceil(high)

            return max(low, 0), min(high, dim - 1)

        def linear_ramp_factor(low, high, dim):
            """
            Linear ramp from 0 → 1 across frequency indices.

            - below low  → 0 (pure extrapolation)
            - above high → 1 (pure interpolation)
            """
            if low == high:
                high += 1e-6

            x = torch.arange(dim, dtype=torch.float32)
            ramp = (x - low) / (high - low)
            return torch.clamp(ramp, 0, 1)

        # --- Base frequencies (log-spaced) ---
        pos_freqs = theta_base ** (
            torch.arange(0, head_dim, 2, dtype=dtype) / head_dim
        )

        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (rope_factor * pos_freqs)

        # --- Find correction region (in frequency space) ---
        low, high = find_correction_range(
            beta_fast, beta_slow, num_freqs, theta_base, rope_orig_max
        )

        # --- Build blending ramp ---
        ramp = linear_ramp_factor(low, high, num_freqs).to(dtype=dtype)

        # --- Blend frequencies ---
        inv_freq = (
            inv_freq_interpolation * ramp
            + inv_freq_extrapolation * (1 - ramp)
        )

    else:
        # Default RoPE (no scaling)
        inv_freq = 1.0 / (
            theta_base
            ** (
                torch.arange(0, head_dim, 2, dtype=dtype)[:num_freqs].float()
                / head_dim
            )
        )

    # --- Positions ---
    positions = torch.arange(context_length, dtype=dtype)

    # --- Outer product: [T, num_freqs] ---
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)

    # --- Expand to full head_dim ---
    angles = torch.cat([angles, angles], dim=1)

    # --- Compute rotation matrices ---
    cos = torch.cos(angles) * attention_factor
    sin = torch.sin(angles) * attention_factor

    return cos, sin




def apply_rope(x, sin, cos, offset=0):
    """
    Apply Rotary Positional Embedding (RoPE) to input tensor.

    Args:
        x   : Tensor of shape [batch, num_heads, seq_len, head_dim]
        sin : Precomputed sine values [context_len, head_dim]
        cos : Precomputed cosine values [context_len, head_dim]
        offset : Starting position index (used for KV-cache / generation)

    Returns:
        x_rotated : Same shape as x, with RoPE applied
    """

    # --- Shapes ---
    # x: [B, H, T, D]
    B, H, T, D = x.shape
    assert D % 2 == 0, "head_dim must be even (RoPE works on pairs)"

    # --- Split into two halves ---
    # Each index i forms a pair: (x1[i], x2[i])
    # This represents real and imaginary parts in complex form
    x1 = x[..., : D // 2]   # [B, H, T, D/2]
    x2 = x[..., D // 2 :]   # [B, H, T, D/2]

    # --- Select correct positional slice ---
    # Handles streaming / KV-cache by shifting position indices
    # Result: [T, D]
    cos = cos[offset : offset + T, :]
    sin = sin[offset : offset + T, :]

    # --- Expand for broadcasting ---
    # [T, D] → [1, 1, T, D]
    # so it can broadcast over batch and heads
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # --- Construct perpendicular (90° rotated) component ---
    # For each pair (x1, x2), this builds (-x2, x1)
    # This is equivalent to multiplying by i in complex space
    rotated = torch.cat((-x2, x1), dim=-1)  # [B, H, T, D]

    # --- Apply rotation ---
    # x*cos → original direction
    # rotated*sin → perpendicular direction
    # Combined → full rotation by angle θ
    x_rotated = (x * cos) + (rotated * sin)

    # --- Ensure dtype consistency (important for mixed precision) ---
    return x_rotated.to(dtype=x.dtype)