"""
Transformer model for factor allocation prediction.

This module implements a MICRO Transformer architecture optimized for
small datasets (~300 monthly samples). Key design choices:
- 1 layer only (minimize overfitting)
- 32 embedding dimension (reduced from 64-128)
- 1 attention head (simplest attention)
- Very high dropout (0.6)
- Causal masking
- Relative positional encoding

Target: <10k parameters to avoid overfitting on limited macro data.
"""

from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import TokenEncoder


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) for encoding relative positions.

    RoPE encodes positions by rotating Q and K vectors, allowing the model
    to capture relative positional information directly in attention scores.
    This is particularly effective for time series data.

    Reference: RoFormer (Su et al., 2021)

    :param dim (int): Dimension of the embeddings (must be divisible by 2)
    :param max_seq_len (int): Maximum sequence length
    :param base (float): Base for frequency computation
    """

    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        """
        Initialize RoPE.

        :param dim (int): Dimension (must be divisible by 2)
        :param max_seq_len (int): Maximum sequence length
        :param base (float): Base for frequency computation
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute sin/cos for all positions
        self._precompute_cache(max_seq_len)

    def _precompute_cache(self, seq_len: int) -> None:
        """
        Precompute sin/cos cache for efficiency.

        :param seq_len (int): Sequence length to cache
        """
        positions = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)

        # Cache sin and cos
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply rotary embeddings to input tensor.

        :param x (torch.Tensor): Input [batch, seq, heads, head_dim]
        :param positions (Optional[torch.Tensor]): Custom positions [batch, seq] (for temporal data)

        :return rotated (torch.Tensor): Rotated embeddings
        """
        batch_size, seq_len, num_heads, head_dim = x.shape

        # Use precomputed cache or compute on-the-fly
        if positions is None:
            # Standard sequential positions: [seq_len, dim//2]
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]
            # Expand to [1, seq_len, 1, dim//2] for broadcasting
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        else:
            # Custom positions (e.g., days offset for temporal data)
            # positions: [batch, seq] -> clamp to valid range
            positions = positions.clamp(0, self.max_seq_len - 1).long()
            # Index into cached values: [batch, seq, dim//2]
            cos = self.cos_cached[positions]
            sin = self.sin_cached[positions]
            # Expand to [batch, seq, 1, dim//2] for broadcasting with heads
            cos = cos.unsqueeze(2)
            sin = sin.unsqueeze(2)

        return self._apply_rotary(x, cos, sin)

    def _apply_rotary(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply rotary transformation.

        :param x (torch.Tensor): Input tensor [batch, seq, heads, head_dim]
        :param cos (torch.Tensor): Cosine values [batch/1, seq, 1, dim//2]
        :param sin (torch.Tensor): Sine values [batch/1, seq, 1, dim//2]

        :return rotated (torch.Tensor): Rotated tensor
        """
        # Split head_dim into pairs for rotation
        half_dim = x.shape[-1] // 2
        x1, x2 = x[..., :half_dim], x[..., half_dim:]

        # Apply rotation: [cos(θ), -sin(θ); sin(θ), cos(θ)] @ [x1; x2]
        rotated = torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin,
        ], dim=-1)

        return rotated


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with causal masking and optional RoPE.

    Includes optional soft temporal decay for attention weights and
    Rotary Position Embeddings (RoPE) for relative position encoding.

    :param d_model (int): Model dimension
    :param num_heads (int): Number of attention heads
    :param dropout (float): Dropout probability
    :param use_temporal_decay (bool): Apply soft temporal decay to attention
    :param use_rope (bool): Use Rotary Position Embeddings
    :param max_seq_len (int): Maximum sequence length for RoPE
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.6,
        use_temporal_decay: bool = True,
        use_rope: bool = True,
        max_seq_len: int = 512,
    ):
        """
        Initialize attention layer.

        :param d_model (int): Model dimension
        :param num_heads (int): Number of attention heads
        :param dropout (float): Dropout probability
        :param use_temporal_decay (bool): Use temporal decay
        :param use_rope (bool): Use Rotary Position Embeddings
        :param max_seq_len (int): Maximum sequence length for RoPE
        """
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_temporal_decay = use_temporal_decay
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Learnable temporal decay rate
        if use_temporal_decay:
            self.decay_rate = nn.Parameter(torch.tensor(0.1))

        # Rotary Position Embeddings
        if use_rope:
            self.rotary_emb = RotaryPositionalEmbedding(
                dim=self.head_dim,
                max_seq_len=max_seq_len,
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        days_offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute multi-head attention with optional RoPE.

        :param x (torch.Tensor): Input [batch, seq, d_model]
        :param mask (Optional[torch.Tensor]): Attention mask [batch, seq, seq]
        :param days_offset (Optional[torch.Tensor]): Days offset for RoPE/decay [batch, seq]

        :return output (torch.Tensor): Output [batch, seq, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply RoPE to Q and K before transposing
        if self.use_rope:
            # Use days_offset as positions if available (for temporal data)
            positions = days_offset.abs() if days_offset is not None else None
            q = self.rotary_emb(q, positions)
            k = self.rotary_emb(k, positions)

        # Transpose for attention: [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply temporal decay if enabled
        if self.use_temporal_decay and days_offset is not None:
            temporal_distance = self._compute_temporal_distance(days_offset)
            decay = torch.exp(-self.decay_rate * temporal_distance)
            scores = scores * decay.unsqueeze(1)

        # Apply mask (causal + padding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        output = torch.matmul(attn_weights, v)

        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        return output

    def _compute_temporal_distance(self, days_offset: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise temporal distance matrix.

        :param days_offset (torch.Tensor): Days offset [batch, seq]

        :return distance (torch.Tensor): Pairwise distance [batch, seq, seq]
        """
        # Expand for pairwise computation
        offset_i = days_offset.unsqueeze(-1)
        offset_j = days_offset.unsqueeze(-2)

        # Absolute difference in days
        distance = torch.abs(offset_i - offset_j).float()

        return distance


class TransformerBlock(nn.Module):
    """
    Single Transformer block with attention and feedforward.

    Includes skip connections (residual) for learning linear relationships.
    Uses Pre-LN architecture (LayerNorm before attention/FFN).

    :param d_model (int): Model dimension
    :param num_heads (int): Number of attention heads
    :param d_ff (int): Feedforward dimension
    :param dropout (float): Dropout probability
    :param use_rope (bool): Use Rotary Position Embeddings
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.6,
        use_rope: bool = True,
    ):
        """
        Initialize Transformer block.

        :param d_model (int): Model dimension
        :param num_heads (int): Number of attention heads
        :param d_ff (int): Feedforward dimension
        :param dropout (float): Dropout probability
        :param use_rope (bool): Use Rotary Position Embeddings
        """
        super().__init__()

        self.attention = MultiHeadAttention(
            d_model, num_heads, dropout, use_rope=use_rope
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        days_offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through Transformer block.

        :param x (torch.Tensor): Input [batch, seq, d_model]
        :param mask (Optional[torch.Tensor]): Attention mask
        :param days_offset (Optional[torch.Tensor]): Days offset

        :return output (torch.Tensor): Output [batch, seq, d_model]
        """
        # Pre-norm attention with residual
        attn_out = self.attention(self.norm1(x), mask, days_offset)
        x = x + attn_out

        # Pre-norm feedforward with residual
        ff_out = self.ff(self.norm2(x))
        x = x + ff_out

        return x


class FactorAllocationTransformer(nn.Module):
    """
    Transformer model for factor allocation prediction.

    Architecture:
    1. Token encoding (macro + market context)
    2. Transformer blocks with causal attention + RoPE
    3. Pooling (mean or CLS token)
    4. Output head for allocation weights

    :param num_indicators (int): Number of unique macro indicators
    :param num_factors (int): Number of output factor categories
    :param d_model (int): Model dimension
    :param num_heads (int): Number of attention heads
    :param num_layers (int): Number of Transformer layers
    :param d_ff (int): Feedforward dimension
    :param dropout (float): Dropout probability
    :param max_seq_len (int): Maximum sequence length
    :param use_rope (bool): Use Rotary Position Embeddings
    """

    def __init__(
        self,
        num_indicators: int,
        num_factors: int = 6,
        d_model: int = 32,
        num_heads: int = 1,
        num_layers: int = 1,
        d_ff: int = 64,
        dropout: float = 0.6,
        max_seq_len: int = 100,
        use_rope: bool = True,
    ):
        """
        Initialize the Transformer model.

        :param num_indicators (int): Number of unique macro indicators
        :param num_factors (int): Number of factor categories
        :param d_model (int): Model dimension
        :param num_heads (int): Number of attention heads
        :param num_layers (int): Number of Transformer layers
        :param d_ff (int): Feedforward dimension
        :param dropout (float): Dropout probability
        :param max_seq_len (int): Maximum sequence length
        :param use_rope (bool): Use Rotary Position Embeddings
        """
        super().__init__()

        self.d_model = d_model
        self.num_factors = num_factors

        # Token encoder
        self.token_encoder = TokenEncoder(
            num_indicators=num_indicators,
            d_model=d_model,
            dropout=dropout,
        )

        # Transformer blocks with RoPE
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout, use_rope=use_rope)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Output heads
        # Binary classification head (phase 1)
        self.binary_head = nn.Linear(d_model, 1)

        # Regression head (phase 2)
        self.regression_head = nn.Linear(d_model, 1)

        # Full allocation head (phase 3) - simplified to single linear layer
        self.allocation_head = nn.Linear(d_model, num_factors)

        # Causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len + 1, max_seq_len + 1)),
        )

    def forward(
        self,
        macro_batch: Dict[str, torch.Tensor],
        market_context: torch.Tensor,
        output_type: str = "binary",
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        :param macro_batch (Dict[str, torch.Tensor]): Macro token features
        :param market_context (torch.Tensor): Market context [batch, 3]
        :param output_type (str): Output type ('binary', 'regression', 'allocation')

        :return output (torch.Tensor): Model predictions
        """
        # Encode tokens
        x = self.token_encoder(macro_batch, market_context)
        batch_size, seq_len, _ = x.shape

        # Get causal mask
        mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0)

        # Get days offset (prepend 0 for market context token)
        days_offset = macro_batch.get("days_offset")
        if days_offset is not None:
            market_offset = torch.zeros(batch_size, 1, device=days_offset.device)
            days_offset = torch.cat([market_offset, days_offset], dim=1)

        # Pass through Transformer layers
        for layer in self.layers:
            x = layer(x, mask, days_offset)

        x = self.final_norm(x)

        # Pool: use mean of all tokens
        pooled = x.mean(dim=1)

        # Output based on training phase
        if output_type == "binary":
            return torch.sigmoid(self.binary_head(pooled))
        elif output_type == "regression":
            return self.regression_head(pooled)
        elif output_type == "allocation":
            return F.softmax(self.allocation_head(pooled), dim=-1)
        else:
            raise ValueError(f"Unknown output_type: {output_type}")

    def get_attention_weights(
        self,
        macro_batch: Dict[str, torch.Tensor],
        market_context: torch.Tensor,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """
        Extract attention weights for interpretability.

        :param macro_batch (Dict[str, torch.Tensor]): Macro token features
        :param market_context (torch.Tensor): Market context
        :param layer_idx (int): Which layer's attention to extract

        :return weights (torch.Tensor): Attention weights [batch, heads, seq, seq]
        """
        x = self.token_encoder(macro_batch, market_context)
        batch_size, seq_len, _ = x.shape

        mask = self.causal_mask[:seq_len, :seq_len].unsqueeze(0)

        # Pass through layers until target
        for i, layer in enumerate(self.layers):
            if i == layer_idx:
                # Get attention weights from this layer
                attn = layer.attention
                q = attn.q_proj(attn.norm1(x) if hasattr(attn, "norm1") else x)
                k = attn.k_proj(attn.norm1(x) if hasattr(attn, "norm1") else x)

                q = q.view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, attn.num_heads, attn.head_dim).transpose(1, 2)

                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(attn.head_dim)
                scores = scores.masked_fill(mask == 0, float("-inf"))
                weights = F.softmax(scores, dim=-1)

                return weights

            x = layer(x, mask)

        return torch.zeros(1)

    def count_parameters(self) -> int:
        """
        Count total trainable parameters.

        :return count (int): Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineRegularization(nn.Module):
    """
    Regularization toward a simple baseline model (ridge regression).

    As per strategy document Section 1.4.3:
    "Baseline regularization: Loss term penalizing deviations from a simple
    econometric model (ridge regression)"

    This helps inject prior economic knowledge by keeping the model
    close to a simple linear baseline, reducing overfitting.

    :param num_features (int): Number of input features for baseline
    :param num_outputs (int): Number of outputs (factors)
    :param alpha (float): Ridge regularization strength
    """

    def __init__(
        self,
        num_features: int,
        num_outputs: int = 6,
        alpha: float = 1.0,
    ):
        """
        Initialize baseline regularization.

        :param num_features (int): Input feature dimension
        :param num_outputs (int): Output dimension
        :param alpha (float): Ridge penalty strength
        """
        super().__init__()
        self.alpha = alpha

        # Simple linear baseline (ridge regression)
        self.baseline = nn.Linear(num_features, num_outputs)

        # Initialize with small weights
        nn.init.xavier_uniform_(self.baseline.weight, gain=0.1)
        nn.init.zeros_(self.baseline.bias)

    def forward(
        self,
        model_output: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute regularization loss.

        :param model_output (torch.Tensor): Model predictions [batch, num_outputs]
        :param features (torch.Tensor): Input features for baseline [batch, num_features]

        :return reg_loss (torch.Tensor): Regularization loss
        """
        # Baseline prediction (no gradient through baseline)
        with torch.no_grad():
            baseline_pred = self.baseline(features)

        # MSE between model output and baseline
        deviation = (model_output - baseline_pred).pow(2).mean()

        # Ridge penalty on baseline weights
        ridge_penalty = self.baseline.weight.pow(2).mean()

        return deviation + self.alpha * ridge_penalty

    def fit_baseline(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        lr: float = 0.01,
        epochs: int = 100,
    ) -> None:
        """
        Fit the baseline model on training data.

        :param features (torch.Tensor): Input features [n_samples, num_features]
        :param targets (torch.Tensor): Target values [n_samples, num_outputs]
        :param lr (float): Learning rate
        :param epochs (int): Number of epochs
        """
        optimizer = torch.optim.Adam(self.baseline.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for _ in range(epochs):
            optimizer.zero_grad()
            pred = self.baseline(features)
            loss = criterion(pred, targets) + self.alpha * self.baseline.weight.pow(2).mean()
            loss.backward()
            optimizer.step()


class SharpeRatioLoss(nn.Module):
    """
    Differentiable Sharpe ratio loss with running std for gradient stability.

    Loss = -mean(R) / std(R) + lambda * turnover + beta * baseline_deviation

    Uses running std estimation across batches to stabilize gradients.

    :param gamma (float): Risk aversion coefficient (0 = pure Sharpe, >0 adds variance penalty)
    :param turnover_penalty (float): Turnover penalty coefficient (calibrated)
    :param baseline_penalty (float): Baseline regularization penalty
    :param use_running_stats (bool): Use running std for stable gradients
    :param momentum (float): Momentum for running std update
    """

    def __init__(
        self,
        gamma: float = 0.0,
        turnover_penalty: float = 0.01,
        baseline_penalty: float = 0.0,
        use_running_stats: bool = True,
        momentum: float = 0.99,
    ):
        """
        Initialize Sharpe ratio loss.

        :param gamma (float): Risk aversion (0 = pure Sharpe, >0 adds variance penalty)
        :param turnover_penalty (float): Penalty for weight changes
        :param baseline_penalty (float): Penalty for deviation from baseline
        :param use_running_stats (bool): Use running std for stable gradients
        :param momentum (float): Momentum for running std update
        """
        super().__init__()
        self.gamma = gamma
        self.turnover_penalty = turnover_penalty
        self.baseline_penalty = baseline_penalty
        self.use_running_stats = use_running_stats
        self.momentum = momentum
        # Running std buffer for stable gradients
        self.register_buffer('running_std', torch.tensor(0.01))

    def forward(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        prev_weights: Optional[torch.Tensor] = None,
        baseline_deviation: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute true Sharpe ratio loss.

        :param weights (torch.Tensor): Predicted weights [batch, num_factors]
        :param returns (torch.Tensor): Factor returns [batch, num_factors]
        :param prev_weights (Optional[torch.Tensor]): Previous weights for turnover
        :param baseline_deviation (Optional[torch.Tensor]): Deviation from baseline

        :return loss (torch.Tensor): Scalar loss
        """
        # Portfolio return
        portfolio_return = (weights * returns).sum(dim=-1)

        # Mean and std
        mean_return = portfolio_return.mean()
        batch_std = portfolio_return.std()

        # Update running std (only in training mode)
        if self.training and self.use_running_stats:
            with torch.no_grad():
                self.running_std = self.momentum * self.running_std + (1 - self.momentum) * batch_std

        # Use running std for stable gradients, or batch std
        std_return = self.running_std + 1e-8 if self.use_running_stats else batch_std + 1e-8

        # TRUE Sharpe ratio (annualized: sqrt(12) for monthly data)
        sharpe_ratio = mean_return / std_return * 3.464  # sqrt(12) = 3.464

        # Base loss: negative Sharpe (minimize loss = maximize Sharpe)
        loss = -sharpe_ratio

        # Optional: additional variance penalty if gamma > 0
        if self.gamma > 0:
            variance_penalty = self.gamma * portfolio_return.var()
            loss = loss + variance_penalty

        # Turnover penalty
        if prev_weights is not None:
            turnover = torch.abs(weights - prev_weights).sum(dim=-1).mean()
            loss = loss + self.turnover_penalty * turnover

        # Baseline regularization
        if baseline_deviation is not None and self.baseline_penalty > 0:
            loss = loss + self.baseline_penalty * baseline_deviation

        return loss


class SortinoLoss(nn.Module):
    """
    Differentiable Sortino ratio loss - penalizes only downside volatility.

    The Sortino ratio is similar to Sharpe but uses downside deviation
    (std of negative returns) instead of total volatility. This is more
    appropriate for investment strategies where upside volatility is desirable.

    Loss = -mean(R) / downside_std + lambda * turnover

    :param target_return (float): Target return for downside calculation (default 0)
    :param turnover_penalty (float): Turnover penalty coefficient
    :param use_running_stats (bool): Use running downside std for stable gradients
    :param momentum (float): Momentum for running stats update
    :param min_downside_samples (int): Minimum negative samples for valid std
    """

    def __init__(
        self,
        target_return: float = 0.0,
        turnover_penalty: float = 0.01,
        use_running_stats: bool = True,
        momentum: float = 0.99,
        min_downside_samples: int = 5,
    ):
        """
        Initialize Sortino loss.

        :param target_return (float): Target return threshold (typically 0 or risk-free rate)
        :param turnover_penalty (float): Penalty for weight changes
        :param use_running_stats (bool): Use running downside std for stable gradients
        :param momentum (float): Momentum for running stats update
        :param min_downside_samples (int): Minimum downside samples for valid std
        """
        super().__init__()
        self.target_return = target_return
        self.turnover_penalty = turnover_penalty
        self.use_running_stats = use_running_stats
        self.momentum = momentum
        self.min_downside_samples = min_downside_samples
        # Running downside std buffer for stable gradients
        self.register_buffer('running_downside_std', torch.tensor(0.01))

    def forward(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        prev_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Sortino ratio loss.

        :param weights (torch.Tensor): Predicted weights [batch, num_factors]
        :param returns (torch.Tensor): Factor returns [batch, num_factors]
        :param prev_weights (Optional[torch.Tensor]): Previous weights for turnover

        :return loss (torch.Tensor): Scalar loss
        """
        # Portfolio return
        portfolio_return = (weights * returns).sum(dim=-1)

        # Mean return
        mean_return = portfolio_return.mean()

        # Downside deviation: std of returns below target
        downside_returns = portfolio_return - self.target_return
        downside_mask = downside_returns < 0

        # Count downside samples
        n_downside = downside_mask.sum()

        if n_downside >= self.min_downside_samples:
            # Compute downside std using only negative returns
            downside_squared = (downside_returns * downside_mask.float()).pow(2)
            downside_variance = downside_squared.sum() / n_downside
            batch_downside_std = torch.sqrt(downside_variance + 1e-8)
        else:
            # Fallback to regular std if not enough downside samples
            batch_downside_std = portfolio_return.std() + 1e-8

        # Update running downside std (only in training mode)
        if self.training and self.use_running_stats:
            with torch.no_grad():
                self.running_downside_std = (
                    self.momentum * self.running_downside_std
                    + (1 - self.momentum) * batch_downside_std
                )

        # Use running downside std for stable gradients
        downside_std = (
            self.running_downside_std + 1e-8
            if self.use_running_stats
            else batch_downside_std
        )

        # Sortino ratio (annualized: sqrt(12) for monthly data)
        sortino_ratio = mean_return / downside_std * 3.464  # sqrt(12) = 3.464

        # Base loss: negative Sortino (minimize loss = maximize Sortino)
        loss = -sortino_ratio

        # Turnover penalty
        if prev_weights is not None:
            turnover = torch.abs(weights - prev_weights).sum(dim=-1).mean()
            loss = loss + self.turnover_penalty * turnover

        return loss


def calibrate_turnover_penalty(
    transaction_cost_bps: float = 10.0,
    expected_turnover: float = 0.3,
    holding_period_months: int = 1,
) -> float:
    """
    Calibrate turnover penalty based on transaction costs.

    As per strategy document Section 1.5.2:
    "The coefficient λ must be calibrated based on actual transaction costs."

    :param transaction_cost_bps (float): One-way transaction cost in basis points
    :param expected_turnover (float): Expected monthly turnover (0-1)
    :param holding_period_months (int): Average holding period

    :return lambda_penalty (float): Calibrated turnover penalty
    """
    # Convert bps to decimal
    tc_decimal = transaction_cost_bps / 10000.0

    # Two-way cost (buy + sell)
    two_way_cost = 2 * tc_decimal

    # Annualized cost impact
    annual_trades = 12 / holding_period_months
    annual_cost = two_way_cost * expected_turnover * annual_trades

    # Scale to match Sharpe ratio magnitude (typical Sharpe ~0.5-1.0)
    # Penalty should make turnover cost visible in loss
    lambda_penalty = annual_cost * 10.0

    return lambda_penalty
