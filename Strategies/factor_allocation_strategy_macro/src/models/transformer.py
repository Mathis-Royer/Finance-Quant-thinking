"""
Transformer model for factor allocation prediction.

This module implements a minimal Transformer architecture as specified
in the strategy document, with:
- 2-4 layers
- 64-128 embedding dimension
- 2-4 attention heads
- High dropout (0.3-0.5)
- Causal masking
- Relative positional encoding
"""

from typing import Dict, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import TokenEncoder


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with causal masking.

    Includes optional soft temporal decay for attention weights.

    :param d_model (int): Model dimension
    :param num_heads (int): Number of attention heads
    :param dropout (float): Dropout probability
    :param use_temporal_decay (bool): Apply soft temporal decay to attention
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.3,
        use_temporal_decay: bool = True,
    ):
        """
        Initialize attention layer.

        :param d_model (int): Model dimension
        :param num_heads (int): Number of attention heads
        :param dropout (float): Dropout probability
        :param use_temporal_decay (bool): Use temporal decay
        """
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_temporal_decay = use_temporal_decay

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

        # Learnable temporal decay rate
        if use_temporal_decay:
            self.decay_rate = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        days_offset: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        :param x (torch.Tensor): Input [batch, seq, d_model]
        :param mask (Optional[torch.Tensor]): Attention mask [batch, seq, seq]
        :param days_offset (Optional[torch.Tensor]): Days offset for decay [batch, seq]

        :return output (torch.Tensor): Output [batch, seq, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)

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

    :param d_model (int): Model dimension
    :param num_heads (int): Number of attention heads
    :param d_ff (int): Feedforward dimension
    :param dropout (float): Dropout probability
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.3,
    ):
        """
        Initialize Transformer block.

        :param d_model (int): Model dimension
        :param num_heads (int): Number of attention heads
        :param d_ff (int): Feedforward dimension
        :param dropout (float): Dropout probability
        """
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
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
    2. Transformer blocks with causal attention
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
    """

    def __init__(
        self,
        num_indicators: int,
        num_factors: int = 6,
        d_model: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.4,
        max_seq_len: int = 100,
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

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Output heads
        # Binary classification head (phase 1)
        self.binary_head = nn.Linear(d_model, 1)

        # Regression head (phase 2)
        self.regression_head = nn.Linear(d_model, 1)

        # Full allocation head (phase 3)
        self.allocation_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_factors),
        )

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


class SharpeRatioLoss(nn.Module):
    """
    Differentiable approximation of negative Sharpe ratio loss.

    Loss = -E[R] + gamma * Var[R] + lambda * turnover

    :param gamma (float): Risk aversion coefficient
    :param turnover_penalty (float): Turnover penalty coefficient
    """

    def __init__(self, gamma: float = 1.0, turnover_penalty: float = 0.01):
        """
        Initialize Sharpe ratio loss.

        :param gamma (float): Risk aversion (higher = more risk-averse)
        :param turnover_penalty (float): Penalty for weight changes
        """
        super().__init__()
        self.gamma = gamma
        self.turnover_penalty = turnover_penalty

    def forward(
        self,
        weights: torch.Tensor,
        returns: torch.Tensor,
        prev_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute Sharpe-based loss.

        :param weights (torch.Tensor): Predicted weights [batch, num_factors]
        :param returns (torch.Tensor): Factor returns [batch, num_factors]
        :param prev_weights (Optional[torch.Tensor]): Previous weights for turnover

        :return loss (torch.Tensor): Scalar loss
        """
        # Portfolio return
        portfolio_return = (weights * returns).sum(dim=-1)

        # Mean and variance
        mean_return = portfolio_return.mean()
        var_return = portfolio_return.var()

        # Base loss (negative Sharpe approximation)
        loss = -mean_return + self.gamma * var_return

        # Turnover penalty
        if prev_weights is not None:
            turnover = torch.abs(weights - prev_weights).sum(dim=-1).mean()
            loss = loss + self.turnover_penalty * turnover

        return loss
