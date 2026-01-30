"""
Embedding layers for macroeconomic token encoding.

This module implements the additive embedding architecture as specified
in the strategy document:
    E_total = E_identity + E_type + E_importance + E_temporal + E_category + E_country
    X_token = LayerNorm(Linear(concat(E_total, [val_norm, surprise, MA5])))
"""

from typing import Dict, List, Optional
import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal offset.

    Uses continuous sinusoidal encoding based on the number of days
    since the reference date, allowing the model to understand
    temporal relationships between tokens.

    :param d_model (int): Embedding dimension
    :param max_days (int): Maximum number of days to encode
    """

    def __init__(self, d_model: int, max_days: int = 365):
        """
        Initialize positional encoding.

        :param d_model (int): Embedding dimension
        :param max_days (int): Maximum days offset
        """
        super().__init__()
        self.d_model = d_model

        # Precompute frequencies
        position = torch.arange(max_days).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_days, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, days_offset: torch.Tensor) -> torch.Tensor:
        """
        Get positional encoding for given day offsets.

        :param days_offset (torch.Tensor): Tensor of day offsets (negative for past)

        :return encoding (torch.Tensor): Positional encodings [batch, seq, d_model]
        """
        # Convert negative offsets to positive indices
        indices = torch.abs(days_offset).long()
        indices = torch.clamp(indices, 0, self.pe.size(0) - 1)

        return self.pe[indices]


class MacroTokenEmbedding(nn.Module):
    """
    Embedding layer for macroeconomic tokens.

    Implements additive embeddings:
    - E_identity: Learned embedding for each unique indicator
    - E_type: Embedding for publication type (consensus, revision, etc.)
    - E_importance: Linear projection of importance score
    - E_temporal: Sinusoidal encoding of days offset
    - E_category: Embedding for macro category
    - E_country: Embedding for country/region

    Numerical features (normalized_value, surprise, MA5) are concatenated
    with the embedding sum and projected through a linear layer.

    :param num_indicators (int): Number of unique indicators
    :param num_pub_types (int): Number of publication types
    :param num_categories (int): Number of macro categories
    :param num_regions (int): Number of regions
    :param d_model (int): Output embedding dimension
    :param d_identity (int): Identity embedding dimension
    :param d_type (int): Publication type embedding dimension
    :param d_importance (int): Importance projection dimension
    :param d_temporal (int): Temporal encoding dimension
    :param d_category (int): Category embedding dimension
    :param d_country (int): Country embedding dimension
    :param dropout (float): Dropout probability
    """

    def __init__(
        self,
        num_indicators: int,
        num_pub_types: int = 6,
        num_categories: int = 8,
        num_regions: int = 6,
        d_model: int = 64,
        d_identity: int = 32,
        d_type: int = 8,
        d_importance: int = 8,
        d_temporal: int = 16,
        d_category: int = 16,
        d_country: int = 8,
        dropout: float = 0.3,
    ):
        """
        Initialize embedding layer.

        :param num_indicators (int): Number of unique indicators
        :param num_pub_types (int): Number of publication types
        :param num_categories (int): Number of macro categories
        :param num_regions (int): Number of regions
        :param d_model (int): Output embedding dimension
        :param d_identity (int): Identity embedding dimension
        :param d_type (int): Publication type embedding dimension
        :param d_importance (int): Importance projection dimension
        :param d_temporal (int): Temporal encoding dimension
        :param d_category (int): Category embedding dimension
        :param d_country (int): Country embedding dimension
        :param dropout (float): Dropout probability
        """
        super().__init__()

        self.d_model = d_model

        # Categorical embeddings
        self.identity_embedding = nn.Embedding(num_indicators, d_identity)
        self.type_embedding = nn.Embedding(num_pub_types, d_type)
        self.category_embedding = nn.Embedding(num_categories, d_category)
        self.country_embedding = nn.Embedding(num_regions, d_country)

        # Importance projection (scalar -> d_importance)
        self.importance_projection = nn.Linear(1, d_importance)

        # Temporal encoding
        self.temporal_encoding = SinusoidalPositionalEncoding(d_temporal)

        # Total embedding dimension before projection
        d_embed_total = d_identity + d_type + d_importance + d_temporal + d_category + d_country

        # Numerical features: normalized_value, surprise, MA5
        num_numerical = 3

        # Final projection to d_model
        self.projection = nn.Linear(d_embed_total + num_numerical, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        indicator_ids: torch.Tensor,
        pub_type_ids: torch.Tensor,
        category_ids: torch.Tensor,
        country_ids: torch.Tensor,
        importance: torch.Tensor,
        days_offset: torch.Tensor,
        normalized_value: torch.Tensor,
        surprise: torch.Tensor,
        ma5: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute token embeddings.

        :param indicator_ids (torch.Tensor): Indicator indices [batch, seq]
        :param pub_type_ids (torch.Tensor): Publication type indices [batch, seq]
        :param category_ids (torch.Tensor): Category indices [batch, seq]
        :param country_ids (torch.Tensor): Country indices [batch, seq]
        :param importance (torch.Tensor): Importance scores [batch, seq]
        :param days_offset (torch.Tensor): Days offset [batch, seq]
        :param normalized_value (torch.Tensor): Normalized values [batch, seq]
        :param surprise (torch.Tensor): Surprises [batch, seq]
        :param ma5 (torch.Tensor): 5-period MAs [batch, seq]

        :return embeddings (torch.Tensor): Token embeddings [batch, seq, d_model]
        """
        # Categorical embeddings
        e_identity = self.identity_embedding(indicator_ids)
        e_type = self.type_embedding(pub_type_ids)
        e_category = self.category_embedding(category_ids)
        e_country = self.country_embedding(country_ids)

        # Importance projection
        e_importance = self.importance_projection(importance.unsqueeze(-1))

        # Temporal encoding
        e_temporal = self.temporal_encoding(days_offset)

        # Sum categorical embeddings (additive approach)
        e_total = torch.cat(
            [e_identity, e_type, e_category, e_country, e_importance, e_temporal],
            dim=-1,
        )

        # Concatenate numerical features
        numerical = torch.stack([normalized_value, surprise, ma5], dim=-1)
        combined = torch.cat([e_total, numerical], dim=-1)

        # Project and normalize
        output = self.projection(combined)
        output = self.layer_norm(output)
        output = self.dropout(output)

        return output


class MarketContextEmbedding(nn.Module):
    """
    Embedding for market context indicators (VIX, credit spread, yield curve).

    These are treated as special tokens that provide market regime context.

    :param d_model (int): Output embedding dimension
    :param dropout (float): Dropout probability
    """

    def __init__(self, d_model: int = 64, dropout: float = 0.3):
        """
        Initialize market context embedding.

        :param d_model (int): Output embedding dimension
        :param dropout (float): Dropout probability
        """
        super().__init__()

        # 3 market context features: VIX, credit_spread, yield_curve
        self.projection = nn.Linear(3, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Learnable type embedding for market context token
        self.type_embedding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, market_context: torch.Tensor) -> torch.Tensor:
        """
        Compute market context embedding.

        :param market_context (torch.Tensor): [batch, 3] (VIX, spread, curve)

        :return embedding (torch.Tensor): [batch, 1, d_model]
        """
        # Normalize inputs
        market_context = market_context / torch.tensor([30.0, 5.0, 2.0]).to(
            market_context.device
        )

        projected = self.projection(market_context).unsqueeze(1)
        output = projected + self.type_embedding.expand(projected.size(0), -1, -1)
        output = self.layer_norm(output)
        output = self.dropout(output)

        return output


class TokenEncoder(nn.Module):
    """
    Combined encoder for macro tokens and market context.

    Prepares the full input sequence for the Transformer.

    :param num_indicators (int): Number of unique indicators
    :param d_model (int): Model dimension
    :param dropout (float): Dropout probability
    """

    def __init__(
        self,
        num_indicators: int,
        d_model: int = 64,
        dropout: float = 0.3,
    ):
        """
        Initialize token encoder.

        :param num_indicators (int): Number of unique indicators
        :param d_model (int): Model dimension
        :param dropout (float): Dropout probability
        """
        super().__init__()

        self.macro_embedding = MacroTokenEmbedding(
            num_indicators=num_indicators,
            d_model=d_model,
            dropout=dropout,
        )
        self.market_embedding = MarketContextEmbedding(
            d_model=d_model,
            dropout=dropout,
        )

    def forward(
        self,
        macro_batch: Dict[str, torch.Tensor],
        market_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode full input sequence.

        :param macro_batch (Dict[str, torch.Tensor]): Dictionary with macro features
        :param market_context (torch.Tensor): Market context [batch, 3]

        :return sequence (torch.Tensor): Full sequence [batch, seq+1, d_model]
        """
        # Encode macro tokens
        macro_embeddings = self.macro_embedding(
            indicator_ids=macro_batch["indicator_ids"],
            pub_type_ids=macro_batch["pub_type_ids"],
            category_ids=macro_batch["category_ids"],
            country_ids=macro_batch["country_ids"],
            importance=macro_batch["importance"],
            days_offset=macro_batch["days_offset"],
            normalized_value=macro_batch["normalized_value"],
            surprise=macro_batch["surprise"],
            ma5=macro_batch["ma5"],
        )

        # Encode market context as a single token
        market_token = self.market_embedding(market_context)

        # Prepend market context token to sequence
        sequence = torch.cat([market_token, macro_embeddings], dim=1)

        return sequence
