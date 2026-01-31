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

    Implements ADDITIVE embeddings as specified in strategy document (Section 1.3):
        E_total = E_identity + E_type + E_importance + E_temporal + E_category + E_country + E_periodicity
        X_token = LayerNorm(Linear(concat(E_total, [val_norm, surprise, MA5])))

    All categorical embeddings share the same dimension (d_embed) so they can be SUMMED.
    This is the BERT-like approach that allows learning shared semantic space.

    :param num_indicators (int): Number of unique indicators
    :param num_pub_types (int): Number of publication types
    :param num_categories (int): Number of macro categories
    :param num_regions (int): Number of regions
    :param num_periodicities (int): Number of periodicity types (daily, weekly, monthly, quarterly, irregular)
    :param d_model (int): Output embedding dimension
    :param d_embed (int): Shared embedding dimension for all categorical embeddings
    :param dropout (float): Dropout probability
    """

    def __init__(
        self,
        num_indicators: int,
        num_pub_types: int = 6,
        num_categories: int = 8,
        num_regions: int = 6,
        num_periodicities: int = 5,
        d_model: int = 32,
        d_embed: int = 16,
        dropout: float = 0.5,
    ):
        """
        Initialize embedding layer with ADDITIVE embeddings.

        :param num_indicators (int): Number of unique indicators
        :param num_pub_types (int): Number of publication types
        :param num_categories (int): Number of macro categories
        :param num_regions (int): Number of regions
        :param num_periodicities (int): Number of periodicity types
        :param d_model (int): Output embedding dimension
        :param d_embed (int): Shared embedding dimension (all embeddings use same dim for addition)
        :param dropout (float): Dropout probability
        """
        super().__init__()

        self.d_model = d_model
        self.d_embed = d_embed

        # All categorical embeddings use SAME dimension for ADDITIVE approach
        self.identity_embedding = nn.Embedding(num_indicators, d_embed)
        self.type_embedding = nn.Embedding(num_pub_types, d_embed)
        self.category_embedding = nn.Embedding(num_categories, d_embed)
        self.country_embedding = nn.Embedding(num_regions, d_embed)
        self.periodicity_embedding = nn.Embedding(num_periodicities, d_embed)

        # Importance projection (scalar -> d_embed for addition)
        self.importance_projection = nn.Linear(1, d_embed)

        # Temporal encoding (same dimension for addition)
        self.temporal_encoding = SinusoidalPositionalEncoding(d_embed)

        # Numerical features: normalized_value, surprise, MA5
        num_numerical = 3

        # Final projection: concat(E_total, numericals) -> d_model
        # E_total has dimension d_embed (after summing), plus 3 numerical features
        self.projection = nn.Linear(d_embed + num_numerical, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        indicator_ids: torch.Tensor,
        pub_type_ids: torch.Tensor,
        category_ids: torch.Tensor,
        country_ids: torch.Tensor,
        periodicity_ids: torch.Tensor,
        importance: torch.Tensor,
        days_offset: torch.Tensor,
        normalized_value: torch.Tensor,
        surprise: torch.Tensor,
        ma5: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute token embeddings using ADDITIVE approach.

        As per strategy document Section 1.3:
            E_total = E_identity + E_type + E_importance + E_temporal + E_category + E_country + E_periodicity
            X_token = LayerNorm(Linear(concat(E_total, [val_norm, surprise, MA5])))

        :param indicator_ids (torch.Tensor): Indicator indices [batch, seq]
        :param pub_type_ids (torch.Tensor): Publication type indices [batch, seq]
        :param category_ids (torch.Tensor): Category indices [batch, seq]
        :param country_ids (torch.Tensor): Country indices [batch, seq]
        :param periodicity_ids (torch.Tensor): Periodicity indices [batch, seq]
        :param importance (torch.Tensor): Importance scores [batch, seq]
        :param days_offset (torch.Tensor): Days offset [batch, seq]
        :param normalized_value (torch.Tensor): Normalized values [batch, seq]
        :param surprise (torch.Tensor): Surprises [batch, seq]
        :param ma5 (torch.Tensor): 5-period MAs [batch, seq]

        :return embeddings (torch.Tensor): Token embeddings [batch, seq, d_model]
        """
        # Categorical embeddings (all same dimension d_embed)
        e_identity = self.identity_embedding(indicator_ids)
        e_type = self.type_embedding(pub_type_ids)
        e_category = self.category_embedding(category_ids)
        e_country = self.country_embedding(country_ids)
        e_periodicity = self.periodicity_embedding(periodicity_ids)

        # Importance projection (to d_embed)
        e_importance = self.importance_projection(importance.unsqueeze(-1))

        # Temporal encoding (d_embed)
        e_temporal = self.temporal_encoding(days_offset)

        # ADDITIVE embeddings: SUM all categorical embeddings (not concatenate!)
        # This follows the BERT-like approach specified in the strategy document
        e_total = e_identity + e_type + e_category + e_country + e_periodicity + e_importance + e_temporal

        # Concatenate numerical features with the summed embedding
        numerical = torch.stack([normalized_value, surprise, ma5], dim=-1)
        combined = torch.cat([e_total, numerical], dim=-1)

        # Project to d_model and normalize
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

    def __init__(self, d_model: int = 32, dropout: float = 0.5):
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
        d_model: int = 32,
        dropout: float = 0.5,
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
            periodicity_ids=macro_batch["periodicity_ids"],
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
