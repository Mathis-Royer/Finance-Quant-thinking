"""
Pre-training module for embedding initialization.

This module implements auxiliary pre-training tasks to improve embedding quality
as specified in the strategy document (Section 1.4.3):
- Classify indicators by category (auxiliary task)
- Learn meaningful representations before main training

Pre-training helps the model learn better indicator embeddings with limited data.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class IndicatorClassificationDataset(Dataset):
    """
    Dataset for indicator category classification pre-training.

    Each sample is an indicator with its category label.

    :param indicator_ids (np.ndarray): Indicator indices
    :param category_ids (np.ndarray): Category labels (targets)
    :param periodicity_ids (np.ndarray): Periodicity indices
    :param importance (np.ndarray): Importance scores
    """

    def __init__(
        self,
        indicator_ids: np.ndarray,
        category_ids: np.ndarray,
        periodicity_ids: np.ndarray,
        importance: np.ndarray,
    ):
        """
        Initialize dataset.

        :param indicator_ids (np.ndarray): Indicator indices
        :param category_ids (np.ndarray): Category labels
        :param periodicity_ids (np.ndarray): Periodicity indices
        :param importance (np.ndarray): Importance scores
        """
        self.indicator_ids = torch.tensor(indicator_ids, dtype=torch.long)
        self.category_ids = torch.tensor(category_ids, dtype=torch.long)
        self.periodicity_ids = torch.tensor(periodicity_ids, dtype=torch.long)
        self.importance = torch.tensor(importance, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.indicator_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        return (
            self.indicator_ids[idx],
            self.periodicity_ids[idx],
            self.importance[idx],
            self.category_ids[idx],  # Target
        )


class EmbeddingPretrainer(nn.Module):
    """
    Pre-training model for indicator embeddings.

    Trains embeddings to predict indicator category from identity embedding.
    This auxiliary task helps the model learn meaningful representations.

    :param num_indicators (int): Number of unique indicators
    :param num_categories (int): Number of macro categories
    :param num_periodicities (int): Number of periodicity types
    :param d_embed (int): Embedding dimension
    :param dropout (float): Dropout probability
    """

    def __init__(
        self,
        num_indicators: int,
        num_categories: int = 8,
        num_periodicities: int = 5,
        d_embed: int = 32,
        dropout: float = 0.3,
    ):
        """
        Initialize pre-training model.

        :param num_indicators (int): Number of indicators
        :param num_categories (int): Number of categories
        :param num_periodicities (int): Number of periodicities
        :param d_embed (int): Embedding dimension
        :param dropout (float): Dropout rate
        """
        super().__init__()

        self.d_embed = d_embed

        # Embeddings to pre-train
        self.identity_embedding = nn.Embedding(num_indicators, d_embed)
        self.periodicity_embedding = nn.Embedding(num_periodicities, d_embed)
        self.importance_projection = nn.Linear(1, d_embed)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_embed, d_embed * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_embed * 2, d_embed),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_embed, num_categories),
        )

    def forward(
        self,
        indicator_ids: torch.Tensor,
        periodicity_ids: torch.Tensor,
        importance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for category prediction.

        :param indicator_ids (torch.Tensor): Indicator indices [batch]
        :param periodicity_ids (torch.Tensor): Periodicity indices [batch]
        :param importance (torch.Tensor): Importance scores [batch]

        :return logits (torch.Tensor): Category logits [batch, num_categories]
        """
        # Get embeddings
        e_identity = self.identity_embedding(indicator_ids)
        e_periodicity = self.periodicity_embedding(periodicity_ids)
        e_importance = self.importance_projection(importance.unsqueeze(-1))

        # Combine embeddings (additive)
        combined = e_identity + e_periodicity + e_importance

        # Classify
        logits = self.classifier(combined)
        return logits

    def get_pretrained_embeddings(self) -> Dict[str, nn.Module]:
        """
        Get pre-trained embedding layers for transfer.

        :return embeddings (Dict): Dictionary of pre-trained embedding modules
        """
        return {
            "identity_embedding": self.identity_embedding,
            "periodicity_embedding": self.periodicity_embedding,
            "importance_projection": self.importance_projection,
        }


def create_pretraining_dataset(
    indicators: List,
    category_to_idx: Dict[str, int],
    periodicity_to_idx: Dict[str, int],
    augment_factor: int = 10,
) -> IndicatorClassificationDataset:
    """
    Create pre-training dataset from indicator definitions.

    :param indicators (List): List of MacroIndicator objects
    :param category_to_idx (Dict): Category to index mapping
    :param periodicity_to_idx (Dict): Periodicity to index mapping
    :param augment_factor (int): Repeat each indicator this many times

    :return dataset (IndicatorClassificationDataset): Pre-training dataset
    """
    indicator_ids = []
    category_ids = []
    periodicity_ids = []
    importance_scores = []

    for idx, ind in enumerate(indicators):
        for _ in range(augment_factor):
            indicator_ids.append(idx)
            category_ids.append(category_to_idx.get(ind.category.value, 0))
            periodicity_ids.append(periodicity_to_idx.get(ind.periodicity, 0))
            importance_scores.append(ind.importance / 3.0)

    return IndicatorClassificationDataset(
        indicator_ids=np.array(indicator_ids),
        category_ids=np.array(category_ids),
        periodicity_ids=np.array(periodicity_ids),
        importance=np.array(importance_scores),
    )


def pretrain_embeddings(
    indicators: List,
    category_to_idx: Dict[str, int],
    periodicity_to_idx: Dict[str, int],
    num_categories: int = 8,
    num_periodicities: int = 5,
    d_embed: int = 32,
    epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    device: torch.device = None,
    verbose: bool = True,
) -> Dict[str, nn.Module]:
    """
    Pre-train embeddings using indicator category classification.

    :param indicators (List): List of MacroIndicator objects
    :param category_to_idx (Dict): Category to index mapping
    :param periodicity_to_idx (Dict): Periodicity to index mapping
    :param num_categories (int): Number of categories
    :param num_periodicities (int): Number of periodicities
    :param d_embed (int): Embedding dimension
    :param epochs (int): Number of training epochs
    :param learning_rate (float): Learning rate
    :param batch_size (int): Batch size
    :param device (torch.device): Device to use
    :param verbose (bool): Print progress

    :return pretrained_modules (Dict): Pre-trained embedding modules
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dataset
    dataset = create_pretraining_dataset(
        indicators, category_to_idx, periodicity_to_idx, augment_factor=20
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = EmbeddingPretrainer(
        num_indicators=len(indicators),
        num_categories=num_categories,
        num_periodicities=num_periodicities,
        d_embed=d_embed,
    ).to(device)

    # Training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if verbose:
        print("\n" + "=" * 60)
        print("EMBEDDING PRE-TRAINING (Indicator Category Classification)")
        print("=" * 60)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for indicator_ids, periodicity_ids, importance, targets in loader:
            indicator_ids = indicator_ids.to(device)
            periodicity_ids = periodicity_ids.to(device)
            importance = importance.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(indicator_ids, periodicity_ids, importance)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

        acc = correct / total
        avg_loss = total_loss / len(loader)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.4f}")

    if verbose:
        print(f"Pre-training complete. Final accuracy: {acc:.4f}")

    return model.get_pretrained_embeddings()


def transfer_pretrained_embeddings(
    pretrained: Dict[str, nn.Module],
    target_model: nn.Module,
) -> None:
    """
    Transfer pre-trained embeddings to a target model.

    :param pretrained (Dict): Dictionary of pre-trained modules
    :param target_model (nn.Module): Target model with embedding layers
    """
    # Access the token encoder's macro embedding
    if hasattr(target_model, "token_encoder"):
        macro_emb = target_model.token_encoder.macro_embedding

        if "identity_embedding" in pretrained:
            macro_emb.identity_embedding.load_state_dict(
                pretrained["identity_embedding"].state_dict()
            )

        if "periodicity_embedding" in pretrained:
            macro_emb.periodicity_embedding.load_state_dict(
                pretrained["periodicity_embedding"].state_dict()
            )

        if "importance_projection" in pretrained:
            macro_emb.importance_projection.load_state_dict(
                pretrained["importance_projection"].state_dict()
            )
