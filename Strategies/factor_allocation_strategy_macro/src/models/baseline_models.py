"""
Baseline models for progressive validation.

This module implements the baseline models for step-by-step validation:
1. Naive baseline (style momentum)
2. Logistic Regression / Ridge
3. Gradient Boosting (XGBoost/LightGBM)
4. Simple LSTM/GRU

Each model must show improvement over the previous before advancing.
"""

from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np

# PyTorch is optional - only needed for LSTMModel
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class BaseModel(ABC):
    """
    Abstract base class for all models.

    Provides a common interface for training and prediction.
    """

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BaseModel":
        """
        Train the model.

        :param X (np.ndarray): Training features
        :param y (np.ndarray): Training targets
        :param X_val (Optional[np.ndarray]): Validation features
        :param y_val (Optional[np.ndarray]): Validation targets

        :return self (BaseModel): Trained model
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        :param X (np.ndarray): Input features

        :return predictions (np.ndarray): Model predictions
        """
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities (for classification).

        :param X (np.ndarray): Input features

        :return probabilities (np.ndarray): Predicted probabilities
        """
        pass


class NaiveBaselineModel(BaseModel):
    """
    Naive baseline: predict based on recent momentum.

    For binary classification (cyclical vs defensive):
    - If cyclicals outperformed defensives recently, predict they will continue
    - Uses simple exponential moving average of past performance

    :param lookback (int): Lookback period for momentum calculation
    :param alpha (float): Exponential smoothing factor
    """

    def __init__(self, lookback: int = 4, alpha: float = 0.3):
        """
        Initialize naive baseline.

        :param lookback (int): Lookback period (weeks)
        :param alpha (float): EMA smoothing factor
        """
        self.lookback = lookback
        self.alpha = alpha
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "NaiveBaselineModel":
        """
        Fit baseline (no actual training, just stores statistics).

        :param X (np.ndarray): Training features (includes past returns)
        :param y (np.ndarray): Training targets

        :return self (NaiveBaselineModel): Model instance
        """
        # Baseline doesn't learn from features, just uses momentum
        self.mean_target = y.mean()
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict based on momentum.

        Assumes X contains columns for recent cyclical and defensive returns.

        :param X (np.ndarray): Features with momentum info

        :return predictions (np.ndarray): Binary predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Assume last 2 columns are recent cyclical and defensive performance
        if X.shape[1] >= 2:
            cyclical_momentum = X[:, -2]
            defensive_momentum = X[:, -1]
            predictions = (cyclical_momentum > defensive_momentum).astype(int)
        else:
            # Fallback to mean
            predictions = (np.random.random(len(X)) > (1 - self.mean_target)).astype(int)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using momentum strength.

        :param X (np.ndarray): Features

        :return probabilities (np.ndarray): Probability of cyclical outperformance
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if X.shape[1] >= 2:
            cyclical_momentum = X[:, -2]
            defensive_momentum = X[:, -1]
            diff = cyclical_momentum - defensive_momentum

            # Sigmoid transformation of momentum difference
            proba = 1 / (1 + np.exp(-diff * 10))
        else:
            proba = np.full(len(X), self.mean_target)

        return proba


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression / Ridge model for binary classification.

    Uses sklearn implementation with L2 regularization.

    :param C (float): Inverse regularization strength
    :param max_iter (int): Maximum iterations
    """

    def __init__(self, C: float = 0.1, max_iter: int = 1000):
        """
        Initialize logistic regression model.

        :param C (float): Inverse regularization strength (lower = more regularization)
        :param max_iter (int): Maximum iterations
        """
        self.C = C
        self.max_iter = max_iter
        self.model = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LogisticRegressionModel":
        """
        Fit logistic regression model.

        :param X (np.ndarray): Training features
        :param y (np.ndarray): Training targets

        :return self (LogisticRegressionModel): Trained model
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=42,
        )
        self.model.fit(X_scaled, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions.

        :param X (np.ndarray): Features

        :return predictions (np.ndarray): Binary predictions
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.

        :param X (np.ndarray): Features

        :return probabilities (np.ndarray): Probability of class 1
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature coefficients.

        :return coefficients (np.ndarray): Model coefficients
        """
        return self.model.coef_[0]


class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting model for capturing non-linear interactions.

    Uses LightGBM for efficiency. Falls back to sklearn if not available.

    :param n_estimators (int): Number of boosting rounds
    :param max_depth (int): Maximum tree depth
    :param learning_rate (float): Learning rate
    :param subsample (float): Subsample ratio
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
    ):
        """
        Initialize gradient boosting model.

        :param n_estimators (int): Number of boosting rounds
        :param max_depth (int): Maximum tree depth
        :param learning_rate (float): Learning rate
        :param subsample (float): Row subsampling ratio
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.model = None
        self.use_lightgbm = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "GradientBoostingModel":
        """
        Fit gradient boosting model.

        :param X (np.ndarray): Training features
        :param y (np.ndarray): Training targets
        :param X_val (Optional[np.ndarray]): Validation features
        :param y_val (Optional[np.ndarray]): Validation targets

        :return self (GradientBoostingModel): Trained model
        """
        try:
            import lightgbm as lgb

            self.use_lightgbm = True
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "verbose": -1,
                "seed": 42,
            }

            self.model = lgb.LGBMClassifier(**params)

            eval_set = [(X_val, y_val)] if X_val is not None else None
            self.model.fit(
                X, y,
                eval_set=eval_set,
            )

        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier

            self.model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=42,
            )
            self.model.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions.

        :param X (np.ndarray): Features

        :return predictions (np.ndarray): Binary predictions
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.

        :param X (np.ndarray): Features

        :return probabilities (np.ndarray): Probability of class 1
        """
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importances.

        :return importances (np.ndarray): Feature importance scores
        """
        return self.model.feature_importances_


# Conditional base class for LSTMModel
_LSTMBase = (BaseModel, nn.Module) if TORCH_AVAILABLE else (BaseModel,)


class LSTMModel(*_LSTMBase):
    """
    Simple LSTM model for sequential modeling.

    Tests if sequential structure adds value over flat features.
    Requires PyTorch to be installed.

    :param input_size (int): Number of input features per timestep
    :param hidden_size (int): LSTM hidden dimension
    :param num_layers (int): Number of LSTM layers
    :param dropout (float): Dropout probability
    :param bidirectional (bool): Use bidirectional LSTM
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ):
        """
        Initialize LSTM model.

        :param input_size (int): Input feature dimension
        :param hidden_size (int): Hidden state dimension
        :param num_layers (int): Number of LSTM layers
        :param dropout (float): Dropout probability
        :param bidirectional (bool): Use bidirectional LSTM
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTMModel. Install with: pip install torch")

        nn.Module.__init__(self)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Output dimension depends on bidirectional
        output_dim = hidden_size * 2 if bidirectional else hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_fitted = False

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """
        Forward pass.

        :param x (torch.Tensor): Input [batch, seq, features]

        :return output (torch.Tensor): Predictions [batch, 1]
        """
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)

        # Use last hidden state
        if self.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden = hidden[-1]

        output = self.classifier(hidden)
        return output

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> "LSTMModel":
        """
        Train LSTM model.

        :param X (np.ndarray): Training sequences [samples, seq_len, features]
        :param y (np.ndarray): Training targets
        :param X_val (Optional[np.ndarray]): Validation sequences
        :param y_val (Optional[np.ndarray]): Validation targets
        :param epochs (int): Number of training epochs
        :param batch_size (int): Batch size
        :param learning_rate (float): Learning rate

        :return self (LSTMModel): Trained model
        """
        self.to(self.device)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(-1).to(self.device)

        if X_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).unsqueeze(-1).to(self.device)

        # Training setup
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        best_val_loss = float("inf")
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            self.train()
            train_loss = 0.0

            # Mini-batch training
            indices = torch.randperm(len(X_tensor))
            for i in range(0, len(X_tensor), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch_X = X_tensor[batch_indices]
                batch_y = y_tensor[batch_indices]

                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            if X_val is not None:
                self.eval()
                with torch.no_grad():
                    val_outputs = self(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()

                scheduler.step(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions.

        :param X (np.ndarray): Input sequences

        :return predictions (np.ndarray): Binary predictions
        """
        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities.

        :param X (np.ndarray): Input sequences

        :return probabilities (np.ndarray): Predicted probabilities
        """
        self.eval()
        self.to(self.device)

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            outputs = self(X_tensor)

        return outputs.cpu().numpy().flatten()
