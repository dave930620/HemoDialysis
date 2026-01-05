"""
Hemodialysis Adequacy Prediction (Kt/V) — SAINT Training and Inference Script
============================================================================

This script contains the end-to-end pipeline used in our study to train a SAINT-based
tabular model for predicting Kt/V (dialysis adequacy) from structured clinical data.

It includes:
- Data preprocessing (imputation, scaling, feature selection).
- SAINT model definition (mixed discrete/continuous features).
- Training loop and evaluation utilities.
- Visualization routines for performance and permutation-based feature importance.
- A lightweight inference helper for loading a trained checkpoint and producing
  predictions on new data using the saved feature configuration.

Reproducibility
---------------
To reproduce results reported in the manuscript, ensure the same:
1) data schema (column names and types),
2) preprocessing configuration (`feature_info.pkl`),
3) random seeds (if set by the caller/environment),
4) library versions (PyTorch / scikit-learn / pandas).

License
-------
Add your project license here (e.g., MIT, Apache-2.0) when publishing on GitHub.
"""

import torch

import torch.nn as nn

import torch.optim as optim

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

import pickle

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GroupShuffleSplit

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error


def preprocess_data(data):
    """Preprocess the raw tabular dataset for SAINT training/inference.

    This function performs median imputation for missing values and standardizes
    continuous features using a fitted `StandardScaler`. Non-feature identifier or
    label columns are excluded as specified by `exclude_columns` in the code.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataframe containing clinical variables and targets.

    Returns
    -------
    X : numpy.ndarray
        Model input matrix after imputation and scaling.
    y : numpy.ndarray
        Target vector used for supervised training (as defined in the code).
    feature_names : list[str]
        Names of the columns used as model inputs (saved for reproducibility).
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler used for continuous feature normalization.
    """
    imputer = SimpleImputer(strategy='median')

    exclude_columns = ['記錄時間', 'PatientID', 'RRF Kt/V', 'total Kt/V',
                       'Kt/V', 'Kt/V (Gotch)', 'Kt/V (Daugirdas)',
                       'URR', 'week total Kt/V', 'week Kt/V']  # 部分column屬於label或不參與訓練
    feature_columns = [col for col in data.columns if col not in exclude_columns]

    X = data[feature_columns].values
    y = data['Kt/V'].values

    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    feature_names = feature_columns

    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)

    return X_scaled, y, feature_names, scaler


class SAINT(nn.Module):
    """SAINT model for tabular learning on mixed discrete/continuous features.

    This implementation follows a transformer-style encoder for tabular data, where
    each feature is treated as a token. Discrete and continuous features are
    embedded separately and then fused before passing through stacked multi-head
    self-attention layers.

    Notes
    -----
    - The exact feature indexing and tensor shapes are governed by
      `discrete_feature_indices` and `continuous_feature_indices`.
    - Training/evaluation behavior (e.g., dropout) depends on `model.train()` vs
      `model.eval()` states controlled by the caller.
    """

    def __init__(self, input_size, hidden_size, output_size,
                 discrete_feature_indices, continuous_feature_indices,
                 num_heads=8, num_layers=6, dropout=0.1):
        """Initialize SAINT modules and projection heads.

        The constructor builds embeddings/projections and the transformer encoder stack
        according to the provided feature partition and hyperparameters.
        """
        super(SAINT, self).__init__()

        self.hidden_size = hidden_size
        self.discrete_feature_indices = discrete_feature_indices
        self.continuous_feature_indices = continuous_feature_indices

        self.num_discrete = len(discrete_feature_indices)
        self.num_continuous = len(continuous_feature_indices)

        self.discrete_embedding = nn.Embedding(100, hidden_size)

        self.continuous_projection = nn.Linear(1, hidden_size)

        self.feature_fusion = nn.Linear(hidden_size, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size,
                                                   nhead=num_heads,
                                                   dropout=dropout,
                                                   batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=num_layers)

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        """Forward pass.

        Converts the raw input matrix into discrete/continuous token embeddings,
        applies the transformer encoder, and produces the final regression output.
        """
        batch_size = x.size(0)

        discrete_features = x[:, self.discrete_feature_indices].long()
        continuous_features = x[:, self.continuous_feature_indices].float()

        discrete_embedded = self.discrete_embedding(discrete_features)

        continuous_features = continuous_features.unsqueeze(-1)
        continuous_embedded = self.continuous_projection(continuous_features)

        all_features = torch.cat([discrete_embedded, continuous_embedded], dim=1)

        fused_features = self.feature_fusion(all_features)

        transformed = self.transformer_encoder(fused_features)

        pooled = transformed.mean(dim=1)

        output = self.output_layer(pooled)

        return output

    def get_feature_importance(self, X, feature_names, device='cpu'):
        """Estimate feature importance via permutation on a held-out matrix.

        For each feature, the column is randomly permuted and the corresponding
        performance degradation is used as an importance proxy.
        """
        self.eval()
        X_tensor = torch.FloatTensor(X).to(device)

        baseline_predictions = self(X_tensor).detach().cpu().numpy().flatten()

        feature_importances = []
        for i in range(X.shape[1]):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])

            X_permuted_tensor = torch.FloatTensor(X_permuted).to(device)
            permuted_predictions = self(X_permuted_tensor).detach().cpu().numpy().flatten()

            importance = np.mean(np.abs(baseline_predictions - permuted_predictions))
            feature_importances.append(importance)

        return np.array(feature_importances)


def evaluate_model(model, X_test, y_test):
    """Compute standard regression metrics for a trained model on a test set."""
    model.eval()

    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    with torch.no_grad():
        predictions = model(X_test_tensor).numpy().flatten()

    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)

    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R²: {r2:.4f}")
    print(f"Test MAPE: {mape:.4f}")

    return predictions


def visualize_results(ground_truth, predictions, threshold=1.7):
    """Visualize prediction performance and threshold-based pass/fail grouping."""
    plt.figure(figsize=(10, 6))
    plt.scatter(ground_truth, predictions, alpha=0.5)
    plt.plot([ground_truth.min(), ground_truth.max()],
             [ground_truth.min(), ground_truth.max()],
             '--', linewidth=2)
    plt.xlabel("Ground Truth Kt/V")
    plt.ylabel("Predicted Kt/V")
    plt.title("Ground Truth vs Predicted Kt/V")
    plt.grid(True)
    plt.show()

    ground_truth_pass = ground_truth >= threshold
    predictions_pass = predictions >= threshold

    accuracy = np.mean(ground_truth_pass == predictions_pass)
    print(f"Threshold-based accuracy (Kt/V >= {threshold}): {accuracy:.4f}")

    plt.figure(figsize=(10, 6))
    plt.hist(predictions[ground_truth_pass], bins=30, alpha=0.7, label='Ground Truth PASS')
    plt.hist(predictions[~ground_truth_pass], bins=30, alpha=0.7, label='Ground Truth FAIL')
    plt.axvline(threshold, linestyle='--', linewidth=2)
    plt.xlabel("Predicted Kt/V")
    plt.ylabel("Frequency")
    plt.title("Predicted Kt/V Distribution by Ground Truth PASS/FAIL")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualize_feature_importance(feature_importances, feature_names, top_n=30):
    """Plot the top-N permutation feature importances."""
    indices = np.argsort(feature_importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = feature_importances[indices]

    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_importances, y=top_features)
    plt.xlabel("Permutation Importance (Mean |Δ Prediction|)")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.show()


def predict_with_model(model_path, data_df, feature_info_path='feature_info.pkl'):
    """Load a saved SAINT checkpoint and run prediction on a provided dataframe.

    This function is intended for reproducible inference using the same feature
    configuration saved during training (`feature_info.pkl`).
    """
    with open(feature_info_path, 'rb') as f:
        feature_info = pickle.load(f)

    feature_names = feature_info['feature_names']
    scaler = feature_info['scaler']
    discrete_feature_indices = feature_info['discrete_feature_indices']
    continuous_feature_indices = feature_info['continuous_feature_indices']
    hidden_size = feature_info['hidden_size']
    output_size = feature_info['output_size']

    X = data_df[feature_names].values

    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    X_scaled = scaler.transform(X_imputed)

    input_size = X_scaled.shape[1]
    model = SAINT(input_size=input_size,
                  hidden_size=hidden_size,
                  output_size=output_size,
                  discrete_feature_indices=discrete_feature_indices,
                  continuous_feature_indices=continuous_feature_indices)

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    X_tensor = torch.FloatTensor(X_scaled)
    with torch.no_grad():
        predictions = model(X_tensor).numpy().flatten()

    return predictions


def main():
    """Entry point for model training, evaluation, and artifact generation."""
    data_path = 'your_data.csv'
    data = pd.read_csv(data_path)

    X, y, feature_names, scaler = preprocess_data(data)

    discrete_feature_indices = []
    continuous_feature_indices = list(range(X.shape[1]))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    groups = data['PatientID'].values
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = 1

    model = SAINT(input_size=input_size,
                  hidden_size=hidden_size,
                  output_size=output_size,
                  discrete_feature_indices=discrete_feature_indices,
                  continuous_feature_indices=continuous_feature_indices)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    batch_size = 32

    for epoch in range(num_epochs):
        model.train()

        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        epoch_loss = 0.0
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i + batch_size]
            y_batch = y_train_shuffled[i:i + batch_size]

            X_batch_tensor = torch.FloatTensor(X_batch)
            y_batch_tensor = torch.FloatTensor(y_batch).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(X_batch_tensor)
            loss = criterion(outputs, y_batch_tensor)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    predictions = evaluate_model(model, X_test, y_test)
    visualize_results(y_test, predictions, threshold=1.7)

    feature_importances = model.get_feature_importance(X_test, feature_names)
    visualize_feature_importance(feature_importances, feature_names, top_n=30)

    model_save_path = 'saint_model.pth'
    torch.save(model.state_dict(), model_save_path)

    feature_info = {
        'feature_names': feature_names,
        'scaler': scaler,
        'discrete_feature_indices': discrete_feature_indices,
        'continuous_feature_indices': continuous_feature_indices,
        'hidden_size': hidden_size,
        'output_size': output_size
    }
    with open('feature_info.pkl', 'wb') as f:
        pickle.dump(feature_info, f)


if __name__ == "__main__":
    main()
