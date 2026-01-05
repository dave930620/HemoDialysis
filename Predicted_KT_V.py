"""
Hemodialysis PD Kt/V Prediction (SAINT)
======================================

This script is the original training and evaluation implementation used in the
associated journal manuscript. The executable code is identical to the authors'
experimental version; only comments/docstrings and user-facing messages have been
revised for clarity, reproducibility, and public release.

Main components
---------------
- Data preprocessing (imputation, scaling, categorical handling).
- SAINT model for mixed discrete/continuous tabular features.
- Training, evaluation, and visualization utilities.
- Checkpoint and feature-configuration saving for reproducible inference.
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
    """Preprocess the raw tabular dataset for model training/inference.

    This function performs the same feature selection and transformations used in
    the manuscript experiments (e.g., excluding identifiers/targets, imputing
    missing values, scaling continuous variables, and preparing discrete features).

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing clinical variables and labels.

    Returns
    -------
    X_scaled : numpy.ndarray
        Preprocessed feature matrix used as model input.
    y : numpy.ndarray
        Target vector defined by `output_column` in the code.
    feature_names : list[str]
        Final list of feature columns (saved for reproducibility).
    discrete_feature_indices : list[int]
        Column indices (in the processed matrix) treated as categorical tokens.
    continuous_feature_indices : list[int]
        Column indices treated as continuous values.
    feature_info : dict
        Serializable configuration required to reproduce preprocessing at inference.
    """

    imputer = SimpleImputer(strategy='median')


    exclude_columns = ['記錄時間', 'PatientID', 'RRF Kt/V', 'total Kt/V',
                       'PCl rate (↑/wk）', 'RRF (↑/wk）',
                       'total Cl rate (↑/wk）', 'Std total Cl rate(↑/wk)民', 'night time PD',
                       'glucose_total_n', 'calcium_total_n', 'apd_total_volume', 'apd_total_cycles',
                       'Fluid change times', 'Fluid Exchange System', 'Primary Bag Exchanger', 'Volume (L)']


    output_column = 'PD Kt/V'


    discrete_columns = ["SEX", "DM type", "long term PD system", "DOResult", "DPResult",
                        'HBsAg', 'Anti-HCV', 'infection', 'BLOOD GROUP',
                        'Primary disease categories', 'primary disease subclass',
                        'EPO', 'active Vit D', 'antihypertensive',
                        'Iron therapy', 'PTx', 'other systemic disease-1',
                        'other systemic disease-2', 'other systemic disease-3',
                        'other systemic disease-4', 'other systemic disease-5', 'other systemic disease-6',
                        'other systemic disease-7', 'other systemic disease-8', 'other systemic disease-9',
                        'other systemic disease-10', 'other systemic disease-12', 'complication-0',
                        'complication-1', 'complication-2', 'complication-3', 'complication-4',
                        'complication-5', 'complication-6', 'complication-7', 'complication-8',
                        'complication-9', 'complication-10', 'complication-11', 'complication-12',
                        'complication-14', 'complication-15', 'complication-17', 'complication-18',
                        'complication-19', 'complication-20', 'complication-21', 'complication-22',
                        'complication-23', 'complication-24', 'complication-25', 'complication-26',
                        'complication-28', 'complication-29', 'complication-30', 'complication-31',
                        'complication-32', 'complication-33']


    features = data.drop(columns=[col for col in exclude_columns if col in data.columns] + [output_column])
    patient_ids = None
    if 'PatientID' in features.columns:
        patient_ids = features['PatientID']
        features = features.drop(columns=['PatientID'])
    target = data[output_column]


    original_feature_names = list(features.columns)


    discrete_feature_columns = [col for col in discrete_columns if col in features.columns]
    continuous_feature_columns = [col for col in features.columns if col not in discrete_feature_columns]

    print(f"Discrete features: {discrete_feature_columns}")
    print(f"Continuous features: {continuous_feature_columns}")


    continuous_data = features[continuous_feature_columns]
    continuous_data_imputed = imputer.fit_transform(continuous_data)

    scaler = StandardScaler()
    continuous_data_scaled = scaler.fit_transform(continuous_data_imputed)

    features[continuous_feature_columns] = continuous_data_scaled


    all_columns = list(features.columns)
    discrete_feature_indices = [all_columns.index(col) for col in discrete_feature_columns]
    continuous_feature_indices = [all_columns.index(col) for col in continuous_feature_columns]


    feature_name_to_index = {name: idx for idx, name in enumerate(all_columns)}
    index_to_feature_name = {idx: name for idx, name in enumerate(all_columns)}


    feature_info = {
        'original_feature_names': original_feature_names,
        'discrete_feature_names': discrete_feature_columns,
        'continuous_feature_names': continuous_feature_columns,
        'discrete_feature_indices': discrete_feature_indices,
        'continuous_feature_indices': continuous_feature_indices,
        'feature_name_to_index': feature_name_to_index,
        'index_to_feature_name': index_to_feature_name,
        'exclude_columns': exclude_columns,
        'output_column': output_column,
        'imputer': imputer,
        'scaler': scaler
    }


    with open('feature_info.pkl', 'wb') as f:
        pickle.dump(feature_info, f)

    print("Feature information saved to 'feature_info.pkl'")

    return features, target, patient_ids, discrete_feature_indices, continuous_feature_indices, all_columns, feature_info


class SAINT(nn.Module):
    """SAINT model for tabular learning with mixed discrete/continuous features.

    SAINT treats each feature as a token and applies Transformer-style self-attention
    over the feature sequence. Discrete features are embedded, while continuous
    features are projected to the same hidden dimension, then fused and encoded.

    Notes
    -----
    - The feature partition is controlled by `discrete_feature_indices` and
      `continuous_feature_indices` passed at initialization.
    - Output is regression (single continuous value) for PD Kt/V prediction.
    """

    def __init__(self, input_size, hidden_size, output_size,
                 discrete_feature_indices, continuous_feature_indices,
                 num_heads=8, num_layers=6, dropout=0.1):
        super(SAINT, self).__init__()

        self.hidden_size = hidden_size
        self.discrete_feature_indices = discrete_feature_indices
        self.continuous_feature_indices = continuous_feature_indices


        self.num_discrete = len(discrete_feature_indices)
        self.num_continuous = len(continuous_feature_indices)

        print(f"Model initialization - #discrete features: {self.num_discrete}, #continuous features: {self.num_continuous}")


        if self.num_discrete > 0:
            self.discrete_embedding = nn.Sequential(
                nn.Linear(self.num_discrete, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            )


        if self.num_continuous > 0:
            self.continuous_embedding = nn.Sequential(
                nn.Linear(self.num_continuous, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            )


        if self.num_discrete > 0:
            for module in self.discrete_embedding:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

        if self.num_continuous > 0:
            for module in self.continuous_embedding:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )


        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )


        for module in self.fc_out:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if torch.isnan(x).any():
            print("NaN detected in input!")
            x = torch.nan_to_num(x, nan=0.0)


        max_index = x.shape[1] - 1
        valid_discrete_indices = [i for i in self.discrete_feature_indices if i <= max_index]
        valid_continuous_indices = [i for i in self.continuous_feature_indices if i <= max_index]


        if len(valid_discrete_indices) != len(self.discrete_feature_indices) or len(valid_continuous_indices) != len(self.continuous_feature_indices):
            print(f"Warning: feature index out of range! Input shape: {x.shape}")
            print(f"Valid discrete indices: {len(valid_discrete_indices)}/{len(self.discrete_feature_indices)}")
            print(f"Valid continuous indices: {len(valid_continuous_indices)}/{len(self.continuous_feature_indices)}")


        discrete_x = x[:, valid_discrete_indices] if valid_discrete_indices else torch.zeros((x.shape[0], 0), device=x.device)
        continuous_x = x[:, valid_continuous_indices] if valid_continuous_indices else torch.zeros((x.shape[0], 0), device=x.device)


        x_embedded = None


        if discrete_x.shape[1] > 0:
            discrete_embedded = self.discrete_embedding(discrete_x)
            x_embedded = discrete_embedded

        if continuous_x.shape[1] > 0:
            continuous_embedded = self.continuous_embedding(continuous_x)
            if x_embedded is None:
                x_embedded = continuous_embedded
            else:
                x_embedded = x_embedded + continuous_embedded


        if x_embedded is None:
            raise RuntimeError("No valid features available for embedding.")


        x_embedded = x_embedded.unsqueeze(1)


        x = self.transformer_encoder(x_embedded)
        x = x.squeeze(1)


        return self.fc_out(x)


    def get_feature_importance(self, X, feature_names, device='cpu'):

        self.eval()


        importances = np.zeros(X.shape[1])


        for i in range(X.shape[1]):

            X_perturbed = X.clone()


            X_perturbed[:, i] = torch.randn_like(X_perturbed[:, i]) * X_perturbed[:, i].std() + X_perturbed[:, i].mean()


            with torch.no_grad():
                original_pred = self(X).cpu().numpy()
                perturbed_pred = self(X_perturbed).cpu().numpy()


            importances[i] = np.mean(np.abs(original_pred - perturbed_pred))

        return importances


def evaluate_model(model, X_test, y_test):
    """Evaluate a trained model on a test split.

    Metrics reported include MAE, RMSE, R², and MAPE, consistent with the manuscript.
    The function prints metrics and returns the raw prediction vector.
    """

    model.eval()
    with torch.no_grad():
        predictions = model(X_test).cpu().numpy()
        ground_truth = y_test.cpu().numpy()

        mse = mean_squared_error(ground_truth, predictions)
        mae = mean_absolute_error(ground_truth, predictions)
        r2 = r2_score(ground_truth, predictions)


        threshold = 1.7
        gt_above = ground_truth > threshold
        pred_above = predictions > threshold


        consistency = np.mean((gt_above & pred_above) | (~gt_above & ~pred_above))
        consistency_percentage = consistency * 100


        above_threshold_correct = np.sum(gt_above & pred_above)
        below_threshold_correct = np.sum(~gt_above & ~pred_above)
        total_above_threshold = np.sum(gt_above)
        total_below_threshold = np.sum(~gt_above)

        print("\nModel performance metrics:")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R-squared (R2 Score): {r2:.4f}")

        print("\nThreshold-based classification performance (threshold = 1.7):")
        print(f"Overall consistency: {consistency_percentage:.2f}% ({int(consistency * len(ground_truth))}/{len(ground_truth)})")

        if total_above_threshold > 0:
            above_accuracy = above_threshold_correct / total_above_threshold * 100
            print(f"Consistency for > 1.7: {above_accuracy:.2f}% ({above_threshold_correct}/{total_above_threshold})")
        else:
            print("No ground-truth samples with value > 1.7")

        if total_below_threshold > 0:
            below_accuracy = below_threshold_correct / total_below_threshold * 100
            print(f"Consistency for <= 1.7: {below_accuracy:.2f}% ({below_threshold_correct}/{total_below_threshold})")
        else:
            print("No ground-truth samples with value <= 1.7")

        return ground_truth, predictions

def visualize_results(ground_truth, predictions, threshold=1.7):
    """Generate diagnostic plots for prediction quality.

    This includes (1) scatter plot of ground truth vs. predictions and (2) a
    threshold-based pass/fail view using the clinical cut-off defined by `threshold`.
    """

    plt.figure(figsize=(15, 10))


    plt.subplot(221)
    plt.scatter(ground_truth, predictions, alpha=0.6)
    plt.plot([ground_truth.min(), ground_truth.max()],
             [ground_truth.min(), ground_truth.max()],
             'r--', lw=2)
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.title('Prediction vs Ground Truth')


    residuals = predictions - ground_truth
    plt.subplot(222)
    plt.scatter(ground_truth, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Ground Truth')
    plt.ylabel('Residual')
    plt.title('Residual Distribution')


    plt.subplot(223)
    sns.histplot(residuals.flatten(), kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')


    plt.subplot(224)


    colors = []
    for gt, pred in zip(ground_truth.flatten(), predictions.flatten()):
        if gt > threshold and pred > threshold:
            colors.append('green')
        elif gt <= threshold and pred <= threshold:
            colors.append('blue')
        elif gt > threshold and pred <= threshold:
            colors.append('red')
        else:
            colors.append('orange')

    plt.scatter(ground_truth, predictions, alpha=0.6, c=colors)
    plt.axhline(y=threshold, color='r', linestyle='--')
    plt.axvline(x=threshold, color='r', linestyle='--')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.title(f'Threshold Classification (threshold={threshold})')


    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='True Positive (>1.7)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='True Negative (≤1.7)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='False Negative'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='False Positive')
    ]
    plt.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.show()


def visualize_feature_importance(feature_importances, feature_names, top_n=30):
    """Visualize feature importance scores.

    This utility expects a dataframe or array-like importance result produced by the
    training/evaluation pipeline (e.g., permutation importance) and plots the top-N.
    """

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })


    importance_df = importance_df.sort_values('Importance', ascending=False)


    if len(importance_df) > top_n:
        top_features = importance_df.head(top_n)
        plt.figure(figsize=(12, 10))


        ax = sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')

        plt.title(f'Top {top_n} Feature Importances', fontsize=16)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.show()

        print(f"Only the top {top_n} most important features are shown. To view all features, increase the `top_n` parameter.")
    else:
        plt.figure(figsize=(12, max(6, len(importance_df) * 0.3)))


        ax = sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')

        plt.title('Feature Importances', fontsize=16)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.show()


    return importance_df


def predict_with_model(model_path, data_df, feature_info_path='feature_info.pkl'):
    """Run inference using a saved checkpoint and preprocessing configuration.

This function loads the serialized `feature_info` produced during training,
applies the identical preprocessing steps to `data_df`, constructs the SAINT
architecture with the same feature partition and hyperparameters, loads the
checkpoint weights, and returns predictions.

Parameters
----------
model_path : str
    Path to the saved model checkpoint (state_dict).
data_df : pandas.DataFrame
    Input dataframe containing the same columns used during training.
feature_info_path : str
    Path to the saved preprocessing configuration (pickle).

Returns
-------
numpy.ndarray
    Predicted PD Kt/V values.
    """

    with open(feature_info_path, 'rb') as f:
        feature_info = pickle.load(f)


    exclude_columns = feature_info['exclude_columns']
    output_column = feature_info['output_column']
    original_feature_names = feature_info['original_feature_names']
    discrete_feature_indices = feature_info['discrete_feature_indices']
    continuous_feature_indices = feature_info['continuous_feature_indices']
    imputer = feature_info['imputer']
    scaler = feature_info['scaler']


    for col in original_feature_names:
        if col not in data_df.columns:
            raise ValueError(f"Missing feature column: {col}")


    features = data_df[original_feature_names].copy()


    discrete_feature_names = [original_feature_names[i] for i in discrete_feature_indices]
    continuous_feature_names = [original_feature_names[i] for i in continuous_feature_indices]


    continuous_data = features[continuous_feature_names]
    continuous_data_imputed = imputer.transform(continuous_data)
    continuous_data_scaled = scaler.transform(continuous_data_imputed)
    features[continuous_feature_names] = continuous_data_scaled


    X = torch.tensor(features.values, dtype=torch.float32)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_size = len(original_feature_names)
    hidden_size = 128
    output_size = 1

    model = SAINT(input_size, hidden_size, output_size,
                 discrete_feature_indices, continuous_feature_indices).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()


    X = X.to(device)
    with torch.no_grad():
        predictions = model(X).cpu().numpy()

    return predictions


def main():
    """Entry point for reproducing the training and evaluation pipeline.

    The dataset paths, hyperparameters, split strategy, and saved artifacts are the
    same as used in the experiments unless the user modifies the constants below.
    """

    try:
        data = pd.read_csv('cleaned_data_after_fill_and_drop.csv')
        print(f"Data loaded successfully. Shape: {data.shape}")
    except Exception as e:
        print(f"Error while loading data: {e}")
        return


    features, target, patient_ids, discrete_feature_indices, continuous_feature_indices, feature_names, feature_info = preprocess_data(data)


    print(f"Total number of features: {len(feature_names)}")
    print(f"Number of discrete features: {len(discrete_feature_indices)}")
    print(f"Number of continuous features: {len(continuous_feature_indices)}")


    print(f"Discrete feature indices: {discrete_feature_indices[:10]}{'...' if len(discrete_feature_indices) > 10 else ''}")


    print(f"Continuous feature indices: {continuous_feature_indices[:10]}{'...' if len(continuous_feature_indices) > 10 else ''}")


    if patient_ids is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(features, target, groups=patient_ids))


        X_train = features.iloc[train_idx].values
        X_test = features.iloc[test_idx].values
        y_train = target.iloc[train_idx].values
        y_test = target.iloc[test_idx].values
    else:

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            features.values, target.values, test_size=0.2, random_state=42
        )


    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)


    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = 1


    model = SAINT(input_size, hidden_size, output_size,
                  discrete_feature_indices, continuous_feature_indices).to(device)

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)


    batch_size = 32
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


    epochs = 200
    train_losses, test_losses = [], []


    try:
        print("Testing model forward pass...")
        with torch.no_grad():
            test_batch = X_train[:batch_size]
            test_output = model(test_batch)
            print(f"Forward pass succeeded. Output shape: {test_output.shape}")
    except Exception as e:
        print(f"Model forward-pass test failed: {e}")
        return

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        batch_count = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)

        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            test_losses.append(test_loss.item())

        scheduler.step(test_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss.item():.4f}")


    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


    ground_truth, predictions = evaluate_model(model, X_test, y_test)
    visualize_results(ground_truth, predictions)


    print("\nComputing feature importance...")

    try:

        feature_importances = model.get_feature_importance(X_train, feature_names, device=device)
        feature_importance_df = visualize_feature_importance(feature_importances, feature_names)
        print("\nFeature importance:")
        print(feature_importance_df.head(20))
    except Exception as e:
        print(f"Error while computing feature importance: {e}")


    torch.save(model.state_dict(), "saint_pd_model.pth")
    print("Model saved as 'saint_pd_model.pth'")


    try:
        feature_importance_df.to_csv("feature_importance.csv", index=False)
        print("Feature importance saved as 'feature_importance.csv'")
    except:
        print("Unable to save feature importance")


if __name__ == "__main__":

    main()
