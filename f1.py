import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import pickle  # 加入用於保存特徵名稱的pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# 數據預處理函數
def preprocess_data(data):
    # 處理缺失值
    imputer = SimpleImputer(strategy='median')
    
    # 要排除的列
    exclude_columns = ['記錄時間', 'PatientID', 'RRF Kt/V', 'total Kt/V', 
                       'PCl rate (↑/wk）', 'RRF (↑/wk）', 
                       'total Cl rate (↑/wk）', 'Std total Cl rate(↑/wk)民','night time PD', 
                       'glucose_total_n', 'calcium_total_n', 'apd_total_volume', 'apd_total_cycles', 
                       'Fluid change times', 'Fluid Exchange System', 'Primary Bag Exchanger','Volume (L)']

    # 輸出目標
    output_column = 'PD Kt/V'

    # 離散列 (已更新)
    discrete_columns = ["SEX", "DM type", "long term PD system", "DOResult", "DPResult", 
                        'HBsAg', 'Anti-HCV', 'infection', 'BLOOD GROUP',
                        'Primary disease categories', 'primary disease subclass',
                        'EPO', 'active Vit D', 'antihypertensive', 
                        'Iron therapy', 'PTx','other systemic disease-1', 
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

    # 分離特徵和目標
    features = data.drop(columns=[col for col in exclude_columns if col in data.columns] + [output_column])
    patient_ids = None
    if 'PatientID' in features.columns:
        patient_ids = features['PatientID']
        features = features.drop(columns=['PatientID'])
    target = data[output_column]

    # 儲存原始特徵名稱列表（非常重要！）
    original_feature_names = list(features.columns)

    # 識別離散和連續列
    discrete_feature_columns = [col for col in discrete_columns if col in features.columns]
    continuous_feature_columns = [col for col in features.columns if col not in discrete_feature_columns]

    print(f"離散特徵: {discrete_feature_columns}")
    print(f"連續特徵: {continuous_feature_columns}")

    # 插補和標準化連續特徵
    continuous_data = features[continuous_feature_columns]
    continuous_data_imputed = imputer.fit_transform(continuous_data)
    
    scaler = StandardScaler()
    continuous_data_scaled = scaler.fit_transform(continuous_data_imputed)
    
    features[continuous_feature_columns] = continuous_data_scaled
    
    # 獲取離散和連續特徵的列索引
    all_columns = list(features.columns)
    discrete_feature_indices = [all_columns.index(col) for col in discrete_feature_columns]
    continuous_feature_indices = [all_columns.index(col) for col in continuous_feature_columns]
    
    # 創建特徵名稱和索引的映射字典
    feature_name_to_index = {name: idx for idx, name in enumerate(all_columns)}
    index_to_feature_name = {idx: name for idx, name in enumerate(all_columns)}
    
    # 保存特徵處理相關資訊，用於後續推理
    feature_info = {
        'original_feature_names': original_feature_names,  # 原始特徵名稱順序
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
    
    # 保存到檔案
    with open('feature_info.pkl', 'wb') as f:
        pickle.dump(feature_info, f)
    
    print("特徵資訊已保存至 'feature_info.pkl'")
    
    return features, target, patient_ids, discrete_feature_indices, continuous_feature_indices, all_columns, feature_info

# SAINT模型定義 (修正了形狀問題)
class SAINT(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, 
                 discrete_feature_indices, continuous_feature_indices, 
                 num_heads=8, num_layers=6, dropout=0.1):
        super(SAINT, self).__init__()
        
        self.hidden_size = hidden_size
        self.discrete_feature_indices = discrete_feature_indices
        self.continuous_feature_indices = continuous_feature_indices
        
        # 計算離散特徵和連續特徵的確切數量
        self.num_discrete = len(discrete_feature_indices)
        self.num_continuous = len(continuous_feature_indices)
        
        print(f"模型初始化 - 離散特徵數量: {self.num_discrete}, 連續特徵數量: {self.num_continuous}")
        
        # 離散特徵的嵌入層
        if self.num_discrete > 0:
            self.discrete_embedding = nn.Sequential(
                nn.Linear(self.num_discrete, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # 連續特徵的嵌入層
        if self.num_continuous > 0:
            self.continuous_embedding = nn.Sequential(
                nn.Linear(self.num_continuous, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        
        # Xavier初始化
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
        
        # Transformer編碼器
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
        
        # 輸出層
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # 輸出層初始化
        for module in self.fc_out:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if torch.isnan(x).any():
            print("NaN detected in input!")
            x = torch.nan_to_num(x, nan=0.0)
        
        # 檢查特徵索引是否有效
        max_index = x.shape[1] - 1
        valid_discrete_indices = [i for i in self.discrete_feature_indices if i <= max_index]
        valid_continuous_indices = [i for i in self.continuous_feature_indices if i <= max_index]
        
        # 打印有效索引的數量，用於調試
        if len(valid_discrete_indices) != len(self.discrete_feature_indices) or len(valid_continuous_indices) != len(self.continuous_feature_indices):
            print(f"警告：特徵索引超出範圍！輸入形狀：{x.shape}")
            print(f"有效離散索引：{len(valid_discrete_indices)}/{len(self.discrete_feature_indices)}")
            print(f"有效連續索引：{len(valid_continuous_indices)}/{len(self.continuous_feature_indices)}")
        
        # 分別提取離散和連續特徵
        discrete_x = x[:, valid_discrete_indices] if valid_discrete_indices else torch.zeros((x.shape[0], 0), device=x.device)
        continuous_x = x[:, valid_continuous_indices] if valid_continuous_indices else torch.zeros((x.shape[0], 0), device=x.device)
        
        # 初始化嵌入變量
        x_embedded = None
        
        # 分別對離散和連續特徵進行嵌入
        if discrete_x.shape[1] > 0:
            discrete_embedded = self.discrete_embedding(discrete_x)
            x_embedded = discrete_embedded
        
        if continuous_x.shape[1] > 0:
            continuous_embedded = self.continuous_embedding(continuous_x)
            if x_embedded is None:
                x_embedded = continuous_embedded
            else:
                x_embedded = x_embedded + continuous_embedded
        
        # 確保x_embedded不為None
        if x_embedded is None:
            raise RuntimeError("沒有有效的特徵可以嵌入")
        
        # 添加序列維度
        x_embedded = x_embedded.unsqueeze(1)
        
        # Transformer處理
        x = self.transformer_encoder(x_embedded)
        x = x.squeeze(1)  # 移除序列維度
        
        # 輸出層
        return self.fc_out(x)
    
    # 獲取特徵重要性的方法
    def get_feature_importance(self, X, feature_names, device='cpu'):
        # 將模型設為評估模式
        self.eval()
        
        # 創建一個空的重要性列表
        importances = np.zeros(X.shape[1])
        
        # 對每個特徵進行擾動
        for i in range(X.shape[1]):
            # 創建一個副本
            X_perturbed = X.clone()
            
            # 隨機擾動該特徵
            X_perturbed[:, i] = torch.randn_like(X_perturbed[:, i]) * X_perturbed[:, i].std() + X_perturbed[:, i].mean()
            
            # 原始預測和擾動後的預測
            with torch.no_grad():
                original_pred = self(X).cpu().numpy()
                perturbed_pred = self(X_perturbed).cpu().numpy()
            
            # 計算擾動對預測的影響
            importances[i] = np.mean(np.abs(original_pred - perturbed_pred))
        
        return importances

# 模型評估函數
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).cpu().numpy()
        ground_truth = y_test.cpu().numpy()
        
        mse = mean_squared_error(ground_truth, predictions)
        mae = mean_absolute_error(ground_truth, predictions)
        r2 = r2_score(ground_truth, predictions)
        
        # 計算大於或小於1.7的一致性
        threshold = 1.7
        gt_above = ground_truth > threshold
        pred_above = predictions > threshold
        
        # 計算預測值和真實值在閾值上表現一致的比例
        consistency = np.mean((gt_above & pred_above) | (~gt_above & ~pred_above))
        consistency_percentage = consistency * 100
        
        # 分別計算大於閾值和小於閾值的一致性
        above_threshold_correct = np.sum(gt_above & pred_above)
        below_threshold_correct = np.sum(~gt_above & ~pred_above)
        total_above_threshold = np.sum(gt_above)
        total_below_threshold = np.sum(~gt_above)
        
        print("\n模型性能指標:")
        print(f"均方誤差 (MSE): {mse:.4f}")
        print(f"平均絕對誤差 (MAE): {mae:.4f}")
        print(f"R平方分數 (R2 Score): {r2:.4f}")
        
        print("\n閾值分類性能 (閾值 = 1.7):")
        print(f"整體一致性: {consistency_percentage:.2f}% ({int(consistency * len(ground_truth))}/{len(ground_truth)})")
        
        if total_above_threshold > 0:
            above_accuracy = above_threshold_correct / total_above_threshold * 100
            print(f"大於1.7的一致性: {above_accuracy:.2f}% ({above_threshold_correct}/{total_above_threshold})")
        else:
            print("數據中沒有大於1.7的真實值")
            
        if total_below_threshold > 0:
            below_accuracy = below_threshold_correct / total_below_threshold * 100
            print(f"小於等於1.7的一致性: {below_accuracy:.2f}% ({below_threshold_correct}/{total_below_threshold})")
        else:
            print("數據中沒有小於等於1.7的真實值")
        
        return ground_truth, predictions

def visualize_results(ground_truth, predictions, threshold=1.7):
    plt.figure(figsize=(15, 10))
    
    # 1. Original plots
    plt.subplot(221)
    plt.scatter(ground_truth, predictions, alpha=0.6)
    plt.plot([ground_truth.min(), ground_truth.max()], 
             [ground_truth.min(), ground_truth.max()], 
             'r--', lw=2)
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.title('Prediction vs Ground Truth')
    
    # 2. 殘差圖
    residuals = predictions - ground_truth
    plt.subplot(222)
    plt.scatter(ground_truth, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Ground Truth')
    plt.ylabel('Residual')
    plt.title('Residual Distribution')
    
    # 3. 殘差直方圖
    plt.subplot(223)
    sns.histplot(residuals.flatten(), kde=True)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    
    # 4. 閾值分類視覺化
    plt.subplot(224)
    
    # 區分四種情況的顏色
    colors = []
    for gt, pred in zip(ground_truth.flatten(), predictions.flatten()):
        if gt > threshold and pred > threshold:
            colors.append('green')  # 都大於閾值 - 正確
        elif gt <= threshold and pred <= threshold:
            colors.append('blue')   # 都小於閾值 - 正確
        elif gt > threshold and pred <= threshold:
            colors.append('red')    # 真實值大於但預測值小於 - 錯誤
        else:  # gt <= threshold and pred > threshold
            colors.append('orange') # 真實值小於但預測值大於 - 錯誤
    
    plt.scatter(ground_truth, predictions, alpha=0.6, c=colors)
    plt.axhline(y=threshold, color='r', linestyle='--')
    plt.axvline(x=threshold, color='r', linestyle='--')
    plt.xlabel('Ground Truth')
    plt.ylabel('Prediction')
    plt.title(f'Threshold Classification (threshold={threshold})')
    
    # 添加圖例
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

# 特徵重要性可視化函數
def visualize_feature_importance(feature_importances, feature_names, top_n=30):
    # 創建一個DataFrame來儲存特徵重要性
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })
    
    # 按重要性排序
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # 只顯示前top_n個特徵
    if len(importance_df) > top_n:
        top_features = importance_df.head(top_n)
        plt.figure(figsize=(12, 10))
        
        # 創建水平條形圖
        ax = sns.barplot(x='Importance', y='Feature', data=top_features, palette='viridis')
        
        plt.title(f'Top {top_n} Feature Importances', fontsize=16)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.show()
        
        print(f"只顯示前{top_n}個最重要的特徵。若要查看所有特徵的重要性，請增加top_n參數。")
    else:
        plt.figure(figsize=(12, max(6, len(importance_df) * 0.3)))
        
        # 創建水平條形圖
        ax = sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        
        plt.title('Feature Importances', fontsize=16)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    # 返回完整的排序後的特徵重要性
    return importance_df

# 創建推理函數 (用於未來預測)
def predict_with_model(model_path, data_df, feature_info_path='feature_info.pkl'):
    """
    使用保存的模型和特徵信息進行預測
    
    Args:
        model_path (str): 模型路徑
        data_df (pd.DataFrame): 要預測的數據
        feature_info_path (str): 特徵信息文件路徑
    
    Returns:
        np.ndarray: 預測值
    """
    # 載入特徵信息
    with open(feature_info_path, 'rb') as f:
        feature_info = pickle.load(f)
    
    # 提取必要信息
    exclude_columns = feature_info['exclude_columns']
    output_column = feature_info['output_column']
    original_feature_names = feature_info['original_feature_names']
    discrete_feature_indices = feature_info['discrete_feature_indices']
    continuous_feature_indices = feature_info['continuous_feature_indices']
    imputer = feature_info['imputer']
    scaler = feature_info['scaler']
    
    # 檢查是否所有需要的特徵都存在
    for col in original_feature_names:
        if col not in data_df.columns:
            raise ValueError(f"缺少特徵: {col}")
    
    # 按照原始特徵順序排列數據
    features = data_df[original_feature_names].copy()
    
    # 分離出離散和連續特徵
    discrete_feature_names = [original_feature_names[i] for i in discrete_feature_indices]
    continuous_feature_names = [original_feature_names[i] for i in continuous_feature_indices]
    
    # 對連續特徵進行轉換
    continuous_data = features[continuous_feature_names]
    continuous_data_imputed = imputer.transform(continuous_data)
    continuous_data_scaled = scaler.transform(continuous_data_imputed)
    features[continuous_feature_names] = continuous_data_scaled
    
    # 轉換為PyTorch張量
    X = torch.tensor(features.values, dtype=torch.float32)
    
    # 載入模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    input_size = len(original_feature_names)
    hidden_size = 128
    output_size = 1
    
    model = SAINT(input_size, hidden_size, output_size, 
                 discrete_feature_indices, continuous_feature_indices).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 進行預測
    X = X.to(device)
    with torch.no_grad():
        predictions = model(X).cpu().numpy()
    
    return predictions

# 主訓練函數
def main():
    # 載入數據
    try:
        data = pd.read_csv('cleaned_data_after_fill_and_drop.csv')
        print(f"成功載入數據，形狀: {data.shape}")
    except Exception as e:
        print(f"載入數據時出錯: {e}")
        return
    
    # 預處理數據 (更新了返回值以包含feature_info)
    features, target, patient_ids, discrete_feature_indices, continuous_feature_indices, feature_names, feature_info = preprocess_data(data)
    
    # 打印數據信息
    print(f"特徵總數: {len(feature_names)}")
    print(f"離散特徵數量: {len(discrete_feature_indices)}")
    print(f"連續特徵數量: {len(continuous_feature_indices)}")
    
    # 打印離散特徵的索引
    print(f"離散特徵索引: {discrete_feature_indices[:10]}{'...' if len(discrete_feature_indices) > 10 else ''}")
    
    # 打印連續特徵的索引
    print(f"連續特徵索引: {continuous_feature_indices[:10]}{'...' if len(continuous_feature_indices) > 10 else ''}")
    
    # 分組切分數據 (如果有patient_ids的話)
    if patient_ids is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(gss.split(features, target, groups=patient_ids))
        
        # 分割數據
        X_train = features.iloc[train_idx].values
        X_test = features.iloc[test_idx].values
        y_train = target.iloc[train_idx].values
        y_test = target.iloc[test_idx].values
    else:
        # 如果沒有patient_ids，使用一般的隨機分割
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            features.values, target.values, test_size=0.2, random_state=42
        )

    # 轉換為PyTorch張量
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    # 打印張量形狀，用於調試
    print(f"X_train形狀: {X_train.shape}")
    print(f"X_test形狀: {X_test.shape}")
    print(f"y_train形狀: {y_train.shape}")
    print(f"y_test形狀: {y_test.shape}")

    # 移動到GPU (如果可用)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    X_train = X_train.to(device)
    X_test = X_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    # 初始化模型參數
    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = 1

    # 初始化模型
    model = SAINT(input_size, hidden_size, output_size, 
                  discrete_feature_indices, continuous_feature_indices).to(device)
    
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01, amsgrad=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # 數據加載器
    batch_size = 32
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 訓練循環
    epochs = 200
    train_losses, test_losses = [], []

    # 首先驗證模型可以進行前向傳播
    try:
        print("測試模型前向傳播...")
        with torch.no_grad():
            test_batch = X_train[:batch_size]
            test_output = model(test_batch)
            print(f"前向傳播成功，輸出形狀: {test_output.shape}")
    except Exception as e:
        print(f"模型前向傳播測試失敗: {e}")
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

    # 繪製損失曲線
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 模型評估
    ground_truth, predictions = evaluate_model(model, X_test, y_test)
    visualize_results(ground_truth, predictions)

    # 計算特徵重要性
    print("\n計算特徵重要性...")
    
    try:
        # 使用模型內建方法計算特徵重要性
        feature_importances = model.get_feature_importance(X_train, feature_names, device=device)
        feature_importance_df = visualize_feature_importance(feature_importances, feature_names)
        print("\n特徵重要性:")
        print(feature_importance_df.head(20))
    except Exception as e:
        print(f"計算特徵重要性時出錯: {e}")

    # 儲存模型
    torch.save(model.state_dict(), "saint_pd_model.pth")
    print("模型已儲存為 'saint_pd_model.pth'")
    
    # 儲存特徵重要性
    try:
        feature_importance_df.to_csv("feature_importance.csv", index=False)
        print("特徵重要性已儲存為 'feature_importance.csv'")
    except:
        print("無法儲存特徵重要性")

# 執行主函數
if __name__ == "__main__":
    main()