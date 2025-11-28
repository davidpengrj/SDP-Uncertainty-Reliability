import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE
from joblib import Parallel, delayed
import warnings

# === Sklearn 模型库导入 ===
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier, 
    AdaBoostClassifier, 
    BaggingClassifier
)

# 1. 数据加载与预处理
def load_and_preprocess(file_path):
    try:
        if not file_path.endswith('.csv'): return None, None, None, None
        df = pd.read_csv(file_path)
        
        possible_labels = ['defects', 'bug', 'class', 'label', 'problems', '<bug', 'is_defective']
        label_col = None
        for target in possible_labels:
            if target in [c.lower() for c in df.columns]:
                for raw_col in df.columns:
                    if raw_col.lower() == target:
                        label_col = raw_col
                        break
            if label_col: break
        if label_col is None: label_col = df.columns[-1]
        
        if df[label_col].dtype == object:
            df[label_col] = df[label_col].map(lambda x: 1 if str(x).lower() in ['true', 'yes', 'buggy', '1', 'y'] else 0)
        else:
            df[label_col] = df[label_col].apply(lambda x: 1 if x > 0 else 0)
            
        y = df[label_col].values
        X_df = df.drop(columns=[label_col]).apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all').fillna(0)
        X = X_df.values
        
        if len(X) < 20 or len(np.unique(y)) < 2: return None, None, None, None
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        try: return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except: return train_test_split(X, y, test_size=0.2, random_state=42)
    except: return None, None, None, None

# 2. 熵计算 (用于 Uncertainty)
def calculate_entropy(probs):
    epsilon = 1e-10
    probs = np.clip(probs, epsilon, 1. - epsilon)
    return -np.sum(probs * np.log(probs), axis=1)

# === 新增：ECE 计算函数 (核心指标) ===
def calculate_ece(y_true, y_prob, n_bins=10):
    """
    计算 Expected Calibration Error (ECE)
    衡量模型置信度与真实准确率的偏差。
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)
    
    for i in range(n_bins):
        # 找到落在当前桶(bin)里的样本
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        
        # 处理边界情况，最后一个桶包含 1.0
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            
        prob_in_bin = y_prob[in_bin]
        
        if len(prob_in_bin) > 0:
            # 平均置信度 (Confidence)
            avg_confidence = np.mean(prob_in_bin)
            # 真实准确率 (Accuracy)
            avg_accuracy = np.mean(y_true[in_bin])
            # 加权误差
            bin_weight = len(prob_in_bin) / n_total
            ece += bin_weight * np.abs(avg_confidence - avg_accuracy)
            
    return ece

# 3. PyTorch MC Dropout
class MC_Dropout_Net(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    def forward(self, x): return self.net(x)

def train_pytorch_mc(X_train, y_train, X_test):
    torch.set_num_threads(1) 
    X_tr = torch.FloatTensor(X_train)
    y_tr = torch.FloatTensor(y_train).unsqueeze(1)
    X_te = torch.FloatTensor(X_test)
    model = MC_Dropout_Net(X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(100):
        optimizer.zero_grad()
        criterion(model(X_tr), y_tr).backward()
        optimizer.step()
    preds = []
    with torch.no_grad():
        for _ in range(20):
            preds.append(torch.sigmoid(model(X_te)).numpy())
    preds = np.array(preds)
    mean_prob = np.mean(preds, axis=0).flatten()
    uncertainty = np.var(preds, axis=0).flatten()
    pred_cls = (mean_prob > 0.5).astype(int)
    return pred_cls, mean_prob, np.mean(uncertainty)

# 4. Sklearn 模型库
def get_sklearn_models():
    models = {}
    models['Naive Bayes'] = GaussianNB()
    models['KNN'] = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    models['Kernel KNN'] = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=1)
    models['Penalized LogReg'] = LogisticRegression(penalty='l2', max_iter=1000, solver='lbfgs', n_jobs=1)
    models['MLP'] = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=500, random_state=42)
    models['MLP WeightDecay'] = MLPClassifier(hidden_layer_sizes=(64,32), alpha=0.1, max_iter=500, random_state=42)
    models['LDA'] = LinearDiscriminantAnalysis()
    models['C4.5 (J48)'] = DecisionTreeClassifier(criterion='entropy', max_depth=10, random_state=42)
    models['CART (rpart)'] = DecisionTreeClassifier(criterion='gini', max_depth=10, random_state=42)
    models['SVM Linear'] = SVC(kernel='linear', probability=True, random_state=42)
    models['SVM Radial'] = SVC(kernel='rbf', probability=True, random_state=42)
    models['Random Forest'] = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)
    models['Bagged AdaBoost'] = BaggingClassifier(
        estimator=AdaBoostClassifier(n_estimators=10),
        n_estimators=10, random_state=42, n_jobs=1
    )
    models['GBM'] = GradientBoostingClassifier(n_estimators=50, random_state=42)
    models['AdaBoost'] = AdaBoostClassifier(n_estimators=50, random_state=42)
    models['LogitBoost (Sim)'] = GradientBoostingClassifier(loss='log_loss', n_estimators=50, random_state=42)
    return models

# 5. 单个文件处理函数
def process_single_dataset(file_path):
    file_name = os.path.basename(file_path)
    print(f"👉 正在处理: {file_name}")
    
    # 强制静音警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_train, X_test, y_train, y_test = load_and_preprocess(file_path)
    
    if X_train is None: return []
    
    try:
        if np.sum(y_train==1) > 1:
            k = min(np.sum(y_train==1)-1, 5)
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train
    except: X_train_res, y_train_res = X_train, y_train

    dataset_results = []
    
    # 遍历 Sklearn 模型
    for name, model in get_sklearn_models().items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_res, y_train_res)
                
                y_pred = model.predict(X_test)
                
                # 初始化指标
                auc = 0.5
                unc = 0.0
                ece = 0.0 # 初始化 ECE
                
                if hasattr(model, "predict_proba"):
                    y_prob_all = model.predict_proba(X_test)
                    y_prob = y_prob_all[:, 1] # 取正类概率
                    
                    # 计算 AUC
                    if len(np.unique(y_test)) > 1:
                        auc = roc_auc_score(y_test, y_prob)
                    
                    # 计算 Uncertainty (Entropy)
                    unc = np.mean(calculate_entropy(y_prob_all))
                    
                    # === 计算 ECE ===
                    ece = calculate_ece(y_test, y_prob)
                
                mcc = matthews_corrcoef(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                # 保存结果，增加 ECE
                dataset_results.append({
                    'Dataset': file_name, 
                    'Model': name, 
                    'MCC': mcc, 
                    'F1': f1, 
                    'AUC': auc, 
                    'Uncertainty': unc,
                    'ECE': ece  # <--- 新增列
                })
        except: pass
        
    # PyTorch 模型
    try:
        pt_pred, pt_prob, pt_unc = train_pytorch_mc(X_train_res, y_train_res, X_test)
        mcc = matthews_corrcoef(y_test, pt_pred)
        f1 = f1_score(y_test, pt_pred)
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, pt_prob)
        else:
            auc = 0.5
            
        # === 计算 PyTorch 模型的 ECE ===
        ece = calculate_ece(y_test, pt_prob)
        
        dataset_results.append({
            'Dataset': file_name, 
            'Model': 'PyTorch MC Dropout', 
            'MCC': mcc, 
            'F1': f1, 
            'AUC': auc, 
            'Uncertainty': pt_unc,
            'ECE': ece # <--- 新增列
        })
    except: pass

    return dataset_results

# 6. 主程序入口
if __name__ == "__main__":
    # 请确保路径正确
    root_dir = "DefectData-master/DefectData-master/inst/extdata/terapromise"
    
    if not os.path.exists(root_dir):
        print(f"❌ 错误：找不到文件夹 {root_dir}")
    else:
        all_files = [os.path.join(r, f) for r, d, fs in os.walk(root_dir) for f in fs if f.endswith('.csv')]
        
        # all_files = all_files[:10] # 调试用
        
        print(f"🚀 开始测试 (含 ECE 指标) - 9900X 20线程版")
        print(f"📂 待处理文件数: {len(all_files)}\n")
        
        # 消除主进程警告
        warnings.filterwarnings("ignore")
        
        results_lists = Parallel(n_jobs=20, verbose=0)(
            delayed(process_single_dataset)(f) for f in all_files
        )
        
        final_results = [item for sublist in results_lists for item in sublist]

        if final_results:
            output_file = "benchmark_results_IVDP.csv"
            df = pd.DataFrame(final_results)
            df.to_csv(output_file, index=False)
            print(f"\n✅ 实验结束！")
            print(f"📊 结果已保存至: {output_file} (包含新增的 'ECE' 列)")
        else:
            print("\n❌ 没有产生结果。")