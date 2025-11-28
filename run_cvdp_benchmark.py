import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

# ==========================================
# 1. 核心工具函数：ECE 与 Entropy
# ==========================================
def calculate_entropy(probs):
    epsilon = 1e-10
    probs = np.clip(probs, epsilon, 1. - epsilon)
    return -np.sum(probs * np.log(probs), axis=1)

def calculate_ece(y_true, y_prob, n_bins=10):
    """ 计算 Expected Calibration Error (ECE) """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        
        prob_in_bin = y_prob[in_bin]
        if len(prob_in_bin) > 0:
            avg_confidence = np.mean(prob_in_bin)
            avg_accuracy = np.mean(y_true[in_bin])
            bin_weight = len(prob_in_bin) / n_total
            ece += bin_weight * np.abs(avg_confidence - avg_accuracy)
    return ece

# ==========================================
# 2. 数据加载逻辑 (强壮版 + 路径修复)
# ==========================================
def load_dataset_raw(file_path):
    """ 读取单个文件，返回 X (DataFrame) 和 y (Series) """
    if not os.path.exists(file_path):
        return None, None

    try:
        df = pd.read_csv(file_path)
        
        # 1. 寻找标签列 (忽略大小写)
        possible_labels = ['defects', 'bug', 'class', 'label', 'problems', '<bug', 'is_defective']
        label_col = None
        df_cols_lower = [c.lower() for c in df.columns]
        
        for target in possible_labels:
            if target in df_cols_lower:
                label_col = df.columns[df_cols_lower.index(target)]
                break
        
        if label_col is None: 
            label_col = df.columns[-1]

        # 2. 处理标签列
        if df[label_col].dtype == object:
            df[label_col] = df[label_col].map(lambda x: 1 if str(x).lower() in ['true', 'yes', 'buggy', '1', 'y'] else 0)
        else:
            df[label_col] = pd.to_numeric(df[label_col], errors='coerce').fillna(0).apply(lambda x: 1 if x > 0 else 0)
            
        y = df[label_col].values
        
        # 3. 处理特征列
        X_df = df.drop(columns=[label_col])
        X_df = X_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        if X_df.shape[1] == 0: return None, None

        return X_df, y

    except: return None, None

def get_cvdp_pairs(root_dir):
    """ 
    自动扫描文件夹，生成 CVDP 对
    【修复】支持递归扫描子文件夹，直接获取绝对路径
    """
    project_dict = {}
    
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if not f.endswith('.csv'): continue
            
            # 获取绝对路径
            full_path = os.path.join(root, f)
            
            # 正则匹配项目名和版本号
            match = re.match(r"([a-zA-Z]+)[-_]?([\d\.]+)\.csv", f)
            if match:
                proj_name = match.group(1)
                version = match.group(2)
                if proj_name not in project_dict:
                    project_dict[proj_name] = []
                project_dict[proj_name].append((version, full_path))
            
    pairs = []
    for proj, versions in project_dict.items():
        # 按版本号排序 (字符串排序)
        versions.sort(key=lambda x: x[0])
        if len(versions) < 2: continue
        
        for i in range(len(versions) - 1):
            pairs.append({
                'project': proj,
                'train_ver': versions[i][0], 'train_path': versions[i][1],
                'test_ver': versions[i+1][0], 'test_path': versions[i+1][1]
            })
    return pairs

# ==========================================
# 3. 模型定义
# ==========================================
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
    models['Bagged AdaBoost'] = BaggingClassifier(estimator=AdaBoostClassifier(n_estimators=10), n_estimators=10, random_state=42, n_jobs=1)
    models['GBM'] = GradientBoostingClassifier(n_estimators=50, random_state=42)
    models['AdaBoost'] = AdaBoostClassifier(n_estimators=50, random_state=42)
    models['LogitBoost (Sim)'] = GradientBoostingClassifier(loss='log_loss', n_estimators=50, random_state=42)
    return models

# ==========================================
# 4. CVDP 处理逻辑
# ==========================================
def process_cvdp_pair(pair_info):
    proj = pair_info['project']
    ver_train = pair_info['train_ver']
    ver_test = pair_info['test_ver']
    task_name = f"{proj} ({ver_train} -> {ver_test})"
    
    # 1. 加载数据
    X_train_df, y_train = load_dataset_raw(pair_info['train_path'])
    X_test_df, y_test = load_dataset_raw(pair_info['test_path'])
    
    if X_train_df is None or X_test_df is None: return []
    if len(y_train) < 10 or len(y_test) < 10: return []
    
    # 2. 特征对齐
    common_cols = list(set(X_train_df.columns) & set(X_test_df.columns))
    if len(common_cols) < 3: return []
        
    X_train = X_train_df[common_cols].values
    X_test = X_test_df[common_cols].values
    
    # 3. 标准化
    scaler = StandardScaler()
    try:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    except: return []
    
    # 4. SMOTE
    try:
        if np.sum(y_train==1) > 1:
            k = min(np.sum(y_train==1)-1, 5)
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train
    except: X_train_res, y_train_res = X_train, y_train

    dataset_results = []
    
    # 5. 运行 Sklearn 模型
    for name, model in get_sklearn_models().items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_res, y_train_res)
                y_pred = model.predict(X_test)
                
                auc = 0.5; unc = 0.0; ece = 0.0
                if hasattr(model, "predict_proba"):
                    y_prob_all = model.predict_proba(X_test)
                    y_prob = y_prob_all[:, 1]
                    if len(np.unique(y_test)) > 1:
                        auc = roc_auc_score(y_test, y_prob)
                    unc = np.mean(calculate_entropy(y_prob_all))
                    ece = calculate_ece(y_test, y_prob)
                
                mcc = matthews_corrcoef(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                
                dataset_results.append({
                    'Dataset': task_name, 'Scenario': 'CVDP',
                    'Model': name, 'MCC': mcc, 'F1': f1, 'AUC': auc, 'Uncertainty': unc, 'ECE': ece
                })
        except: pass
        
    # 6. 运行 PyTorch 模型
    try:
        pt_pred, pt_prob, pt_unc = train_pytorch_mc(X_train_res, y_train_res, X_test)
        mcc = matthews_corrcoef(y_test, pt_pred)
        f1 = f1_score(y_test, pt_pred)
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, pt_prob)
        else: auc = 0.5
        ece = calculate_ece(y_test, pt_prob)
        
        dataset_results.append({
            'Dataset': task_name, 'Scenario': 'CVDP',
            'Model': 'PyTorch MC Dropout', 'MCC': mcc, 'F1': f1, 'AUC': auc, 'Uncertainty': pt_unc, 'ECE': ece
        })
    except: pass

    if len(dataset_results) > 0:
        print(f"   ✅ [Success] {task_name} 完成，生成 {len(dataset_results)} 条结果")
    
    return dataset_results

# ==========================================
# 5. 主程序入口
# ==========================================
if __name__ == "__main__":
    # --- 请确认路径 ---
    root_dir = "DefectData-master/DefectData-master/inst/extdata/terapromise"
    
    # === 这里的开关已经设为 False，火力全开 ===
    DEBUG_MODE = False
    
    if not os.path.exists(root_dir):
        print(f"❌ 错误：找不到文件夹 {root_dir}")
    else:
        cvdp_pairs = get_cvdp_pairs(root_dir)
        print(f"🚀 开始 CVDP 跨版本测试 - 共识别出 {len(cvdp_pairs)} 对版本")
        
        if DEBUG_MODE:
            print("🛑 [调试模式] 只运行前 5 对，单线程运行...")
            pairs_to_run = cvdp_pairs[:5]
            n_jobs_config = 1
        else:
            print("🔥 [全速模式] 运行所有数据，20 线程 (CPU 利用率将飙升)...")
            pairs_to_run = cvdp_pairs
            n_jobs_config = 20
            
        warnings.filterwarnings("ignore")
        
        results_lists = Parallel(n_jobs=n_jobs_config, verbose=0)(
            delayed(process_cvdp_pair)(pair) for pair in pairs_to_run
        )
        
        final_results = [item for sublist in results_lists for item in sublist]

        if final_results:
            output_file = "benchmark_results_CVDP.csv"
            df = pd.DataFrame(final_results)
            df.to_csv(output_file, index=False)
            print(f"\n✅ CVDP 实验结束！结果已保存至: {output_file}")
            print(f"📊 总共生成结果数: {len(df)}")
        else:
            print("\n❌ 没有产生结果。")