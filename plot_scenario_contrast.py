import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === 绘图设置 ===
sns.set(style="whitegrid")
# 防止字体乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False 

def load_and_merge_data():
    """ 加载并合并 IVDP 和 CVDP 数据 """
    df_list = []
    
    # 1. 加载 IVDP 数据 (智能查找: 优先找新名字，再找旧名字)
    possible_ivdp_files = ["benchmark_results_IVDP.csv", "benchmark_results_with_f1.csv"]
    file_ivdp = None
    
    for f in possible_ivdp_files:
        if os.path.exists(f):
            file_ivdp = f
            break
            
    if file_ivdp:
        print(f"✅ 找到 IVDP 数据: {file_ivdp}")
        df1 = pd.read_csv(file_ivdp)
        df1['Scenario'] = 'IVDP (Within-Version)'
        # 兼容性处理: 如果旧文件没有 ECE，填 NaN
        if 'ECE' not in df1.columns:
            print(f"⚠️ 警告: {file_ivdp} 中没有 ECE 列。建议重新运行 run_ivdp_benchmark.py！")
            df1['ECE'] = np.nan
        df_list.append(df1)
    else:
        print(f"❌ 错误: 找不到 IVDP 数据文件。请检查是否存在 {possible_ivdp_files}")

    # 2. 加载 CVDP 数据
    file_cvdp = "benchmark_results_CVDP.csv"
    if os.path.exists(file_cvdp):
        print(f"✅ 找到 CVDP 数据: {file_cvdp}")
        df2 = pd.read_csv(file_cvdp)
        df2['Scenario'] = 'CVDP (Cross-Version)'
        df_list.append(df2)
    else:
        print(f"❌ 错误: 找不到 {file_cvdp}")
        
    if not df_list: 
        return None
    
    return pd.concat(df_list, ignore_index=True)

def plot_contrast_tradeoff(df):
    """ 
    绘制核心对比图：Performance vs Reliability 
    左图: IVDP, 右图: CVDP
    """
    print(">>> 正在绘制场景对比图 (Contrast Analysis)...")
    
    # 1. 聚合数据
    summary = df.groupby(['Model', 'Scenario'])[['AUC', 'ECE']].mean().reset_index()
    summary = summary.dropna(subset=['ECE'])
    
    if summary.empty:
        print("❌ 数据不足，无法绘图 (可能缺少 ECE 数据)")
        return

    # 2. 绘图
    g = sns.relplot(
        data=summary,
        x="AUC", y="ECE",
        col="Scenario",      
        hue="Model",         
        style="Model",       
        kind="scatter",
        s=300,               
        alpha=0.8,
        palette="tab20",
        height=6, aspect=1.1,
        facet_kws={'sharex': True, 'sharey': True} 
    )
    
    # 3. 标注
    for ax in g.axes.flat:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        ax.text(x_max*0.95, y_min + (y_max-y_min)*0.05, 
                'Ideal Zone\n(High Accuracy, Honest)', 
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize=12, color='green', weight='bold')
        
        ax.grid(True, linestyle='--', alpha=0.5)

    g.fig.suptitle("Reliability Shift Analysis: IVDP vs. CVDP\n(Does your model lie when distributions shift?)", 
                   y=1.05, fontsize=16, weight='bold')
    
    g.set_axis_labels("Predictive Performance (AUC) $\\rightarrow$", "Calibration Error (ECE) $\\leftarrow$")
    
    output_file = "analysis_contrast_ivdp_cvdp.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 对比图已保存: {output_file}")

def plot_shift_arrows(df):
    """
    绘制‘漂移’图：用箭头展示每个模型从 IVDP -> CVDP 的变化
    """
    summary = df.groupby(['Model', 'Scenario'])[['AUC', 'ECE']].mean().reset_index()
    summary = summary.dropna(subset=['ECE']) # 过滤没有 ECE 的行
    models = summary['Model'].unique()
    
    plt.figure(figsize=(12, 10))
    
    # 背景点
    sns.scatterplot(data=summary, x='AUC', y='ECE', hue='Model', style='Scenario', 
                    s=100, alpha=0.6, legend=False, palette="tab20")
    
    print(">>> 正在绘制模型漂移路径 (Shift Arrows)...")
    
    # 绘制箭头
    for model in models:
        subset = summary[summary['Model'] == model]
        if len(subset) != 2: continue
        
        # 找到起点 (IVDP) 和终点 (CVDP)
        row_ivdp = subset[subset['Scenario'].str.contains('IVDP')].iloc[0]
        row_cvdp = subset[subset['Scenario'].str.contains('CVDP')].iloc[0]
        
        plt.arrow(
            row_ivdp['AUC'], row_ivdp['ECE'], 
            row_cvdp['AUC'] - row_ivdp['AUC'], row_cvdp['ECE'] - row_ivdp['ECE'],
            color='gray', alpha=0.5, 
            head_width=0.005, length_includes_head=True
        )
        
        # 标记名称
        plt.text(row_cvdp['AUC'], row_cvdp['ECE'], model, fontsize=9)

    plt.title("Model Robustness: Trajectory from IVDP to CVDP\n(Short arrow = Robust; Upward arrow = Overconfidence Increase)", 
              fontsize=14)
    plt.xlabel("AUC (Performance)")
    plt.ylabel("ECE (Calibration Error)")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    output_file = "analysis_robustness_arrows.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 漂移图已保存: {output_file}")

if __name__ == "__main__":
    df_all = load_and_merge_data()
    
    if df_all is not None:
        plot_contrast_tradeoff(df_all)
        plot_shift_arrows(df_all)
        print("\n🎉 分析完成！请查看生成的 .png 图片。")