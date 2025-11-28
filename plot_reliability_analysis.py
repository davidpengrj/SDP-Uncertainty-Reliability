import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os

# 设置绘图风格
sns.set(style="whitegrid")
# 设置通用字体，防止中文或特殊符号乱码
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False 

def plot_correlation_ranking(df, metric_name):
    """
    绘制不确定性与性能指标的相关性排行榜 (Spearman Correlation)
    回答问题：当模型变不确定时，它的性能真的下降了吗？
    """
    print(f"\n>>> 正在分析可靠性排行: Uncertainty vs {metric_name} ...")
    
    # 1. 数据清洗
    sub_df = df.dropna(subset=[metric_name, 'Uncertainty'])
    if len(sub_df) < 10:
        print(f"  [跳过] {metric_name} 数据不足。")
        return

    # 2. 计算每个模型的相关性
    model_corrs = []
    for model in sub_df['Model'].unique():
        m_df = sub_df[sub_df['Model'] == model]
        if len(m_df) < 5: continue
        
        # 计算 Spearman 相关系数 (Uncertainty vs Metric)
        # 如果 Metric 是性能(AUC/MCC)，我们希望它是负相关 (Unc高 -> Perf低)
        r, _ = spearmanr(m_df['Uncertainty'], m_df[metric_name])
        if np.isnan(r): r = 0
        model_corrs.append({'Model': model, 'Correlation': r})
    
    if not model_corrs:
        return

    corr_df = pd.DataFrame(model_corrs).sort_values('Correlation')
    
    # 3. 绘制排行榜
    plt.figure(figsize=(10, max(6, len(corr_df) * 0.5)))
    
    # 颜色编码：蓝色表示负相关（理想，Reliable），红色表示正相关（反常，Unreliable）
    # 注意：这仅适用于 AUC/MCC/F1 等“越高越好”的指标
    palette = ['#4e79a7' if x < 0 else '#e15759' for x in corr_df['Correlation']]
    
    sns.barplot(x='Correlation', y='Model', data=corr_df, palette=palette)
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    
    plt.title(f'Reliability Ranking: Uncertainty vs {metric_name}\n(Blue/Negative is Better: Model knows when it fails)', fontsize=14)
    plt.xlabel(f'Spearman Correlation')
    plt.tight_layout()
    
    filename = f'analysis_ranking_{metric_name}.png'
    plt.savefig(filename, dpi=300)
    print(f"  -> 排行榜已保存: {filename}")

def plot_auc_vs_ece(df):
    """
    绘制 AUC vs ECE 权衡图 (CCF-A 级核心图表)
    回答问题：哪些模型在保持高性能的同时，还能保持低误校准率？
    """
    if 'ECE' not in df.columns:
        print("\n[跳过] 数据集中没有 ECE 列，无法绘制 AUC vs ECE 图。")
        return

    print(f"\n>>> 正在绘制 Trade-off 分析: AUC (Performance) vs ECE (Calibration) ...")
    
    # 1. 计算每个模型的平均 AUC 和 ECE
    summary = df.groupby('Model')[['AUC', 'ECE']].mean().reset_index()
    
    # 2. 绘图
    plt.figure(figsize=(12, 9))
    
    # 使用散点图，点的大小可以表示样本数量或其他权重，这里统一大小
    # hue='Model' 给每个点上色，style='Model' 给不同形状
    scatter = sns.scatterplot(
        data=summary, 
        x='AUC', 
        y='ECE', 
        hue='Model', 
        style='Model',
        s=200, 
        alpha=0.9,
        palette='tab20' # 使用颜色丰富的调色板
    )
    
    # 3. 添加文字标签 (避免重叠是个难点，这里简单处理)
    for i in range(summary.shape[0]):
        plt.text(
            summary.AUC[i] + 0.001,  # X轴稍微偏移
            summary.ECE[i],          # Y轴
            summary.Model[i], 
            horizontalalignment='left', 
            size='small', 
            color='black', 
            weight='semibold'
        )

    # 4. 划分象限/区域
    # 获取当前轴的范围
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    
    # 理想区域：右下角 (AUC高，ECE低)
    plt.title('Performance vs. Reliability Trade-off\n(Target: Bottom-Right Corner)', fontsize=16)
    plt.xlabel('Predictive Performance (Average AUC) $\\rightarrow$ Higher is Better', fontsize=12)
    plt.ylabel('Calibration Error (Average ECE) $\\leftarrow$ Lower is Better', fontsize=12)
    
    # 添加背景网格
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 标注理想方向
    plt.annotate('Ideal Zone\n(High AUC, Low ECE)', 
                 xy=(x_max, y_min), 
                 xytext=(x_max - (x_max-x_min)*0.2, y_min + (y_max-y_min)*0.2),
                 arrowprops=dict(facecolor='green', shrink=0.05),
                 fontsize=12, color='green', weight='bold',
                 ha='center')

    plt.tight_layout()
    plt.savefig('analysis_tradeoff_auc_ece.png', dpi=300)
    print(f"  -> Trade-off 图已保存: analysis_tradeoff_auc_ece.png")

if __name__ == "__main__":
    result_file = "benchmark_results_IVDP.csv"
    
    if os.path.exists(result_file):
        print(f"✅ 读取结果文件: {result_file}")
        df = pd.read_csv(result_file)
        
        # 1. 传统的 Spearman 排序分析 (针对性能指标)
        # 注意：不建议对 ECE 做 Spearman 分析，因为 ECE 本身就是越低越好，逻辑不一样
        metrics_to_rank = ['MCC', 'AUC', 'F1'] 
        
        for metric in metrics_to_rank:
            if metric in df.columns:
                plot_correlation_ranking(df, metric)
        
        # 2. 高级的 Trade-off 分析 (AUC vs ECE)
        plot_auc_vs_ece(df)
        
        print("\n✅ 所有分析图表绘制完毕！")
    else:
        print(f"❌ 找不到 {result_file}，请先运行 run_caret_benchmark.py 生成数据！")