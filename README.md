
# ReliableSDP: 软件缺陷预测可靠性基准测试框架 (Reliability-aware SDP Benchmark)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Active-brightgreen)

🚀 **ReliableSDP** 是一个专注于软件缺陷预测 (SDP) **可靠性 (Reliability)** 与 **不确定性 (Uncertainty)** 的基准测试框架。

传统的 SDP 研究往往只关注 AUC、MCC 或 F1 等性能指标，而忽略了模型的**校准度 (Calibration)**。本框架通过引入 **ECE (Expected Calibration Error)** 和 **认知不确定性 (Epistemic Uncertainty)**，深入探究模型在面对**项目内 (IVDP)** 和 **跨版本 (CVDP)** 数据分布漂移时，是否还能保持“诚实”。

---

## 📊 核心实验结论 (Key Insights)

基于本框架在 TeraPROMISE 数据集上的实验，我们在 **Uncertainty** 与 **Performance (MCC/AUC)** 的相关性研究中取得了以下关键发现：

### 1. 性能与可靠性的倒挂现象 (The Reliability Gap)
> **研究问题**：当模型表现出高不确定性时，它的预测性能是否真的下降了？（理想情况下应呈负相关：不确定性越高 $\rightarrow$ 性能越差）

通过 Spearman 相关性分析，我们发现：
* **反直觉的结论**：绝大多数模型（如 **C4.5, SVM Radial, LDA**）呈现出**正相关 (Red Bars)**。这意味着模型在“错误”的时候往往非常自信（低不确定性），而在“正确”的时候却犹豫不决。
* **可靠的模型**：仅有 **Naive Bayes** 和 **Penalized LogReg** 等少数模型呈现出理想的**负相关 (Blue Bars)**。这表明简单的贝叶斯方法虽然预测精度不高，但由于其基于概率的特性，在“自我认知”方面比复杂的黑盒模型更诚实。

![Uncertainty vs MCC Ranking](figures_ivdp/analysis_ranking_MCC.png)
*(图注：蓝色代表模型“有自知之明”（理想），红色代表模型“盲目自信”或“过度焦虑”。)*

### 2. 性能与校准度的权衡 (Trade-off Analysis)
> **研究问题**：是否存在既准确又校准良好的模型？

从下面的 Trade-off 散点图来看：
* **最佳实践**：**Random Forest** 位于图表的右下角（Ideal Zone），展现了最高的 AUC (>0.80) 和最低的 ECE (<0.15)。它是目前在该基准测试中综合表现最稳健的模型。
* **极端情况**：Naive Bayes 虽然不确定性排序很好（结论1），但其绝对校准误差 (ECE) 极大，处于左上角。这说明它虽然知道相对的难易程度，但给出的具体概率值偏差严重。

![Performance vs Reliability Trade-off](figures_ivdp/analysis_tradeoff_auc_ece.png)

### 3. 跨版本场景下的分布漂移 (Robustness under Drift)
> **研究问题**：当从旧版本 (IVDP) 迁移到新版本 (CVDP) 时，模型发生了什么？

漂移路径图显示：
* 几乎所有模型的箭头都指向了 **左上方**。
* **左移 (AUC $\downarrow$)**：随着软件版本的迭代，数据分布发生概念漂移，导致预测性能普遍下降。
* **上移 (ECE $\uparrow$)**：更为严重的是，模型的校准误差显著增加。这表明在面对未知的新版本数据时，模型不仅变笨了，而且变得更加**“过度自信” (Overconfident)**。

![Model Robustness Trajectory](figures_contrast/analysis_robustness_arrows.png)

---

## ✨ 主要特性

* **多维度评估**：不仅计算 AUC/MCC/F1，还计算 **ECE** 和 **Entropy (Uncertainty)**。
* **双场景覆盖**：支持 **IVDP (Within-Version)** 和 **CVDP (Cross-Version)** 自动化测试。
* **模型丰富**：集成 15+ 种 Sklearn 经典模型及 **PyTorch MC Dropout** 深度学习基线。
* **自动化绘图**：一键生成上述的 Ranking 图、Trade-off 散点图和漂移路径图。

## 📂 项目结构

```text
.
├── figures_ivdp/              # [图片] 存放 IVDP 相关的分析图表
│   ├── analysis_ranking_MCC.png
│   └── analysis_tradeoff_auc_ece.png
├── figures_contrast/          # [图片] 存放 CVDP 对比相关的分析图表
│   └── analysis_robustness_arrows.png
├── run_ivdp_benchmark.py      # [主程序] 运行项目内缺陷预测 (IVDP)
├── run_cvdp_benchmark.py      # [主程序] 运行跨版本缺陷预测 (CVDP)
├── plot_reliability_analysis.py # [绘图] 生成 Ranking 图和 Trade-off 图
├── plot_scenario_contrast.py  # [绘图] 生成 IVDP vs CVDP 对比图
├── requirements.txt           # 依赖库列表
└── README.md                  # 说明文档
````

## 🛠️ 快速开始

### 1\. 环境依赖

```bash
pip install -r requirements.txt
```

### 2\. 运行基准测试

确保数据集路径正确（默认支持 PROMISE/TeraPROMISE 格式），然后运行：

```bash
# 1. 运行 IVDP 实验
python run_ivdp_benchmark.py

# 2. 运行 CVDP 跨版本实验
python run_cvdp_benchmark.py
```

### 3\. 复现分析图表

当 CSV 结果文件生成后，运行可视化脚本以复现本文档中的结论图表：

```bash
# 分析不确定性与 MCC/AUC 的相关性 (生成 Ranking 图)
python plot_reliability_analysis.py

# 分析跨版本漂移现象 (生成 Arrow 图)
python plot_scenario_contrast.py
```

## 📝 引用

如果你在研究中使用了本代码或受到启发，请考虑引用本项目。

```
```
