
# Reliability of Uncertainty Quantification in Software Defect Prediction

This repository contains the source code and experimental data for our research on the **reliability of uncertainty quantification (UQ)** in Software Defect Prediction (SDP).

Unlike traditional studies that focus solely on predictive performance (e.g., AUC, F1-score), this work systematically evaluates **confidence calibration** across two key scenarios:
1.  **IVDP** (Inner-Version Defect Prediction)
2.  **CVDP** (Cross-Version Defect Prediction)

We analyze 28 Machine Learning models across 53 dataset versions to reveal the trade-offs between **Performance** and **Reliability** (measured by Expected Calibration Error, ECE).

## 📂 Repository Structure

```text
├── run_ivdp_benchmark.py       # Training script for IVDP scenario (Within-Version)
├── run_cvdp_benchmark.py       # Training script for CVDP scenario (Cross-Version)
├── plot_reliability_analysis.py # Analysis script for ranking (Spearman) and Trade-off (AUC vs ECE)
├── plot_scenario_contrast.py   # Analysis script for IVDP vs CVDP contrast (Drift Arrows)
├── requirements.txt            # Python dependencies
├── DefectData-master/          # Dataset directory (PROMISE, NASA, etc.)
└── figures_contrast/           # Generated result figures
````

## 🚀 Quick Start

### 1\. Prerequisites

Ensure you have Python 3.8+ installed. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2\. Run Experiments

The experiments are divided into two scenarios. You can run them sequentially:

**Step 1: Run IVDP Benchmark**
Generates `benchmark_results_IVDP.csv`.

```bash
python run_ivdp_benchmark.py
```

**Step 2: Run CVDP Benchmark**
Generates `benchmark_results_CVDP.csv`.

```bash
python run_cvdp_benchmark.py
```

### 3\. Generate Analysis Plots

After obtaining the `.csv` result files, run the analysis scripts to generate visualizations:

**Reliability Ranking & Trade-off (Single Scenario Analysis):**

```bash
python plot_reliability_analysis.py
```

**Scenario Contrast & Robustness Shift (IVDP vs. CVDP):**

```bash
python plot_scenario_contrast.py
```

## 📊 Key Results

### 1\. The "Drift" of Reliability (IVDP -\> CVDP)

The figure below illustrates how models shift when moving from a stable environment (IVDP) to a distribution-shifted environment (CVDP).

  * **Upward Arrows** indicate an increase in **ECE (Overconfidence)**.
  * **Random Forest** demonstrates the highest robustness (shortest drift).

### 2\. Performance vs. Reliability Trade-off

We identify the "Ideal Zone" (High AUC, Low ECE) for SDP models. While **Naive Bayes** excels at ranking (Spearman correlation), it suffers from poor calibration. **Bagged AdaBoost** and **Random Forest** achieve the best balance.

## 🛠️ Metrics Used

  * **Performance:** AUC, MCC, F1-Score
  * **Uncertainty:** Shannon Entropy (for Sklearn models), Prediction Variance (for PyTorch MC Dropout)
  * **Reliability:** Expected Calibration Error (ECE), Spearman Correlation (Uncertainty vs. Metric)

## 📁 Dataset

The datasets used in this study are sourced from the [DefectData](https://github.com/awsm-research/DefectData) repository, covering projects from PROMISE, NASA, and Apache ecosystems.

## 📧 Contact

If you have any questions about the code or the paper, please feel free to open an issue.

````

---

### 如何在本地添加这个文件？

1.  在你的 **WSL 终端**（或者 Windows 文件夹）里，创建一个新文件叫 `README.md`。
    ```bash
    nano README.md
    ```
2.  把上面那一长串英文代码**复制粘贴**进去。
3.  按 `Ctrl+O` 保存，`Ctrl+X` 退出。
4.  **提交到 GitHub**：
    ```bash
    git add README.md
    git commit -m "Add README documentation"
    git push
    ```

### 💡 一个小技巧：让图片显示出来
在上面的 Markdown 代码中，我写了这样的语法：
`![Robustness Analysis](figures_contrast/analysis_robustness_arrows.png)`

这要求你的仓库里有一个叫 `figures_contrast` 的文件夹，并且里面有这张图片。如果你之前没有建文件夹，而是直接放在根目录下，你需要：
1.  **修改 README** 里的路径（把 `figures_contrast/` 删掉，直接写文件名）。
2.  **或者**（推荐），把你生成的漂亮图片也 `git add` 并 `push` 上去，这样别人打开你的 GitHub 主页就能直接看到那张震撼的箭头图了！这非常吸睛！
````
