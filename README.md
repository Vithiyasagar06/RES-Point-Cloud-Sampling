# Reconstruction-Based Sampling (RES) for 3D Point Cloud Learning

An end-to-end PyTorch implementation of the **Reconstruction-Based Sampling (RES)** framework for 3D point clouds, designed to run purely on standard CPU hardware[cite: 1]. This project implements the core point reconstruction mechanisms, downstream feature extraction, and classification evaluation benchmarks on the **ModelNet40** dataset based on the paper:

> **RES: Reconstruction-based sampling for point cloud learning**
> *Guoqing Zhang, Wenbo Zhao, Junjun Jiang, Xianming Liu*
> *Displays 92 (2026) 103322, Elsevier*[cite: 1]

---

## 📌 Project Architecture & Theoretical Foundation

### The Core Problem

Point clouds generated from 3D scanners are dense and computationally heavy[cite: 1]. Conventional downsampling strategies (such as **Farthest Point Sampling / FPS**) prioritize uniform spatial coverage[cite: 1]. As a result, they discard critical fine-grained geometric features like sharp edges, corners, and contours[cite: 1].

```
+---------------------------------------------------------------------------------------+
|                                    INPUT POINT CLOUD                                  |
|                                     (N = 1024 points)                                 |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                              RECONSTRUCTION SCORING (RES)                             |
|                                                                                       |
|   1. Query k-Nearest Neighbors (k = 16) for every point p_i                           |
|   2. Pass relative neighbor coordinates into a lightweight MLP                        |
|   3. Reconstruct center point coordinate: \hat{p}_i = MLP(N(p_i))                     |
|   4. Compute Reconstruction Error (Salience Score):                                   |
|                         D_point = || p_i - \hat{p}_i ||_2                             |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                               TOP-K FEATURE DOWNSAMPLING                              |
|                                                                                       |
|   Select M = 256 points with the highest salience errors (Top 25% retained)           |
|   - Low Error  --> Planar/Smooth surface (Redundant -> Culled)                        |
|   - High Error --> Sharp Edge/Corner/Contour (Vital -> Retained)                      |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                             DOWNSTREAM CLASSIFICATION HEAD                            |
|                                                                                       |
|   - Point-wise Feature Extractor: Linear(3 -> 64 -> 128 -> 256) + BatchNorm + ReLU    |
|   - Symmetric Aggregation: Global Max Pooling over M = 256 points                     |
|   - Prediction Head: MLP(256 -> 128 -> 40 Classes) + Dropout(0.3)                    |
+---------------------------------------------------------------------------------------+

```

---

## 📂 Repository Structure

```text
RES-Point-Cloud-Sampling/
│
├── README.md                              # Comprehensive project documentation
├── requirements.txt                       # Exact dependency environment specifications
├── RES_ModelNet40_Pipeline.ipynb          # End-to-end 12-step implementation notebook
│
└── assets/
    ├── coordinate_eda.png                 # Coordinate density & 3D normalized shape
    ├── confusion_matrix.png               # Downstream 40-class evaluation matrix
    └── before_after_res.png               # Before vs. After 3D downsampling comparison

```

---

## 🔬 The 12-Step Implementation Lifecycle

This repository follows a 12-step data science and deep learning lifecycle:

1. **Understand the Data & Goal:** Formulate the objective to downsample raw CAD point clouds while retaining structural identity[cite: 1].
2. **Load Dataset:** Parse native `.off` 3D mesh files into raw NumPy arrays.
3. **Inspect Data:** Extract category distributions, vertex counts, and file structures across 40 classes.
4. **Clean Data:** Resample meshes to a fixed budget ($N=1024$), shift centroids to $(0,0,0)$ for translation invariance, and scale points to fit a unit sphere ($R \le 1.0$).
5. **Exploratory Data Analysis (EDA):** Generate descriptive coordinate metrics ($X, Y, Z$) and plot spatial density distributions[cite: 5].
6. **Find Patterns (RES Module):** Implement $k\text{NN}$-based local neighborhood reconstruction to extract geometric salience[cite: 1].
7. **Prepare Data:** Construct lightweight in-memory PyTorch DataLoaders with stratified batching.
8. **Choose Architecture:** Assemble the end-to-end `RESClassifier` coupling dynamic downsampling with a PointNet classification backbone[cite: 1].
9. **Train Model:** Train the network on CPU using Cross-Entropy loss and the Adam optimizer.
10. **Evaluate Model:** Compute standard point cloud evaluation metrics (Overall Accuracy and Mean Class Accuracy)[cite: 1].
11. **Diagnostic Improvement:** Generate a full $40 \times 40$ Confusion Matrix to identify inter-class confusion boundaries[cite: 3].
12. **Final Insights & Visualization:** Render side-by-side 3D comparisons of full point sets versus the downsampled subsets colored by reconstruction salience[cite: 1, 4].

---

## 📊 Visualizations & Benchmark Results

### 1. Preprocessing & Exploratory Data Analysis

Centering the centroid to $(0, 0, 0)$ and scaling coordinates within a radius of $1.0$ ensures translation and scale invariance, preventing gradient explosion in neural layers:

| Coordinate Axis | Mean | Std Dev | Min | Median (50%) | Max |
| --- | --- | --- | --- | --- | --- |
| **X** | $-1.31 \times 10^{-7}$[cite: 5] | $0.2471$[cite: 5] | $-0.8632$[cite: 5] | $0.0041$[cite: 5] | $0.8552$[cite: 5] |
| **Y** | $3.25 \times 10^{-8}$[cite: 5] | $0.2715$[cite: 5] | $-0.9529$[cite: 5] | $0.0892$[cite: 5] | $0.4134$[cite: 5] |
| **Z** | $-2.05 \times 10^{-7}$[cite: 5] | $0.1186$[cite: 5] | $-0.3500$[cite: 5] | $0.0563$[cite: 5] | $0.1748$[cite: 5] |

---

### 2. Before vs. After RES Downsampling

* **Before ($N=1024$ points):** Color represents reconstruction error $D_{point}$[cite: 1, 4]. Points in yellow/orange ($\sim 0.13$) have the highest reconstruction difficulty and indicate structural boundaries[cite: 4].
* **After ($M=256$ points):** Retaining the top 25% salient points preserves the fine silhouette, tips, and edges while eliminating redundant coplanar vertices[cite: 1, 4].

---

### 3. Downstream Evaluation Matrix (Confusion Matrix)

Classification performance across ModelNet40 categories, tracking true versus predicted labels:

---

## 📐 Mathematical Formulation: LSVR-Style Evaluation

To quantify downsampling performance without relying solely on task accuracy, this repository implements an **LSVR-style regularized performance functional** ($\mathcal{J}_{\text{RES}}$)[cite: 1]:

$$\mathcal{J}_{\text{RES}} = \underbrace{\frac{1}{B}\sum_{b=1}^{B} \mathcal{L}_{\text{task}}\big(y_b, \hat{y}_b(Q_b)\big)}_{\text{Empirical Classification Risk}} + \lambda_1 \underbrace{\frac{1}{B}\sum_{b=1}^{B} \mathcal{D}_{\text{Chamfer}}\big(\mathcal{P}_b, Q_b\big)}_{\text{Geometric Fidelity Penalty}} + \lambda_2 \underbrace{\frac{1}{B}\sum_{b=1}^{B} \mathcal{R}_{\text{salience}}(Q_b)}_{\text{Salience Margin Regularizer}}$$

### Formulation Components:

1. **Empirical Task Risk ($\mathcal{L}_{\text{task}}$):** Cross-entropy classification loss evaluated on the downsampled point cloud $Q_b$[cite: 1].
2. **Geometric Fidelity Penalty ($\mathcal{D}_{\text{Chamfer}}$):** Symmetric Chamfer Distance measuring shape distortion between the original dense cloud $\mathcal{P}$ and downsampled set $Q$:

$$\mathcal{D}_{\text{Chamfer}}(\mathcal{P}, Q) = \frac{1}{\vert{}\mathcal{P}\vert{}}\sum_{p \in \mathcal{P}} \min_{q \in Q} \Vert{}p - q\Vert{}_2^2 + \frac{1}{\vert{}Q\vert{}}\sum_{q \in Q} \min_{p \in \mathcal{P}} \Vert{}q - p\Vert{}_2^2$$


3. **Salience Margin Regularizer ($\mathcal{R}_{\text{salience}}$):** Ensures selected points maximize reconstruction salience:

$$\mathcal{R}_{\text{salience}}(Q) = \max\left(0, 1 - \frac{1}{M}\sum_{q \in Q} D_{\text{point}}(q)\right)$$



---

## 🛠 Installation & Usage

### 1. Prerequisites

This repository requires Python 3.9+ and is configured with stable NumPy 1.x compatibility to prevent binary C-API mismatches:

```bash
git clone https://github.com/Vithiyasagar06/RES-Point-Cloud-Sampling.git
cd RES-Point-Cloud-Sampling

```

### 2. Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt

```

#### `requirements.txt`

```text
numpy==1.26.4
pandas==2.2.2
torch>=2.2.0
matplotlib>=3.7.0

```

### 3. Running the Pipeline

Open Jupyter Notebook and execute the step-by-step cells:

```bash
jupyter notebook RES_ModelNet40_Pipeline.ipynb

```

Set your local dataset path in the first configuration cell:

```python
DATASET_ROOT = r"C:\path\to\your\ModelNet40"

```

---

## 💡 Key Takeaways & Practical Insights

* **Reconstruction as Salience:** Redundancy is inversely related to reconstruction difficulty[cite: 1]. Evaluating point-level reconstruction error isolates edge boundaries without explicit edge annotations[cite: 1].
* **Computational Efficiency:** Reducing point density from $1024$ to $256$ points achieves a **75% reduction in data volume**, speeding up downstream feature extraction while preserving classification-critical contours[cite: 1, 4].
* **Pure CPU Feasibility:** Vectorizing pairwise distance calculations and relative neighbor gathering in PyTorch allows testing and running the pipeline on standard hardware.

---

## 📜 Citation & References

```bibtex
@article{zhang2026res,
  title={RES: Reconstruction-based sampling for point cloud learning},
  author={Zhang, Guoqing and Zhao, Wenbo and Jiang, Junjun and Liu, Xianming},
  journal={Displays},
  volume={92},
  pages={103322},
  year={2026},
  publisher={Elsevier}
}

```

---

## 📄 License

This project is open-source under the [MIT License](https://www.google.com/search?q=LICENSE&utm_source=gemini).
