# Institutional Investment Graph Analysis 📊📍  
**Temporal Knowledge Graph + Graph Neural Networks to uncover institutional capital flow patterns in S&P 500 companies**

---

## 🧩 Problem Statement

Institutional investment trends are buried inside complex SEC 13F filings and static financial datasets.

- SEC filings are information-heavy and difficult to interpret  
- Institutional capital flow patterns are not easily visualized  
- Traditional competitive analysis misses nonlinear investment relationships  
- Sector rotation and fund concentration shifts are hard to detect over time  
- Static dashboards fail to capture temporal structural changes  

**What’s missing is a temporal investment knowledge graph system that can power institutional capital flow insights before strategic market shifts occur.**

---

## ⚙️ What This Project Does

This project builds a **temporal knowledge graph + Graph Neural Network pipeline** to analyze institutional investment behavior across S&P 500 companies.

- Collects institutional holdings data from SEC EDGAR (13F filings)  
- Integrates historical financial data from Yahoo Finance  
- Cleans and aligns data across quarterly time windows  
- Constructs a temporal knowledge graph with static & dynamic edges  
- Models fund-to-company relationships across time  
- Applies Graph Attention Networks (GAT / GATv2) for embedding learning  
- Uses unsupervised clustering (K-Means) to detect investment patterns  
- Deploys an interactive Streamlit dashboard for exploration  

### This system can be used by:

- 🏦 Investment analysts for capital flow analysis  
- 📊 Business consultants for competitive benchmarking  
- 💼 Institutional investors for sector rotation insights  
- 📈 Financial researchers for structural market studies  
- 🧠 AI practitioners exploring temporal graph modeling  

---

## 🏗️ Technical Architecture

### Data Sources
- SEC EDGAR (13F institutional filings)  
- Yahoo Finance (S&P 500 historical data)  

### Graph Construction
- **Nodes:** Companies, Funds, Sectors, Industries  
- **Static Edges:** Company → Sector / Industry  
- **Dynamic Edges:** Fund → Company (quarterly holdings)  

### Modeling
- Graph Attention Networks (GAT / GATv2)  
- Dropout (p=0.2) + Weight Decay regularization  
- Grid search over:
  - Learning rate
  - Hidden channels
  - Attention heads
  - Number of layers
- Evaluation using silhouette score on embeddings  

Validation improvements:
- Silhouette score improved from 0.18 → 0.27  
- Reduced overfitting via dropout  
- Improved stability across temporal windows  

---

## ▶️ How to Run (Streamlit App)

1️⃣ Clone the repository  
<code>git clone https://github.com/dhruvin18/Institutional_Investment_Graph_Analysis.git</code>

2️⃣ Navigate to project directory  
<code>cd Institutional_Investment_Graph_Analysis</code>

3️⃣ Install dependencies  
<code>pip install -r requirements.txt</code>

4️⃣ Launch the Streamlit application  
<code>streamlit run app.py</code>

5️⃣ Open the browser  
Streamlit will automatically open at:  
<code>http://localhost:8501</code>

---

## 🛠️ Tech Stack

- Python  
- Streamlit  
- PyTorch Geometric  
- NetworkX  
- Pandas  
- SEC EDGAR Downloader  
- Yahoo Finance API  
- Scikit-learn
