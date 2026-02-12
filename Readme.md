# Temporal Investment Graph with GNN 🚀📍
**Turn raw SEC filings into graph intelligence that reveals how institutional capital flows across companies and sectors over time**

> Based on the system design and methods described in the project report :contentReference[oaicite:0]{index=0}

## 🧩 Problem Statement
- SEC 13F filings are information-heavy, cryptic, and extremely difficult to analyze at scale
- Institutional investment patterns are scattered across thousands of quarterly filings
- Traditional competitor analysis ignores how capital *actually* moves across industries
- Consultants and analysts lack a way to visualize sector rotation and fund behavior over time
- Static dashboards cannot represent temporal relationships between funds and companies
- What’s missing is a **temporal graph intelligence system** that can power **strategic investment and competitive decisions** before **market shifts happen**.

## ⚙️ What This Project Does
Builds a temporal knowledge graph from SEC filings and applies Graph Neural Networks to learn investment behavior patterns.

- Downloads and parses SEC EDGAR 13F filings for institutional holdings
- Merges Yahoo Finance company metadata (sector, industry, indicators)
- Constructs a **temporal knowledge graph** with companies, sectors, and funds as nodes
- Models **static edges** (sector/industry) and **dynamic edges** (quarterly fund investments)
- Applies **GAT / GATv2 Graph Neural Networks** to learn node embeddings
- Uses K-Means + silhouette scoring to discover clusters of similar investment behavior
- Detects sector rotation, fund concentration, and capital flow trends
- Visualizes evolving relationships through an animated graph interface

### This system can be used by:
- 🧑‍💼 Business Consultants
- 📊 Investment Analysts
- 🏦 Institutional Research Teams
- 📈 Portfolio Strategists
- 🧠 Financial Data Scientists

## ▶️ How to Run

1. Clone the repository
<code>
git clone https://github.com/your-repo/temporal-gnn-sec.git
cd temporal-gnn-sec
</code>

2. Install dependencies
<code>
pip install -r requirements.txt
</code>

3. Download SEC filings & financial data
<code>
python data_pipeline/download_sec_data.py
python data_pipeline/fetch_yfinance_data.py
</code>

4. Build the temporal graph
<code>
python graph/build_temporal_graph.py
</code>

5. Train the GNN model
<code>
python gnn/train_gatv2.py
</code>

6. Launch visualization
<code>
python app/visualize_graph.py
</code>

## 🖥️ Output
![Graph Visualization](docs/graph_demo.gif)

Animated graph showing how institutional investments shift across companies and sectors over time. Clusters reveal hidden relationships driven by capital flow rather than direct competition.

## 🧰 Tech Stack

**Data & ML**
- PyTorch Geometric (GAT, GATv2)
- scikit-learn (KMeans, silhouette score)
- NetworkX

**Streaming/Backend**
- SEC EDGAR Downloader
- yFinance
- Pandas

**Visualization**
- Matplotlib / Plotly
- Animated Network Graph

**Language**
- Python
