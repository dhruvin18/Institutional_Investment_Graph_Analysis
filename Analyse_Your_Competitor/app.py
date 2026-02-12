import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import io
import imageio
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Page config
st.set_page_config(page_title="Competitor Analysis", layout="wide")
st.title("Competitor Analysis")
st.write("Select a company and analysis parameters.")

@st.cache_data
def load_data():
    return (
        pd.read_csv("nodes.csv"),
        pd.read_csv("edges.csv"),
        pd.read_csv("funds.csv"),
        pd.read_csv("SP500Enriched.csv"),
        pd.read_csv("FundHoldings.csv"),
        pd.read_csv("company_embeddings.csv"),
    )

nodes_df, edges_df, funds_df, sp500_df, holdings_df, embeds_df = load_data()
company_dict = dict(zip(sp500_df["Name"], sp500_df["Ticker"]))

# Session state defaults
if "selected_company" not in st.session_state:
    st.session_state.selected_company = "Apple Inc."
if "num_clusters" not in st.session_state:
    st.session_state.num_clusters = 3
if "num_competitors" not in st.session_state:
    st.session_state.num_competitors = 5

def analyse_competitor(ticker, num_clusters, num_competitors):
    # 1) Fit KMeans on the *full* embedding matrix
    H_full = embeds_df.values  # shape = (N_nodes, D)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(H_full)
    labels = kmeans.labels_    # length N_nodes

    # 2) Annotate all nodes with their cluster
    nodes = nodes_df.copy()
    nodes["cluster"] = labels

    # 3) Pull out only the company nodes
    company_nodes = (nodes[nodes["node_type"] == "company"].reset_index(drop=True).copy())

    # 4) Extract their embeddings & labels
    idxs         = company_nodes["node_id"].values         # rows into H_full
    H_companies  = H_full[idxs]                            # (M_companies × D)
    comp_labels  = labels[idxs]                            # length M_companies

    # 5) Compute distance to each company’s assigned centroid
    centroids         = kmeans.cluster_centers_            # (num_clusters × D)
    dists_to_centroid = np.linalg.norm(
        H_companies - centroids[comp_labels], axis=1
    )
    company_nodes["dist_to_centroid"] = dists_to_centroid

    # 6) (Optional) Build cosine-similarity matrix
    sim_matrix = cosine_similarity(H_companies)            # (M × M)

    # 7) Pick out the top-k competitors by Euclidean distance
    competitors = get_competitors(company_nodes, sim_matrix, ticker, num_competitors)

    # 8) Return enriched nodes + competitors table
    return {
        "company_nodes":  company_nodes,
        "competitors":    competitors,
        "target_cluster": int(company_nodes.loc[
                                company_nodes["entity_id"] == ticker, 
                                "cluster"
                            ].iloc[0]),
        "cluster_sizes":  company_nodes["cluster"].value_counts().to_dict()
    }


def get_competitors(company_nodes, sim_matrix, ticker, k):
    """
    Return the k companies whose embeddings are closest (Euclidean)
    to the selected ticker’s embedding, sorted by distance ascending.
    """
    # 1) Rebuild the M×D embedding matrix for just these companies
    H_full      = embeds_df.values
    idxs        = company_nodes["node_id"].values      # integer rows into H_full
    comp_embeds = H_full[idxs]                         # (M_companies × D)

    # 2) Locate the target’s row position
    pos_list = company_nodes.index[company_nodes["entity_id"] == ticker].tolist()
    if not pos_list:
        return pd.DataFrame(columns=["Ticker","Cluster","Distance to Centroid"])
    target_pos = pos_list[0]

    # 3) Compute Euclidean distances to the target vector
    target_vec = comp_embeds[target_pos]
    dists      = np.linalg.norm(comp_embeds - target_vec, axis=1)

    # 4) Exclude the target itself
    dists[target_pos] = np.inf

    # 5) Pick the k smallest distances and sort them
    top_idxs = np.argpartition(dists, k)[:k]
    top_idxs = top_idxs[np.argsort(dists[top_idxs])]

    # 6) Slice out those rows and attach distances
    df = company_nodes.iloc[top_idxs][["entity_id","cluster"]].copy()
    df["Distance to Centroid"] = dists[top_idxs]

    # 7) Rename columns and reset index
    return (
        df
        .rename(columns={
            "entity_id": "Ticker",
            "cluster":    "Cluster"
        })
        .reset_index(drop=True)
    )



def display_company_info(name):
    info = sp500_df[sp500_df["Name"] == name]
    if info.empty:
        st.warning("No metadata found.")
        return
    cols = {
        "Ticker":        "Ticker",
        "Name":          "Name",
        "GICS Sector":   "Sector",
        "GICS Sub-Industry": "Sub-Industry",
        "marketCap":     "Market Cap",
        "country":       "Country",
        "fullTimeEmployees": "Employees"
    }
    info = info[list(cols)].rename(columns=cols)
    info["Market Cap"] = info["Market Cap"].apply(lambda x: f"${x:,.0f}")
    info["Employees"]   = info["Employees"].apply(lambda x: f"{int(x):,}")
    st.subheader("Company Metadata")
    st.table(info.set_index("Ticker"))

def display_top_competitors(analysis):
    # Extract the competitors DataFrame
    df = analysis.get("competitors")
    if df is None or df.empty:
        st.warning("No competitors to display.")
        return

    # Merge with SP500 metadata
    meta = sp500_df[[
        "Ticker",
        "Name",
        "GICS Sector",
        "GICS Sub-Industry",
        "marketCap"
    ]]
    merged = df.merge(meta, on="Ticker", how="left")

    # Select and reorder the columns to show
    display_cols = [
        "Ticker",
        "Name",
        "GICS Sector",
        "GICS Sub-Industry",
        "marketCap",
        "Cluster",
        "Distance to Centroid"
    ]
    display_df = merged[display_cols].rename(columns={
        "Name": "Company Name",
        "marketCap": "Market Cap"
    })

    # Format numeric fields
    display_df["Market Cap"] = display_df["Market Cap"].apply(lambda x: f"${x:,.0f}")
    display_df["Distance to Centroid"] = display_df["Distance to Centroid"].apply(lambda x: f"{x:.4f}")

    # Display the table
    st.subheader("Top Fund Competitors")
    st.table(display_df.reset_index(drop=True))



def plot_shared_funds_graph(ticker: str, competitors: pd.DataFrame):
    """
    Draw a bipartite graph of the selected company + its competitors
    vs. the funds that hold them in the latest quarter, using fund_name
    and company Name for labels.
    """
    # 1) Build company list
    companies = [ticker] + competitors["Ticker"].tolist()

    # 2) Latest-period edges
    latest_t     = edges_df["t"].max()
    newest_edges = edges_df[edges_df["t"] == latest_t]

    # 3) Map node_id → entity_id for all nodes
    node2entity = dict(zip(nodes_df["node_id"], nodes_df["entity_id"]))

    # 4) Collect the set of fund-CIKs that hold any target company
    relevant_funds = set()
    for _, row in newest_edges.iterrows():
        src_ent = node2entity[row["src"]]
        dst_ent = node2entity[row["dst"]]
        if dst_ent in companies:
            relevant_funds.add(src_ent)

    # 5) Build the bipartite graph
    B = nx.Graph()
    B.add_nodes_from(companies, bipartite="company")
    B.add_nodes_from(relevant_funds, bipartite="fund")
    for _, row in newest_edges.iterrows():
        f = node2entity[row["src"]]
        c = node2entity[row["dst"]]
        if c in companies and f in relevant_funds:
            B.add_edge(f, c)

    # 6) Layout: companies on top, funds on bottom
    pos = {}
    for i, c in enumerate(companies):
        pos[c] = (i * 2, 1)
    for i, f in enumerate(sorted(relevant_funds)):
        pos[f] = (i * 2, 0)

    # 7) Build label maps from your CSVs, zero-padding CIKs to 10 digits
    #    so they match the entity_id format in nodes_df.
    cik2name = {
        str(int(cik)).zfill(10): name
        for cik, name in zip(funds_df["fund_cik"], funds_df["fund_name"])
    }
    tkr2name = dict(zip(sp500_df["Ticker"], sp500_df["Name"]))

    labels = {}
    for c in companies:
        labels[c] = tkr2name.get(c, c)
    for f in relevant_funds:
        labels[f] = cik2name.get(f, f)

    # 8) Draw the graph
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw_networkx_nodes(
        B, pos,
        nodelist=companies,
        node_color="skyblue",
        node_size=800,
        label="Company",
        ax=ax
    )
    nx.draw_networkx_nodes(
        B, pos,
        nodelist=list(relevant_funds),
        node_color="lightgreen",
        node_size=400,
        label="Fund",
        ax=ax
    )
    nx.draw_networkx_edges(B, pos, alpha=0.6, ax=ax)
    nx.draw_networkx_labels(B, pos, labels=labels, font_size=8, ax=ax)
    ax.legend(scatterpoints=1)
    ax.set_axis_off()
    ax.set_title(f"{tkr2name.get(ticker, ticker)} Top Competitors Latest Shared-Fund Graph")

    # 9) Render in Streamlit
    st.pyplot(fig)


def display_cluster_composition(analysis):
    """
    Draw stacked bar charts of GICS Sector and Sub-Industry counts per cluster.
    """
    comp_nodes = analysis.get("company_nodes")
    if comp_nodes is None or comp_nodes.empty:
        st.warning("Run the analysis first!")
        return

    # merge in the SP500 metadata
    merged = comp_nodes.merge(
        sp500_df[["Ticker","GICS Sector","GICS Sub-Industry"]],
        left_on="entity_id",
        right_on="Ticker",
        how="left"
    )

    # --- 1) Sector composition ---
    sector_ct = (
        merged
        .groupby(["cluster","GICS Sector"])["entity_id"]
        .count()
        .unstack(fill_value=0)
    )

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sector_ct.plot(
        kind="bar",
        stacked=True,
        ax=ax1,
        legend=False  # we'll put legend below
    )
    ax1.set_title("Cluster Composition by GICS Sector")
    ax1.set_xlabel("Cluster")
    ax1.set_ylabel("Number of Companies")
    ax1.legend(title="Sector", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    st.pyplot(fig1)

    # --- 2) Sub-Industry composition ---
    subind_ct = (
        merged
        .groupby(["cluster","GICS Sub-Industry"])["entity_id"]
        .count()
        .unstack(fill_value=0)
    )
    # if there are *too many* sub-industries, you might want to pick the top N:
    top_sub = subind_ct.sum().sort_values(ascending=False).head(10).index
    subind_ct = subind_ct[top_sub]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    subind_ct.plot(
        kind="bar",
        stacked=True,
        ax=ax2,
        legend=False
    )
    ax2.set_title("Cluster Composition by GICS Sub-Industry")
    ax2.set_xlabel("Cluster")
    ax2.set_ylabel("Number of Companies")
    ax2.legend(title="Sub-Industry", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    st.pyplot(fig2)


def animate_company_holdings(ticker: str):
    """
    Animate the bipartite star-graph of `ticker` vs. its holding funds
    over time, labeling frames from 2020Q1 onward.
    """
    # 1) Map node_id → entity_id
    node2ent = dict(zip(nodes_df["node_id"], nodes_df["entity_id"]))

    # 2) Build holdings_by_t (t → sorted list of fund-CIKs)
    holdings_by_t = {}
    for t, grp in edges_df.groupby("t"):
        funds = {
            node2ent[row["src"]]
            for _, row in grp.iterrows()
            if node2ent[row["dst"]] == ticker
        }
        holdings_by_t[t] = sorted(funds)
    times = sorted(holdings_by_t.keys())
    n_frames = len(times)

    # 3) Create quarter labels starting at 2020Q1
    quarters = pd.period_range("2020Q1", periods=n_frames, freq="Q").astype(str).tolist()

    # 4) Precompute fund CIK → name map (zero-padded)
    cik2name = {
        str(int(cik)).zfill(10): name
        for cik, name in zip(funds_df["fund_cik"], funds_df["fund_name"])
    }

    # 5) Prepare figure
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.close(fig)  # we will draw on it in the loop

    def draw_frame(idx):
        ax.clear()
        quarter = quarters[idx]
        funds   = holdings_by_t[times[idx]]
        m = len(funds)

        # Compute positions
        pos = {ticker: (0.5, 0.8)}
        xs = np.linspace(0.1, 0.9, m) if m > 0 else []
        for x, f in zip(xs, funds):
            pos[f] = (x, 0.2)

        # Build bipartite star graph
        G = nx.Graph()
        G.add_node(ticker, type="company")
        for f in funds:
            G.add_node(f, type="fund")
            G.add_edge(f, ticker)

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=[ticker],
            node_color="#66c2a5",
            node_size=800,
            ax=ax
        )
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=funds,
            node_color="#fc8d62",
            node_size=500,
            ax=ax
        )

        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=2, ax=ax)

        # Draw labels
        ax.text(
            0.5, 0.87, ticker,
            ha="center", va="bottom",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold"
        )
        for f in funds:
            name = cik2name.get(f, f)
            ax.text(
                pos[f][0], pos[f][1] - 0.05,
                name,
                ha="center", va="top",
                fontsize=8,
                rotation=30
            )

        # Frame title & styling
        ax.set_title(f"Holdings of {ticker} — {quarter}", pad=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("white")

    # 6) Render each frame to an image array
    images = []
    for i in range(n_frames):
        draw_frame(i)
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        img = buf.reshape((h, w, 3))
        images.append(img)
    plt.close(fig)

    # 7) Save all frames as an in-memory GIF
    gif_buf = io.BytesIO()
    imageio.mimsave(gif_buf, images, format="GIF", fps=1)
    gif_buf.seek(0)

    # 8) Display in Streamlit
    st.image(gif_buf.getvalue())


# Sidebar inputs
st.sidebar.header("Analysis Parameters")
company_names = sorted(sp500_df["Name"].tolist())
st.sidebar.selectbox("Select a Company", company_names, key="selected_company")
st.sidebar.slider("Number of Clusters", 2, 7, key="num_clusters")
st.sidebar.slider("Number of Competitors", 3, 10, key="num_competitors")
if st.sidebar.button("Start Analysis", type="primary"):
    analysis = analyse_competitor(
        company_dict[st.session_state.selected_company],
        st.session_state.num_clusters,
        st.session_state.num_competitors
    )
    st.subheader("Selected Parameters")
    st.write(f"- Company: {st.session_state.selected_company}")
    st.write(f"- Clusters: {st.session_state.num_clusters}")
    st.write(f"- Competitors: {st.session_state.num_competitors}")
    ticker = company_dict[st.session_state.selected_company]
    animate_company_holdings(ticker)
    display_company_info(st.session_state.selected_company)
    display_top_competitors(analysis)
    
    plot_shared_funds_graph(
        company_dict[st.session_state.selected_company],
        analysis["competitors"]
    )
    display_cluster_composition(analysis)
   