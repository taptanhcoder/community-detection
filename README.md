
# OSNClusters — Community Detection in Location-Based Social Networks (LBSNs)
---

## Abstract
Community detection in social networks is often performed using only graph topology (friendship edges), which may miss real-world similarity in users’ spatio-temporal behavior. This project studies community discovery in **Location-Based Social Networks (LBSN: social graph + check-in trajectories)** by integrating **user mobility/temporal features** with **graph representation learning**. We first engineer per-user behavioral features from check-ins, then learn user embeddings using **GraphSAGE (Graph Sample-and-Aggregate)** trained with neighbor sampling and link prediction. Communities are obtained by building an embedding-based **kNN similarity graph (k-Nearest Neighbors)** and applying **Leiden clustering**. We evaluate community quality via **structural cohesion** (e.g., modularity, intra-density) and **spatial cohesion** (distance-to-community-centroid), including a **random baseline** via size-preserving label shuffling. A Streamlit dashboard supports qualitative inspection through embedding, geographic, and graph-layout views.

---

## Method 
1. **Parse & Normalize**: load datasets via adapters and enforce unified schema for edges/check-ins.  
2. **Preprocess & Clean**: normalize edges (self-loops, dedup, undirected) + clean check-ins (timestamps, geo validity).  
3. **Induced Filtering**: iteratively filter users by `min_checkins` and induced-graph `min_degree` until stable.  
4. **User Feature Engineering**: build `X_users` from check-ins (activity, spatial footprint, temporal routine, entropy).  
5. **Self-supervised GraphSAGE**: neighbor sampling + negative sampling under link prediction objective → embeddings `Z`.  
6. **Community Detection**: cosine kNN similarity graph from `Z` + Leiden clustering → `community_id`.  
7. **Evaluation & Visualization**: structural + spatial metrics + random baseline; Streamlit visualization.

---

## Datasets & Schema
### Supported datasets (typical)
- **Brightkite (SNAP)**
- **Gowalla (SNAP)**
- **LBSN2Vec++ (Foursquare-based)**

### Normalized schema
**Edges**
- `u`: user_id (string)  
- `v`: user_id (string)

**Check-ins**
- `user_id`: user_id (string)  
- `ts`: timestamp (datetime)  
- `lat`: latitude (float)  
- `lon`: longitude (float)  
- optional: `venue_id`, `category` (string)

---

---

## How to Run

> This project provides CLI modules under `src/osnclusters/cli`. Commands below assume your repo is executed from the project root.

### 1) Install

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Validate environment + datasets

```bash
python -m osnclusters.cli.validate --config configs/default.yaml --dataset brightkite
```

### 3) Run the full pipeline

```bash
python -m osnclusters.cli.run --config configs/default.yaml --dataset brightkite
```

### 4) Run Streamlit visualization

```bash
streamlit run visualization/app.py
```

---

## Outputs (Artifacts)

Typical artifacts produced per dataset/run (exact filenames may vary by config/run manager):

* `feat_df.parquet` — per-user feature table
* `X_users.npy` — user feature matrix `(N×F)`
* `Z.npy` — GraphSAGE embeddings `(N×d)`
* `comm_df.parquet` — community assignments `(user_id, community_id)`
* `comm_metrics.parquet` — per-community metrics (structural + spatial)
* `metrics_global.json` — global summary + random baseline comparison
* `run_config.json` — resolved config snapshot for reproducibility

---

## Reproducibility

### Libraries (minimum)

* `numpy`, `pandas`
* `torch`
* `scikit-learn`
* `python-igraph`, `leidenalg`
* `pyyaml`, `pyarrow`
* `streamlit`

> Pin exact versions in `requirements.txt` for consistent reproduction.

### Configuration

* Default config: `configs/default.yaml`
* Each run should save:

  * `run_config.json`
  * metrics + artifacts under a run directory managed by `core/run_manager.py`

### Reproduce (example)

```bash
pip install -r requirements.txt
python -m osnclusters.cli.validate --config configs/default.yaml --dataset brightkite
python -m osnclusters.cli.run      --config configs/default.yaml --dataset brightkite
streamlit run visualization/app.py
```

---

