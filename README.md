## Abstract
Community detection in social networks is often performed using only graph topology (friendship edges), which may miss real-world similarity in users’ spatio-temporal behavior. This project studies community discovery in **Location-Based Social Networks (LBSN: social graph + check-in trajectories)** by integrating **user mobility/temporal features** with **graph representation learning**. We first engineer per-user behavioral features from check-ins, then learn user embeddings using **GraphSAGE (Graph Sample-and-Aggregate: inductive GNN embeddings)** trained with neighbor sampling and link prediction. Communities are obtained by building an embedding-based **kNN similarity graph (k-Nearest Neighbors)** and applying **Leiden clustering**. We evaluate community quality from two angles: **structural cohesion** (e.g., modularity, intra-density) and **spatial cohesion** (distance-to-community-centroid), including a **random baseline** via label shuffling to verify non-random geographic coherence. An interactive Streamlit dashboard supports qualitative inspection through embedding, geographic, and graph-layout views.

---

## Method (6-Step Pipeline)

1. **Parse & Normalize**
   - Parse raw dataset files and enforce a unified schema for edges and check-ins.
   - Output standardized `edges_final` and `checkins_final`.

2. **User Feature Engineering**
   - Build per-user feature matrix `X_users` from check-ins, capturing:
     - activity intensity (counts, active days, unique venues)
     - spatial mobility (centroid, dispersion metrics)
     - temporal routine (hour-of-day / day-of-week distributions)
     - entropy-based regularity (time/venue entropy)

3. **Graph Construction**
   - Create the base user graph `G=(V,E)` from friendship edges.
   - Align `X_users` with node ordering to form node attributes.

4. **Graph Representation Learning (GraphSAGE)**
   - Train GraphSAGE with neighbor sampling and negative sampling under a link prediction objective.
   - Produce user embeddings `Z ∈ R^(N×d)`.

5. **Community Detection (kNN + Leiden)**
   - Build a similarity graph from `Z` using cosine-based kNN.
   - Run Leiden clustering on the kNN graph to assign `community_id` per user.

6. **Evaluation & Visualization**
   - Compute structural metrics (modularity, density, conductance approx.).
   - Compute spatial cohesion (distance-to-community-centroid) and compare to random baselines.
   - Visualize communities via Streamlit (embedding/geo/graph layout).

---

## Datasets & Schema

### Supported datasets (typical)
- **Brightkite (SNAP)**
- **Gowalla (SNAP)**
- **LBSN2Vec++ (Foursquare-based)**

### Normalized schema

**Edges table**
- Required columns:
  - `u`: user_id (string)
  - `v`: user_id (string)

**Check-ins table**
- Required columns:
  - `user_id`: user_id (string)
  - `ts`: timestamp (datetime)
  - `lat`: latitude (float)
  - `lon`: longitude (float)
- Optional columns:
  - `venue_id` (string)
  - `category` (string)

---

## How to Run

> If your project is notebook-based (Step 0 → Step 8), run notebooks in order, then start Streamlit.  
> If your project provides python entrypoints, use the commands below (adapt module names to your repo).

### 1) Install

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
````

### 2) Run pipeline (conceptual commands)

**(A) Feature engineering**

```bash
python -m src.features.build_features --config configs/default.yaml --dataset brightkite
```

**(B) Train GraphSAGE embeddings**

```bash
python -m src.models.train_graphsage --config configs/default.yaml --dataset brightkite
```

**(C) kNN + Leiden clustering**

```bash
python -m src.clustering.run_leiden --config configs/default.yaml --dataset brightkite
```

**(D) Evaluation**

```bash
python -m src.evaluation.run_metrics --config configs/default.yaml --dataset brightkite
```

### 3) Run Streamlit visualization

```bash
streamlit run visualization/app.py
```

---

## Outputs (Artifacts)

The pipeline typically produces the following artifacts per dataset/run:

* `X_users.npy` — user feature matrix (N×F)
* `feat_df.parquet` — feature table with `user_id` index and named columns
* `Z.npy` — GraphSAGE embeddings (N×d)
* `comm_df.parquet` — community assignments: `(user_id, community_id)`
* `comm_metrics.parquet` — per-community metrics (structural + spatial)
* `metrics_global.json` — global summary metrics + random-baseline comparison
* `run_config.json` — saved configuration for the run (for reproducibility)

---

## Reproducibility

### Libraries (suggested minimal list)

* `numpy`
* `pandas`
* `scikit-learn`
* `torch`
* `streamlit`
* `pyyaml` (if using YAML configs)
* `pyarrow` (parquet I/O)
* `python-igraph` + `leidenalg` (for Leiden; optional fallback may exist)

> The exact versions should be pinned in `requirements.txt`.

### Configuration

* Default config: `configs/default.yaml`
* Each run should save:

  * `run_config.json` (resolved config)
  * metrics + artifacts under `data/processed/_runs/<run_id>/...` (or equivalent)

### Commands to reproduce a run

```bash
# 1) Install
pip install -r requirements.txt

# 2) Run the pipeline (example dataset)
python -m src.features.build_features --config configs/default.yaml --dataset brightkite
python -m src.models.train_graphsage --config configs/default.yaml --dataset brightkite
python -m src.clustering.run_leiden --config configs/default.yaml --dataset brightkite
python -m src.evaluation.run_metrics --config configs/default.yaml --dataset brightkite

# 3) Visualize
streamlit run visualization/app.py
```

```
```
