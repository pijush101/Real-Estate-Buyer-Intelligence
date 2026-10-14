# Machine Learning-Based Buyer Segmentation and Investment Profiling

## Project scope
A reproducible real-estate buyer analytics workflow based on the supplied PRD and `clients.csv` / `properties.csv`. It cleans and joins buyer and property data, explores transaction patterns, compares K-Means solutions with silhouette scores, fits an interpretable clustering model, and serves interactive results in Streamlit.

## Important data limitation
The dataset has no income, net-worth, mortgage amount, or verified investment-capital field. Therefore, this project does **not** claim to identify high-income or high-net-worth buyers. Segment names in the dashboard are descriptive, data-derived labels based on observed behavior, not verified financial status.

## Run locally
Use Python 3.10+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
python src/run_analysis.py
streamlit run app.py
```

The app runs the pipeline on first load and caches its result. Outputs are written to `outputs/`:
- `buyer_level_features.csv`
- `buyer_segments.csv`
- `cluster_metrics.csv`
- `data_quality_report.json`

## Pipeline design
1. Validate schema and normalize categorical values.
2. Parse mixed date formats and currency strings.
3. Remove duplicate IDs and report missing/unmatched records.
4. Derive age at the latest transaction date in the supplied property data.
5. Aggregate sold-property activity at buyer level (count, total spend, mean sale price, mean area). Buyers with no matched sold transactions retain zero counts and missing purchase averages.
6. Cluster buyers using age, satisfaction, client type, country, region, acquisition purpose, loan indicator, and referral channel. Direct identifiers and names are excluded.
7. Compare K-Means cluster counts using inertia and silhouette score. Use the best silhouette score among tested `k` values when valid.
8. Fit an agglomerative clustering model on a deterministic sample as a secondary comparison.
9. Produce the dashboard and CSV outputs.

## Interpretation and responsible use
- Clustering is exploratory and does not prove causal relationships.
- Cluster numbers are arbitrary and can change after data/model updates.
- Geography and demographic fields may encode sensitive or proxy information. Use segment-level summaries, not automated individual-level eligibility or pricing decisions.
- `listing_status == "Sold"` is used for transaction-derived spend calculations. Listed prices on available inventory are not treated as completed sales.
- This dataset contains synthetic-looking identifiers and should be treated as provided project data; do not assume it represents the full real-estate market.

## Deliverables
- `app.py`: Streamlit analytics dashboard
- `src/run_analysis.py`: reproducible preprocessing and clustering pipeline
- `research_report.md`: report structure, method, known dataset findings, and limitations
- `data/`: source CSVs


## Run validation tests
```bash
python -m unittest discover -s tests -v
```
