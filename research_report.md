# Machine Learning-Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence

## Abstract
This project develops an exploratory buyer-segmentation workflow for real-estate market intelligence. The supplied client and property tables are cleaned, joined at client level, and analyzed using K-Means clustering. Cluster count is compared using inertia (Elbow diagnostic) and silhouette score; agglomerative clustering is used as a secondary comparison on a reproducible sample. The Streamlit application presents segment distribution, investment-purpose patterns, loan application behavior, geographic distribution, and descriptive segment statistics.

## 1. Business problem
The PRD identifies a need to understand differences between buyer types, acquisition motivations, geographic patterns, and financing behavior. A single generic marketing strategy can obscure meaningful differences in the customer base. The intended output is descriptive intelligence to support segment-level marketing and property recommendations.

## 2. Data
Two supplied CSVs are used:
- `clients.csv`: client ID, client type, date of birth, gender, country, region, acquisition purpose, satisfaction score, loan application indicator, and referral channel (plus names, which are excluded from modeling).
- `properties.csv`: listing ID, tower, transaction date, unit category, unit number, floor area, sale price, listing status, and client reference.

Initial file inspection found 2,000 client rows and 10,000 property rows. The client table has no missing values in the initial file and no repeated client IDs. The property table has 2,695 missing `client_ref` values, corresponding to the 2,695 records marked `Available`; the 7,305 non-null client references align with records marked `Sold`. Listing IDs are unique in the supplied file. Currency values are stored as strings with dollar signs and commas and are parsed to numeric values before analysis.

These are observations about the provided files, not claims about the wider market.

## 3. Data preparation
1. Validate the expected schema and fail clearly if required columns are absent.
2. Remove duplicate client and listing IDs, retaining the first record and recording counts.
3. Normalize whitespace and categorical label capitalization.
4. Parse date strings with mixed separators and remove currency formatting from sale prices.
5. Derive age at the latest transaction date found in the property data; implausible ages outside 18–110 are treated as missing and imputed for clustering.
6. Aggregate only `Sold` records linked to a known client to calculate sold-property count, total linked sold value, mean sold price, mean area, and latest purchase date.
7. Retain clients with no matched sold activity. Their sold count and total sold value are set to zero; average purchase measures remain missing.

## 4. Feature engineering and model
The clustering input includes age and satisfaction score (standardized) plus one-hot encoded client type, country, region, acquisition purpose, loan application, and referral channel. Names, client IDs, gender, and transaction-derived purchase aggregates are excluded from the clustering features. Excluding transaction aggregates prevents a segment from being defined directly by the outcomes later used to describe it.

K-Means is evaluated for `k = 2` through `min(8, n-1)`, using fixed random seeds and multiple initializations. The Elbow diagnostic reports inertia, while silhouette score assesses cohesion/separation. The best silhouette score among tested values is selected automatically. This is a reproducible heuristic, not proof that the selected cluster count is objectively correct. Agglomerative clustering with Ward linkage is fit on a deterministic sample and its silhouette score is reported as a secondary comparison.

## 5. Evaluation and interpretation
Run `python src/run_analysis.py` to create:
- `outputs/cluster_metrics.csv`: inertia and silhouette scores by k.
- `outputs/cluster_summary.csv`: cluster sizes and profile statistics.
- `outputs/buyer_segments.csv`: buyer-level records with cluster assignment and descriptive segment label.
- `outputs/data_quality_report.json`: data quality and model diagnostics.

The Streamlit app exposes the results interactively. Segment names are generated from observable proportions such as investment-purpose share, loan-use share, company share, and satisfaction. Names are descriptive and may be changed as business review evolves. Cluster numbers are arbitrary.

## 6. Business recommendations
- Tailor communication to observed acquisition purpose and referral channel rather than assuming every buyer has the same needs.
- Use country/region summaries to prioritize market research, while checking sample size and local context before reallocating budget.
- For loan-using segments, provide clear financing information and a human-led follow-up; do not infer creditworthiness from cluster membership.
- Use satisfaction patterns to identify where service feedback or post-sale support needs further investigation.
- Track conversion, cost per qualified lead, repeat transactions, and campaign response in a future version to test whether segmentation improves outcomes.

## 7. Limitations and responsible use
- No income, net worth, financing amount, risk tolerance, or verified investment size is present. The model cannot validate labels such as “high-income”, “high-net-worth”, or “luxury investor”.
- `acquisition_purpose` records Home/Investment and is not a detailed motivation measure.
- Country and region categories may be imbalanced; geographic associations are descriptive and not causal.
- One-hot encoding high-cardinality geography can affect distance-based clustering. Consider sensitivity analysis with grouped geography, Gower distance, k-prototypes, or embeddings if this becomes a production system.
- K-Means assumes Euclidean geometry and can be sensitive to scaling, encoding, and cluster shape.
- A silhouette score may be low even when clusters are useful for operational segmentation; validate profiles with domain experts and campaign experiments.
- Personal data should be access-controlled, retained only as necessary, and used in line with applicable privacy and anti-discrimination requirements.
- This project is decision support, not an automated mechanism for pricing, lending, or eligibility decisions.

## 8. Conclusion
The project provides a repeatable path from raw client/property files to interpretable buyer profiles and a filterable dashboard. Its value is a structured view of observable differences in acquisition purpose, financing indicators, geography, satisfaction, and linked property activity. Before operational deployment, segments should be reviewed with business stakeholders and validated against real campaign outcomes.

## Reproducibility
See `README.md` for installation and execution instructions. The random seeds and preprocessing choices are fixed in `src/run_analysis.py`.
