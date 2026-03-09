# Data Dictionary — AAA Northeast Member Sample

**Source:** `data/raw/member_sample.csv`  
**Raw grain:** individual member × roadside service call (21,344 rows, 113 columns)  
**Modeling grain:** household (3,500 rows after preprocessing)

---

## Column Exclusions (documented reasons)

| Column | Reason excluded |
|--------|----------------|
| `Race` | Protected attribute — using in a model is an illegal fairness violation |
| `Reason Joined` | >95% missing — not recoverable |
| `Cancel Date` | **Target leakage** — only populated after a member cancels |
| `Cancel Reason` | **Target leakage** — only populated after a member cancels |
| `Months from Join to Cancel` | **Target leakage** — only populated after cancellation |
| `SC Vehicle Manufacturer/Model` | Vehicle-level, cannot aggregate to household meaningfully |
| `Education` | Cannot reliably aggregate to household level |

---

## Product Flags (Classification Targets)

| Raw Column | Clean Name | Raw Adoption | After Filter |
|-----------|-----------|-------------|--------------|
| FSV CMSI Flag | FSV CMSI | 4.5% | varies |
| FSV Credit Card Flag | FSV Credit Card | 6.0% | varies |
| FSV Deposit Program Flag | FSV Deposit Program | 0.2% | below threshold |
| FSV Home Equity Flag | FSV Home Equity | 0.0% | below threshold |
| FSV ID Theft Flag | FSV ID Theft | 2.3% | varies |
| FSV Mortgage Flag | FSV Mortgage | 0.1% | below threshold |
| INS Client Flag | INS Client | 17.6% | ✅ modeled |
| TRV Globalware Flag | TRV Globalware | 9.2% | ✅ modeled |
| New Mover Flag | New Mover | 3.9% | varies |

Products below `min_product_penetration` (5% by default, set in `configs/settings.yaml`) are not modeled.

---

## Aggregation Rules (member → household)

| Column type | Rule | Rationale |
|-------------|------|-----------|
| Categorical (ZIP, Mosaic) | Mode | Most common value in household |
| Continuous numeric | Mean | Household average |
| ERS counts/costs | Sum + Mean | Both total volume and per-member rate carry signal |
| Product flags | Max | Household holds product if ANY member holds it |