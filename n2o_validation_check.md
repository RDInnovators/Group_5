# N2O Dataset Validation Check (arXiv 2407.05959)

## Objective

Check whether the N2O full-scale WWTP dataset provides a usable **timestamped pH channel** with genuine variation across the compliance window (6.5-8.5), suitable for open-loop validation of the Section 4.2.1 pH titration simulator.

## 1) Access and download method that worked

- Paper/data source identified from arXiv 2407.05959: **Mendeley Data DOI `10.17632/xmbxhscgpr.1`** (latest public dataset object currently resolves to v4).
- Successful programmatic access method:
  1. Query dataset metadata JSON from  
     `https://data.mendeley.com/public-api/datasets/xmbxhscgpr`
  2. Read `files[*].content_details.download_url`
  3. Download files directly from those URLs.
- Downloaded real files:
  - `aved_raw.csv` (245,765,293 bytes)
  - `aved_raw_semicolon.csv` (245,607,114 bytes)

## 2) Actual schema inspection (real files)

Inspected `aved_raw.csv` header and parsed timestamp column.

- Total columns: **49**
- Timestamp column: **`time`** (present and parseable)
- Timestamp coverage:
  - Date range: **2022-06-11 22:01:00+00:00 -> 2024-06-11 21:59:00+00:00**
  - Median step: **60 s** (p05=60 s, p95=120 s)

### pH channel check (explicit)

- No true pH measurement column exists in the dataset header.
- Literal/token checks found:
  - No column containing standalone `pH`
  - No column with `.PH.` token
  - Only similarly named process-phase fields (e.g., `...PROCESSPHASE...`, `...PHASECODE...`), which are **control state codes**, not pH chemistry.

Therefore:
- (a) **pH channel present?** **No**
- (b) **with timestamps?** Timestamps exist, but **not with a pH channel**
- (c) **pH varies/crosses 6.5-8.5?** Not assessable because no pH channel is present.

## 3) Open-loop simulator validation feasibility

Section 4.2.1 simulator predicts pH dynamics and requires observed pH as validation target.

Because this N2O dataset has **no pH measurement column**, open-loop MAE/MAPE against pH **cannot be computed** from this source.

No simulator run was executed on this dataset for pH validation, because required target variable is missing.

## Verdict

**NO USABLE REAL VALIDATION DATA EXISTS in any inspected source**

Given inspected sources so far:
- DS-3 has pH but no timestamps (confounded for time-step simulator validation),
- this N2O full-scale dataset has timestamps but no pH channel,
- DS-1/DS-5 timestamped pH are too flat for dosing-scale validation.

So there is currently no inspected source that provides both:
1. timestamped pH suitable for open-loop dynamic validation, and
2. genuine dosing-scale pH variation.
