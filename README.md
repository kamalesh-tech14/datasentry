# DataSentry — Automated Dataset Health & Bias Auditor

Upload any CSV. DataSentry tells you (1) how healthy the data is, (2) whether it treats groups unfairly,
(3) which columns secretly leak sensitive attributes, and (4) whether a fix works, all in one dashboard,
with a downloadable report and a cleaned CSV.

## Run it (5 minutes, from zero)

1. Open a terminal in this folder (`datasentry/`).
2. `pip install -r requirements.txt`
3. `streamlit run app.py`
4. Your browser opens at http://localhost:8501. The synthetic loan dataset loads automatically.

To audit your own data: sidebar → **Upload a CSV**, then pick the outcome column (what is being decided),
the favourable value (e.g. `approved`, `1`), and check the auto-detected sensitive columns.

Optional AI-written report: paste an Anthropic API key in the sidebar (or `export ANTHROPIC_API_KEY=...`).
Without a key everything still works; you get the built-in template report.

## Files

| File | What it is |
|---|---|
| `app.py` | The dashboard (Streamlit). |
| `audit_engine.py` | All the analysis logic. Importable from a notebook: `from audit_engine import run_audit`. |
| `make_sample_data.py` | Creates the synthetic loan dataset with *planted* problems, so you know what the tool should find. |
| `sample_data/` | The generated CSV. |

## How each part works (read this before you present)

**Health score (0–100).** Weighted average of five sub-scores: completeness 15%, uniqueness 10%, validity 15%,
class balance 10%, fairness 50%. Fairness is weighted highest because bias is the point of the tool.
Grades: A ≥ 90, B ≥ 80, C ≥ 65, D ≥ 50, F below.

**Data health checks.** Missing values, exact duplicate rows, constant and ID-like columns, extreme outliers
(beyond 3 × IQR from the quartiles), numbers stored as text, inconsistent labels ("Male" vs " male"),
high-cardinality text columns, target class imbalance, tiny sample size.

**Bias audit.** For each sensitive attribute, we compute the favourable-outcome rate per group.
- **Fairness ratio (disparate impact)** = lowest group rate ÷ highest group rate. 1.0 = perfectly equal.
  Below **0.80** fails the "four-fifths rule" (a standard red flag from US employment law). 0.80–0.90 is a warning.
- **Gap** = highest rate − lowest rate, in percentage points.
- A **chi-square test** says whether the gap is statistically significant or could be chance. A gap that fails the test (p ≥ 0.05) is labelled **Inconclusive**, not High, and does not lower the score, so random noise doesn't trigger false alarms.
- Labels that differ only by case or spacing ("Male", " male") are merged before groups are compared.
- **Intersectional check:** gender × race combined, because bias often hides at intersections.
- Groups with fewer than 30 rows are flagged "too few rows to judge" and excluded from the ratio.

**Proxy scan.** Deleting the `race` column doesn't remove racial bias if `zip_code` reveals race.
We measure how strongly every other column is associated with each sensitive attribute using
**Cramér's V** (0 = unrelated, 1 = identical). ≥ 0.30 = strong proxy.

**Fix demo (reweighing).** We train a logistic-regression model *without* the sensitive column, then again with
**reweighing** (Kamiran & Calders, 2012): each (group, outcome) combination gets weight
`P(group) × P(outcome) / P(group, outcome)`, so the model learns from data where group tells you nothing about outcome.
Both models are tested on held-out rows and approve the same share of people, so the comparison is fair.

**What the demo shows on the sample data (real output):**
- `race`: labels have ratio 0.75, but the model **amplifies** it to 0.53 through the `zip_code_group` proxy even though race was removed.
  Reweighing lifts it to **0.92** for a −0.1 point accuracy change.
- `gender`: labels are strongly biased (0.51) but no other column reveals gender, so the model without it is already at 0.85 and
  reweighing does little. DataSentry says so plainly: "the risk is in the historical labels; audit how they were assigned."
  Being honest about when a fix does *not* help is a strength; say it in your pitch.

## Suggested 3-minute pitch

1. **Problem (30s).** Models inherit the flaws of their data, and teams find out after deployment. Nobody audits the data first.
2. **Demo (90s).** Load the sample → score 55/100, Grade D → Bias tab: women approved at half the rate → Proxies tab:
   zip code reveals race → Fix tab: baseline 0.53 → 0.92 after reweighing → download the report.
3. **Then run it on the datathon's own dataset** (30s): this is where you win. Show a real finding from *their* data.
4. **Close (30s).** Works on any CSV, no code needed, explains itself in plain English; roadmap: more fixes
   (threshold tuning, resampling), drift monitoring, PDF export.

## Likely judge questions

- **"Is a fairness ratio of 0.8 proof of discrimination?"** No. It is a statistical warning sign. Differences can have legitimate causes;
  the tool flags where a human should look.
- **"Why remove the sensitive column and still measure bias?"** Because in practice it is often legally removed but still leaks through proxies. That is what the proxy scan and the 0.53 result demonstrate.
- **"Why reweighing?"** It is a well-known pre-processing method that needs no model changes, and it is transparent and quick to explain.
- **"Does fixing bias cost accuracy?"** Usually a little. Here −0.1 points. We show both numbers side by side.
- **"How is the score calculated?"** Weighted average of five sub-scores (see above); the weights are a design choice and are shown in the UI.
- **"Limits?"** Binary outcomes only; correlation-based proxy detection; one mitigation method; equal-split benchmark for under-representation.

## Known limitations (be upfront)

- Outcome must be a 2–20 value column; the favourable value is user-selected.
- The sample data is synthetic (planted bias), used so results can be verified. Real datasets like Adult Income or COMPAS can be uploaded as CSV.
- Group sizes below 30 are ignored; very small datasets give noisy results.
