"""DataSentry audit engine.

Pure-Python (pandas / numpy / scipy / scikit-learn). No web framework in here, so you
can import it from a notebook or the Streamlit app.

Main entry point:  run_audit(df, target, favorable, sensitive_cols)

The four questions the engine answers
-------------------------------------
1. Is the data HEALTHY?      -> health_checks()   (missing, duplicates, outliers, ...)
2. Is it BIASED?             -> bias_audit()      (representation + outcome gaps per group)
3. Is bias HIDING?           -> proxy_scan()      (features that secretly encode a sensitive attribute)
4. Can we FIX it?            -> mitigation_demo() (train a model before/after "reweighing")
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
SENSITIVE_TOKENS = {
    "gender", "sex", "race", "ethnicity", "ethnic", "religion", "age", "marital",
    "nationality", "disability", "caste", "orientation", "citizenship", "native",
}
SIGNIFICANCE = 0.05          # gaps with p >= this are labelled "Inconclusive" (could be chance)
MIN_GROUP_SIZE = 30          # groups smaller than this are too small to judge
FOUR_FIFTHS = 0.80           # "80% rule" used in US employment law as a bias red flag
DI_WARN = 0.90
OUTLIER_IQR_MULT = 3.0       # only *extreme* outliers
PROXY_STRONG = 0.30          # Cramer's V thresholds
PROXY_MODERATE = 0.15
MAX_ROWS_FOR_MODEL = 30000


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t]


def is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def is_id_like(df: pd.DataFrame, col: str) -> bool:
    toks = _tokens(col)
    name_hit = "id" in toks or str(col).lower().endswith("id") or "uuid" in toks
    s = df[col]
    unique_all = s.nunique(dropna=True) >= 0.98 * len(df) and len(df) > 20
    return bool(name_hit and s.nunique() > 0.5 * len(df)) or bool(unique_all and not is_numeric(s))


def detect_sensitive(df: pd.DataFrame) -> list[str]:
    """Guess which columns describe protected attributes, from their names."""
    found = []
    for c in df.columns:
        if any(t in SENSITIVE_TOKENS for t in _tokens(c)):
            found.append(c)
    return found


def guess_target(df: pd.DataFrame) -> str | None:
    """Guess a binary outcome column (last-column / keyword heuristic)."""
    keywords = ("target", "label", "outcome", "approved", "approval", "hired", "income",
                "default", "churn", "survived", "class", "decision", "admit", "risk")
    cands = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    for c in cands:
        if any(k in str(c).lower() for k in keywords):
            return c
    return cands[-1] if cands else None


def as_groups(s: pd.Series, max_bins: int = 4) -> pd.Series:
    """Turn any column into string groups. Continuous numerics are cut into quantile bins."""
    if is_numeric(s) and s.nunique(dropna=True) > 8:
        try:
            g = pd.qcut(s, q=min(max_bins, 4), duplicates="drop")
            return g.astype(str).where(s.notna())
        except Exception:
            pass
    txt = s.astype(str).where(s.notna())
    if not is_numeric(s):
        # merge labels that differ only by case/whitespace ("Male", " male") under their most common spelling
        norm = txt.str.strip().str.lower()
        canon = txt.groupby(norm).agg(lambda x: x.str.strip().mode().iat[0])
        txt = norm.map(canon).where(s.notna())
    return txt


def to_binary(y: pd.Series, favorable) -> pd.Series:
    return (y.astype(str) == str(favorable)).astype(int)


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Association strength between two categorical columns: 0 = unrelated, 1 = identical."""
    tab = pd.crosstab(a, b)
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(tab, correction=False)[0]
    n = tab.values.sum()
    return float(np.sqrt(chi2 / (n * (min(tab.shape) - 1)))) if n else 0.0


# --------------------------------------------------------------------------------------
# 1. DATA HEALTH
# --------------------------------------------------------------------------------------
def health_checks(df: pd.DataFrame, target: str | None = None) -> dict:
    n_rows, n_cols = df.shape
    issues: list[dict] = []

    # Missing values
    miss_pct = (df.isna().mean() * 100).round(2)
    overall_missing = float(df.isna().sum().sum() / max(df.size, 1) * 100)
    for c, p in miss_pct[miss_pct > 0].sort_values(ascending=False).items():
        sev = "High" if p >= 20 else "Medium" if p >= 5 else "Low"
        issues.append(dict(category="Missing values", severity=sev, column=c,
                           title=f"'{c}' has {p:.1f}% missing values",
                           fix="Impute (median/mode), add a 'was missing' flag, or drop if unusable."))

    # Duplicates
    dup_n = int(df.duplicated().sum())
    dup_pct = dup_n / max(n_rows, 1) * 100
    if dup_n:
        issues.append(dict(category="Duplicates", severity="High" if dup_pct >= 5 else "Medium", column="(rows)",
                           title=f"{dup_n} exact duplicate rows ({dup_pct:.1f}%)",
                           fix="Drop duplicates before training/evaluation; they inflate accuracy and leak between splits."))

    # Constant + ID columns
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    id_cols = [c for c in df.columns if c not in constant_cols and is_id_like(df, c)]
    for c in constant_cols:
        issues.append(dict(category="Useless column", severity="Low", column=c,
                           title=f"'{c}' is constant (a single value)",
                           fix="Drop it; it carries no information."))
    for c in id_cols:
        issues.append(dict(category="Useless column", severity="Low", column=c,
                           title=f"'{c}' looks like an identifier",
                           fix="Exclude from modelling; IDs can cause leakage or memorisation."))

    # Outliers (extreme, IQR-based)
    outlier_cols: dict[str, dict] = {}
    outlier_row_mask = pd.Series(False, index=df.index)
    for c in df.columns:
        if c in id_cols or c in constant_cols or not is_numeric(df[c]) or df[c].nunique() < 10:
            continue
        s = df[c].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - OUTLIER_IQR_MULT * iqr, q3 + OUTLIER_IQR_MULT * iqr
        mask = (df[c] < lo) | (df[c] > hi)
        k = int(mask.sum())
        if k:
            outlier_cols[c] = dict(count=k, pct=k / n_rows * 100, low=float(lo), high=float(hi),
                                   max=float(s.max()), min=float(s.min()))
            outlier_row_mask |= mask.fillna(False)
            # show a sensible lower bound (a negative bound for an always-positive column is confusing)
            lo_show = max(lo, float(s.min())) if s.min() >= 0 else lo
            noun = "extreme outlier" if k == 1 else "extreme outliers"
            issues.append(dict(category="Outliers", severity="Medium" if k / n_rows > 0.005 else "Low", column=c,
                               title=f"'{c}' has {k} {noun} (typical range ~{lo_show:,.0f} to {hi:,.0f}, max seen {s.max():,.0f})",
                               fix="Check for entry errors; cap (winsorise), correct, or remove."))
    outlier_row_pct = float(outlier_row_mask.mean() * 100)

    # Numbers stored as text + inconsistent labels
    text_numeric_cols, messy_label_cols = [], []
    for c in df.columns:
        s = df[c]
        if is_numeric(s) or pd.api.types.is_bool_dtype(s) or c in id_cols:
            continue
        nn = s.dropna()
        if len(nn) == 0:
            continue
        parsed = pd.to_numeric(nn, errors="coerce").notna().mean()
        if parsed >= 0.9:
            text_numeric_cols.append(c)
            issues.append(dict(category="Wrong type", severity="Medium", column=c,
                               title=f"'{c}' is numeric but stored as text ({parsed*100:.0f}% parseable)",
                               fix="Convert with pd.to_numeric(errors='coerce') and inspect the failures."))
            continue
        raw_u = nn.astype(str).nunique()
        norm_u = nn.astype(str).str.strip().str.lower().nunique()
        if norm_u < raw_u:
            messy_label_cols.append(c)
            issues.append(dict(category="Inconsistent labels", severity="Medium", column=c,
                               title=f"'{c}' has labels differing only by case/whitespace ({raw_u} raw vs {norm_u} normalised)",
                               fix="Normalise with .str.strip().str.lower() before analysis."))
        if nn.nunique() > 50 and nn.nunique() > 0.5 * len(nn):
            issues.append(dict(category="High cardinality", severity="Low", column=c,
                               title=f"'{c}' has {nn.nunique()} distinct text values",
                               fix="Group rare values or use target/embedding encoding."))

    # Target balance
    balance = None
    if target is not None and target in df.columns:
        vc = df[target].value_counts(normalize=True, dropna=True)
        if len(vc) >= 2:
            minority = float(vc.min())
            balance = dict(shares={str(k): float(v) for k, v in vc.items()}, minority_share=minority,
                           minority_label=str(vc.idxmin()))
            if minority < 0.30:
                issues.append(dict(category="Class imbalance", severity="High" if minority < 0.10 else "Medium",
                                   column=target,
                                   title=f"Target '{target}' is imbalanced: '{vc.idxmin()}' is only {minority*100:.1f}% of rows",
                                   fix="Use stratified splits, class weights or resampling; judge with recall/F1, not accuracy."))
    if n_rows < 500:
        issues.append(dict(category="Small sample", severity="Medium", column="(rows)",
                           title=f"Only {n_rows} rows — results (especially group statistics) will be noisy",
                           fix="Collect more data or treat findings as indicative only."))

    # ---- sub-scores (0-100, higher = healthier) ----------------------------------------
    completeness = float(np.clip(100 - 3 * overall_missing, 0, 100))
    uniqueness = float(np.clip(100 - 5 * dup_pct, 0, 100))
    validity = float(np.clip(100 - 8 * outlier_row_pct - 4 * (len(constant_cols) + len(id_cols))
                             - 6 * len(text_numeric_cols) - 6 * len(messy_label_cols), 0, 100))
    bal_score = None
    if balance:
        bal_score = float(100 * min(1.0, (balance["minority_share"] / 0.5) ** 0.5))

    sev_rank = {"High": 0, "Medium": 1, "Low": 2}
    issues.sort(key=lambda d: sev_rank[d["severity"]])
    return dict(
        n_rows=n_rows, n_cols=n_cols, overall_missing_pct=overall_missing, missing_pct=miss_pct,
        duplicates=dup_n, duplicate_pct=dup_pct, constant_cols=constant_cols, id_cols=id_cols,
        outliers=outlier_cols, outlier_row_pct=outlier_row_pct, text_numeric_cols=text_numeric_cols,
        messy_label_cols=messy_label_cols, balance=balance, issues=issues,
        scores=dict(completeness=completeness, uniqueness=uniqueness, validity=validity, balance=bal_score),
    )


# --------------------------------------------------------------------------------------
# 2. BIAS AUDIT
# --------------------------------------------------------------------------------------
def group_stats(groups: pd.Series, y: pd.Series) -> pd.DataFrame:
    """Per-group size, share, favourable-outcome rate and disparate-impact ratio."""
    d = pd.DataFrame({"g": groups, "y": y}).dropna()
    rows = []
    for g, sub in d.groupby("g"):
        rows.append(dict(group=str(g), n=len(sub), share=len(sub) / len(d), positive_rate=float(sub["y"].mean())))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    valid = out[out["n"] >= MIN_GROUP_SIZE]
    ref_rate = valid["positive_rate"].max() if len(valid) else out["positive_rate"].max()
    out["di_ratio"] = out["positive_rate"] / ref_rate if ref_rate > 0 else np.nan
    out["gap_vs_best"] = ref_rate - out["positive_rate"]
    out["too_small"] = out["n"] < MIN_GROUP_SIZE
    k = len(out)
    out["under_represented"] = (out["share"] < 0.7 * (1 / k)) & (k <= 6)
    return out.sort_values("positive_rate", ascending=False).reset_index(drop=True)


def _bias_entry(name: str, groups: pd.Series, y: pd.Series) -> dict:
    stats = group_stats(groups, y)
    valid = stats[~stats["too_small"]] if len(stats) else stats
    if len(valid) < 2:
        return dict(attribute=name, stats=stats, di=np.nan, spd=np.nan, p_value=np.nan, severity="n/a",
                    note="Fewer than two groups with enough rows to compare.")
    di = float(valid["positive_rate"].min() / valid["positive_rate"].max()) if valid["positive_rate"].max() > 0 else np.nan
    spd = float(valid["positive_rate"].max() - valid["positive_rate"].min())
    d = pd.DataFrame({"g": groups, "y": y}).dropna()
    d = d[d["g"].isin(valid["group"])]
    tab = pd.crosstab(d["g"], d["y"])
    p = float(chi2_contingency(tab)[1]) if tab.shape[0] >= 2 and tab.shape[1] >= 2 else np.nan
    sev = "High" if di < FOUR_FIFTHS else "Medium" if di < DI_WARN else "OK"
    # A gap that could easily be random noise must not raise a red flag (multiple small groups => chance gaps).
    if sev != "OK" and not np.isnan(p) and p >= SIGNIFICANCE:
        sev = "Inconclusive"
    return dict(attribute=name, stats=stats, di=di, spd=spd, p_value=p, severity=sev,
                worst_group=str(valid.sort_values("positive_rate").iloc[0]["group"]),
                best_group=str(valid.sort_values("positive_rate").iloc[-1]["group"]),
                note="")


def bias_audit(df: pd.DataFrame, target: str, favorable, sensitive_cols: list[str]) -> dict:
    y = to_binary(df[target], favorable)
    results = []
    for c in sensitive_cols:
        results.append(_bias_entry(c, as_groups(df[c]), y))
    # intersectional (first two sensitive attributes)
    if len(sensitive_cols) >= 2:
        a, b = sensitive_cols[:2]
        ga, gb = as_groups(df[a]), as_groups(df[b])
        combo = (ga + " × " + gb).where(ga.notna() & gb.notna())
        e = _bias_entry(f"{a} × {b}", combo, y)
        e["intersectional"] = True
        results.append(e)
    return dict(overall_positive_rate=float(y.mean()), results=results)


# --------------------------------------------------------------------------------------
# 3. PROXY SCAN
# --------------------------------------------------------------------------------------
def proxy_scan(df: pd.DataFrame, sensitive_col: str, exclude: list[str]) -> pd.DataFrame:
    """Which other columns can 'reveal' the sensitive attribute? (Cramér's V, 0-1)"""
    d = df.sample(min(len(df), 20000), random_state=0) if len(df) > 20000 else df
    sens = as_groups(d[sensitive_col])
    rows = []
    for c in d.columns:
        if c == sensitive_col or c in exclude:
            continue
        feat = as_groups(d[c], max_bins=10) if is_numeric(d[c]) else d[c].astype(str).where(d[c].notna())
        if feat.nunique() > 60:
            continue
        m = sens.notna() & feat.notna()
        if m.sum() < 50:
            continue
        v = cramers_v(sens[m], feat[m])
        level = "Strong" if v >= PROXY_STRONG else "Moderate" if v >= PROXY_MODERATE else "Weak"
        rows.append(dict(feature=c, association=v, level=level))
    out = pd.DataFrame(rows)
    return out.sort_values("association", ascending=False).reset_index(drop=True) if len(out) else out


# --------------------------------------------------------------------------------------
# 4. MITIGATION DEMO (train a model before / after reweighing)
# --------------------------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each numeric column to its 1st-99th percentile, learned on training data only."""
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, 1, axis=0)
        self.hi_ = np.nanpercentile(X, 99, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def _reweighing_weights(groups: pd.Series, y: pd.Series) -> np.ndarray:
    """Kamiran & Calders (2012): weight = P(group) * P(y) / P(group, y).
    Up-weights rare (group, outcome) combos so group and outcome look statistically independent."""
    d = pd.DataFrame({"g": groups.values, "y": y.values})
    n = len(d)
    p_g = d["g"].value_counts(normalize=True)
    p_y = d["y"].value_counts(normalize=True)
    p_gy = d.groupby(["g", "y"]).size() / n
    w = d.apply(lambda r: p_g[r["g"]] * p_y[r["y"]] / p_gy[(r["g"], r["y"])], axis=1)
    return w.values


def _fit_model(X_tr, y_tr, num_cols, cat_cols, sample_weight=None):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("win", Winsorizer()), ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10))]), cat_cols),
    ])
    pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X_tr, y_tr, clf__sample_weight=sample_weight)
    return pipe


def _pred_at_base_rate(proba: np.ndarray, base_rate: float) -> np.ndarray:
    """Threshold chosen so the model approves the same share of people as the data does.
    Makes before/after comparisons fair and avoids 'predict nobody' models on imbalanced data."""
    thr = np.quantile(proba, 1 - base_rate)
    return (proba >= thr).astype(int)


def _fairness_of_predictions(pred, y_true, groups) -> dict:
    d = pd.DataFrame({"pred": pred, "y": np.asarray(y_true), "g": np.asarray(groups)})
    rates, tprs = {}, {}
    for g, sub in d.groupby("g"):
        if len(sub) < MIN_GROUP_SIZE:
            continue
        rates[str(g)] = float(sub["pred"].mean())
        pos = sub[sub["y"] == 1]
        tprs[str(g)] = float(pos["pred"].mean()) if len(pos) >= 10 else np.nan
    if len(rates) < 2 or max(rates.values()) == 0:
        return dict(di=np.nan, spd=np.nan, tpr_gap=np.nan, rates=rates)
    tp = [v for v in tprs.values() if not np.isnan(v)]
    return dict(di=min(rates.values()) / max(rates.values()), spd=max(rates.values()) - min(rates.values()),
                tpr_gap=(max(tp) - min(tp)) if len(tp) >= 2 else np.nan, rates=rates)


def mitigation_demo(df: pd.DataFrame, target: str, favorable, sensitive_col: str,
                    drop_cols: list[str], other_sensitive: list[str] | None = None) -> dict:
    """Train a plain model (sensitive attribute removed — 'fairness through unawareness')
    and a reweighed model, then compare accuracy and fairness on held-out data."""
    d = df.copy()
    if len(d) > MAX_ROWS_FOR_MODEL:
        d = d.sample(MAX_ROWS_FOR_MODEL, random_state=0)
    d = d.drop_duplicates()
    y = to_binary(d[target], favorable)
    groups = as_groups(d[sensitive_col])
    keep = groups.notna() & d[target].notna()
    d, y, groups = d[keep], y[keep], groups[keep]
    if len(d) < 300 or y.nunique() < 2:
        return dict(skipped=True, reason="Not enough rows / only one outcome class for a model demo.")

    # features: everything except target, ids/constants, and the sensitive attribute itself
    feat_cols = [c for c in d.columns if c not in set(drop_cols) | {target, sensitive_col}]
    X = d[feat_cols]
    num_cols = [c for c in feat_cols if is_numeric(X[c])]
    cat_cols = [c for c in feat_cols if c not in num_cols]
    X = X.copy()
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan).astype(str).replace("nan", np.nan)

    strat = y.astype(str) + "|" + groups.astype(str)
    if strat.value_counts().min() < 2:
        strat = y
    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(X, y, groups, test_size=0.3, random_state=7, stratify=strat)
    base_rate = float(y_tr.mean())

    # (a) baseline model
    m0 = _fit_model(X_tr, y_tr, num_cols, cat_cols)
    p0 = m0.predict_proba(X_te)[:, 1]
    pred0 = _pred_at_base_rate(p0, base_rate)

    # (b) reweighed model
    w = _reweighing_weights(g_tr, y_tr)
    m1 = _fit_model(X_tr, y_tr, num_cols, cat_cols, sample_weight=w)
    p1 = m1.predict_proba(X_te)[:, 1]
    pred1 = _pred_at_base_rate(p1, base_rate)

    before = _fairness_of_predictions(pred0, y_te, g_te)
    after = _fairness_of_predictions(pred1, y_te, g_te)
    before["accuracy"] = float(accuracy_score(y_te, pred0))
    after["accuracy"] = float(accuracy_score(y_te, pred1))

    # (c) what the *data itself* says (labels), for reference
    data_fair = _fairness_of_predictions(y_te.values, y_te, g_te)

    return dict(skipped=False, sensitive=sensitive_col, n_train=len(X_tr), n_test=len(X_te),
                before=before, after=after, data_di=data_fair["di"], features_used=feat_cols)


# --------------------------------------------------------------------------------------
# 5. HEALTH SCORE
# --------------------------------------------------------------------------------------
WEIGHTS = dict(completeness=15, uniqueness=10, validity=15, balance=10, fairness=50)


def _di_to_score(di: float) -> float:
    """DI 0.9+ -> 100, 0.8 -> ~80 (borderline), 0.4 or less -> 0."""
    if di is None or np.isnan(di):
        return np.nan
    if di >= 0.9:
        return 100.0
    return float(np.clip((di - 0.4) / 0.5 * 100, 0, 100)) if di < 0.9 else 100.0


def health_score(health: dict, bias: dict | None) -> dict:
    comp = dict(health["scores"])
    fair = None
    if bias:
        # only statistically meaningful gaps lower the fairness score; chance-level gaps ("Inconclusive") don't
        dis = [r["di"] if r["severity"] != "Inconclusive" else 1.0
               for r in bias["results"] if not r.get("intersectional") and not np.isnan(r["di"])]
        if dis:
            fair = _di_to_score(min(dis))
    comp["fairness"] = fair
    avail = {k: v for k, v in comp.items() if v is not None and not (isinstance(v, float) and np.isnan(v))}
    tot_w = sum(WEIGHTS[k] for k in avail)
    overall = sum(avail[k] * WEIGHTS[k] for k in avail) / tot_w if tot_w else 0
    grade = "A" if overall >= 90 else "B" if overall >= 80 else "C" if overall >= 65 else "D" if overall >= 50 else "F"
    return dict(overall=float(overall), grade=grade, components=avail, weights={k: WEIGHTS[k] for k in avail})


# --------------------------------------------------------------------------------------
# 6. ORCHESTRATION + REPORT
# --------------------------------------------------------------------------------------
@dataclass
class Audit:
    target: str | None
    favorable: object
    sensitive_cols: list
    health: dict
    bias: dict | None
    proxies: dict = field(default_factory=dict)
    mitigation: dict = field(default_factory=dict)   # {sensitive_col: mitigation_demo result}
    score: dict = field(default_factory=dict)
    recommendations: list = field(default_factory=list)


def run_audit(df: pd.DataFrame, target: str | None, favorable, sensitive_cols: list[str],
              run_mitigation: bool = True) -> Audit:
    health = health_checks(df, target)
    bias = bias_audit(df, target, favorable, sensitive_cols) if target and sensitive_cols else None

    skip = list(set(health["id_cols"]) | set(health["constant_cols"]))
    proxies = {}
    if target and sensitive_cols:
        for s in sensitive_cols:
            pr = proxy_scan(df, s, exclude=[target] + skip + [c for c in sensitive_cols if c != s])
            proxies[s] = pr

    mitigation = {}
    if run_mitigation and bias:
        # run the demo for EVERY single sensitive attribute (worst first) and report all results honestly
        singles = [r for r in bias["results"] if not r.get("intersectional") and not np.isnan(r["di"])]
        for r in sorted(singles, key=lambda r: r["di"])[:3]:
            mitigation[r["attribute"]] = mitigation_demo(df, target, favorable, r["attribute"], drop_cols=skip)

    a = Audit(target, favorable, sensitive_cols, health, bias, proxies, mitigation)
    a.score = health_score(health, bias)
    a.recommendations = build_recommendations(a)
    return a


def interpret_mitigation(m: dict) -> tuple[str, str]:
    """Plain-English reading of a mitigation_demo result. Returns (verdict_key, sentence)."""
    if not m or m.get("skipped"):
        return "skipped", (m or {}).get("reason", "Demo skipped.")
    s, b, af, data = m["sensitive"], m["before"], m["after"], m["data_di"]
    if np.isnan(b["di"]) or np.isnan(af["di"]):
        return "skipped", "Not enough data per group to measure fairness on predictions."
    dacc = (af["accuracy"] - b["accuracy"]) * 100
    if af["di"] >= b["di"] + 0.03:
        amplified = b["di"] < data - 0.05
        lead = (f"Even with '{s}' removed, the model became MORE biased than the data (ratio {b['di']:.2f} vs {data:.2f} in the labels) "
                f"because other columns act as proxies. " if amplified else "")
        return "improved", (f"{lead}Reweighing raised the fairness ratio from {b['di']:.2f} to {af['di']:.2f} "
                            f"for an accuracy change of {dacc:+.1f} points.")
    if b["di"] >= data + 0.05:
        return "not_learned", (f"The labels are biased on '{s}' (ratio {data:.2f}), but a model without '{s}' does not reproduce the gap "
                               f"(ratio {b['di']:.2f}) because no other column reveals it. Reweighing changes little here; "
                               f"the risk sits in the historical labels, so audit how they were assigned.")
    return "no_change", (f"Reweighing did not change fairness for '{s}' (ratio {b['di']:.2f} → {af['di']:.2f}). "
                         f"Consider other mitigations (threshold tuning, better features, more data).")


def build_recommendations(a: Audit) -> list[str]:
    recs = []
    h = a.health
    if a.bias:
        for r in a.bias["results"]:
            if r["severity"] == "High":
                recs.append(f"**Investigate bias on '{r['attribute']}'**: '{r['worst_group']}' receives the favourable outcome at "
                            f"{r['di']*100:.0f}% of the rate of '{r['best_group']}' (below the 80% rule). Check whether the historical "
                            f"labels reflect real merit or past discrimination before training on them.")
        for s, pr in a.proxies.items():
            strong = pr[pr["level"] == "Strong"] if len(pr) else pr
            if len(strong):
                recs.append(f"**Watch for proxy features for '{s}'**: {', '.join(strong['feature'].head(3))} strongly predict it. "
                            f"Removing '{s}' alone will not remove its influence.")
    for attr, m in a.mitigation.items():
        verdict, text = interpret_mitigation(m)
        if verdict == "improved":
            recs.append(f"**Apply reweighing for '{attr}'**: {text}")
        elif verdict == "not_learned":
            recs.append(f"**Audit the labels for '{attr}'**: {text}")
    if h["duplicates"]:
        recs.append(f"Remove the {h['duplicates']} duplicate rows before splitting into train/test.")
    for c, p in h["missing_pct"][h["missing_pct"] >= 5].items():
        recs.append(f"Handle missing values in '{c}' ({p:.1f}%); check whether missingness differs across groups.")
    if h["outliers"]:
        recs.append("Review extreme values in: " + ", ".join(f"'{c}'" for c in h["outliers"]) + " (likely entry errors).")
    if h["balance"] and h["balance"]["minority_share"] < 0.30:
        recs.append("Use stratified splitting and report recall/precision, since the target is imbalanced.")
    if a.bias:
        for r in a.bias["results"]:
            if r["stats"] is not None and len(r["stats"]) and r["stats"]["under_represented"].any():
                ur = ", ".join(r["stats"][r["stats"]["under_represented"]]["group"].astype(str))
                recs.append(f"Under-representation in '{r['attribute']}': {ur}. Compare against the real population "
                            f"you intend to serve and collect more data if they differ.")
                break
    return recs


def template_report(a: Audit, dataset_name: str = "dataset") -> str:
    """Rule-based markdown report. Works offline; can be rewritten by an LLM afterwards."""
    h, s = a.health, a.score
    L = []
    L.append(f"# DataSentry Audit — {dataset_name}\n")
    L.append(f"**Overall Health Score: {s['overall']:.0f}/100 (Grade {s['grade']})**  ")
    L.append(f"{h['n_rows']:,} rows × {h['n_cols']} columns · target: `{a.target}` · sensitive attributes: "
             f"{', '.join('`'+c+'`' for c in a.sensitive_cols) or 'none selected'}\n")
    L.append("## Score breakdown\n")
    for k, v in s["components"].items():
        L.append(f"- {k.capitalize()}: {v:.0f}/100 (weight {s['weights'][k]}%)")
    L.append("\n## Data health findings\n")
    if h["issues"]:
        for i in h["issues"][:15]:
            L.append(f"- [{i['severity']}] {i['title']}. *Fix:* {i['fix']}")
    else:
        L.append("- No data-quality issues detected.")
    if a.bias:
        L.append("\n## Bias findings\n")
        L.append(f"Overall favourable-outcome rate: {a.bias['overall_positive_rate']*100:.1f}%.\n")
        for r in a.bias["results"]:
            if np.isnan(r["di"]):
                L.append(f"- `{r['attribute']}`: {r['note']}")
                continue
            note = (" (statistically significant)" if r["p_value"] < 0.01 else
                    " (not statistically significant: may be chance)" if r["severity"] == "Inconclusive" else "")
            L.append(f"- `{r['attribute']}` — disparate impact **{r['di']:.2f}** ({r['severity']}); gap of "
                     f"{r['spd']*100:.1f} percentage points between '{r['best_group']}' and '{r['worst_group']}'{note}.")
        for sname, pr in a.proxies.items():
            if len(pr):
                top = pr.head(3)
                L.append(f"\n**Proxy check for `{sname}`:** " + "; ".join(
                    f"`{row.feature}` ({row.level.lower()}, V={row.association:.2f})" for row in top.itertuples()))
    real = {k: m for k, m in a.mitigation.items() if m and not m.get("skipped")}
    if real:
        L.append("\n## Fix demo: reweighing\n")
        L.append("Logistic-regression model trained with the sensitive attribute removed from the inputs, then retrained "
                 "with reweighing. Evaluated on held-out rows.\n")
        for attr, m in real.items():
            L.append(f"**`{attr}`** — label fairness ratio {m['data_di']:.2f}\n")
            L.append("| | Baseline model | After reweighing |\n|---|---|---|")
            L.append(f"| Fairness ratio (1.0 = equal) | {m['before']['di']:.2f} | {m['after']['di']:.2f} |")
            L.append(f"| Outcome-rate gap | {m['before']['spd']*100:.1f} pts | {m['after']['spd']*100:.1f} pts |")
            L.append(f"| Accuracy | {m['before']['accuracy']*100:.1f}% | {m['after']['accuracy']*100:.1f}% |\n")
            L.append(f"*{interpret_mitigation(m)[1]}*\n")
    L.append("\n## Recommended actions\n")
    for i, r in enumerate(a.recommendations, 1):
        L.append(f"{i}. {r}")
    L.append("\n---\n*Limitations: fairness metrics describe statistical gaps, not proof of discrimination. Groups are compared "
             "to each other, not to the true population. Always combine with domain judgement.*")
    return "\n".join(L)


def llm_report(template_md: str, api_key: str | None, model: str = "claude-sonnet-4-5") -> tuple[str, str]:
    """Ask an LLM to turn the technical audit into a clear executive narrative.
    Returns (markdown, status). Falls back to the template report on any failure."""
    if not api_key:
        return template_md, "No API key: showing the built-in report."
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model, max_tokens=1800,
            messages=[{"role": "user", "content":
                       "You are a data-ethics analyst. Rewrite the following automated dataset audit as a clear report for a "
                       "non-technical decision maker: a 3-sentence executive summary, then the key risks in plain English, "
                       "then prioritised actions. Do NOT invent numbers; use only figures present below. Keep the caveat "
                       "that statistical gaps are not proof of discrimination.\n\n" + template_md}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return (text or template_md), "AI-written report."
    except Exception as e:  # network, key, quota...
        return template_md, f"AI report unavailable ({type(e).__name__}); showing the built-in report."
