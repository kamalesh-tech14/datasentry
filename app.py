"""DataSentry — Automated Dataset Health & Bias Auditor (Streamlit app).

Run:  streamlit run app.py
"""
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import audit_engine as ae

# Validated categorical slots 1-3 (blue, orange, aqua) — safe for colour-blind viewers.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRID = "rgba(128,128,128,0.25)"
SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "sample_data", "loan_approvals_synthetic.csv")
SAMPLE_NAME = "Sample: synthetic loan approvals"

st.set_page_config(page_title="DataSentry", page_icon="🛡️", layout="wide")


# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes | None, path: str | None) -> pd.DataFrame:
    if file_bytes is not None:
        import io
        return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    return pd.read_csv(path)


def style(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def default_favorable(values: list[str], counts: pd.Series) -> str:
    for pref in ("1", "1.0", "yes", "true", "approved", "hired", "admitted", ">50k", "good", "low risk"):
        for v in values:
            if v.lower() == pref:
                return v
    return str(counts.idxmin())  # otherwise the minority class


def clean_dataset(df: pd.DataFrame, health: dict) -> pd.DataFrame:
    """Conservative auto-clean: drop duplicates and useless columns, tidy text labels, cap extreme outliers.
    Missing values are left alone (imputing is a modelling decision)."""
    out = df.drop_duplicates().copy()
    out = out.drop(columns=[c for c in health["constant_cols"] + health["id_cols"] if c in out.columns])
    for c in out.columns:
        if not ae.is_numeric(out[c]) and not pd.api.types.is_bool_dtype(out[c]):
            out[c] = out[c].where(out[c].isna(), out[c].astype(str).str.strip())
    for c, info in health["outliers"].items():
        if c in out.columns:
            out[c] = out[c].clip(lower=info["low"], upper=info["high"])
    return out


def badge(sev: str) -> str:
    return {"High": "HIGH", "Medium": "MEDIUM", "Low": "LOW", "OK": "OK", "Inconclusive": "INCONCLUSIVE"}.get(sev, sev)


# ------------------------------------------------------------------ header
st.title("🛡️ DataSentry")
st.caption("Automated dataset health & bias auditor — upload a CSV, get a health score, hidden-bias findings and a tested fix.")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1 · Data")
    source = st.radio("Source", [SAMPLE_NAME, "Upload a CSV"], label_visibility="collapsed")
    if source == "Upload a CSV":
        up = st.file_uploader("CSV file", type=["csv"])
        if up is None:
            st.info("Upload a CSV to begin, or switch back to the sample.")
            st.stop()
        df = load_csv(up.getvalue(), None)
        dataset_name = up.name
    else:
        df = load_csv(None, SAMPLE_PATH)
        dataset_name = SAMPLE_NAME

    st.header("2 · What to audit")
    cols = list(df.columns)
    guess_t = ae.guess_target(df)
    target = st.selectbox("Outcome column (what is being decided?)", cols,
                          index=cols.index(guess_t) if guess_t in cols else len(cols) - 1)
    vc = df[target].value_counts(dropna=True)
    if len(vc) < 2 or len(vc) > 20:
        st.error(f"'{target}' has {len(vc)} distinct values. Pick a yes/no style outcome column (2–20 values).")
        st.stop()
    vals = [str(v) for v in vc.index]
    favorable = st.selectbox("Which value is the FAVOURABLE outcome?", sorted(vals),
                             index=sorted(vals).index(default_favorable(vals, vc)),
                             help="E.g. 'approved', 'hired', '1'. Bias is measured as differences in how often each group gets this.")
    sens_guess = [c for c in ae.detect_sensitive(df) if c != target]
    sensitive = st.multiselect("Sensitive attributes (auto-detected, editable)", [c for c in cols if c != target],
                               default=sens_guess,
                               help="Columns describing protected characteristics such as gender, race, age.")
    run_mit = st.checkbox("Run fix demo (trains small models)", value=True)

    st.header("3 · Report")
    api_key = st.text_input("Anthropic API key (optional)", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""),
                            help="Only used to rewrite the report in plain English. Without it you get the built-in report.")
    run = st.button("Run audit", type="primary", width="stretch")

sig = (dataset_name, target, favorable, tuple(sensitive), run_mit, len(df))
if run or st.session_state.get("sig") != sig:
    with st.spinner("Auditing…"):
        st.session_state["audit"] = ae.run_audit(df, target, favorable, sensitive, run_mitigation=run_mit)
        st.session_state["sig"] = sig
        st.session_state.pop("ai_report", None)
a: ae.Audit = st.session_state["audit"]
h, score = a.health, a.score

# ------------------------------------------------------------------ score header
c1, c2 = st.columns([1, 2])
with c1:
    st.metric("Dataset Health Score", f"{score['overall']:.0f} / 100 · Grade {score['grade']}")
    verdict = ("Healthy, low risk." if score["overall"] >= 80 else
               "Usable, but fix the flagged issues first." if score["overall"] >= 65 else
               "Risky — do not train or deploy on this data as-is.")
    (st.success if score["overall"] >= 80 else st.warning if score["overall"] >= 65 else st.error)(verdict)
    st.caption(f"{h['n_rows']:,} rows × {h['n_cols']} columns · {dataset_name}")
with c2:
    comp = score["components"]
    labels = [f"{k.capitalize()} ({score['weights'][k]}%)" for k in comp]
    fig = go.Figure(go.Bar(y=labels[::-1], x=list(comp.values())[::-1], orientation="h", marker_color=BLUE,
                           text=[f"{v:.0f}" for v in list(comp.values())[::-1]], textposition="outside",
                           hovertemplate="%{y}: %{x:.0f}/100<extra></extra>"))
    fig.update_xaxes(range=[0, 115], title="Sub-score (100 = healthy) · weight in brackets")
    st.plotly_chart(style(fig, 230), width="stretch")

tab_health, tab_bias, tab_proxy, tab_fix, tab_report = st.tabs(
    ["Data health", "Bias audit", "Hidden proxies", "Fix it", "Report"])

# ------------------------------------------------------------------ Data health
with tab_health:
    m = st.columns(5)
    m[0].metric("Missing cells", f"{h['overall_missing_pct']:.1f}%")
    m[1].metric("Duplicate rows", f"{h['duplicates']:,}")
    m[2].metric("Rows with extreme outliers", f"{h['outlier_row_pct']:.2f}%")
    m[3].metric("Useless columns", len(h["constant_cols"]) + len(h["id_cols"]))
    m[4].metric("Target minority share", f"{h['balance']['minority_share']*100:.0f}%" if h["balance"] else "n/a")

    st.subheader("Issues found")
    if h["issues"]:
        tbl = pd.DataFrame([dict(Severity=badge(i["severity"]), Type=i["category"], Issue=i["title"], **{"Suggested fix": i["fix"]})
                            for i in h["issues"]])
        st.dataframe(tbl, hide_index=True, width="stretch")
    else:
        st.success("No data-quality issues detected.")

    miss = h["missing_pct"][h["missing_pct"] > 0].sort_values()
    if len(miss):
        st.subheader("Missing values by column")
        fig = go.Figure(go.Bar(y=miss.index, x=miss.values, orientation="h", marker_color=BLUE,
                               text=[f"{v:.1f}%" for v in miss.values], textposition="outside",
                               hovertemplate="%{y}: %{x:.1f}% missing<extra></extra>"))
        fig.update_xaxes(title="% missing", range=[0, max(miss.max() * 1.25, 5)])
        st.plotly_chart(style(fig, 60 + 34 * len(miss)), width="stretch")

    st.subheader("Quick clean")
    cleaned = clean_dataset(df, h)
    st.caption(f"Drops duplicates and useless columns, trims text labels and caps extreme outliers "
               f"({len(df):,} → {len(cleaned):,} rows). Missing values are left for you to handle.")
    st.download_button("Download cleaned CSV", cleaned.to_csv(index=False).encode(), "cleaned_data.csv", "text/csv")

# ------------------------------------------------------------------ Bias
with tab_bias:
    if not a.bias:
        st.info("Select at least one sensitive attribute in the sidebar to run the bias audit.")
    else:
        st.markdown(f"Overall, **{a.bias['overall_positive_rate']*100:.1f}%** of rows get the favourable outcome "
                    f"(`{target} = {favorable}`). The **fairness ratio** (disparate impact) compares the worst-treated group's rate "
                    f"to the best-treated group's. Below **0.80** fails the common *four-fifths rule*.")
        for r in a.bias["results"]:
            st.markdown("---")
            title = r["attribute"] + ("  (intersection)" if r.get("intersectional") else "")
            st.subheader(title)
            if r["severity"] == "n/a":
                st.info(r["note"]); continue
            msg = (f"Fairness ratio **{r['di']:.2f}** — '{r['worst_group']}' gets the favourable outcome at "
                   f"{r['di']*100:.0f}% of the rate of '{r['best_group']}' (gap {r['spd']*100:.1f} pts"
                   f"{', statistically significant' if r['p_value'] < 0.01 else ''}).")
            if r["severity"] == "Inconclusive":
                msg += " **Not statistically significant, so this gap may be random noise.** Treat as inconclusive."
            {"High": st.error, "Medium": st.warning, "Inconclusive": st.info, "OK": st.success}[r["severity"]](
                f"**{badge(r['severity'])}** · " + msg)
            s = r["stats"]
            fig = go.Figure(go.Bar(x=s["group"], y=s["positive_rate"] * 100, marker_color=BLUE,
                                   text=[f"{v*100:.1f}%" for v in s["positive_rate"]], textposition="outside",
                                   customdata=np.c_[s["n"], s["share"] * 100, s["di_ratio"]],
                                   hovertemplate="%{x}<br>Favourable rate: %{y:.1f}%<br>n = %{customdata[0]:,} (%{customdata[1]:.0f}% of data)"
                                                 "<br>Ratio vs best: %{customdata[2]:.2f}<extra></extra>"))
            best = s[~s["too_small"]]["positive_rate"].max() * 100 if (~s["too_small"]).any() else s["positive_rate"].max() * 100
            fig.add_hline(y=best * ae.FOUR_FIFTHS, line_dash="dash", line_color="gray",
                          annotation_text="80% of best group", annotation_position="top right")
            fig.update_yaxes(title="Favourable outcome rate (%)", range=[0, max(best * 1.25, 5)])
            st.plotly_chart(style(fig, 300), width="stretch")
            show = s.rename(columns={"group": "Group", "n": "Rows", "share": "Share of data", "positive_rate": "Favourable rate",
                                     "di_ratio": "Ratio vs best"}).copy()
            flags = []
            for _, row in s.iterrows():
                f = []
                if row["too_small"]: f.append("too few rows to judge")
                if row["under_represented"]: f.append("under-represented")
                flags.append(", ".join(f))
            show["Flags"] = flags
            show["Share of data"] = (show["Share of data"] * 100).round(1).astype(str) + "%"
            show["Favourable rate"] = (show["Favourable rate"] * 100).round(1).astype(str) + "%"
            show["Ratio vs best"] = show["Ratio vs best"].round(2)
            st.dataframe(show[["Group", "Rows", "Share of data", "Favourable rate", "Ratio vs best", "Flags"]],
                         hide_index=True, width="stretch")
        st.caption("Under-representation is judged against an equal split; compare with the real population you serve. "
                   "Statistical gaps are warning signs, not proof of discrimination.")

# ------------------------------------------------------------------ Proxies
with tab_proxy:
    if not a.proxies:
        st.info("Select at least one sensitive attribute to scan for proxies.")
    else:
        st.markdown("Removing a sensitive column does **not** remove its influence if other columns reveal it. "
                    "This scan measures how strongly each column is associated with each sensitive attribute "
                    "(Cramér's V: 0 = unrelated, 1 = identical).")
        for sname, pr in a.proxies.items():
            st.subheader(f"Proxies for '{sname}'")
            if pr is None or not len(pr):
                st.info("No comparable columns."); continue
            top = pr.head(8).iloc[::-1]
            strong = pr[pr["level"] == "Strong"]
            if len(strong):
                st.error(f"**Strong proxy:** {', '.join(strong['feature'])} can reconstruct '{sname}'. A model may discriminate through it.")
            elif (pr["level"] == "Moderate").any():
                st.warning("Moderate proxies present: " + ", ".join(pr[pr['level'] == 'Moderate']['feature']))
            else:
                st.success("No column strongly reveals this attribute.")
            fig = go.Figure(go.Bar(y=top["feature"], x=top["association"], orientation="h", marker_color=BLUE,
                                   text=[f"{v:.2f}" for v in top["association"]], textposition="outside",
                                   hovertemplate="%{y}: V = %{x:.2f}<extra></extra>"))
            for x, lab in ((ae.PROXY_MODERATE, "moderate"), (ae.PROXY_STRONG, "strong")):
                fig.add_vline(x=x, line_dash="dash", line_color="gray", annotation_text=lab, annotation_position="top")
            fig.update_xaxes(range=[0, 1.1], title="Association strength (Cramér's V)")
            st.plotly_chart(style(fig, 60 + 34 * len(top)), width="stretch")

# ------------------------------------------------------------------ Fix it
with tab_fix:
    real = {k: v for k, v in a.mitigation.items() if v and not v.get("skipped")}
    if not run_mit:
        st.info("Tick “Run fix demo” in the sidebar.")
    elif not real:
        st.info("The fix demo needs a sensitive attribute and enough rows (≥300).")
    else:
        st.markdown("We train a simple model **without** the sensitive column, then again with **reweighing** "
                    "(rare group-outcome combinations get more weight so outcome stops depending on group). "
                    "Both are scored on held-out rows and approve the same share of people, so the comparison is fair.")
        for attr, mres in real.items():
            st.subheader(f"'{attr}'")
            verdict, text = ae.interpret_mitigation(mres)
            {"improved": st.success, "not_learned": st.info, "no_change": st.warning, "skipped": st.info}[verdict](text)
            b, af = mres["before"], mres["after"]
            fig = go.Figure(go.Bar(x=["Historical labels", "Baseline model", "After reweighing"],
                                   y=[mres["data_di"], b["di"], af["di"]], marker_color=[BLUE, ORANGE, AQUA],
                                   text=[f"{v:.2f}" for v in (mres["data_di"], b["di"], af["di"])], textposition="outside",
                                   hovertemplate="%{x}: fairness ratio %{y:.2f}<extra></extra>"))
            fig.add_hline(y=ae.FOUR_FIFTHS, line_dash="dash", line_color="gray", annotation_text="0.80 threshold",
                          annotation_position="top left")
            fig.update_yaxes(range=[0, 1.15], title="Fairness ratio (1.0 = equal)")
            st.plotly_chart(style(fig, 300), width="stretch")
            k = st.columns(3)
            k[0].metric("Accuracy (baseline)", f"{b['accuracy']*100:.1f}%")
            k[1].metric("Accuracy (reweighed)", f"{af['accuracy']*100:.1f}%", f"{(af['accuracy']-b['accuracy'])*100:+.1f} pts")
            k[2].metric("Outcome-rate gap", f"{af['spd']*100:.1f} pts", f"{(af['spd']-b['spd'])*100:+.1f} pts vs baseline", delta_color="inverse")
        with st.expander("How does reweighing work?"):
            st.markdown("For each (group, outcome) pair we compute `weight = P(group) × P(outcome) / P(group, outcome)`. "
                        "Combinations that are rarer than they would be if group and outcome were independent get weight > 1, "
                        "over-represented ones get weight < 1. The model then learns from a dataset where group tells you nothing about the outcome "
                        "(Kamiran & Calders, 2012).")

# ------------------------------------------------------------------ Report
with tab_report:
    base_md = ae.template_report(a, dataset_name)
    left, right = st.columns([3, 1])
    with right:
        if st.button("Write with AI", disabled=not api_key, help="Needs an Anthropic API key in the sidebar."):
            with st.spinner("Writing report…"):
                st.session_state["ai_report"] = ae.llm_report(base_md, api_key)
        if not api_key:
            st.caption("Add an API key in the sidebar to enable AI-written reports.")
    md, status = st.session_state.get("ai_report", (base_md, "Built-in report."))
    with left:
        st.caption(status)
        st.markdown(md)
    st.download_button("Download report (.md)", md.encode(), "datasentry_report.md", "text/markdown")
