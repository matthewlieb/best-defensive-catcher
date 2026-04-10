import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

st.set_page_config(
    page_title="MLB Catcher Defensive Analysis",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚾ Filters")
    season = st.selectbox(
        "Season",
        options=list(range(2024, 2019, -1)),
        index=0,
        help="Select the MLB season to analyze.",
    )
    min_opportunities = st.slider(
        "Min Framing Opportunities",
        min_value=100,
        max_value=1500,
        value=500,
        step=50,
        help="Minimum called pitches (framing opportunities) required to include a catcher.",
    )
    st.markdown("---")
    st.caption(
        "Data from Baseball Savant & FanGraphs via "
        "[pybaseball](https://github.com/jldbc/pybaseball)"
    )


# ── Data loading helpers ──────────────────────────────────────────────────────

def _first_match(df, *candidates):
    """Return the first column name that exists in df, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


@st.cache_data(ttl=86400, show_spinner="Fetching framing data from Baseball Savant…")
def _load_framing(season):
    from pybaseball import statcast_catcher_framing
    df = statcast_catcher_framing(min_year=season, max_year=season)
    df.columns = df.columns.str.lower()
    if "year" in df.columns:
        df = df[df["year"] == season]
    return df


@st.cache_data(ttl=86400, show_spinner="Fetching pop-time data from Baseball Savant…")
def _load_poptime(season):
    try:
        from pybaseball import statcast_catcher_poptime
        df = statcast_catcher_poptime(season)
        df.columns = df.columns.str.lower()
        return df
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner="Fetching blocking data from Baseball Savant…")
def _load_blocking(season):
    try:
        from pybaseball import statcast_catcher_fielding_run_value
        df = statcast_catcher_fielding_run_value(min_year=season, max_year=season, type="blocking")
        df.columns = df.columns.str.lower()
        if "year" in df.columns:
            df = df[df["year"] == season]
        return df
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner="Fetching WAR data from FanGraphs…")
def _load_war(season):
    from pybaseball import batting_stats
    df = batting_stats(season, season, qual=0)
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(ttl=86400, show_spinner="Building catcher dataset…")
def build_dataset(season):
    framing = _load_framing(season)
    poptime = _load_poptime(season)
    blocking = _load_blocking(season)
    war_raw = _load_war(season)

    # Build player_name from framing data
    fn = _first_match(framing, "first_name")
    ln = _first_match(framing, "last_name")
    if fn and ln:
        framing["player_name"] = framing[fn].str.strip() + " " + framing[ln].str.strip()
    elif _first_match(framing, "name"):
        framing["player_name"] = framing[_first_match(framing, "name")]
    else:
        raise ValueError(
            "Cannot identify player name in framing data. "
            "Columns found: " + str(framing.columns.tolist())
        )

    # Rename framing metric columns to stable names
    rename_map = {}
    opp_col = _first_match(framing, "n_called_pitches", "called_pitches", "n_frames")
    runs_col = _first_match(framing, "runs_extra_strikes", "framing_runs")
    if opp_col:
        rename_map[opp_col] = "framing_opportunities"
    if runs_col:
        rename_map[runs_col] = "framing_runs"
    framing = framing.rename(columns=rename_map)

    keep = [c for c in ["player_id", "player_name", "framing_opportunities", "framing_runs"]
            if c in framing.columns]
    df = framing[keep].copy()

    # Merge pop-time / arm data
    if (poptime is not None
            and "player_id" in df.columns
            and "player_id" in poptime.columns):
        arm_cols = ["player_id"] + [
            c for c in ["pop_2b_sba", "arm_2b_3b_sba", "exchange_2b_3b_sba", "maxeff_arm_2b_3b_sba"]
            if c in poptime.columns
        ]
        df = df.merge(poptime[arm_cols], on="player_id", how="left")

    # Merge blocking data
    if (blocking is not None
            and "player_id" in df.columns
            and "player_id" in blocking.columns):
        blk_col = _first_match(blocking, "runs_bpv", "blocking_runs", "blk_runs")
        if blk_col:
            blocking = blocking.rename(columns={blk_col: "blocking_runs"})
            df = df.merge(blocking[["player_id", "blocking_runs"]], on="player_id", how="left")

    # Merge WAR from FanGraphs
    wname = _first_match(war_raw, "name", "player name")
    wcol = _first_match(war_raw, "war")
    if wname and wcol:
        war_sub = (
            war_raw[[wname, wcol]]
            .copy()
            .rename(columns={wname: "player_name", wcol: "WAR"})
        )
        # Traded players appear multiple times — keep the highest-WAR row
        war_sub = war_sub.sort_values("WAR", ascending=False).drop_duplicates("player_name")
        df = df.merge(war_sub, on="player_name", how="left")

    return df


# ── Load and filter ───────────────────────────────────────────────────────────

st.title("⚾ MLB Catcher Defensive Analysis")
st.markdown(
    f"Real MLB Statcast data · **{season}** season · "
    "Source: Baseball Savant & FanGraphs"
)

try:
    raw_df = build_dataset(season)
except Exception as exc:
    st.error(f"Data loading failed: {exc}")
    st.info("Ensure pybaseball is installed: `pip install pybaseball`")
    st.stop()

if "framing_opportunities" in raw_df.columns:
    df = raw_df[raw_df["framing_opportunities"] >= min_opportunities].copy()
else:
    df = raw_df.copy()

if df.empty:
    st.warning("No catchers meet the current filter. Try lowering the minimum framing opportunities.")
    st.stop()

# ── Feature catalog ───────────────────────────────────────────────────────────
FEATURES = {
    "framing_runs":          "Framing Runs",
    "framing_opportunities": "Framing Opps",
    "pop_2b_sba":            "Pop Time 2B (s)",
    "arm_2b_3b_sba":         "Arm Strength (mph)",
    "blocking_runs":         "Blocking Runs",
}

# Keep only features present with enough non-null data
feature_cols = [
    c for c in FEATURES
    if c in df.columns and df[c].notna().mean() > 0.4
]
feature_labels = [FEATURES[c] for c in feature_cols]
has_war = "WAR" in df.columns and df["WAR"].notna().sum() >= 5

# ── Display DataFrame ─────────────────────────────────────────────────────────
disp_cols = ["player_name"] + feature_cols + (["WAR"] if has_war else [])
disp_df = df[disp_cols].dropna(subset=feature_cols[:1]).copy()
disp_df = disp_df.rename(columns={"player_name": "Catcher", **FEATURES})
if has_war:
    disp_df = disp_df.sort_values("WAR", ascending=False).reset_index(drop=True)

# ── Model training ────────────────────────────────────────────────────────────
modeled = False
model_df = pd.DataFrame()

if has_war and len(disp_df) >= 5:
    model_df = disp_df.dropna(subset=feature_labels + ["WAR"]).copy()
    if len(model_df) >= 5:
        X_raw = model_df[feature_labels].values
        y = model_df["WAR"].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X_raw)

        lr = LinearRegression().fit(X, y)
        lr_pred = lr.predict(X)
        lr_r2 = r2_score(y, lr_pred)
        lr_rmse = np.sqrt(mean_squared_error(y, lr_pred))

        rf = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42).fit(X, y)
        rf_pred = rf.predict(X)
        rf_r2 = r2_score(y, rf_pred)
        rf_rmse = np.sqrt(mean_squared_error(y, rf_pred))

        model_df = model_df.copy()
        model_df["LR Predicted WAR"] = lr_pred
        model_df["RF Predicted WAR"] = rf_pred

        lr_imp = (
            pd.DataFrame({"Feature": feature_labels, "Coefficient": lr.coef_})
            .sort_values("Coefficient", ascending=False)
        )
        rf_imp = (
            pd.DataFrame({"Feature": feature_labels, "Importance": rf.feature_importances_})
            .sort_values("Importance", ascending=False)
        )

        modeled = True


# ── Shared render helper ──────────────────────────────────────────────────────

def render_model_tab(mdf, pred_col, r2_val, rmse_val, imp_df, imp_col, title):
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{r2_val:.3f}")
    c2.metric("RMSE", f"{rmse_val:.3f}")
    c3.metric("Sample Size", len(mdf))

    color_scale = "RdBu" if imp_col == "Coefficient" else "Viridis"
    fig_imp = px.bar(
        imp_df, x="Feature", y=imp_col,
        color=imp_col,
        color_continuous_scale=color_scale,
        title=f"{title} — Feature Importance",
    )
    fig_imp.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    war_min, war_max = mdf["WAR"].min(), mdf["WAR"].max()
    fig_sc = px.scatter(
        mdf, x="WAR", y=pred_col,
        hover_name="Catcher",
        title=f"Actual vs Predicted WAR — {title}",
        labels={"WAR": "Actual WAR", pred_col: "Predicted WAR"},
        color=pred_col,
        color_continuous_scale="Blues",
    )
    fig_sc.add_shape(
        type="line",
        x0=war_min, y0=war_min,
        x1=war_max, y1=war_max,
        line=dict(color="red", dash="dash", width=2),
    )
    fig_sc.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_sc, use_container_width=True)

    best = mdf.loc[mdf[pred_col].idxmax()]
    st.success(
        f"🏆 **{best['Catcher']}** leads with a predicted WAR of **{best[pred_col]:.2f}**"
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Data & Rankings", "📈 Linear Regression", "🌲 Random Forest"])

with tab1:
    st.subheader(f"{season} Season — {len(disp_df)} catchers")

    float_fmt = {
        col: st.column_config.NumberColumn(format="%.2f")
        for col in disp_df.select_dtypes("float").columns
    }
    st.dataframe(disp_df, use_container_width=True, height=420, column_config=float_fmt)

    if has_war:
        fig_war = px.bar(
            disp_df.head(20), x="Catcher", y="WAR",
            color="WAR", color_continuous_scale="Blues",
            title=f"Top Catchers by WAR — {season}",
        )
        fig_war.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
        st.plotly_chart(fig_war, use_container_width=True)

    if "Framing Runs" in disp_df.columns:
        fig_fr = px.bar(
            disp_df.sort_values("Framing Runs", ascending=False).head(20),
            x="Catcher", y="Framing Runs",
            color="Framing Runs", color_continuous_scale="Greens",
            title=f"Top Catchers by Framing Runs — {season}",
        )
        fig_fr.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
        st.plotly_chart(fig_fr, use_container_width=True)

with tab2:
    st.subheader("Linear Regression")
    if modeled:
        render_model_tab(model_df, "LR Predicted WAR", lr_r2, lr_rmse, lr_imp, "Coefficient", "Linear Regression")
    else:
        st.info("Modeling requires WAR data and at least 5 catchers meeting the filter threshold.")

with tab3:
    st.subheader("Random Forest")
    if modeled:
        render_model_tab(model_df, "RF Predicted WAR", rf_r2, rf_rmse, rf_imp, "Importance", "Random Forest")

        st.markdown("---")
        st.subheader("Model Comparison")
        cmp = pd.DataFrame({
            "Model": ["Linear Regression", "Random Forest"],
            "R²": [lr_r2, rf_r2],
            "RMSE": [lr_rmse, rf_rmse],
        })
        st.dataframe(
            cmp.style.format({"R²": "{:.4f}", "RMSE": "{:.4f}"}),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Modeling requires WAR data and at least 5 catchers meeting the filter threshold.")
