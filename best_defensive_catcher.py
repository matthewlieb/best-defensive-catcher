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
    min_innings = st.slider(
        "Min Innings Played",
        min_value=50,
        max_value=1000,
        value=200,
        step=25,
        help="Minimum innings behind the plate to include a catcher.",
    )
    st.markdown("---")
    st.caption(
        "Data from FanGraphs & Baseball Savant via "
        "[pybaseball](https://github.com/jldbc/pybaseball)"
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _first_match(df, *candidates):
    """Return the first column name present in df, or None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ── Data loading (each step cached independently) ─────────────────────────────

@st.cache_data(ttl=86400, show_spinner="Fetching catcher fielding stats from FanGraphs…")
def _load_fg_fielding(season):
    from pybaseball import fielding_stats
    df = fielding_stats(season, season, qual=0)
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(ttl=86400, show_spinner="Fetching WAR from FanGraphs…")
def _load_fg_batting(season):
    from pybaseball import batting_stats
    df = batting_stats(season, season, qual=0)
    df.columns = df.columns.str.lower()
    return df


@st.cache_data(ttl=86400, show_spinner="Fetching Statcast framing data…")
def _load_statcast_framing(season):
    try:
        from pybaseball import statcast_catcher_framing
        df = statcast_catcher_framing(season, season)
        df.columns = df.columns.str.lower()
        if "year" in df.columns:
            df = df[df["year"] == season]
        return df
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner="Fetching pop-time data…")
def _load_statcast_poptime(season):
    try:
        from pybaseball import statcast_catcher_poptime
        df = statcast_catcher_poptime(season)
        df.columns = df.columns.str.lower()
        return df
    except Exception:
        return None


@st.cache_data(ttl=86400, show_spinner="Building catcher dataset…")
def build_dataset(season):
    # ── Primary: FanGraphs fielding (always works) ────────────────────────
    fld = _load_fg_fielding(season)

    # Filter to catchers
    pos_col = _first_match(fld, "pos", "position")
    if pos_col:
        fld = fld[fld[pos_col].astype(str).str.strip() == "C"].copy()

    # Resolve player name
    name_col = _first_match(fld, "name", "player name", "playername", "player_name")
    if name_col is None:
        raise ValueError(
            "No player name column found in fielding data. "
            "Available columns: " + str(fld.columns.tolist())
        )
    fld = fld.rename(columns={name_col: "player_name"})

    # Innings — keep row with most innings for traded players
    inn_col = _first_match(fld, "inn", "innings", "ip")
    if inn_col:
        fld = fld.sort_values(inn_col, ascending=False).drop_duplicates("player_name")
        fld = fld.rename(columns={inn_col: "innings"})
    else:
        fld = fld.drop_duplicates("player_name")

    # Rename CS% to avoid % in column names
    if "cs%" in fld.columns:
        fld = fld.rename(columns={"cs%": "cs_pct"})

    # ── WAR from FanGraphs batting ────────────────────────────────────────
    bat = _load_fg_batting(season)
    bat_name = _first_match(bat, "name", "player name", "playername")
    war_col = _first_match(bat, "war")
    if bat_name and war_col:
        war_sub = (
            bat[[bat_name, war_col]]
            .rename(columns={bat_name: "player_name", war_col: "WAR"})
            .sort_values("WAR", ascending=False)
            .drop_duplicates("player_name")
        )
        fld = fld.merge(war_sub, on="player_name", how="left")

    # ── Optional: Baseball Savant framing ─────────────────────────────────
    framing = _load_statcast_framing(season)
    if framing is not None and not framing.empty:
        fn = _first_match(framing, "first_name")
        ln = _first_match(framing, "last_name")
        nm = _first_match(framing, "name")
        if fn and ln:
            framing["player_name"] = framing[fn].str.strip() + " " + framing[ln].str.strip()
        elif nm:
            framing["player_name"] = framing[nm]

        if "player_name" in framing.columns:
            opp = _first_match(framing, "n_called_pitches", "called_pitches")
            runs = _first_match(framing, "runs_extra_strikes", "framing_runs")
            rename_map = {}
            if opp:
                rename_map[opp] = "framing_opportunities"
            if runs:
                rename_map[runs] = "framing_runs"
            framing = framing.rename(columns=rename_map)
            sv_cols = ["player_name"] + [
                c for c in ["framing_opportunities", "framing_runs"] if c in framing.columns
            ]
            if len(sv_cols) > 1:
                fld = fld.merge(
                    framing[sv_cols].drop_duplicates("player_name"),
                    on="player_name", how="left",
                )

    # ── Optional: Baseball Savant pop time ────────────────────────────────
    poptime = _load_statcast_poptime(season)
    if poptime is not None and not poptime.empty:
        fn = _first_match(poptime, "first_name")
        ln = _first_match(poptime, "last_name")
        nm = _first_match(poptime, "name")
        if fn and ln:
            poptime["player_name"] = poptime[fn].str.strip() + " " + poptime[ln].str.strip()
        elif nm:
            poptime["player_name"] = poptime[nm]

        if "player_name" in poptime.columns:
            pop_cols = ["player_name"] + [
                c for c in ["pop_2b_sba", "arm_2b_3b_sba"] if c in poptime.columns
            ]
            if len(pop_cols) > 1:
                fld = fld.merge(
                    poptime[pop_cols].drop_duplicates("player_name"),
                    on="player_name", how="left",
                )

    return fld


# ── Load & filter ─────────────────────────────────────────────────────────────

st.title("⚾ MLB Catcher Defensive Analysis")
st.markdown(
    f"Real MLB data · **{season}** season · "
    "Source: FanGraphs & Baseball Savant"
)

try:
    raw_df = build_dataset(season)
except Exception as exc:
    st.error(f"Data loading failed: {exc}")
    st.stop()

if "innings" in raw_df.columns:
    df = raw_df[raw_df["innings"] >= min_innings].copy()
else:
    df = raw_df.copy()

if df.empty:
    st.warning("No catchers meet the current filter. Try lowering the minimum innings threshold.")
    st.stop()

# ── Feature catalog (in priority order) ──────────────────────────────────────
FEATURES = {
    # FanGraphs (primary — always available)
    "cs_pct":   "CS%",
    "drs":      "DRS",
    "arm":      "ARM",
    "innings":  "Innings",
    "pb":       "Passed Balls",
    # FanGraphs Statcast page (available some seasons)
    "framing":  "Framing (FG)",
    # Baseball Savant (optional enrichment)
    "framing_runs":        "Framing Runs (SV)",
    "framing_opportunities": "Framing Opps",
    "pop_2b_sba":          "Pop Time 2B (s)",
    "arm_2b_3b_sba":       "Arm Strength (mph)",
}

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

# ── Train models ──────────────────────────────────────────────────────────────
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


# ── Render helper ─────────────────────────────────────────────────────────────

def render_model_tab(mdf, pred_col, r2_val, rmse_val, imp_df, imp_col, title):
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{r2_val:.3f}")
    c2.metric("RMSE", f"{rmse_val:.3f}")
    c3.metric("Sample Size", len(mdf))

    fig_imp = px.bar(
        imp_df, x="Feature", y=imp_col,
        color=imp_col,
        color_continuous_scale="RdBu" if imp_col == "Coefficient" else "Viridis",
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
        x0=war_min, y0=war_min, x1=war_max, y1=war_max,
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

    if "DRS" in disp_df.columns:
        fig_drs = px.bar(
            disp_df.sort_values("DRS", ascending=False).head(20),
            x="Catcher", y="DRS",
            color="DRS", color_continuous_scale="Greens",
            title=f"Top Catchers by Defensive Runs Saved — {season}",
        )
        fig_drs.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
        st.plotly_chart(fig_drs, use_container_width=True)

    if "Framing Runs (SV)" in disp_df.columns or "Framing (FG)" in disp_df.columns:
        frm_col = "Framing Runs (SV)" if "Framing Runs (SV)" in disp_df.columns else "Framing (FG)"
        fig_frm = px.bar(
            disp_df.sort_values(frm_col, ascending=False).head(20),
            x="Catcher", y=frm_col,
            color=frm_col, color_continuous_scale="Purples",
            title=f"Top Catchers by Framing — {season}",
        )
        fig_frm.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
        st.plotly_chart(fig_frm, use_container_width=True)

with tab2:
    st.subheader("Linear Regression")
    if modeled:
        render_model_tab(model_df, "LR Predicted WAR", lr_r2, lr_rmse, lr_imp, "Coefficient", "Linear Regression")
    else:
        st.info("Modeling requires WAR data and at least 5 catchers meeting the innings threshold.")

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
        st.info("Modeling requires WAR data and at least 5 catchers meeting the innings threshold.")
