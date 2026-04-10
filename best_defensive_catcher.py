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
        "Data from MLB Stats API & Baseball Savant via "
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

def _parse_innings(inn_str):
    """Convert baseball innings string '450.1' → decimal (450.333…)."""
    if inn_str is None:
        return 0.0
    s = str(inn_str).strip()
    if "." in s:
        whole, frac = s.split(".", 1)
        return int(whole or 0) + int(frac or 0) / 3.0
    try:
        return float(s)
    except ValueError:
        return 0.0


@st.cache_data(ttl=86400, show_spinner="Fetching catcher fielding stats from MLB Stats API…")
def _load_fg_fielding(season):
    import requests
    url = (
        "https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&group=fielding&season={season}"
        "&gameType=R&playerPool=ALL&position=C&limit=500"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for stat_group in data.get("stats", []):
        for split in stat_group.get("splits", []):
            player = split.get("player", {})
            stat = split.get("stat", {})
            sb = int(stat.get("stolenBases") or 0)
            cs = int(stat.get("caughtStealing") or 0)
            rows.append({
                "name": player.get("fullName", ""),
                "pos": "C",
                "inn": _parse_innings(stat.get("innings")),
                "pb": int(stat.get("passedBall") or 0),
                "cs_attempts": sb + cs,
                "cs": cs,
            })

    df = pd.DataFrame(rows)
    # Keep totals row per player (MLB API may split by team for traded players)
    df = df.sort_values("inn", ascending=False).drop_duplicates("name")
    df["cs_pct"] = df.apply(
        lambda r: r["cs"] / r["cs_attempts"] if r["cs_attempts"] > 0 else np.nan, axis=1
    )
    return df


@st.cache_data(ttl=86400, show_spinner="Fetching WAR from Baseball Reference…")
def _load_fg_batting(season):
    try:
        from pybaseball import batting_stats_bref
        df = batting_stats_bref(season)
        df.columns = df.columns.str.lower()
        return df
    except Exception:
        return pd.DataFrame()


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
    # ── Primary: MLB Stats API fielding (always works) ───────────────────
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

    # ── WAR from Baseball Reference ──────────────────────────────────────
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

    # ── Composite score fallback when WAR is unavailable ─────────────────
    if "WAR" not in fld.columns or fld["WAR"].notna().sum() < 5:
        components = []
        if "cs_pct" in fld.columns and fld["cs_pct"].notna().sum() >= 5:
            s = fld["cs_pct"]
            components.append((s - s.mean()) / s.std())
        if "pb" in fld.columns and "innings" in fld.columns and fld["innings"].gt(0).sum() >= 5:
            pb_rate = fld["pb"] / (fld["innings"] / 9).replace(0, np.nan)
            components.append(-(pb_rate - pb_rate.mean()) / pb_rate.std())
        if "framing_runs" in fld.columns and fld["framing_runs"].notna().sum() >= 5:
            s = fld["framing_runs"]
            components.append((s - s.mean()) / s.std())
        if components:
            fld["WAR"] = sum(c.fillna(0) for c in components) / len(components)
            fld["_composite_score"] = True

    return fld


# ── Load & filter ─────────────────────────────────────────────────────────────

st.title("⚾ MLB Catcher Defensive Analysis")
st.markdown(
    f"Real MLB data · **{season}** season · "
    "Source: MLB Stats API & Baseball Savant"
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
    # MLB Stats API (always available)
    "cs_pct":   "CS%",
    "innings":  "Innings",
    "pb":       "Passed Balls",
    # Baseball Savant (optional enrichment)
    "framing_runs":          "Framing Runs",
    "framing_opportunities": "Framing Opps",
    "pop_2b_sba":            "Pop Time 2B (s)",
    "arm_2b_3b_sba":         "Arm Strength (mph)",
}

feature_cols = [
    c for c in FEATURES
    if c in df.columns and df[c].notna().mean() > 0.4
]
feature_labels = [FEATURES[c] for c in feature_cols]

has_war = "WAR" in df.columns and df["WAR"].notna().sum() >= 5
is_composite = has_war and df.get("_composite_score", pd.Series(False)).any()
score_label = "Def. Score" if is_composite else "WAR"

# ── Display DataFrame ─────────────────────────────────────────────────────────
disp_cols = ["player_name"] + feature_cols + (["WAR"] if has_war else [])
disp_df = df[disp_cols].dropna(subset=feature_cols[:1]).copy()
disp_df = disp_df.rename(columns={"player_name": "Catcher", "WAR": score_label, **FEATURES})
if has_war:
    disp_df = disp_df.sort_values(score_label, ascending=False).reset_index(drop=True)

# ── Train models ──────────────────────────────────────────────────────────────
modeled = False
model_df = pd.DataFrame()

if has_war and len(disp_df) >= 5:
    model_df = disp_df.dropna(subset=feature_labels + [score_label]).copy()
    if len(model_df) >= 5:
        X_raw = model_df[feature_labels].values
        y = model_df[score_label].values

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
        model_df[f"LR Predicted {score_label}"] = lr_pred
        model_df[f"RF Predicted {score_label}"] = rf_pred

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

def render_model_tab(mdf, pred_col, actual_col, r2_val, rmse_val, imp_df, imp_col, title):
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

    v_min, v_max = mdf[actual_col].min(), mdf[actual_col].max()
    fig_sc = px.scatter(
        mdf, x=actual_col, y=pred_col,
        hover_name="Catcher",
        title=f"Actual vs Predicted {actual_col} — {title}",
        labels={actual_col: f"Actual {actual_col}", pred_col: f"Predicted {actual_col}"},
        color=pred_col,
        color_continuous_scale="Blues",
    )
    fig_sc.add_shape(
        type="line",
        x0=v_min, y0=v_min, x1=v_max, y1=v_max,
        line=dict(color="red", dash="dash", width=2),
    )
    fig_sc.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_sc, use_container_width=True)

    best = mdf.loc[mdf[pred_col].idxmax()]
    st.success(
        f"**{best['Catcher']}** leads with a predicted {actual_col} of **{best[pred_col]:.2f}**"
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

    if is_composite:
        st.info(
            "WAR data unavailable from external sources. "
            "Showing a **Defensive Score** (composite of CS%, Passed Ball rate"
            + (", Framing Runs" if "Framing Runs" in disp_df.columns else "")
            + ") — used as the model target.",
        )

    if has_war:
        fig_war = px.bar(
            disp_df.head(20), x="Catcher", y=score_label,
            color=score_label, color_continuous_scale="Blues",
            title=f"Top Catchers by {score_label} — {season}",
        )
        fig_war.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
        st.plotly_chart(fig_war, use_container_width=True)

    if "Framing Runs" in disp_df.columns:
        fig_frm = px.bar(
            disp_df.sort_values("Framing Runs", ascending=False).head(20),
            x="Catcher", y="Framing Runs",
            color="Framing Runs", color_continuous_scale="Purples",
            title=f"Top Catchers by Framing Runs — {season}",
        )
        fig_frm.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
        st.plotly_chart(fig_frm, use_container_width=True)

with tab2:
    st.subheader("Linear Regression")
    if modeled:
        render_model_tab(model_df, f"LR Predicted {score_label}", score_label, lr_r2, lr_rmse, lr_imp, "Coefficient", "Linear Regression")
    else:
        st.info("Not enough data for modeling. Try lowering the minimum innings threshold.")

with tab3:
    st.subheader("Random Forest")
    if modeled:
        render_model_tab(model_df, f"RF Predicted {score_label}", score_label, rf_r2, rf_rmse, rf_imp, "Importance", "Random Forest")

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
        st.info("Not enough data for modeling. Try lowering the minimum innings threshold.")
