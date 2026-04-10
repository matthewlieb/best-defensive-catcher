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


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _savant_name_to_fullname(name_series):
    """Convert 'Last, First' → 'First Last'."""
    def _flip(n):
        if pd.isna(n):
            return n
        parts = str(n).split(",", 1)
        if len(parts) == 2:
            return parts[1].strip() + " " + parts[0].strip()
        return n.strip()
    return name_series.apply(_flip)


@st.cache_data(ttl=86400, show_spinner="Fetching Statcast framing data…")
def _load_statcast_framing(season):
    import io
    url = (
        f"https://baseballsavant.mlb.com/leaderboard/catcher-framing"
        f"?type=catcher&seasonStart={season}&seasonEnd={season}"
        f"&team=&min=q&sortColumn=rv_tot&sortDirection=desc&csv=true"
    )
    try:
        resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=30)
        if resp.status_code != 200:
            st.sidebar.warning(f"Framing: HTTP {resp.status_code}")
            return None
        raw = resp.content.decode("utf-8-sig")
        if not raw.strip().startswith('"id"') and not raw.strip().startswith('id'):
            st.sidebar.warning(f"Framing: unexpected response — {raw[:120]}")
            return None
        df = pd.read_csv(io.StringIO(raw))
        df.columns = df.columns.str.lower()
        if "name" in df.columns:
            df["player_name"] = _savant_name_to_fullname(df["name"])
            df = df[df["player_name"].notna()].copy()
        return df
    except Exception as e:
        st.sidebar.warning(f"Framing error: {e}")
        return None


@st.cache_data(ttl=86400, show_spinner="Fetching pop-time data…")
def _load_statcast_poptime(season):
    import io
    try:
        url = (
            f"https://baseballsavant.mlb.com/leaderboard/poptime"
            f"?year={season}&team=&min2b=5&min3b=0&csv=true"
        )
        resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.content.decode("utf-8-sig")))
        df.columns = df.columns.str.lower()
        # columns: entity_name (Last, First), entity_id, ...
        name_col = _first_match(df, "entity_name", "name")
        if name_col:
            df["player_name"] = _savant_name_to_fullname(df[name_col])
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
    if framing is not None and not framing.empty and "player_name" in framing.columns:
        # Direct CSV columns: id, name, pitches, rv_tot, pct_tot, ...
        rename_map = {}
        if "rv_tot" in framing.columns:
            rename_map["rv_tot"] = "framing_runs"
        if "pitches" in framing.columns:
            rename_map["pitches"] = "framing_opportunities"
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
    if poptime is not None and not poptime.empty and "player_name" in poptime.columns:
        # Direct CSV columns: entity_name, pop_2b_sba, maxeff_arm_2b_3b_sba, ...
        rename_map = {}
        if "maxeff_arm_2b_3b_sba" in poptime.columns:
            rename_map["maxeff_arm_2b_3b_sba"] = "arm_2b_3b_sba"
        poptime = poptime.rename(columns=rename_map)
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

# ── All display columns ───────────────────────────────────────────────────────
ALL_COLS = {
    "cs_pct":              "CS%",
    "innings":             "Innings",
    "pb":                  "Passed Balls",
    "framing_runs":        "Framing Runs",
    "framing_opportunities": "Framing Opps",
    "pop_2b_sba":          "Pop Time 2B (s)",
    "arm_2b_3b_sba":       "Arm Strength (mph)",
}

# Option B: traditional box-score stats → predict Statcast framing runs
# Features must be independent of the target; framing is computed separately by Statcast
MODEL_FEATURE_COLS = ["cs_pct", "innings", "pb"]
MODEL_FEATURE_LABELS = ["CS%", "Innings", "Passed Balls"]
MODEL_TARGET_COL = "framing_runs"
MODEL_TARGET_LABEL = "Framing Runs"

disp_cols_present = ["player_name"] + [c for c in ALL_COLS if c in df.columns]
disp_df = df[disp_cols_present].copy()
disp_df = disp_df.rename(columns={"player_name": "Catcher", **ALL_COLS})
# Sort by framing runs if available, else CS%
sort_col = MODEL_TARGET_LABEL if MODEL_TARGET_LABEL in disp_df.columns else "CS%"
if sort_col in disp_df.columns:
    disp_df = disp_df.sort_values(sort_col, ascending=False).reset_index(drop=True)

# ── Train models (Option B: predict framing from traditional stats) ───────────
modeled = False
model_df = pd.DataFrame()

has_framing = (
    MODEL_TARGET_COL in df.columns
    and df[MODEL_TARGET_COL].notna().sum() >= 5
)
features_present = all(c in df.columns for c in MODEL_FEATURE_COLS)

if has_framing and features_present:
    model_df = df[MODEL_FEATURE_COLS + [MODEL_TARGET_COL, "player_name"]].dropna().copy()
    model_df = model_df.rename(columns={"player_name": "Catcher"})
    if len(model_df) >= 5:
        X_raw = model_df[MODEL_FEATURE_COLS].values
        y = model_df[MODEL_TARGET_COL].values

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

        model_df["LR Predicted Framing"] = lr_pred
        model_df["RF Predicted Framing"] = rf_pred
        # rename for display
        model_df = model_df.rename(columns={
            c: lbl for c, lbl in zip(MODEL_FEATURE_COLS, MODEL_FEATURE_LABELS)
        })
        model_df = model_df.rename(columns={MODEL_TARGET_COL: MODEL_TARGET_LABEL})

        lr_imp = (
            pd.DataFrame({"Feature": MODEL_FEATURE_LABELS, "Coefficient": lr.coef_})
            .sort_values("Coefficient", ascending=False)
        )
        rf_imp = (
            pd.DataFrame({"Feature": MODEL_FEATURE_LABELS, "Importance": rf.feature_importances_})
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

    if MODEL_TARGET_LABEL in disp_df.columns:
        fig_frm = px.bar(
            disp_df.dropna(subset=[MODEL_TARGET_LABEL]).head(20),
            x="Catcher", y=MODEL_TARGET_LABEL,
            color=MODEL_TARGET_LABEL, color_continuous_scale="Purples",
            title=f"Top Catchers by Framing Runs — {season}",
        )
        fig_frm.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
        st.plotly_chart(fig_frm, use_container_width=True)

    fig_cs = px.bar(
        disp_df.dropna(subset=["CS%"]).sort_values("CS%", ascending=False).head(20),
        x="Catcher", y="CS%",
        color="CS%", color_continuous_scale="Blues",
        title=f"Top Catchers by Caught Stealing % — {season}",
    )
    fig_cs.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
    st.plotly_chart(fig_cs, use_container_width=True)

_NO_MODEL_MSG = (
    "Models require Statcast framing data (Baseball Savant) as the target variable. "
    "Framing data wasn't available for this season/filter — try a different season."
)

with tab2:
    st.subheader("Linear Regression")
    if modeled:
        st.caption(
            "**What this model does:** uses traditional box-score stats (CS%, Innings, Passed Balls) "
            "to predict Statcast Framing Runs — an independently-computed advanced metric. "
            "This tells us how well old-school scouting stats correlate with modern pitch framing."
        )
        render_model_tab(model_df, "LR Predicted Framing", MODEL_TARGET_LABEL, lr_r2, lr_rmse, lr_imp, "Coefficient", "Linear Regression")
    else:
        st.info(_NO_MODEL_MSG)

with tab3:
    st.subheader("Random Forest")
    if modeled:
        st.caption(
            "**What this model does:** same prediction as Linear Regression but using a Random Forest — "
            "an ensemble of decision trees that can capture non-linear relationships between stats."
        )
        render_model_tab(model_df, "RF Predicted Framing", MODEL_TARGET_LABEL, rf_r2, rf_rmse, rf_imp, "Importance", "Random Forest")

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
        st.info(_NO_MODEL_MSG)
