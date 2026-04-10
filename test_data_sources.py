"""
Run this before pushing to verify all data sources work.
Usage: python3 test_data_sources.py [season]
"""
import sys
import requests
import io
import pandas as pd

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2024

BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://baseballsavant.mlb.com/",
}

errors = []

def check(label, fn):
    try:
        result = fn()
        print(f"  ✅ {label}: {result}")
    except Exception as e:
        print(f"  ❌ {label}: {e}")
        errors.append(label)

# ── 1. MLB Stats API ──────────────────────────────────────────────────────────
print(f"\n[1] MLB Stats API — {SEASON} catchers")
r = requests.get(
    f"https://statsapi.mlb.com/api/v1/stats?stats=season&group=fielding"
    f"&season={SEASON}&gameType=R&playerPool=ALL&position=C&limit=500",
    timeout=30,
)
r.raise_for_status()
rows = []
for sg in r.json().get("stats", []):
    for split in sg.get("splits", []):
        p = split["player"]; s = split["stat"]
        rows.append({"name": p.get("fullName"), "inn": s.get("innings"),
                     "pb": int(s.get("passedBall") or 0),
                     "cs": int(s.get("caughtStealing") or 0),
                     "sb": int(s.get("stolenBases") or 0)})
df_fld = pd.DataFrame(rows).drop_duplicates("name")
check("Row count",   lambda: f"{len(df_fld)} catchers")
check("Has innings", lambda: f"innings present={('inn' in df_fld.columns)}")
check("Sample",      lambda: df_fld['name'].head(3).tolist())

# ── 2. Statcast Framing ───────────────────────────────────────────────────────
print(f"\n[2] Statcast Framing — {SEASON}")
r = requests.get(
    f"https://baseballsavant.mlb.com/leaderboard/catcher-framing"
    f"?type=catcher&seasonStart={SEASON}&seasonEnd={SEASON}"
    f"&team=&min=q&sortColumn=rv_tot&sortDirection=desc&csv=true",
    headers=BROWSER_HEADERS, timeout=30,
)
check("HTTP status", lambda: r.status_code)
df_frm = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")))
def flip(n):
    parts = str(n).split(",", 1)
    return parts[1].strip() + " " + parts[0].strip() if len(parts) == 2 else str(n)
df_frm["player_name"] = df_frm["name"].apply(flip)
merged = df_fld.merge(df_frm[["player_name", "rv_tot"]], left_on="name", right_on="player_name", how="inner")
check("Framing rows",   lambda: f"{len(df_frm)} rows")
check("Merge count",    lambda: f"{len(merged)} matched to fielding data")
check("Top framer",     lambda: merged.nlargest(1, "rv_tot")[["name", "rv_tot"]].to_dict("records"))

# ── 3. Pop Time ───────────────────────────────────────────────────────────────
print(f"\n[3] Pop Time — {SEASON}")
r = requests.get(
    f"https://baseballsavant.mlb.com/leaderboard/poptime"
    f"?year={SEASON}&team=&min2b=5&min3b=0&csv=true",
    headers=BROWSER_HEADERS, timeout=30,
)
df_pop = pd.read_csv(io.StringIO(r.content.decode("utf-8-sig")))
check("HTTP status", lambda: r.status_code)
check("Row count",   lambda: f"{len(df_pop)} catchers")
check("Columns",     lambda: df_pop.columns.tolist()[:4])

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"❌ FAILED: {errors} — do not push until fixed")
    sys.exit(1)
else:
    print("✅ All data sources OK — safe to push")
