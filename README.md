![portfolio-13-large](https://github.com/user-attachments/assets/db268439-bcc6-4d85-a28c-f6355d55fa3c)

# MLB Catcher Defensive Analysis ⚾

Live at [phillies-analytics.streamlit.app](https://phillies-analytics.streamlit.app).

A Streamlit app that pulls real MLB Statcast data via [pybaseball](https://github.com/jldbc/pybaseball) to evaluate catcher defense and predict WAR using Linear Regression and Random Forest models.

## Features

- **Real Statcast data** — framing runs, pop time, arm strength, and blocking runs pulled directly from Baseball Savant for any season from 2020–2024
- **Sidebar filters** — select season and minimum framing opportunities (default: 500) to control sample size
- **Sortable data table** — all qualifying catchers with their raw defensive metrics
- **Rankings** — bar charts for WAR and Framing Runs
- **Linear Regression** — coefficients, R², RMSE, and actual vs predicted WAR scatter plot
- **Random Forest** — feature importances, R², RMSE, and actual vs predicted WAR scatter plot
- **Model comparison** — side-by-side R² and RMSE for both models
- **Data caching** — Statcast data is cached for 24 hours so the app doesn't re-query on every load

## Running Locally

1. Clone this repository:
```bash
git clone https://github.com/matthewlieb/best-defensive-catcher.git
cd best-defensive-catcher
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
streamlit run best_defensive_catcher.py
```

The app will open in your default browser at `http://localhost:8501`.

## Data Sources

- **[Baseball Savant](https://baseballsavant.mlb.com)** — catcher framing, pop time, arm strength, and blocking metrics via the Statcast system
- **[FanGraphs](https://www.fangraphs.com)** — WAR (Wins Above Replacement)
- **[pybaseball](https://github.com/jldbc/pybaseball)** — Python wrapper for Baseball Savant and FanGraphs data

## Metrics Used

| Metric | Source | Description |
|---|---|---|
| Framing Runs | Baseball Savant | Runs saved by converting borderline pitches to strikes |
| Framing Opportunities | Baseball Savant | Total called pitches (used as sample-size filter) |
| Pop Time 2B | Baseball Savant | Avg time from glove to 2B on steal attempts (seconds) |
| Arm Strength | Baseball Savant | Arm strength in mph on throws to 2B/3B |
| Blocking Runs | Baseball Savant | Run value saved by blocking pitches in the dirt |
| WAR | FanGraphs | Wins Above Replacement (prediction target) |

## Contributing

Pull requests are welcome. Open an issue to discuss any changes before submitting.
