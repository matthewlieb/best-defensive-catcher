# MLB Catcher Defensive Analysis ⚾📊

Test it out at [phillies-analytics.streamlit.app](https://phillies-analytics.streamlit.app). This Streamlit application analyzes the defensive performance of MLB catchers using various metrics and a linear regression model to predict their overall value (WAR).

## Features ✨

- Displays raw data of catcher defensive metrics
- Performs linear regression to predict WAR based on defensive metrics
- Visualizes feature importance of different defensive skills
- Compares actual vs. predicted WAR for each catcher
- Identifies the best defensive catcher based on the model
- Provides insights and recommendations for decision-making

## Installation 🛠️

1. Clone this repository:
`git clone https://github.com/yourusername/best-defensive-catcher.git
cd best-defensive-catcher`


2. Create a virtual environment (optional but recommended):
`python -m venv venv`
`source venv/bin/activate # On Windows, use venv\Scripts\activate`


3. Install the required packages:
`pip install -r requirements.txt`


*Note: If there is no `requirements.txt` file, you can manually install the necessary packages by running:*
'pip install streamlit pandas numpy scikit-learn plotly


## Usage 🎯

Run the Streamlit app:
`streamlit run best_defensive_catcher.py`


The app will open in your default web browser.

## Data 📊

The current version uses a sample dataset of 8 top MLB catchers from the 2023 season. In a production environment, you should replace this with real, up-to-date data from official MLB statistics or reputable sports data providers.

## Modifying the Analysis 🔍

To update the analysis:

1. Edit the `data` dictionary in `best_defensive_catcher.py` to include more catchers or different metrics.
2. Modify the `features` list if you change the metrics being analyzed.
3. Adjust the visualizations or add new ones as needed.

## Contributing 🤝

Contributions to improve the analysis or extend its capabilities are welcome. Please feel free to submit pull requests or open issues to discuss potential changes.

## License 📜

This project is open source and available under the [MIT License](LICENSE).

## License 📜

This project is open source and available under the [MIT License](LICENSE).
