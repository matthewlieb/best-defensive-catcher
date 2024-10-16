import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Real 2023 MLB Catcher Data
data = {
    'Catcher': ['J.T. Realmuto', 'Will Smith', 'Sean Murphy', 'Adley Rutschman', 'Cal Raleigh', 'Jonah Heim', 'William Contreras', 'Alejandro Kirk'],
    'Caught_Stealing_Rate': [0.36, 0.19, 0.28, 0.22, 0.26, 0.39, 0.16, 0.26],
    'Framing_Runs': [6.0, 6.0, 11.0, 9.0, 9.0, 13.0, 4.0, -2.0],
    'Defensive_Runs_Saved': [11, 0, 8, 13, 15, 21, -5, -5],
    'Fielding_Percentage': [0.996, 0.996, 0.996, 0.997, 0.995, 0.996, 0.994, 0.993],
    'WAR': [5.0, 4.3, 4.0, 5.0, 4.4, 4.8, 5.0, 1.9]
}

df = pd.DataFrame(data)

# Standardize the features
scaler = StandardScaler()
features = ['Caught_Stealing_Rate', 'Framing_Runs', 'Defensive_Runs_Saved', 'Fielding_Percentage']
df[features] = scaler.fit_transform(df[features])

# Linear Regression Model
X = df[features]
y = df['WAR']
model = LinearRegression()
model.fit(X, y)

# Calculate predicted WAR and residuals
df['Predicted_WAR'] = model.predict(X)
df['Residual'] = df['WAR'] - df['Predicted_WAR']

# Calculate feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.coef_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Streamlit App
st.title('2023 MLB Catcher Defensive Analysis')

# Create tabs
tab1, tab2 = st.tabs(["Analysis", "Source Code"])

with tab1:
    st.write("This app evaluates MLB catchers based on their 2023 defensive metrics and predicts their overall value.")

    # Display raw data
    st.subheader("Raw Data")
    st.dataframe(df)

    # Display model results
    st.subheader("Linear Regression Results")
    st.write(f"R-squared: {model.score(X, y):.4f}")

    # Feature importance plot
    st.subheader("Feature Importance")
    fig_importance = px.bar(feature_importance, x='Feature', y='Importance', title='Feature Importance in Predicting WAR')
    st.plotly_chart(fig_importance)

    # Scatter plot of Actual vs Predicted WAR
    st.subheader("Actual vs Predicted WAR")
    fig_scatter = px.scatter(df, x='WAR', y='Predicted_WAR', hover_name='Catcher',
                             title='Actual vs Predicted WAR',
                             labels={'WAR': 'Actual WAR', 'Predicted_WAR': 'Predicted WAR'})
    fig_scatter.add_shape(type='line', x0=df['WAR'].min(), y0=df['WAR'].min(),
                          x1=df['WAR'].max(), y1=df['WAR'].max(),
                          line=dict(color='red', dash='dash'))
    st.plotly_chart(fig_scatter)

    # Residual plot
    st.subheader("Residual Plot")
    fig_residual = px.scatter(df, x='Predicted_WAR', y='Residual', hover_name='Catcher',
                              title='Residual Plot',
                              labels={'Predicted_WAR': 'Predicted WAR', 'Residual': 'Residual'})
    fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_residual)

    # Identify the best defensive catcher
    best_catcher = df.loc[df['Predicted_WAR'].idxmax()]
    st.subheader("Best Defensive Catcher")
    st.write(f"Based on our model, the best defensive catcher is {best_catcher['Catcher']} with a predicted WAR of {best_catcher['Predicted_WAR']:.2f}.")

    # Recommendations
    st.subheader("Insights and Recommendations")
    st.write("""
    1. The model shows varying levels of importance for different defensive metrics. Teams should focus on developing catchers' skills in areas with higher importance.
    2. There's a discrepancy between predicted and actual WAR for some catchers, suggesting that other factors (possibly offensive contributions or unmeasured defensive skills) play a significant role.
    3. Catchers with positive residuals (performing above predicted WAR) might have intangible skills or leadership qualities worth investigating.
    4. This model should be used in conjunction with scouting reports and other analytical tools for a comprehensive evaluation of catchers.
    5. Regular updates to the model with new data and potentially additional metrics could improve its predictive power.
    6. Consider the specific needs of your pitching staff when evaluating catchers, as some skills may be more valuable depending on your team's pitchers.
    """)

with tab2:
    st.subheader("Source Code")
    st.code('''
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.express as px

# Real 2023 MLB Catcher Data
data = {
    'Catcher': ['J.T. Realmuto', 'Will Smith', 'Sean Murphy', 'Adley Rutschman', 'Cal Raleigh', 'Jonah Heim', 'William Contreras', 'Alejandro Kirk'],
    'Caught_Stealing_Rate': [0.36, 0.19, 0.28, 0.22, 0.26, 0.39, 0.16, 0.26],
    'Framing_Runs': [6.0, 6.0, 11.0, 9.0, 9.0, 13.0, 4.0, -2.0],
    'Defensive_Runs_Saved': [11, 0, 8, 13, 15, 21, -5, -5],
    'Fielding_Percentage': [0.996, 0.996, 0.996, 0.997, 0.995, 0.996, 0.994, 0.993],
    'WAR': [5.0, 4.3, 4.0, 5.0, 4.4, 4.8, 5.0, 1.9]
}

df = pd.DataFrame(data)

# Standardize the features
scaler = StandardScaler()
features = ['Caught_Stealing_Rate', 'Framing_Runs', 'Defensive_Runs_Saved', 'Fielding_Percentage']
df[features] = scaler.fit_transform(df[features])

# Linear Regression Model
X = df[features]
y = df['WAR']
model = LinearRegression()
model.fit(X, y)

# Calculate predicted WAR and residuals
df['Predicted_WAR'] = model.predict(X)
df['Residual'] = df['WAR'] - df['Predicted_WAR']

# Calculate feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.coef_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# Streamlit App
st.title('2023 MLB Catcher Defensive Analysis')

# Create tabs
tab1, tab2 = st.tabs(["Analysis", "Source Code"])

with tab1:
    st.write("This app evaluates MLB catchers based on their 2023 defensive metrics and predicts their overall value.")

    # Display raw data
    st.subheader("Raw Data")
    st.dataframe(df)

    # Display model results
    st.subheader("Linear Regression Results")
    st.write(f"R-squared: {model.score(X, y):.4f}")

    # Feature importance plot
    st.subheader("Feature Importance")
    fig_importance = px.bar(feature_importance, x='Feature', y='Importance', title='Feature Importance in Predicting WAR')
    st.plotly_chart(fig_importance)

    # Scatter plot of Actual vs Predicted WAR
    st.subheader("Actual vs Predicted WAR")
    fig_scatter = px.scatter(df, x='WAR', y='Predicted_WAR', hover_name='Catcher',
                             title='Actual vs Predicted WAR',
                             labels={'WAR': 'Actual WAR', 'Predicted_WAR': 'Predicted WAR'})
    fig_scatter.add_shape(type='line', x0=df['WAR'].min(), y0=df['WAR'].min(),
                          x1=df['WAR'].max(), y1=df['WAR'].max(),
                          line=dict(color='red', dash='dash'))
    st.plotly_chart(fig_scatter)

    # Residual plot
    st.subheader("Residual Plot")
    fig_residual = px.scatter(df, x='Predicted_WAR', y='Residual', hover_name='Catcher',
                              title='Residual Plot',
                              labels={'Predicted_WAR': 'Predicted WAR', 'Residual': 'Residual'})
    fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_residual)

    # Identify the best defensive catcher
    best_catcher = df.loc[df['Predicted_WAR'].idxmax()]
    st.subheader("Best Defensive Catcher")
    st.write(f"Based on our model, the best defensive catcher is {best_catcher['Catcher']} with a predicted WAR of {best_catcher['Predicted_WAR']:.2f}.")

    # Recommendations
    st.subheader("Insights and Recommendations")
    st.write("""
    1. The model shows varying levels of importance for different defensive metrics. Teams should focus on developing catchers' skills in areas with higher importance.
    2. There's a discrepancy between predicted and actual WAR for some catchers, suggesting that other factors (possibly offensive contributions or unmeasured defensive skills) play a significant role.
    3. Catchers with positive residuals (performing above predicted WAR) might have intangible skills or leadership qualities worth investigating.
    4. This model should be used in conjunction with scouting reports and other analytical tools for a comprehensive evaluation of catchers.
    5. Regular updates to the model with new data and potentially additional metrics could improve its predictive power.
    6. Consider the specific needs of your pitching staff when evaluating catchers, as some skills may be more valuable depending on your team's pitchers.
    """))
    ''', language='python')