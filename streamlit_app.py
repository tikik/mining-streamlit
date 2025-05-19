
# iron_concentrate_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv("your_data.csv")  # Replace with your file

st.title("Iron Concentrate Prediction Dashboard")

# Feature selection
features = ['air_flow', 'pulp_density', 'reagent_1', 'reagent_2']  # Example features
X = df[features]
y = df['% Iron Concentrate']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model fitting
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature importance
importances = model.feature_importances_
importance_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)

# Bar plot for feature importance
plt.figure(figsize=(10, 6))
plt.bar(importance_df['Feature'], importance_df['Importance'])
plt.xticks(rotation=45, ha='right')
plt.title('Feature Importance for Iron Concentrate Prediction')
plt.tight_layout()
st.pyplot(plt)

# Model performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Model Mean Squared Error: {mse:.2f}")

# Actual vs Predicted plot
fig_pred = px.scatter(x=y_test, y=y_pred,
                      labels={'x': 'Actual Iron Concentrate %', 'y': 'Predicted Iron Concentrate %'},
                      title='Actual vs Predicted Iron Concentrate')
fig_pred.add_scatter(x=[y_test.min(), y_test.max()],
                     y=[y_test.min(), y_test.max()],
                     mode='lines', name='Perfect Prediction')
st.plotly_chart(fig_pred)

# Time series plot
fig_time = px.line(df, x='date', y='% Iron Concentrate',
                   title='Iron Concentrate % Over Time')
st.plotly_chart(fig_time)

# Optimization suggestion
st.subheader("Optimization Suggestion")
best_conditions = df.loc[df["% Iron Concentrate"].idxmax(), features]

# Radar chart
categories = features
values = best_conditions.values.tolist()
values.append(values[0])

angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
angles = np.concatenate((angles, [angles[0]]))

fig_radar = plt.figure(figsize=(10, 10))
ax = fig_radar.add_subplot(111, polar=True)
ax.plot(angles, values)
ax.fill(angles, values, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, rotation=45)
plt.title("Best Operating Conditions Profile")
st.pyplot(fig_radar)

st.write("Best operating conditions from data:")
st.write(best_conditions)
st.success("Try adjusting your air flow and pulp levels to match these for best iron concentrate output.")
