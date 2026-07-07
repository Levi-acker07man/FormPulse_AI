import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
#main setup
st.set_page_config(page_title="FormPulse AI", layout="wide")
st.title("🏀 FormPulse AI — Player Trajectory Forecast")
st.write("Adjust the matchday conditions below to see how physical fatigue and workload impact projected scoring output.")
#loading the ai model
@st.cache_resource
def load_engine():
    model = tf.keras.models.load_model('formpulse_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_engine()

#sidebar control
st.sidebar.header("⚙️ Matchday Parameters")
sim_minutes = st.sidebar.slider("Projected Minutes Played", 15.0, 48.0, 34.0, step=0.5)
sim_rest = st.sidebar.slider("Days of Rest Prior to Game", 1.0, 7.0, 2.0, step=1.0)
sim_fga = st.sidebar.slider("Expected Field Goal Attempts", 10.0, 30.0, 20.0, step=1.0)
sim_fta = st.sidebar.slider("Expected Free Throw Attempts", 0.0, 15.0, 5.0, step=1.0)

historical_sequence = np.array([
    [32.0, 35.5, 2.0, 22.0, 6.0],  # Game T-4
    [24.0, 32.0, 1.0, 18.0, 4.0],  # Game T-3
    [28.0, 36.0, 3.0, 20.0, 5.0],  # Game T-2
    [41.0, 38.5, 1.0, 26.0, 9.0],  # Game T-1
    [30.0, sim_minutes, sim_rest, sim_fga, sim_fta] 
])

# Scale the data using the exact same scaler from training
scaled_sequence = scaler.transform(historical_sequence)
input_tensor = np.reshape(scaled_sequence, (1, 5, 5))
prediction_scaled = model.predict(input_tensor)
# Reverse the scaling to get real-world points back
dummy_matrix = np.zeros((1, 5))
dummy_matrix[0, 0] = prediction_scaled[0][0]
projected_pts = scaler.inverse_transform(dummy_matrix)[0, 0]
#displaying metrics dashboords
col1, col2, col3 = st.columns(3)
col1.metric("Recent Average", "31.2 PTS")
col2.metric("FormPulse Forecast", f"{round(projected_pts, 1)} PTS")
col3.metric("Fatigue Status", "High Risk" if sim_minutes > 38 and sim_rest <= 1 else "Optimal")
#rendering trajectory path
st.subheader("Trajectory Analysis")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=["Game T-4", "Game T-3", "Game T-2", "Game T-1", "Matchday (Forecast)"],
    y=[32, 24, 28, 41, projected_pts],
    mode='lines+markers',
    marker=dict(size=[8, 8, 8, 8, 16], color=['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4', '#ff7f0e']),
    name='Points'
))
st.plotly_chart(fig, use_container_width=True)
