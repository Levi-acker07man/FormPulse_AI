import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="FormPulse AI", layout="wide")
st.markdown("""
    <style>
    /* Force light mode feel */
    .stApp { background-color: #f8f9fa; color: #333; }
    
    /* Style headers */
    h1 { color: #2c3e50; text-align: center; }
    h2 { color: #34495e; }
    
    /* Style buttons */
    div.stButton > button {
        background-color: #3498db;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] { color: #2c3e50; font-size: 30px; }
    </style>
""", unsafe_allow_html=True)
st.title("🏀 FormPulse AI: Performance Engine")
st.write("---")


@st.cache_resource
def load_engine():
    model = tf.keras.models.load_model('formpulse_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_engine()


st.sidebar.header("📊 Matchday Controls")
st.sidebar.info("Adjust player metrics to forecast performance.")

sim_minutes = st.sidebar.slider("Projected Minutes Played", 15.0, 48.0, 34.0, step=0.5)
sim_rest = st.sidebar.slider("Days of Rest", 1.0, 7.0, 2.0, step=1.0)
sim_fga = st.sidebar.slider("Field Goal Attempts", 10.0, 30.0, 20.0, step=1.0)
sim_fta = st.sidebar.slider("Free Throw Attempts", 0.0, 15.0, 5.0, step=1.0)


historical_sequence = np.array([[32.0, 35.5, 2.0, 22.0, 6.0], [24.0, 32.0, 1.0, 18.0, 4.0],
                                [28.0, 36.0, 3.0, 20.0, 5.0], [41.0, 38.5, 1.0, 26.0, 9.0],
                                [30.0, sim_minutes, sim_rest, sim_fga, sim_fta]])

scaled_sequence = scaler.transform(historical_sequence)
input_tensor = np.reshape(scaled_sequence, (1, 5, 5))
prediction_scaled = model.predict(input_tensor)

dummy_matrix = np.zeros((1, 5))
dummy_matrix[0, 0] = prediction_scaled[0][0]
projected_pts = scaler.inverse_transform(dummy_matrix)[0, 0]


col1, col2, col3 = st.columns(3)
col1.metric("Recent Scoring Avg", "31.2 PTS")
col2.metric("FormPulse Forecast", f"{round(projected_pts, 1)} PTS")
col3.metric("Fatigue Status", "High Risk" if sim_minutes > 38 and sim_rest <= 1 else "Optimal")

st.subheader("Performance Trajectory")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=["Game T-4", "Game T-3", "Game T-2", "Game T-1", "Matchday (Forecast)"],
    y=[32, 24, 28, 41, float(projected_pts)],
    mode='lines+markers',
    line=dict(color='#3498db', width=3),
    marker=dict(size=[8, 8, 8, 8, 14], color=['#2c3e50', '#2c3e50', '#2c3e50', '#2c3e50', '#e74c3c'])
))
fig.update_yaxes(range=[10, 50])
fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
st.plotly_chart(fig, use_container_width=True)
