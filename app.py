import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go

# --- 1. PAGE CONFIGURATION & HACKATHON STYLING ---
st.set_page_config(page_title="FormPulse AI", page_icon="🏀", layout="wide")

# Custom Dark Mode SaaS CSS
st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #0b0f19; color: #ffffff; }
    
    /* Neumorphic Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #151a28;
        border: 1px solid #232b3e;
        padding: 5% 5% 5% 10%;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Neon Accents for Headers */
    h1 { color: #00E676 !important; font-weight: 800; }
    h2, h3 { color: #ffffff !important; font-weight: 600; }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] { background-color: #111520; border-right: 1px solid #232b3e; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] label { color: #a3b1c6 !important; }
    </style>
""", unsafe_allow_html=True)

# --- 2. ENGINE INITIALIZATION ---
@st.cache_resource
def load_engine():
    model = tf.keras.models.load_model('formpulse_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_engine()

# --- 3. SIDEBAR INTERACTIVITY ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/808/808439.png", width=50)
st.sidebar.title("Simulation Controls")
st.sidebar.markdown("Drag the sliders to adjust real-time match variables.")

sim_minutes = st.sidebar.slider("Projected Minutes Played", 15.0, 48.0, 34.0, step=0.5)
sim_rest = st.sidebar.slider("Days of Rest", 1.0, 7.0, 2.0, step=1.0)
sim_fga = st.sidebar.slider("Field Goal Attempts", 10.0, 30.0, 20.0, step=1.0)
sim_fta = st.sidebar.slider("Free Throw Attempts", 0.0, 15.0, 5.0, step=1.0)

# --- 4. DEEP LEARNING LOGIC ---
historical_sequence = np.array([
    [32.0, 35.5, 2.0, 22.0, 6.0], [24.0, 32.0, 1.0, 18.0, 4.0],
    [28.0, 36.0, 3.0, 20.0, 5.0], [41.0, 38.5, 1.0, 26.0, 9.0],
    [30.0, sim_minutes, sim_rest, sim_fga, sim_fta]
])

scaled_sequence = scaler.transform(historical_sequence)
input_tensor = np.reshape(scaled_sequence, (1, 5, 5))
prediction = model.predict(input_tensor, verbose=0)

dummy_matrix = np.zeros((1, 5))
dummy_matrix[0, 0] = prediction[0][0]
projected_pts = scaler.inverse_transform(dummy_matrix)[0, 0]

# FIXED: Workload index math now dynamically listens to ALL 4 sliders
base_workload = (sim_minutes * 1.2) + (sim_fga * 1.5) + (sim_fta * 0.8)
rest_recovery = sim_rest * 8.0
fatigue_score = min(100, max(0, (base_workload - rest_recovery)))

# --- 5. TOP DASHBOARD METRICS ---
st.title("FormPulse AI // Workload Engine")
st.markdown("Predictive analytics modeling for NBA player momentum and fatigue.")
st.write("---")

col1, col2, col3 = st.columns(3)
col1.metric("Historical Baseline", "31.2 PTS", "Last 4 Games")
col2.metric("LSTM Projected Output", f"{round(projected_pts, 2)} PTS", "Live AI Calculation", delta_color="off")
col3.metric("System Status", "High Alert" if fatigue_score > 75 else ("Warning" if fatigue_score > 45 else "Optimal"))

st.write("---")

# --- 6. INTERACTIVE VISUALIZATIONS ---
chart_col1, chart_col2 = st.columns([2, 1])

with chart_col1:
    st.subheader("📈 5-Game Scoring Trajectory")
    fig_line = go.Figure()
    
    # Historical Data + Live Prediction
    fig_line.add_trace(go.Scatter(
        x=["T-4", "T-3", "T-2", "T-1", f"Matchday ({round(projected_pts,1)}pt)"],
        y=[32, 24, 28, 41, float(projected_pts)],
        mode='lines+markers',
        name='Points Scored',
        line=dict(color='#00E676', width=4, shape='spline'),
        marker=dict(size=[10, 10, 10, 10, 20], color='#ffffff', line=dict(width=2, color='#e74c3c' if fatigue_score > 75 else '#00E676'))
    ))
    
    # FIXED: Removed the hardcoded [10, 50] range so the graph auto-zooms and shows every micro-movement cleanly
    fig_line.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#a3b1c6'),
        yaxis=dict(gridcolor='#232b3e', zeroline=False, automargin=True),
        xaxis=dict(gridcolor='#232b3e'),
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_line, use_container_width=True)

with chart_col2:
    st.subheader("🔋 Combined Workload Index")
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = fatigue_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Stress & Fatigue %", 'font': {'color': '#a3b1c6'}},
        number = {'font': {'color': '#ffffff'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#a3b1c6"},
            'bar': {'color': "#ffffff"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#232b3e",
            'steps': [
                {'range': [0, 45], 'color': '#00E676'},
                {'range': [45, 75], 'color': '#FFD600'},
                {'range': [75, 100], 'color': '#FF1744'}
            ]
        }
    ))
    fig_gauge.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=300
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

# Add an expandable section for judges to read about the tech
with st.expander("⚙️ View Architecture & Methodology"):
    st.markdown("""
    **FormPulse AI** uses a Bidirectional Long Short-Term Memory (LSTM) neural network. 
    Unlike static linear regressions, this model analyzes chronological 5-game sequences to capture the compounding physiological effects of high minutes and low rest days.
    
    * **Data Normalization:** Scikit-Learn `MinMaxScaler` (0 to 1 range)
    * **Network Architecture:** Bi-LSTM -> Dropout -> Dense ReLU -> Linear Output
    * **UI Framework:** Streamlit Cloud + Plotly Graph Objects
    """)
