import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import plotly.express as px

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Premium Predictor AI",
    page_icon="💸",
    layout="wide"
)

# --- CSS FOR STYLING ---
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# --- TITLE SECTION ---
st.title("💸 Insurance Premium Predictor")
st.markdown("Use this Simple Linear Regression model to estimate insurance premiums based on age.")
st.markdown("---")

# --- DATA HANDLING ---
@st.cache_data
def load_data():
    # Since we don't have the original CSV, we reconstruct a dataset 
    # based on the statistics found in the PDF (Min Age 18, Max 33, Min Prem 10k, Max 27k)
    data = {
        'Age': [18, 20, 22, 23, 26, 28, 31, 33],
        'Premium': [10000, 12000, 15000, 16500, 19000, 22000, 26500, 27000]
    }
    return pd.DataFrame(data)

# Sidebar for file upload or using default data
st.sidebar.header("⚙️ Configuration")
uploaded_file = st.sidebar.file_uploader("Upload CSV (Optional)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom data loaded!")
else:
    df = load_data()
    st.sidebar.info("Using demo dataset (matches PDF stats)")

# --- SIDEBAR INPUT ---
st.sidebar.subheader("Make a Prediction")
min_age = int(df['Age'].min())
max_age = int(df['Age'].max()) + 20 # Allow predicting slightly outside range

input_age = st.sidebar.slider("Select Age", min_value=18, max_value=60, value=30)

# --- MODEL TRAINING ---
X = df[['Age']]
y = df['Premium']

lr = LinearRegression()
lr.fit(X, y)

# Prediction for user input
prediction = lr.predict(pd.DataFrame([[input_age]], columns=['Age']))[0]

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Prediction Result")
    
    # Displaying the result in a nice metric style
    st.markdown(f"""
    <div class="metric-card">
        <h3>Estimated Premium</h3>
        <h1 style="color: #27ae60; font-size: 40px;">${prediction:,.2f}</h1>
        <p style="color: gray;">For Age: {input_age}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Model Details")
    st.write(f"**Coefficient (Slope):** {lr.coef_[0]:.2f}")
    st.write(f"**Intercept:** {lr.intercept_:.2f}")
    
    with st.expander("View Raw Data"):
        st.dataframe(df, height=200)

with col2:
    st.subheader("📈 Regression Visualization")
    
    # Generate regression line points
    x_range = np.linspace(df['Age'].min(), max(df['Age'].max(), input_age), 100).reshape(-1, 1)
    y_range = lr.predict(x_range)
    
    # Create interactive plot using Plotly
    fig = go.Figure()

    # 1. Scatter plot of actual data
    fig.add_trace(go.Scatter(
        x=df['Age'], 
        y=df['Premium'], 
        mode='markers',
        name='Training Data',
        marker=dict(color='#3498db', size=12, opacity=0.8)
    ))

    # 2. Regression Line
    fig.add_trace(go.Scatter(
        x=x_range.flatten(), 
        y=y_range, 
        mode='lines',
        name='Regression Line',
        line=dict(color='#e74c3c', width=3, dash='dash')
    ))

    # 3. User Prediction Point
    fig.add_trace(go.Scatter(
        x=[input_age], 
        y=[prediction], 
        mode='markers',
        name='Your Prediction',
        marker=dict(color='#27ae60', size=20, symbol='star')
    ))

    fig.update_layout(
        title="Age vs Premium Analysis",
        xaxis_title="Age",
        yaxis_title="Premium ($)",
        template="plotly_white",
        height=500,
        hovermode="x"
    )

    st.plotly_chart(fig, use_container_width=True)

# --- FOOTER ---
st.markdown("---")
st.markdown("*Built with Streamlit & Scikit-Learn*")
