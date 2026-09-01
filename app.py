import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# 1. MODEL DEFINITION
# ---------------------------------------------------------
class SimpleECGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(32 * 43, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------------------------------------------------
# 2. CACHED DATA & MODEL LOADERS
# ---------------------------------------------------------
@st.cache_resource
def load_trained_model():
    model = SimpleECGModel()
    model.load_state_dict(torch.load("best_model.pt", map_location=torch.device("cpu"), weights_only=True))
    model.eval()
    return model

@st.cache_data
def load_dataset():
    df_normal = pd.read_csv("ptbdb_normal.csv", header=None)
    df_abnormal = pd.read_csv("ptbdb_abnormal.csv", header=None)
    
    # Calculate dataset-wide baseline means and stds for comparison
    normal_signals = df_normal.iloc[:, :-1].values
    abnormal_signals = df_abnormal.iloc[:, :-1].values
    
    baselines = {
        "normal_mean": np.mean(normal_signals, axis=0),
        "normal_std": np.std(normal_signals, axis=0),
        "abnormal_mean": np.mean(abnormal_signals, axis=0),
        "abnormal_std": np.std(abnormal_signals, axis=0),
    }
    return df_normal, df_abnormal, baselines

# ---------------------------------------------------------
# 3. EXPLAINABILITY: 1D GRADIENT SALIENCY
# ---------------------------------------------------------
def compute_saliency(model, signal_tensor, target_class):
    signal_tensor = signal_tensor.clone().detach().requires_grad_(True)
    output = model(signal_tensor)
    score = output[0, target_class]
    score.backward()
    
    # Gradients absolute value represents importance
    saliency = signal_tensor.grad.data.abs().squeeze().numpy()
    # Normalize between 0 and 1
    if saliency.max() > saliency.min():
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    return saliency

# ---------------------------------------------------------
# 4. STREAMLIT UI SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="ECG Diagnostic & Explainability Dashboard", layout="wide")

st.title("🫀 ECG Diagnostic & Explainability Dashboard")
st.markdown("Interpret 1D-CNN predictions on PTB ECG leads with feature attribution maps and baseline morphology overlays.")

try:
    model = load_trained_model()
    df_normal, df_abnormal, baselines = load_dataset()
except Exception as e:
    st.error(f"Error loading model or datasets: {e}. Please ensure 'best_model.pt', 'ptbdb_normal.csv', and 'ptbdb_abnormal.csv' exist.")
    st.stop()

# ---------------------------------------------------------
# 5. SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("Signal Selection")
input_source = st.sidebar.radio("Choose Input Mode", ["Sample from Dataset", "Upload Custom CSV / Array"])

sample_signal = None
ground_truth = None

if input_source == "Sample from Dataset":
    category = st.sidebar.selectbox("Select Class", ["Normal", "Abnormal"])
    df_selected = df_normal if category == "Normal" else df_abnormal
    
    sample_idx = st.sidebar.slider("Sample Index", min_value=0, max_value=len(df_selected)-1, value=42)
    sample_signal = df_selected.iloc[sample_idx, :-1].values.astype(np.float32)
    ground_truth = int(df_selected.iloc[sample_idx, -1])
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV containing 187 float values", type=["csv", "txt"])
    if uploaded_file is not None:
        raw_data = pd.read_csv(uploaded_file, header=None)
        sample_signal = raw_data.iloc[0, :187].values.astype(np.float32)

if sample_signal is None:
    st.info("Select or upload an ECG waveform from the sidebar to begin analysis.")
    st.stop()

# ---------------------------------------------------------
# 6. INFERENCE & SALIENCY COMPUTATION
# ---------------------------------------------------------
tensor_in = torch.tensor(sample_signal).unsqueeze(0).unsqueeze(0) # Shape: (1, 1, 187)
logits = model(tensor_in)
probs = torch.softmax(logits, dim=1).detach().squeeze().numpy()
pred_class = int(np.argmax(probs))
pred_label = "Abnormal" if pred_class == 1 else "Normal"
confidence = probs[pred_class] * 100

saliency_map = compute_saliency(model, tensor_in, pred_class)

# ---------------------------------------------------------
# 7. METRICS ROW
# ---------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Predicted Diagnosis", value=pred_label, delta=f"{confidence:.1f}% confidence")

with col2:
    if ground_truth is not None:
        actual_label = "Abnormal" if ground_truth == 1 else "Normal"
        is_match = "Match ✅" if actual_label == pred_label else "Mismatch ⚠️"
        st.metric(label="Ground Truth", value=actual_label, delta=is_match)
    else:
        st.metric(label="Ground Truth", value="N/A")

with col3:
    st.metric(label="Normal Class Probability", value=f"{probs[0]*100:.2f}%")

with col4:
    st.metric(label="Abnormal Class Probability", value=f"{probs[1]*100:.2f}%")

st.divider()

# ---------------------------------------------------------
# 8. EXPLAINABILITY VISUALIZATIONS
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["🔍 Why Did the Model Predict This? (Attention Map)", "📊 Morphology vs Population Baselines"])

with tab1:
    st.subheader("1D Saliency Heatmap (Feature Importance)")
    st.write(
        "The background heatmap and line markers show the exact segments of the heartbeat (e.g., QRS complex, ST segment, T wave) "
        "that contributed most heavily to the classification."
    )

    x_axis = np.arange(len(sample_signal))
    
    fig_saliency = go.Figure()

    # Base ECG line
    fig_saliency.add_trace(go.Scatter(
        x=x_axis,
        y=sample_signal,
        mode="lines+markers",
        line=dict(color="#1f77b4", width=2),
        marker=dict(
            size=6,
            color=saliency_map,
            colorscale="Reds",
            showscale=True,
            colorbar=dict(title="Importance Weight")
        ),
        name="ECG Lead Signal",
        hovertemplate="Time index: %{x}<br>Amplitude: %{y:.3f}<br>Importance: %{marker.color:.3f}<extra></extra>"
    ))

    fig_saliency.update_layout(
        title=f"Sample Signal Saliency Attribution (Target: {pred_label})",
        xaxis_title="Normalized Time Step (Sampling points)",
        yaxis_title="Normalized Voltage Amplitude",
        template="plotly_white",
        height=450
    )
    st.plotly_chart(fig_saliency, use_container_width=True)

with tab2:
    st.subheader("Patient Signal vs Statistical Baselines")
    st.write("Compare the active heartbeat morphology directly against the population average ±1 standard deviation.")

    fig_comp = make_subplots(rows=1, cols=2, subplot_titles=("Comparison with Normal Baseline", "Comparison with Abnormal Baseline"))

    # Left plot: Normal Baseline
    fig_comp.add_trace(go.Scatter(
        x=x_axis, y=baselines["normal_mean"] + baselines["normal_std"],
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ), row=1, col=1)
    fig_comp.add_trace(go.Scatter(
        x=x_axis, y=baselines["normal_mean"] - baselines["normal_std"],
        mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(46, 204, 113, 0.2)',
        name='Normal ±1σ Band', hoverinfo='skip'
    ), row=1, col=1)
    fig_comp.add_trace(go.Scatter(
        x=x_axis, y=baselines["normal_mean"],
        mode='lines', line=dict(color='#27ae60', dash='dash', width=2), name='Mean Normal'
    ), row=1, col=1)
    fig_comp.add_trace(go.Scatter(
        x=x_axis, y=sample_signal,
        mode='lines', line=dict(color='#2c3e50', width=2), name='Current Sample'
    ), row=1, col=1)

    # Right plot: Abnormal Baseline
    fig_comp.add_trace(go.Scatter(
        x=x_axis, y=baselines["abnormal_mean"] + baselines["abnormal_std"],
        mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
    ), row=1, col=2)
    fig_comp.add_trace(go.Scatter(
        x=x_axis, y=baselines["abnormal_mean"] - baselines["abnormal_std"],
        mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(231, 76, 60, 0.2)',
        name='Abnormal ±1σ Band', hoverinfo='skip'
    ), row=1, col=2)
    fig_comp.add_trace(go.Scatter(
        x=x_axis, y=baselines["abnormal_mean"],
        mode='lines', line=dict(color='#c0392b', dash='dash', width=2), name='Mean Abnormal'
    ), row=1, col=2)
    fig_comp.add_trace(go.Scatter(
        x=x_axis, y=sample_signal,
        mode='lines', line=dict(color='#2c3e50', width=2), showlegend=False
    ), row=1, col=2)

    fig_comp.update_layout(
        template="plotly_white",
        height=480,
        xaxis_title="Time Step",
        yaxis_title="Normalized Amplitude"
    )
    st.plotly_chart(fig_comp, use_container_width=True)