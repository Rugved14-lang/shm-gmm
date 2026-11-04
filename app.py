# =============================================================================
# GMM Clustering for Mode Identification in Structural Health Monitoring
# =============================================================================
# A Streamlit application that:
#   1. Generates (or loads) acceleration signals from a structure
#   2. Computes the Power Spectral Density (Welch method)
#   3. Detects spectral peaks and estimates damping via half-power bandwidth
#   4. Clusters peaks using a Gaussian Mixture Model (GMM) selected by BIC
#   5. Visualises results interactively with Plotly
#
# Synthetic data mimics the LANL 3-story shear-frame SHM benchmark.
# =============================================================================

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import welch, find_peaks
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SHM GMM Clustering",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for a cleaner look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container { padding-top: 1.5rem; }
        h1 { color: #1f4e79; }
        h2 { color: #2e75b6; border-bottom: 2px solid #2e75b6; padding-bottom: 4px; }
        h3 { color: #2e75b6; }
        .stMetric > div { background: #f0f4f8; border-radius: 8px; padding: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# SECTION 1 — Header
# ===========================================================================
st.title("🏗️ GMM Clustering for Mode Identification in SHM")
st.markdown(
    """
    **Structural Health Monitoring (SHM)** aims to detect damage in civil or mechanical 
    structures by continuously analysing sensor data. A key task is **operational modal 
    analysis (OMA)**: identifying the natural frequencies, mode shapes, and damping ratios 
    of a structure from ambient vibration measurements — without a known excitation force.

    This application demonstrates how **Gaussian Mixture Models (GMM)** can automatically 
    cluster spectral peaks extracted from acceleration signals and identify structural 
    vibration modes. The workflow mirrors research conducted with the 
    *Los Alamos National Laboratory (LANL) 3-story building benchmark* dataset.
    """,
    unsafe_allow_html=False,
)

# ===========================================================================
# SECTION 2 — Sidebar controls
# ===========================================================================
with st.sidebar:
    st.header("⚙️ Controls")

    # --- Data source ---
    st.subheader("Data Source")
    data_source = st.radio(
        "Choose data source",
        ["Synthetic SHM Signal", "Upload CSV"],
        index=0,
    )

    uploaded_file = None
    csv_column = None
    csv_fs = 200.0

    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload acceleration CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                _preview_df = pd.read_csv(uploaded_file)
                uploaded_file.seek(0)
                numeric_cols = _preview_df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    csv_column = st.selectbox("Select acceleration column", numeric_cols)
                    csv_fs = st.number_input(
                        "Sampling rate (Hz)", min_value=10.0, max_value=10000.0, value=200.0, step=10.0
                    )
                else:
                    st.error("❌ No numeric columns found in the uploaded CSV.")
                    csv_column = None
            except Exception as e:
                st.error(f"❌ Could not read CSV: {e}")

    st.divider()

    # --- Signal parameters (synthetic) ---
    st.subheader("Signal Parameters")
    duration = st.slider("Duration (s)", min_value=10, max_value=300, value=60, step=10)
    fs = st.slider("Sampling rate (Hz)", min_value=50, max_value=1000, value=200, step=50)
    noise_level = st.slider(
        "Noise level (σ)", min_value=0.001, max_value=0.5, value=0.05, step=0.005, format="%.3f"
    )
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    st.divider()

    # --- Frequency analysis ---
    st.subheader("Frequency Analysis")
    nperseg_exp = st.slider(
        "FFT window (2^n samples)", min_value=8, max_value=15, value=11, step=1
    )
    nperseg = 2**nperseg_exp
    st.caption(f"→ nperseg = {nperseg} samples  ({nperseg/fs:.2f} s window)")

    peak_prominence = st.slider(
        "Peak prominence threshold", min_value=0.01, max_value=10.0, value=0.5, step=0.01, format="%.2f"
    )

    st.divider()

    # --- GMM ---
    st.subheader("GMM Settings")
    max_components = st.slider(
        "Max GMM components", min_value=2, max_value=10, value=6, step=1
    )

# ===========================================================================
# SECTION 3 — Data Generation / Loading
# ===========================================================================

@st.cache_data
def generate_synthetic_shm(duration=60, fs=200, noise_level=0.05, seed=42):
    """
    Mimics a 3-DOF building system excited by ambient / white noise.

    Modal frequencies: ~2.5, 7.1, 12.4 Hz (approx LANL 3-story benchmark).
    Damping ratios:    1%, 1.5%, 2%.

    Each mode is modelled as a damped harmonic oscillator with 20 random 
    phase restarts to simulate stationary ambient vibration conditions.
    Additive white Gaussian noise models sensor measurement noise.
    """
    np.random.seed(seed)
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)

    # Mode parameters: (natural frequency Hz, damping ratio, amplitude)
    modes = [(2.5, 0.01, 1.0), (7.1, 0.015, 0.6), (12.4, 0.02, 0.35)]

    signal = np.zeros_like(t)
    for f0, zeta, amp in modes:
        omega0 = 2 * np.pi * f0
        # Primary decayed harmonic — contributes initial energy
        phase = np.random.uniform(0, 2 * np.pi)
        env = np.exp(-zeta * omega0 * t)
        signal += amp * env * np.cos(omega0 * t + phase)

        # 20 random restarts simulate stationary ambient excitation
        for restart in np.random.uniform(0, duration, 20):
            idx = int(restart * fs)
            if idx >= len(t):
                continue
            t_local = t[idx:] - t[idx]
            env_r = np.exp(-zeta * omega0 * t_local)
            phase_r = np.random.uniform(0, 2 * np.pi)
            pad = np.zeros(idx)
            contribution = np.concatenate(
                [pad, amp * env_r * np.cos(omega0 * t_local + phase_r)]
            )[: len(t)]
            signal += 0.3 * contribution

    # Additive measurement noise
    signal += noise_level * np.random.randn(len(t))
    return t, signal


@st.cache_data
def load_csv_data(file_bytes, column, fs):
    """Load acceleration data from an uploaded CSV file."""
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    acc = df[column].dropna().values.astype(float)
    t = np.arange(len(acc)) / fs
    return t, acc


# Acquire time-series data
if data_source == "Synthetic SHM Signal":
    t, acc = generate_synthetic_shm(
        duration=int(duration), fs=int(fs), noise_level=float(noise_level), seed=int(seed)
    )
    signal_fs = fs
    st.success(
        f"✅ Synthetic signal generated — {len(t):,} samples at {fs} Hz over {duration} s"
    )
else:
    if uploaded_file is not None and csv_column is not None:
        file_bytes = uploaded_file.read()
        t, acc = load_csv_data(file_bytes, csv_column, csv_fs)
        signal_fs = csv_fs
        st.success(f"✅ Loaded '{csv_column}' — {len(t):,} samples at {csv_fs} Hz")
    else:
        st.warning("⚠️ Please upload a CSV file and select a column, or switch to Synthetic mode.")
        st.stop()

# ===========================================================================
# SECTION 4 — Time-domain Plot
# ===========================================================================
st.header("📈 Time-Domain Signal")

# Show at most 5 s of data for readability; full signal used for PSD
max_display_samples = int(min(5.0, t[-1]) * signal_fs)
fig_time = go.Figure()
fig_time.add_trace(
    go.Scattergl(
        x=t[:max_display_samples],
        y=acc[:max_display_samples],
        mode="lines",
        line=dict(color="#2e75b6", width=1),
        name="Acceleration",
    )
)
fig_time.update_layout(
    xaxis_title="Time (s)",
    yaxis_title="Acceleration (g or units)",
    title=f"Acceleration Signal (first {min(5, int(t[-1]))} s shown)",
    height=320,
    margin=dict(l=60, r=20, t=50, b=50),
    hovermode="x unified",
)
st.plotly_chart(fig_time, use_container_width=True)

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.metric("Total Duration", f"{t[-1]:.1f} s")
col_m2.metric("Samples", f"{len(t):,}")
col_m3.metric("Sampling Rate", f"{signal_fs:.0f} Hz")
col_m4.metric("RMS Amplitude", f"{np.sqrt(np.mean(acc**2)):.4f}")

# ===========================================================================
# SECTION 5 — Frequency Analysis (Welch PSD)
# ===========================================================================
st.header("🔊 Power Spectral Density (Welch Method)")

@st.cache_data
def compute_psd(acc_arr, fs, nperseg):
    """Compute one-sided Welch PSD of the acceleration signal."""
    freqs, psd = welch(acc_arr, fs=fs, nperseg=min(nperseg, len(acc_arr) // 2))
    return freqs, psd


freqs, psd = compute_psd(acc, signal_fs, nperseg)

# Display only up to Nyquist / 2 for relevance (keep full for damping calc)
nyq = signal_fs / 2.0
mask = freqs <= nyq
f_plot = freqs[mask]
p_plot = psd[mask]

fig_psd = go.Figure()
fig_psd.add_trace(
    go.Scatter(
        x=f_plot,
        y=p_plot,
        mode="lines",
        line=dict(color="#1a7f4e", width=1.5),
        name="PSD",
    )
)
fig_psd.update_layout(
    xaxis_title="Frequency (Hz)",
    yaxis_title="PSD [(units)²/Hz]",
    yaxis_type="log",
    title="Welch Power Spectral Density (log scale)",
    height=350,
    margin=dict(l=60, r=20, t=50, b=50),
    hovermode="x unified",
)
st.plotly_chart(fig_psd, use_container_width=True)

with st.expander("ℹ️ About the Welch Method"):
    st.markdown(
        f"""
        The **Welch method** estimates the PSD by dividing the signal into overlapping 
        segments of `nperseg = {nperseg}` samples (≈ {nperseg/signal_fs:.2f} s), applying a 
        Hann window to each, computing the FFT, and averaging the squared magnitudes.  
        Longer windows give **higher frequency resolution** but more variance per estimate; 
        shorter windows give **smoother** estimates at lower resolution.  
        Frequency resolution here: **Δf = {signal_fs/nperseg:.4f} Hz**.
        """
    )

# ===========================================================================
# SECTION 6 — Peak Feature Extraction
# ===========================================================================
st.header("🔍 Spectral Peak Detection & Feature Extraction")

@st.cache_data
def extract_peaks(freqs, psd, prominence_threshold, fs):
    """
    Detect peaks in PSD and compute per-peak features:
      - frequency (Hz)
      - log10 amplitude
      - half-power bandwidth → damping ratio ζ ≈ Δf / (2 f₀)
    """
    log_psd = np.log10(np.maximum(psd, 1e-30))
    peak_indices, props = find_peaks(
        log_psd,
        prominence=prominence_threshold,
        height=np.percentile(log_psd, 50),  # ignore very low-energy peaks
    )

    records = []
    for idx in peak_indices:
        f0 = freqs[idx]
        p0 = psd[idx]
        half_power = p0 / 2.0

        # Scan left for lower half-power crossing
        left_idx = idx
        while left_idx > 0 and psd[left_idx] > half_power:
            left_idx -= 1

        # Scan right for upper half-power crossing
        right_idx = idx
        while right_idx < len(psd) - 1 and psd[right_idx] > half_power:
            right_idx += 1

        delta_f = freqs[right_idx] - freqs[left_idx]
        if f0 > 0:
            zeta = delta_f / (2.0 * f0)
        else:
            zeta = np.nan

        records.append(
            {
                "peak_idx": int(idx),
                "frequency_hz": float(f0),
                "psd_amplitude": float(p0),
                "log_amplitude": float(np.log10(max(p0, 1e-30))),
                "bandwidth_hz": float(delta_f),
                "damping_ratio": float(np.clip(zeta, 0, 0.5)),
            }
        )

    return pd.DataFrame(records)


peaks_df = extract_peaks(freqs, psd, float(peak_prominence), signal_fs)

if peaks_df.empty:
    st.warning(
        "⚠️ No peaks detected. Try reducing the **Peak prominence threshold** in the sidebar."
    )
    st.stop()

# Overlay peaks on PSD chart
fig_psd_peaks = go.Figure()
fig_psd_peaks.add_trace(
    go.Scatter(
        x=f_plot,
        y=p_plot,
        mode="lines",
        line=dict(color="#1a7f4e", width=1.5),
        name="PSD",
    )
)
fig_psd_peaks.add_trace(
    go.Scatter(
        x=peaks_df["frequency_hz"],
        y=peaks_df["psd_amplitude"],
        mode="markers",
        marker=dict(color="red", size=10, symbol="triangle-up"),
        name="Detected Peaks",
        hovertemplate=(
            "<b>f₀</b> = %{x:.3f} Hz<br>"
            "<b>PSD</b> = %{y:.3e}<br>"
            "<extra></extra>"
        ),
    )
)
fig_psd_peaks.update_layout(
    xaxis_title="Frequency (Hz)",
    yaxis_title="PSD [(units)²/Hz]",
    yaxis_type="log",
    title="Welch PSD with Detected Peaks",
    height=380,
    margin=dict(l=60, r=20, t=50, b=50),
)
st.plotly_chart(fig_psd_peaks, use_container_width=True)

col_p1, col_p2, col_p3 = st.columns(3)
col_p1.metric("Peaks Detected", len(peaks_df))
col_p2.metric("Freq. Range", f"{peaks_df['frequency_hz'].min():.2f} – {peaks_df['frequency_hz'].max():.2f} Hz")
col_p3.metric("Mean Damping ζ", f"{peaks_df['damping_ratio'].mean()*100:.2f} %")

with st.expander("📋 Raw Peak Data"):
    st.dataframe(
        peaks_df[["frequency_hz", "log_amplitude", "bandwidth_hz", "damping_ratio"]]
        .rename(
            columns={
                "frequency_hz": "Frequency (Hz)",
                "log_amplitude": "log₁₀(PSD)",
                "bandwidth_hz": "Bandwidth Δf (Hz)",
                "damping_ratio": "Damping ζ",
            }
        )
        .style.format(precision=4),
        use_container_width=True,
    )

# ===========================================================================
# SECTION 7 — GMM Clustering + BIC Model Selection
# ===========================================================================
st.header("🔵 Gaussian Mixture Model Clustering")

st.info(
    """
    **What is GMM doing here?**  
    Each detected spectral peak is represented as a point in a 3D feature space: 
    *[frequency, log-amplitude, damping ratio]*. 
    A **Gaussian Mixture Model** fits a probabilistic mixture of multivariate Gaussians 
    to this feature space. Each Gaussian component can represent a **structural vibration 
    mode** (or harmonic / noise cluster). The optimal number of modes is chosen automatically 
    by the **Bayesian Information Criterion (BIC)** — lower BIC means a better model 
    considering both fit quality and complexity.
    """
)

@st.cache_data
def fit_gmm_models(freq_arr, log_amp_arr, damp_arr, max_comp):
    """
    Fit GaussianMixture for n_components in 1..max_comp.
    Returns fitted models, BIC scores, and feature scaler.
    """
    # Build and normalise feature matrix
    X = np.column_stack([freq_arr, log_amp_arr, damp_arr])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    bic_scores = []
    models = []

    # Clamp max components to number of peaks available
    actual_max = min(max_comp, len(freq_arr))

    for n in range(1, actual_max + 1):
        try:
            gmm = GaussianMixture(
                n_components=n,
                covariance_type="full",
                n_init=5,
                random_state=42,
                max_iter=300,
            )
            gmm.fit(X_scaled)
            bic_scores.append(gmm.bic(X_scaled))
            models.append(gmm)
        except Exception:
            bic_scores.append(np.inf)
            models.append(None)

    best_n = int(np.argmin(bic_scores)) + 1  # 1-indexed
    best_model = models[best_n - 1]
    labels = best_model.predict(X_scaled) if best_model is not None else np.zeros(len(freq_arr), dtype=int)

    return bic_scores, best_n, labels, scaler, X_scaled, actual_max


bic_scores, best_n, labels, scaler, X_scaled, actual_max = fit_gmm_models(
    peaks_df["frequency_hz"].values,
    peaks_df["log_amplitude"].values,
    peaks_df["damping_ratio"].values,
    max_components,
)

peaks_df = peaks_df.copy()
peaks_df["cluster"] = labels.astype(str)

col_g1, col_g2, col_g3 = st.columns(3)
col_g1.metric("Best n_components (BIC)", best_n)
col_g2.metric("Min BIC", f"{min(bic_scores):.2f}")
col_g3.metric("Peaks Clustered", len(peaks_df))

# ===========================================================================
# SECTION 8 — BIC Curve
# ===========================================================================
st.subheader("📉 BIC vs Number of GMM Components")

n_range = list(range(1, actual_max + 1))
fig_bic = go.Figure()
fig_bic.add_trace(
    go.Scatter(
        x=n_range,
        y=bic_scores,
        mode="lines+markers",
        line=dict(color="#7030a0", width=2),
        marker=dict(size=8, color="#7030a0"),
        name="BIC",
    )
)
# Mark best
fig_bic.add_trace(
    go.Scatter(
        x=[best_n],
        y=[bic_scores[best_n - 1]],
        mode="markers",
        marker=dict(size=15, color="red", symbol="star", line=dict(color="darkred", width=1)),
        name=f"Best (n={best_n})",
    )
)
fig_bic.update_layout(
    xaxis_title="Number of GMM Components",
    yaxis_title="BIC Score",
    title="BIC Model Selection — lower is better",
    xaxis=dict(tickmode="linear", tick0=1, dtick=1),
    height=320,
    margin=dict(l=60, r=20, t=50, b=50),
    legend=dict(orientation="h", y=1.1),
)
st.plotly_chart(fig_bic, use_container_width=True)

# ===========================================================================
# SECTION 9 — Cluster Scatter Plot
# ===========================================================================
st.subheader("🫧 Mode Cluster Scatter (Frequency × Amplitude × Damping)")

PLOTLY_COLORS = px.colors.qualitative.Plotly
n_colors = len(PLOTLY_COLORS)

peaks_df["bubble_size"] = peaks_df["damping_ratio"] * 5000 + 5
peaks_df["cluster_label"] = "Cluster " + peaks_df["cluster"]

fig_scatter = go.Figure()
for cid in sorted(peaks_df["cluster"].unique(), key=int):
    sub = peaks_df[peaks_df["cluster"] == cid]
    color = PLOTLY_COLORS[int(cid) % n_colors]
    fig_scatter.add_trace(
        go.Scatter(
            x=sub["frequency_hz"],
            y=sub["log_amplitude"],
            mode="markers",
            marker=dict(
                size=sub["bubble_size"],
                color=color,
                opacity=0.75,
                line=dict(color="white", width=1),
            ),
            name=f"Mode {int(cid)+1}",
            hovertemplate=(
                "<b>Cluster %{text}</b><br>"
                "f₀ = %{x:.3f} Hz<br>"
                "log(PSD) = %{y:.3f}<br>"
                "ζ = %{customdata:.4f}<br>"
                "<extra></extra>"
            ),
            text=sub["cluster"],
            customdata=sub["damping_ratio"],
        )
    )

fig_scatter.update_layout(
    xaxis_title="Frequency (Hz)",
    yaxis_title="log₁₀(PSD Amplitude)",
    title=(
        "GMM Clusters in Frequency–Amplitude Space  "
        "<br><sup>Bubble size ∝ damping ratio ζ</sup>"
    ),
    height=450,
    margin=dict(l=60, r=20, t=70, b=50),
    legend_title="Mode Cluster",
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ===========================================================================
# SECTION 10 — Results Table
# ===========================================================================
st.header("📊 Cluster Summary — Modal Parameters")

summary_rows = []
for cid in sorted(peaks_df["cluster"].unique(), key=int):
    sub = peaks_df[peaks_df["cluster"] == cid]
    summary_rows.append(
        {
            "Cluster": f"Mode {int(cid)+1}",
            "Count (peaks)": len(sub),
            "Mean Frequency (Hz)": sub["frequency_hz"].mean(),
            "Std Frequency (Hz)": sub["frequency_hz"].std(),
            "Mean log₁₀(PSD)": sub["log_amplitude"].mean(),
            "Mean Damping ζ (%)": sub["damping_ratio"].mean() * 100,
            "Std Damping ζ (%)": sub["damping_ratio"].std() * 100,
        }
    )

summary_df = pd.DataFrame(summary_rows)

st.dataframe(
    summary_df.style.format(
        {
            "Mean Frequency (Hz)": "{:.3f}",
            "Std Frequency (Hz)": "{:.4f}",
            "Mean log₁₀(PSD)": "{:.3f}",
            "Mean Damping ζ (%)": "{:.3f}",
            "Std Damping ζ (%)": "{:.4f}",
        }
    ).background_gradient(cmap="Blues", subset=["Mean Frequency (Hz)"]),
    use_container_width=True,
)

# Comparison against known values (synthetic only)
if data_source == "Synthetic SHM Signal":
    st.markdown("#### ✅ Comparison with Ground-Truth Modal Frequencies")
    truth_data = {
        "Mode": ["Mode 1", "Mode 2", "Mode 3"],
        "True Frequency (Hz)": [2.5, 7.1, 12.4],
        "True Damping ζ (%)": [1.0, 1.5, 2.0],
    }
    truth_df = pd.DataFrame(truth_data)
    st.table(truth_df)
    st.caption(
        "Compare the cluster centroids in the summary table above with these "
        "ground-truth values to validate peak detection and clustering accuracy."
    )

# ===========================================================================
# SECTION 11 — Joint PSD + cluster colour overlay
# ===========================================================================
st.header("🎨 PSD with Mode-Coloured Peak Markers")

fig_final = go.Figure()
fig_final.add_trace(
    go.Scatter(
        x=f_plot,
        y=p_plot,
        mode="lines",
        line=dict(color="#888888", width=1.2),
        name="PSD",
        opacity=0.7,
    )
)
for cid in sorted(peaks_df["cluster"].unique(), key=int):
    sub = peaks_df[peaks_df["cluster"] == cid]
    color = PLOTLY_COLORS[int(cid) % n_colors]
    fig_final.add_trace(
        go.Scatter(
            x=sub["frequency_hz"],
            y=sub["psd_amplitude"],
            mode="markers",
            marker=dict(
                size=14,
                color=color,
                symbol="diamond",
                line=dict(color="white", width=1),
            ),
            name=f"Mode {int(cid)+1}",
            hovertemplate=(
                f"<b>Mode {int(cid)+1}</b><br>"
                "f₀ = %{x:.3f} Hz<br>"
                "PSD = %{y:.3e}<br>"
                "<extra></extra>"
            ),
        )
    )
fig_final.update_layout(
    xaxis_title="Frequency (Hz)",
    yaxis_title="PSD [(units)²/Hz]",
    yaxis_type="log",
    title="PSD with GMM-Identified Mode Peaks",
    height=400,
    margin=dict(l=60, r=20, t=50, b=50),
)
st.plotly_chart(fig_final, use_container_width=True)

# ===========================================================================
# SECTION 12 — Dataset Citation & Explanation
# ===========================================================================
st.header("📚 Dataset & Methodology Reference")

with st.expander("🏛️ LANL SHM Benchmark Dataset", expanded=True):
    st.markdown(
        """
        ### Los Alamos National Laboratory SHM Benchmark

        The **LANL 3-story shear-frame building** is a widely used benchmark for SHM 
        algorithm development and validation. It is a small-scale laboratory structure 
        with three stories, instrumented with accelerometers on each floor. Damage states 
        are simulated by removing bolt connections at specific joints.

        #### Synthetic Data Generation (this app)

        Because downloading the original dataset requires institutional access, this 
        application **synthetically generates** a structurally realistic acceleration signal 
        that reproduces the key modal characteristics of the LANL building:

        | Parameter | Mode 1 | Mode 2 | Mode 3 |
        |-----------|--------|--------|--------|
        | Natural Frequency | 2.5 Hz | 7.1 Hz | 12.4 Hz |
        | Damping Ratio ζ   | 1.0 %  | 1.5 %  | 2.0 %  |
        | Relative Amplitude | 1.0   | 0.6    | 0.35   |

        Each mode is modelled as a **damped harmonic oscillator** (impulse response) 
        driven by white noise, with 20 random phase restarts per mode to simulate 
        stationarity under ambient excitation. Additive Gaussian noise models sensor noise.

        #### Why GMM for OMA?

        Traditional Operational Modal Analysis methods (e.g., SSI, FDD) are well-validated 
        but computationally intensive. GMM clustering offers:
        - **Data-driven** mode identification without assuming mode count *a priori*
        - **Probabilistic** cluster assignments reflecting uncertainty
        - **Automatic** model order selection via BIC
        - Ability to group harmonics, noise clusters, and true structural modes

        #### Citations

        > **K. Worden, G. Manson, and N. Fieller**, "Damage detection using outlier analysis," 
        > *Journal of Sound and Vibration*, 229(3), pp. 647–667, 2000.

        > **Los Alamos National Laboratory SHM Benchmark Dataset**:  
        > http://institute.lanl.gov/ei/software-and-data/

        > **S. W. Doebling, C. R. Farrar, M. B. Prime, and D. W. Shevitz**, 
        > "Damage identification and health monitoring of structural and mechanical systems 
        > from changes in their vibration characteristics: A literature review,"  
        > *Los Alamos National Laboratory Report LA-13070-MS*, 1996.

        > **R. Brincker, L. Zhang, and P. Andersen**, "Modal identification of output-only 
        > systems using frequency domain decomposition,"  
        > *Smart Materials and Structures*, 10(3), pp. 441–445, 2001.
        """
    )

with st.expander("⚙️ Algorithm Pipeline"):
    st.markdown(
        """
        ```
        Raw Acceleration Signal
               │
               ▼
        Welch PSD Estimation
        (scipy.signal.welch, Hann window)
               │
               ▼
        Peak Detection (scipy.signal.find_peaks)
        → Frequency f₀, log-PSD, prominence filter
               │
               ▼
        Half-Power Bandwidth → Damping ζ ≈ Δf / (2·f₀)
               │
               ▼
        Feature Matrix  X = [f₀_norm, log_PSD_norm, ζ_norm]
               │
               ▼
        GMM Fit: n = 1, 2, ..., max_comp
        → BIC score per model
        → Select n* = argmin BIC
               │
               ▼
        Cluster Labels + Probabilistic Assignments
               │
               ▼
        Modal Parameter Summary per Cluster
        ```
        """
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "🏗️ **SHM GMM Clustering App** — Built with Streamlit · Plotly · SciPy · scikit-learn  |  "
    "Synthetic data inspired by the LANL 3-story SHM Benchmark"
)
