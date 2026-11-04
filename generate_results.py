# =============================================================================
# generate_results.py
# =============================================================================
# Runs the same pipeline as app.py (without Streamlit) and writes all data
# needed by the static website to web/data/results.json.
#
# Usage:
#   python generate_results.py
# =============================================================================

import json
import os
import numpy as np
from scipy.signal import welch, find_peaks
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Parameters (defaults matching app.py sidebar defaults)
# ---------------------------------------------------------------------------
DURATION        = 60        # seconds
FS              = 200       # Hz
NOISE_LEVEL     = 0.05
SEED            = 42
NPERSEG         = 2**11     # 2048 samples
PEAK_PROMINENCE = 0.5
MAX_COMPONENTS  = 6

OUT_DIR = os.path.join(os.path.dirname(__file__), "web", "data")
OUT_FILE = os.path.join(OUT_DIR, "results.json")

# ---------------------------------------------------------------------------
# 1. Synthetic signal generation (identical logic to app.py)
# ---------------------------------------------------------------------------

def generate_synthetic_shm(duration=60, fs=200, noise_level=0.05, seed=42):
    """
    Mimics a 3-DOF building system excited by ambient / white noise.

    Modal frequencies: ~2.5, 7.1, 12.4 Hz (approx LANL 3-story benchmark).
    Damping ratios:    1%, 1.5%, 2%.
    """
    np.random.seed(seed)
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)

    modes = [(2.5, 0.01, 1.0), (7.1, 0.015, 0.6), (12.4, 0.02, 0.35)]

    signal = np.zeros_like(t)
    for f0, zeta, amp in modes:
        omega0 = 2 * np.pi * f0
        phase = np.random.uniform(0, 2 * np.pi)
        env = np.exp(-zeta * omega0 * t)
        signal += amp * env * np.cos(omega0 * t + phase)

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

    signal += noise_level * np.random.randn(len(t))
    return t, signal


# ---------------------------------------------------------------------------
# 2. PSD (Welch)
# ---------------------------------------------------------------------------

def compute_psd(acc, fs, nperseg):
    freqs, psd = welch(acc, fs=fs, nperseg=min(nperseg, len(acc) // 2))
    return freqs, psd


# ---------------------------------------------------------------------------
# 3. Peak feature extraction (identical logic to app.py)
# ---------------------------------------------------------------------------

def extract_peaks(freqs, psd, prominence_threshold):
    log_psd = np.log10(np.maximum(psd, 1e-30))
    peak_indices, _ = find_peaks(
        log_psd,
        prominence=prominence_threshold,
        height=np.percentile(log_psd, 50),
    )

    records = []
    for idx in peak_indices:
        f0 = freqs[idx]
        p0 = psd[idx]
        half_power = p0 / 2.0

        left_idx = idx
        while left_idx > 0 and psd[left_idx] > half_power:
            left_idx -= 1

        right_idx = idx
        while right_idx < len(psd) - 1 and psd[right_idx] > half_power:
            right_idx += 1

        delta_f = freqs[right_idx] - freqs[left_idx]
        zeta = delta_f / (2.0 * f0) if f0 > 0 else float("nan")

        records.append(
            {
                "peak_idx":      int(idx),
                "frequency_hz":  float(f0),
                "psd_amplitude": float(p0),
                "log_amplitude": float(np.log10(max(p0, 1e-30))),
                "bandwidth_hz":  float(delta_f),
                "damping_ratio": float(np.clip(zeta, 0, 0.5)),
            }
        )
    return records


# ---------------------------------------------------------------------------
# 4. GMM + BIC
# ---------------------------------------------------------------------------

def fit_gmm(peaks, max_comp):
    freq_arr     = np.array([p["frequency_hz"]  for p in peaks])
    log_amp_arr  = np.array([p["log_amplitude"]  for p in peaks])
    damp_arr     = np.array([p["damping_ratio"]  for p in peaks])

    X = np.column_stack([freq_arr, log_amp_arr, damp_arr])
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    actual_max = min(max_comp, len(peaks))
    bic_scores = []
    models     = []

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
            bic_scores.append(float(gmm.bic(X_scaled)))
            models.append(gmm)
        except Exception:
            bic_scores.append(float("inf"))
            models.append(None)

    best_n     = int(np.argmin(bic_scores)) + 1
    best_model = models[best_n - 1]
    labels     = (
        best_model.predict(X_scaled).tolist()
        if best_model is not None
        else [0] * len(peaks)
    )
    return bic_scores, best_n, labels, actual_max


# ---------------------------------------------------------------------------
# 5. Build cluster summary
# ---------------------------------------------------------------------------

def cluster_summary(peaks, labels):
    clusters = {}
    for peak, lbl in zip(peaks, labels):
        clusters.setdefault(lbl, []).append(peak)

    rows = []
    for cid in sorted(clusters.keys()):
        group = clusters[cid]
        freqs  = [p["frequency_hz"]  for p in group]
        amps   = [p["log_amplitude"]  for p in group]
        damps  = [p["damping_ratio"]  for p in group]
        rows.append(
            {
                "cluster":          int(cid),
                "label":            f"Mode {cid + 1}",
                "count":            len(group),
                "mean_freq_hz":     float(np.mean(freqs)),
                "std_freq_hz":      float(np.std(freqs)),
                "mean_log_amp":     float(np.mean(amps)),
                "mean_damping_pct": float(np.mean(damps) * 100),
                "std_damping_pct":  float(np.std(damps) * 100),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 6. Time series sub-sample (first 5 s, max 2000 pts for JSON size)
# ---------------------------------------------------------------------------

def subsample_time(t, acc, fs, max_seconds=5, max_points=2000):
    n = min(int(max_seconds * fs), len(t), max_points)
    step = max(1, int(max_seconds * fs) // max_points)
    idx  = np.arange(0, min(int(max_seconds * fs), len(t)), step)[:max_points]
    return t[idx].tolist(), acc[idx].tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("⚙  Generating synthetic SHM signal …")
    t, acc = generate_synthetic_shm(
        duration=DURATION, fs=FS, noise_level=NOISE_LEVEL, seed=SEED
    )

    print("⚙  Computing Welch PSD …")
    freqs, psd = compute_psd(acc, FS, NPERSEG)

    # Restrict to Nyquist for display
    nyq  = FS / 2.0
    mask = freqs <= nyq
    f_plot = freqs[mask].tolist()
    p_plot = psd[mask].tolist()

    print("⚙  Extracting spectral peaks …")
    peaks = extract_peaks(freqs, psd, PEAK_PROMINENCE)

    if not peaks:
        raise RuntimeError("No peaks detected — lower PEAK_PROMINENCE or increase DURATION.")

    print(f"   → {len(peaks)} peaks detected")

    print("⚙  Fitting GMM models (BIC selection) …")
    bic_scores, best_n, labels, actual_max = fit_gmm(peaks, MAX_COMPONENTS)

    # Attach cluster labels to peaks
    for peak, lbl in zip(peaks, labels):
        peak["cluster"] = int(lbl)

    summary = cluster_summary(peaks, labels)

    print("⚙  Sub-sampling time series …")
    t_sub, acc_sub = subsample_time(t, acc, FS)

    # --- Assemble JSON payload ---
    payload = {
        "meta": {
            "duration_s":    DURATION,
            "fs_hz":         FS,
            "noise_level":   NOISE_LEVEL,
            "seed":          SEED,
            "nperseg":       NPERSEG,
            "peak_prominence": PEAK_PROMINENCE,
            "max_components": MAX_COMPONENTS,
            "n_samples":     len(t),
            "rms_amplitude": float(np.sqrt(np.mean(acc**2))),
        },
        "time_series": {
            "t":   t_sub,
            "acc": acc_sub,
        },
        "psd": {
            "freqs": f_plot,
            "psd":   p_plot,
        },
        "peaks":          peaks,
        "bic": {
            "n_range":    list(range(1, actual_max + 1)),
            "bic_scores": bic_scores,
            "best_n":     best_n,
        },
        "cluster_summary": summary,
        "ground_truth_modes": [
            {"mode": "Mode 1", "freq_hz": 2.5,  "damping_pct": 1.0},
            {"mode": "Mode 2", "freq_hz": 7.1,  "damping_pct": 1.5},
            {"mode": "Mode 3", "freq_hz": 12.4, "damping_pct": 2.0},
        ],
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_FILE, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"✅ Results written to {OUT_FILE}")
    print(f"   Peaks: {len(peaks)}  |  Best n_components: {best_n}  |  Min BIC: {min(bic_scores):.2f}")


if __name__ == "__main__":
    main()
