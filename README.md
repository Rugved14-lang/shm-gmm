# GMM Clustering for Mode Identification in Structural Health Monitoring

![SHM-GMM Banner](https://via.placeholder.com/900x200?text=SHM+GMM+Clustering+App)

> **A Streamlit application that applies Gaussian Mixture Model (GMM) clustering to identify structural vibration modes from acceleration signals — a core task in Structural Health Monitoring (SHM).**

---

## 📸 Screenshot

*Run the app and explore the interactive dashboard in your browser.*

---

## ✨ Features

- **Synthetic SHM signal generation** mimicking a 3-DOF building system (LANL benchmark-inspired)
- **CSV upload** for real acceleration data
- **Welch PSD** frequency analysis with interactive log-scale plot
- **Automatic peak detection** using `scipy.signal.find_peaks`
- **Half-power bandwidth** estimation for damping ratios (ζ ≈ Δf / 2f₀)
- **GMM clustering** with automatic model selection via BIC
- **BIC curve** visualization to show optimal number of components
- **Cluster scatter plot** with frequency × amplitude × damping bubble chart
- **Per-cluster summary table** with modal statistics
- Fully interactive **Plotly** charts throughout

---

## 🚀 Installation & Running

```bash
# 1. Clone or download this repository
git clone https://github.com/yourname/shm-gmm.git
cd shm-gmm

# 2. (Optional) Create a virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 📊 Dataset

### Synthetic Data (Default)

The default synthetic signal mimics the **Los Alamos National Laboratory (LANL) 3-story building SHM benchmark** dataset. It models a 3-DOF shear-frame structure with:

| Mode | Frequency | Damping Ratio |
|------|-----------|---------------|
| 1    | ~2.5 Hz   | 1.0 %         |
| 2    | ~7.1 Hz   | 1.5 %         |
| 3    | ~12.4 Hz  | 2.0 %         |

Each mode is generated as a damped harmonic oscillator response to white noise excitation, with random phase restarts to simulate stationary ambient vibration. Additive Gaussian noise models sensor noise.

### Real Data

Upload any CSV file with a column of acceleration values recorded at a known sampling rate.

### Citations

> K. Worden, G. Manson, and N. Fieller, "Damage detection using outlier analysis," *Journal of Sound and Vibration*, 229(3), pp. 647–667, 2000.

> Los Alamos National Laboratory SHM Benchmark Dataset:  
> http://institute.lanl.gov/ei/software-and-data/

> S. W. Doebling, C. R. Farrar, M. B. Prime, and D. W. Shevitz, "Damage identification and health monitoring of structural and mechanical systems from changes in their vibration characteristics: A literature review," Los Alamos National Laboratory Report LA-13070-MS, 1996.

---

## ⚙️ How It Works

```
Acceleration Signal
       │
       ▼
  Welch PSD (scipy.signal.welch)
       │
       ▼
  Peak Detection (scipy.signal.find_peaks)
       │  → Frequency, Log-Amplitude, Damping Ratio per peak
       ▼
  Feature Normalization
       │
       ▼
  GMM Fit (n=1..max_components)
       │  → BIC for each n → pick best n
       ▼
  Cluster Assignment & Visualization
       │
       ▼
  Modal Parameter Summary Table
```

The **Gaussian Mixture Model** treats each detected spectral peak as a data point with features `[frequency, log-amplitude, damping]`. It groups peaks into clusters that correspond to structural vibration modes. BIC (Bayesian Information Criterion) automatically selects the number of clusters, penalizing model complexity.

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you would like to change.
