# 🚀 PROJECT EXECUTION GUIDE

Complete step-by-step guide to run the House Price Intelligence System.

---

## 📋 Prerequisites

- **Python 3.10+**
- **8GB RAM minimum** (16GB recommended for embeddings)
- **~2GB disk space** for artifacts
- **Internet connection** (for downloading DistilBERT on first run)

---

## ⚙️ Setup Instructions

### 1. Create Virtual Environment

```bash
# Navigate to project directory
cd house-price-intelligence

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Installation time**: ~3-5 minutes

---

## 🎯 Phase-by-Phase Execution

### Phase 1: Train Baseline Models (Structured Features Only)

**Purpose**: Establish performance baseline before adding NLP features.

```bash
python src/training/train_baseline.py
```

**Expected runtime**: 2-3 minutes

**What happens**:
1. Loads California Housing dataset (20,640 rows)
2. Generates property descriptions
3. Splits into train/val/test (70/15/15)
4. Engineers features (ratios, distances, location clusters)
5. Trains 4 baseline models (Linear, Ridge, ElasticNet, Random Forest)
6. Evaluates on validation set
7. Saves models and metrics

**Expected output**:
```
BASELINE TRAINING COMPLETE
Best model: random_forest
  Val RMSE: $65,000
  Val R²: 0.68
```

**Artifacts created**:
- `data/processed/train.csv`, `val.csv`, `test.csv`
- `artifacts/models/baseline_*.pkl`
- `artifacts/reports/baseline_metrics.json`
- `artifacts/reports/pred_vs_actual_*.png`
- `artifacts/preprocessors/structured_preprocessor.pkl`
- `artifacts/preprocessors/location_kmeans.pkl`

---

### Phase 2: Train Hybrid Models (Structured + NLP)

**Purpose**: Add transformer embeddings and build the full hybrid system.

```bash
python src/training/train_hybrid.py
```

**Expected runtime**: 
- **First run**: 10-15 minutes (downloads DistilBERT, generates embeddings)
- **Subsequent runs**: 3-5 minutes (uses cached embeddings)

**What happens**:
1. Loads preprocessed splits
2. Generates DistilBERT embeddings for descriptions
3. PCA-reduces embeddings (768 → 50 dims)
4. Fuses structured + text features
5. Trains hybrid models (ElasticNet, Random Forest, MLP)
6. Builds retrieval index
7. Generates SHAP explanations
8. Estimates prediction intervals
9. Compares baseline vs hybrid

**Expected output**:
```
HYBRID TRAINING COMPLETE
Best model: hybrid_random_forest
  Val RMSE: $62,000
  Val R²: 0.71
Text vs Structured contribution: {'text_share_pct': 15.2}
```

**Artifacts created**:
- `artifacts/embeddings/embeddings_*.npy` (cached)
- `artifacts/embeddings/embeddings_*_pca.npy`
- `artifacts/models/hybrid_*.pkl`
- `artifacts/retrieval/listing_retriever.pkl`
- `artifacts/models/shap_explainer.pkl`
- `artifacts/reports/full_comparison.json`
- `artifacts/reports/shap_importance.png`
- `artifacts/reports/model_comparison_*.png`

---

### Phase 3: Run Inference Demo

**Purpose**: Test the end-to-end inference pipeline.

```bash
python src/inference/predictor.py
```

**Expected output**:
```
============================================================
INFERENCE DEMO
============================================================
Predicted Price: $450,000
Price Interval (80%): $410,000 – $490,000

Top Similar Listings:
  #1: $445,000 (similarity: 94.5%)
  #2: $458,000 (similarity: 92.1%)
  #3: $432,000 (similarity: 89.7%)

Top Explanation Features:
  median_income: +0.3245 ↑
  distance_to_coast: -0.1823 ↓
  location_cluster: +0.1456 ↑
============================================================
```

---

### Phase 4: Launch Streamlit Demo (Optional)

**Purpose**: Interactive web UI for exploring predictions.

```bash
streamlit run app/streamlit_app.py
```

**Expected output**:
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.X:8501
```

Open the URL in your browser. The app provides:
- Interactive input form for property features
- Real-time predictions with intervals
- Similar listings table
- SHAP explanations

---

### Phase 5: Run Tests

**Purpose**: Validate code correctness.

```bash
pytest tests/ -v
```

**Expected output**:
```
tests/test_data.py::test_load_california_housing PASSED
tests/test_data.py::test_validate_dataset PASSED
tests/test_features.py::test_ratio_features PASSED
tests/test_features.py::test_preprocessor_pipeline PASSED
tests/test_models.py::test_baseline_models_build PASSED
tests/test_models.py::test_listing_retriever PASSED

============ 6 passed in 12.3s ============
```

---

## 📊 Validating Results

### Check Model Performance

```python
from src.utils.io import load_json

# Load full comparison
results = load_json("artifacts/reports/full_comparison.json")

# Print all models
for model, metrics in results.items():
    print(f"{model}: RMSE=${metrics['val_rmse']:,.0f}, R²={metrics['val_r2']:.4f}")
```

### Expected Metrics Ranges

| Metric | Baseline Range | Hybrid Range | Target |
|--------|---------------|--------------|--------|
| Val RMSE | $63K - $70K | $60K - $66K | <$65K |
| Val R² | 0.62 - 0.68 | 0.68 - 0.73 | >0.65 |
| Val MAE | $43K - $50K | $41K - $47K | <$48K |
| Val MAPE | 19% - 23% | 18% - 21% | <20% |

**If metrics are outside these ranges**:
- Check for errors in console output
- Verify data files in `data/processed/`
- Ensure random seeds are consistent (42)

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution**: Ensure you're running from the project root and `src/` exists.

```bash
# Verify you're in the right directory
pwd  # should end with house-price-intelligence

# Check src exists
ls -la src/
```

### Issue: "OSError: [Errno 28] No space left on device"

**Solution**: DistilBERT model is ~250MB. Free up disk space or use a different cache dir.

```bash
export TRANSFORMERS_CACHE=/path/to/large/disk
```

### Issue: "RuntimeError: CUDA out of memory"

**Solution**: DistilBERT runs on CPU by default. If you modified the device setting:

```python
# In src/config/config.py, ensure:
device: str = "cpu"
```

### Issue: Embeddings generation is very slow

**Expected**: ~8-10 minutes for 20K descriptions on CPU, ~2-3 minutes on GPU.

**To speed up**: 
- Use GPU if available (set `device: "cuda"` in config)
- Embeddings are cached after first run

### Issue: "FileNotFoundError: No such file or directory: 'artifacts/models/...'"

**Solution**: Run `train_baseline.py` before `train_hybrid.py`.

```bash
# Correct order:
python src/training/train_baseline.py   # First
python src/training/train_hybrid.py     # Second
```

---

## 📈 Artifacts Directory Structure (After Full Run)

```
artifacts/
├── models/                          (~50 MB)
│   ├── baseline_linear_regression.pkl
│   ├── baseline_ridge.pkl
│   ├── baseline_elasticnet.pkl
│   ├── baseline_random_forest.pkl
│   ├── hybrid_elasticnet.pkl
│   ├── hybrid_random_forest.pkl
│   ├── hybrid_mlp.pt
│   └── shap_explainer.pkl
├── embeddings/                      (~400 MB)
│   ├── embeddings_train.npy
│   ├── embeddings_val.npy
│   ├── embeddings_test.npy
│   ├── embeddings_train_pca.npy
│   ├── embeddings_val_pca.npy
│   └── embeddings_test_pca.npy
├── preprocessors/                   (~5 MB)
│   ├── structured_preprocessor.pkl
│   ├── location_kmeans.pkl
│   ├── embedding_pca.pkl
│   └── interval_bounds.json
├── retrieval/                       (~100 MB)
│   └── listing_retriever.pkl
└── reports/                         (~10 MB)
    ├── baseline_metrics.json
    ├── full_comparison.json
    ├── feature_importance.json
    ├── text_contribution.json
    ├── interval_stats.json
    ├── training_metadata.json
    ├── pred_vs_actual_*.png
    ├── residuals_*.png
    ├── shap_importance.png
    └── model_comparison_*.png
```

**Total size**: ~565 MB

---

## 🎓 Next Steps After Successful Run

1. **Explore artifacts**: Check plots in `artifacts/reports/`
2. **Read metrics**: Review `full_comparison.json`
3. **Try inference**: Modify inputs in `predictor.py` demo
4. **Launch Streamlit**: Interactive demo for presentations
5. **Review interview guide**: `INTERVIEW_GUIDE.md`

---

## ⏱️ Expected Timeline Summary

| Phase | Runtime | Cumulative |
|-------|---------|------------|
| Setup (install deps) | 5 min | 5 min |
| Train baseline | 3 min | 8 min |
| Train hybrid (first run) | 12 min | 20 min |
| Run inference | 30 sec | 20.5 min |
| Run tests | 15 sec | 21 min |
| Launch Streamlit | 10 sec | 21 min |

**Total first-run time**: ~21 minutes

**Subsequent runs** (embeddings cached): ~6 minutes

---

## 💾 Disk Space Requirements

- Project code: ~5 MB
- Dependencies (venv): ~1.5 GB
- DistilBERT model (cached): ~250 MB
- Generated artifacts: ~565 MB
- **Total**: ~2.3 GB

---

## 🔄 Retraining from Scratch

To start fresh (delete all artifacts and retrain):

```bash
# Delete artifacts
rm -rf artifacts/models/*
rm -rf artifacts/embeddings/*
rm -rf artifacts/preprocessors/*
rm -rf artifacts/retrieval/*
rm -rf artifacts/reports/*
rm -rf data/interim/*
rm -rf data/processed/*

# Retrain
python src/training/train_baseline.py
python src/training/train_hybrid.py
```

---

## 📞 Support

If you encounter issues not covered here:

1. Check console output for error messages
2. Verify file paths are correct
3. Ensure dependencies are fully installed
4. Check Python version: `python --version` (should be 3.10+)

---

**End of Execution Guide. You're ready to build! 🚀**
