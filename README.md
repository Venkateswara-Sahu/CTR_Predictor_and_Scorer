# 🎯 CTR Prediction and Ad Scoring System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io/)

A comprehensive **Click-Through Rate (CTR) Prediction** and **Ad Scoring System** developed for the Predictive Analysis course. This project implements a complete machine learning pipeline from data processing to production deployment.

## 📊 Project Overview

This system predicts click-through rates for online display advertisements using advanced machine learning techniques and provides intelligent ad scoring and ranking capabilities.

### 🎓 Academic Project Details
- **Course**: Predictive Analysis (Term 7)
- **Dataset**: Criteo Display Advertising Challenge (10M samples)
- **Dataset Link**: https://drive.google.com/drive/folders/1uKuB31FGgp3ldpWEvWvgyTOSAdrWegHl?usp=sharing
- **Live Deployment**: https://ctrpredictor.streamlit.app/
- **Objective**: Build production-ready CTR prediction and ad ranking system
- **Date**: November 2025

## ✨ Key Features

### 🤖 Machine Learning Pipeline
- **Advanced Models**: XGBoost, LightGBM, and Ensemble methods
- **Feature Engineering**: 150 engineered features from 39 raw features
- **High Performance**: AUC 0.8984 on test set
- **CTR Lift**: +265.6% improvement for top 10% ads

### 🚀 Production Deployment
- **REST API**: Flask-based web service with 5 endpoints
- **Web Interface**: Interactive Streamlit application
- **Real-time Predictions**: Sub-second inference times
- **Scalable Architecture**: Production-ready deployment

### 📊 Comprehensive Analytics
- **Multi-dimensional Scoring**: Placement, device, time, engagement factors
- **Batch Processing**: Handle thousands of ads simultaneously
- **Advanced Visualizations**: Performance dashboards and insights
- **Model Interpretability**: Feature importance and what-if analysis

## 📁 Project Structure

```
Predictive_Analysis_Project/
│
├── 📁 src/                          # Source code
├── 📁 notebooks/                    # Research notebooks
├── 📁 docs/                         # Documentation
├── 📁 tests/                        # Unit tests
├── 📁 data/                         # Datasets
│   ├── raw/                        # Raw data
│   └── processed/                  # Processed features
├── 📁 models/                      # Trained models
├── 📁 results/                     # Outputs & visualizations
├── 📁 deployment/                  # Production code
│
├── 📄 README.md                    # This file
├── 📄 requirements.txt             # Dependencies
└── 📁 .venv/                      # Virtual environment
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip package manager
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Venkateswara-Sahu/CTR_Predictor_and_Scorer.git
   cd CTR_Predictor_and_Scorer
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Web Interface (Recommended)
```bash
streamlit run deployment/streamlit_app.py
```
Navigate to `http://localhost:8501` for the interactive web application.

#### REST API
```bash
python deployment/app.py
```
API will be available at `http://localhost:5000`

#### Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

## 📊 Model Performance

| Model | AUC Score | Log Loss | CTR Lift |
|-------|-----------|----------|----------|
| XGBoost | 0.9067 | 0.3105 | +265.6% |
| LightGBM | 0.9024 | 0.3180 | +265.6% |
| Ensemble | 0.9067 | 0.3106 | +265.6% |

- **Dataset Size**: 10M samples (7M train, 2M test, 1M validation)
- **Features**: 150 engineered features
- **Training Time**: ~30 minutes
- **Inference Speed**: < 0.5 seconds per prediction

## 🔧 Technology Stack

### Machine Learning
- **Frameworks**: scikit-learn, XGBoost, LightGBM
- **Optimization**: Optuna for hyperparameter tuning
- **Evaluation**: Comprehensive metrics (AUC, Log Loss, Precision@K)

### Production
- **API Framework**: Flask with CORS support
- **Web Interface**: Streamlit
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly

### Development
- **Version Control**: Git
- **Environment**: Python virtual environments
- **Documentation**: Markdown, Jupyter notebooks

## 📚 Documentation

- **[Project Report](docs/project_report.md)** - Complete technical report
- **[API Documentation](deployment/README.md)** - REST API reference
- **[Deployment Guide](docs/STREAMLIT_DEPLOYMENT.md)** - Production deployment
- **[Research Notebooks](notebooks/)** - Detailed implementation

## 🎯 Business Impact

### Problem Solved
- **Challenge**: Predicting which ads users are likely to click
- **Solution**: Machine learning system with 89.84% AUC accuracy
- **Impact**: Optimize ad display to increase click-through rates

### Key Benefits
1. **Revenue Optimization**: Better ad targeting and bidding
2. **Cost Reduction**: Reduced wasted impressions
3. **User Experience**: More relevant advertisements
4. **Competitive Advantage**: State-of-the-art CTR prediction

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Criteo Labs (Display Advertising Challenge)
- **Libraries**: scikit-learn, XGBoost, LightGBM, Flask, Streamlit
- **Course**: Predictive Analysis - Term 7

## 📞 Contact

**Student**: Venkateswara Sahu
**Course**: Predictive Analysis - Term 7
**Project**: CTR Prediction and Ad Scoring System
**Repository**: https://github.com/Venkateswara-Sahu/CTR_Predictor_and_Scorer

---

**⭐ Star this repository if you find it helpful!**