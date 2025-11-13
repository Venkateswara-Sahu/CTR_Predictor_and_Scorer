# Streamlit Deployment Guide

Complete guide to run and deploy your CTR Prediction System with Streamlit.

---

## Local Testing

### Step 1: Install Streamlit

```powershell
pip install streamlit
```

### Step 2: Run the Streamlit App

```powershell
cd deployment
streamlit run streamlit_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

---

## Streamlit Cloud Deployment (FREE!)

Streamlit Cloud is **completely FREE** and easier than Heroku!

### Step 1: Push to GitHub

```powershell
cd deployment

# Initialize git if not already done
git init

# Add files
git add .
git commit -m "Add Streamlit app"

# Push to GitHub
git remote add origin https://github.com/yourusername/ctr-prediction.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set main file path: `deployment/streamlit_app.py`
6. Click "Deploy"

That's it! Your app will be live at: `https://yourusername-ctr-prediction.streamlit.app`

---

## App Features

### 1. Single Ad Prediction
- Enter individual ad features (I1-I13, C1-C26)
- Get instant CTR prediction
- See quality score and interpretation

### 2. Batch Prediction
- Upload CSV with multiple ads
- Get predictions for all ads
- Download results as CSV

### 3. Ad Ranking
- Generate random ads
- Compare and rank them
- Visualize performance

### 4. Model Info
- View model architecture
- See performance metrics
- Check feature engineering details

---

## Advantages of Streamlit vs Heroku

| Feature | Streamlit Cloud | Heroku |
|---------|----------------|--------|
| **Cost** | FREE forever | $5/month minimum |
| **Setup Time** | 5 minutes | 20+ minutes |
| **UI** | Beautiful built-in | Need to build |
| **Sleep Mode** | No sleep | Sleeps after 30 min |
| **Demo Friendly** | Perfect for demos | API only |
| **Deployment** | One click | Git push + config |

---

## Configuration Files

### For Streamlit Cloud:

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
headless = true
port = 8501
enableCORS = false
```

### For Secrets (if needed):

Create `.streamlit/secrets.toml`:

```toml
# Add any API keys or secrets here
# This file is automatically secure on Streamlit Cloud
```

---

## Testing Locally

```powershell
# Navigate to deployment folder
cd deployment

# Run streamlit
streamlit run streamlit_app.py

# App opens at http://localhost:8501
```

Test all features:
- ✅ Single prediction
- ✅ Batch upload
- ✅ Ad ranking
- ✅ Model info page

---

## Troubleshooting

### Issue: Models not loading

**Solution**: Ensure model files are in `app/models/` directory:
```
deployment/
  app/
    models/
      lightgbm_model.txt
      xgboost_model.json
      feature_engineering_artifacts.pkl
      training_features.json
```

### Issue: Import errors

**Solution**: Install all requirements:
```powershell
pip install -r requirements.txt
```

### Issue: Slow first prediction

**Explanation**: First prediction loads models. Subsequent predictions are fast.

**Solution**: Models are cached using `@st.cache_resource`

---

## Custom Domain (Optional)

On Streamlit Cloud:
1. Go to app settings
2. Click "Custom subdomain"
3. Set: `your-app-name.streamlit.app`

---

## Sharing Your App

After deployment, share your app:
```
https://yourusername-ctr-prediction.streamlit.app
```

Anyone can access it instantly - no login required!

---

## Local vs Cloud Comparison

### Local (Development):
```powershell
streamlit run streamlit_app.py
```
- Fast iteration
- Local files
- Debug mode

### Cloud (Production):
- Public URL
- Always available
- Automatic SSL
- No server management

---

## Best Practices

1. **Test Locally First**: Always test with `streamlit run` before deploying
2. **Keep Models Small**: Streamlit has file size limits (~1GB)
3. **Use Caching**: Use `@st.cache_resource` for models
4. **Add Loading States**: Use `st.spinner()` for long operations
5. **Error Handling**: Wrap predictions in try-except blocks

---

## Quick Start Commands

```powershell
# Install Streamlit
pip install streamlit

# Run locally
cd deployment
streamlit run streamlit_app.py

# Deploy to cloud:
# 1. Push to GitHub
# 2. Go to https://share.streamlit.io
# 3. Connect repository
# 4. Deploy!
```

---

## Resources

- **Streamlit Docs**: https://docs.streamlit.io
- **Streamlit Cloud**: https://share.streamlit.io
- **Community Forum**: https://discuss.streamlit.io
- **Gallery**: https://streamlit.io/gallery

---

## Demo Script

When presenting:

1. **Open App**: Show clean, professional UI
2. **Single Prediction**: 
   - Enter sample ad features
   - Show CTR prediction (e.g., 0.8522 = 85.22%)
   - Explain quality score
3. **Batch Processing**:
   - Upload sample CSV
   - Show bulk predictions
   - Download results
4. **Ad Ranking**:
   - Generate 5 random ads
   - Show ranking visualization
   - Compare top performers
5. **Model Info**:
   - Show AUC 0.8888
   - Explain 150 features
   - Display architecture

---

**Streamlit is perfect for your B.Tech project demo!** 🎓✨

It's:
- ✅ Easy to use
- ✅ Professional looking
- ✅ Free to deploy
- ✅ Perfect for presentations
- ✅ No server management needed
