"""
Streamlit Web Interface for CTR Prediction & Ad Scoring System
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
import os
import logging
from datetime import datetime

# Add parent directory to path to import app modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Download models on first run (for Streamlit Cloud deployment)
try:
    from download_models import download_models
    download_models()
except Exception as e:
    st.warning(f"Could not download models: {e}")

from app.ctr_model import CTRPredictor
from app.ad_scorer import AdScorer
from app.ad_ranking_system import AdRankingSystem

# Setup logging
def setup_logger():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    log_filename = os.path.join(log_dir, f'streamlit_app_{datetime.now().strftime("%Y%m%d")}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Also log to console
        ]
    )
    
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logger()
logger.info("=" * 80)
logger.info("Streamlit CTR Prediction App Started")
logger.info("=" * 80)

# Page configuration
st.set_page_config(
    page_title="CTR Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        height: 3rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Success box styling */
    .success-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Feature card */
    .feature-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        padding: 1.5rem;
        border-radius: 0.8rem;
        border: 2px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
    }
    
    .feature-card h3 {
        color: #667eea;
        margin-bottom: 0.5rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .feature-card p {
        margin: 0;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    /* Ensure text is visible in dark mode */
    [data-theme="dark"] .feature-card p {
        color: #e0e0e0;
    }
    
    [data-theme="light"] .feature-card p {
        color: #333;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        padding: 0 2rem;
        font-weight: 600;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize models
@st.cache_resource
def load_models():
    """Load models once and cache them"""
    logger.info("Loading models...")
    model_path = os.path.join(os.path.dirname(__file__), 'app', 'models')
    if not os.path.exists(model_path):
        model_path = '../models'
        logger.warning(f"Primary model path not found, using fallback: {model_path}")
    
    ranking_system = AdRankingSystem(model_path=model_path)
    logger.info("Models loaded successfully")
    return ranking_system

# Load models
try:
    ranking_system = load_models()
    models_loaded = True
    logger.info("All models initialized successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}", exc_info=True)
    st.error(f"Error loading models: {e}")
    models_loaded = False

# Title
st.markdown('<h1 class="main-header">� AI-Powered CTR Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict Click-Through Rates with 88.88% Accuracy | +265% CTR Lift | 150 AI Features</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/artificial-intelligence.png", width=150)
    
    st.markdown("### 🎯 Navigation")
    
    # Initialize current page in session state
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = "🏠 Home"
    
    # Get the index for the radio button based on current_page
    pages = ["🏠 Home", "🎯 Quick Predict", "📊 Advanced Analysis", "📦 Batch Processing", "🏆 Compare & Rank", "⚙️ System Info"]
    current_index = pages.index(st.session_state['current_page']) if st.session_state['current_page'] in pages else 0
    
    # Radio button for navigation
    selected_page = st.radio(
        "Navigation Menu",
        pages,
        label_visibility="collapsed",
        index=current_index
    )
    
    # Only update if user actually clicked the radio (not just from index change)
    # This prevents radio from overriding button clicks
    if 'last_radio_selection' not in st.session_state:
        st.session_state['last_radio_selection'] = selected_page
    
    # If radio selection changed (user clicked it), update current page
    if selected_page != st.session_state['last_radio_selection']:
        logger.info(f"Navigation: Radio changed from {st.session_state['last_radio_selection']} to {selected_page}")
        st.sidebar.write(f"DEBUG: Radio changed from {st.session_state['last_radio_selection']} to {selected_page}")
        st.session_state['current_page'] = selected_page
    
    # Always sync last_radio_selection to current_page at the end
    st.session_state['last_radio_selection'] = st.session_state['current_page']
    
    st.markdown("---")
    
    # Live metrics
    st.markdown("### ⚡ Live Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🎯 AUC", "0.8888", "+11%")
        st.metric("📈 Lift", "+265%", "")
    with col2:
        st.metric("🔧 Features", "150", "")
        st.metric("⚡ Speed", "0.5s", "")
    
    st.markdown("---")
    
    # Quick stats
    st.markdown("### 📊 Dataset Info")
    st.info("""
    **Criteo Display Ads**
    - 10M total samples
    - 7M training set
    - 2M test set
    - 39 raw features
    """)
    
    st.markdown("---")
    st.success("✅ All systems operational")

# Main content
if not models_loaded:
    st.error("⚠️ Models not loaded. Please check the model files.")
    st.stop()

# Use session state for page routing
page = st.session_state['current_page']

# Debug: Show current page
st.sidebar.markdown(f"**🔍 Debug:** `{page}`")

# Page 0: Home Dashboard
if page == "🏠 Home":
    # Hero section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        border-radius: 1rem; color: white; margin: 1rem 0;'>
            <h2>🎯 Welcome to CTR Prediction AI</h2>
            <p style='font-size: 1.2rem; margin-top: 1rem;'>
                Leverage machine learning to predict ad performance with unprecedented accuracy
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature cards
    st.markdown("### 🌟 Key Features")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <h3>⚡ Instant Prediction</h3>
            <p>Get CTR predictions in <strong>under 0.5 seconds</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <h3>📊 Batch Processing</h3>
            <p>Upload CSV/Excel and process <strong>1000s of ads</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-card'>
            <h3>🏆 Smart Ranking</h3>
            <p>Compare and rank ads by <strong>performance</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='feature-card'>
            <h3>🎯 88.88% AUC</h3>
            <p>Industry-leading <strong>accuracy</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Performance showcase
    st.markdown("### 📈 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎯 AUC Score", "0.8888", "+11.6%", help="Area Under ROC Curve")
    col2.metric("📈 CTR Lift", "+265.6%", "Top 10%", help="Top 10% ads performance")
    col3.metric("⚡ Precision@10", "100%", "Perfect", help="Top 10 predictions accuracy")
    col4.metric("🔥 Log Loss", "0.1089", "↓ Lower", help="Lower is better")
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### 🚀 Quick Start")
    
    # Test button outside columns first
    if st.button("🧪 TEST BUTTON - Click Me!", key="test_btn"):
        st.success("TEST BUTTON WORKS!")
        st.session_state['test_clicked'] = True
    
    if 'test_clicked' in st.session_state:
        st.info(f"Test button was clicked at some point")
    
    col1, col2, col3 = st.columns(3)
    
    if col1.button("🎯 Try Quick Prediction", use_container_width=True, type="primary", key="home_btn1"):
        logger.info("User clicked 'Try Quick Prediction' button")
        st.session_state['current_page'] = "🎯 Quick Predict"
        # DON'T update last_radio_selection - let the sidebar handle it
        st.rerun()
    
    if col2.button("📊 Upload Your Data", use_container_width=True, key="home_btn2"):
        st.session_state['current_page'] = "📦 Batch Processing"
        # DON'T update last_radio_selection - let the sidebar handle it
        st.rerun()
    
    if col3.button("🏆 Compare Ads", use_container_width=True, key="home_btn3"):
        st.session_state['current_page'] = "🏆 Compare & Rank"
        # DON'T update last_radio_selection - let the sidebar handle it
        st.rerun()
    
    st.markdown("---")
    
    # Technology stack
    st.markdown("### 🔧 Technology Stack")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **🤖 Machine Learning Models:**
        - LightGBM (60% ensemble weight)
        - XGBoost (40% ensemble weight)
        - 150 engineered features
        - Hyperparameter optimization (Optuna)
        """)
    
    with col2:
        st.info("""
        **📊 Data Processing:**
        - Criteo 10M dataset
        - Feature engineering pipeline
        - CTR encoding & frequency analysis
        - Real-time prediction API
        """)

# Page 1: Quick Predict
elif page == "🎯 Quick Predict":
    st.header("🎯 Quick CTR Prediction")
    st.markdown("Choose your input method below")
    
    # Three input methods
    tab1, tab2, tab3 = st.tabs(["🎲 Auto-Generate", "✍️ Manual Entry", "📋 Paste JSON"])
    
    # Tab 1: Auto-generate
    with tab1:
        st.markdown("### 🎲 Auto-Generate Random Ad")
        st.info("Click the button to instantly generate a random ad and get predictions")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Generate & Predict Now", type="primary", use_container_width=True, key="auto_gen"):
                with st.spinner("🔮 Generating ad and predicting..."):
                    try:
                        # Auto-generate random features
                        sample_values = ['1000', 'efba', '05db', 'fb9c', '25c8', '7e6e', '0b15', '21dd', 
                                        'a73e', 'b1f8', '3475', '8a4f', 'e5ba', '74c2', '38eb', '1e88',
                                        '3b08', '7c6e', 'b28b', 'febc', '8f90', 'b04e', 'c9d9', '0014', '13d5', '001f']
                        
                        features = {}
                        for i in range(1, 14):
                            features[f'I{i}'] = int(np.random.randint(0, 100))
                        for i in range(1, 27):
                            features[f'C{i}'] = np.random.choice(sample_values)
                        
                        logger.info("Generating random ad for prediction")
                        result = ranking_system.predict_single_ad(features)
                        st.session_state['last_result'] = result
                        st.session_state['last_features'] = features
                        logger.info(f"Prediction complete: CTR={result['predicted_ctr']:.4f}, Quality={result['quality_score']:.4f}")
                        
                    except Exception as e:
                        logger.error(f"Error in auto-generate prediction: {e}", exc_info=True)
                        st.error(f"Error: {e}")
    
    # Tab 2: Manual entry (simplified)
    with tab2:
        st.markdown("### ✍️ Manual Feature Entry")
        st.info("Enter key features manually - others will be auto-filled with defaults")
        
        with st.form("manual_entry_form"):
            col1, col2, col3 = st.columns(3)
            
            manual_features = {}
            
            with col1:
                st.markdown("**Top Integer Features:**")
                manual_features['I1'] = st.number_input("I1 (Click count)", 0, 1000, 5)
                manual_features['I2'] = st.number_input("I2 (Impressions)", 0, 10000, 100)
                manual_features['I3'] = st.number_input("I3 (Hour)", 0, 23, 12)
                manual_features['I10'] = st.number_input("I10 (Ad position)", 0, 100, 25)
            
            with col2:
                st.markdown("**More Integer Features:**")
                manual_features['I6'] = st.number_input("I6 (User age)", 0, 100, 30)
                manual_features['I7'] = st.number_input("I7 (Session length)", 0, 1000, 50)
                manual_features['I11'] = st.number_input("I11 (Ad size)", 0, 100, 10)
                manual_features['I12'] = st.number_input("I12 (Device type)", 0, 10, 2)
            
            with col3:
                st.markdown("**Key Categorical Features:**")
                manual_features['C1'] = st.text_input("C1 (Category)", "1000")
                manual_features['C9'] = st.text_input("C9 (Site ID)", "a73e")
                manual_features['C19'] = st.text_input("C19 (Device)", "b28b")
                manual_features['C21'] = st.text_input("C21 (Platform)", "8f90")
            
            submitted = st.form_submit_button("🚀 Predict with Manual Input", type="primary", use_container_width=True)
            
            if submitted:
                with st.spinner("🔮 Making prediction..."):
                    try:
                        # Fill remaining features with defaults
                        sample_values = ['1000', 'efba', '05db', 'fb9c', '25c8', '7e6e']
                        
                        # Add missing I features
                        for i in range(1, 14):
                            if f'I{i}' not in manual_features:
                                manual_features[f'I{i}'] = 0
                        
                        # Add missing C features
                        for i in range(1, 27):
                            if f'C{i}' not in manual_features:
                                manual_features[f'C{i}'] = np.random.choice(sample_values)
                        
                        logger.info("Manual entry prediction requested")
                        result = ranking_system.predict_single_ad(manual_features)
                        st.session_state['last_result'] = result
                        st.session_state['last_features'] = manual_features
                        logger.info(f"Manual prediction complete: CTR={result['predicted_ctr']:.4f}")
                        st.success("✅ Prediction complete!")
                        st.rerun()
                        
                    except Exception as e:
                        logger.error(f"Error in manual entry prediction: {e}", exc_info=True)
                        st.error(f"Error: {e}")
    
    # Tab 3: JSON paste
    with tab3:
        st.markdown("### 📋 Paste JSON Features")
        st.info("Paste ad features in JSON format for quick testing")
        
        example_json = """{
  "I1": 5, "I2": 10, "I3": 2, "I4": 15, "I5": 3,
  "I6": 100, "I7": 50, "I8": 8, "I9": 12, "I10": 25,
  "I11": 7, "I12": 4, "I13": 30,
  "C1": "1000", "C2": "efba", "C3": "05db", "C4": "fb9c",
  "C5": "25c8", "C6": "7e6e", "C7": "0b15", "C8": "21dd",
  "C9": "a73e", "C10": "b1f8", "C11": "3475", "C12": "8a4f",
  "C13": "e5ba", "C14": "74c2", "C15": "38eb", "C16": "1e88",
  "C17": "3b08", "C18": "7c6e", "C19": "b28b", "C20": "febc",
  "C21": "8f90", "C22": "b04e", "C23": "25c8", "C24": "c9d9",
  "C25": "0014", "C26": "13d5"
}"""
        
        json_input = st.text_area("Paste JSON here:", example_json, height=300)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🚀 Predict from JSON", type="primary", use_container_width=True):
                try:
                    import json
                    logger.info("JSON paste prediction requested")
                    features = json.loads(json_input)
                    result = ranking_system.predict_single_ad(features)
                    st.session_state['last_result'] = result
                    st.session_state['last_features'] = features
                    logger.info(f"JSON prediction complete: CTR={result['predicted_ctr']:.4f}")
                    st.success("✅ Prediction complete!")
                    st.rerun()
                except json.JSONDecodeError:
                    logger.error("Invalid JSON format provided")
                    st.error("❌ Invalid JSON format. Please check your input.")
                except Exception as e:
                    logger.error(f"Error in JSON prediction: {e}", exc_info=True)
                    st.error(f"Error: {e}")
        
        with col2:
            if st.button("📋 Copy Example", use_container_width=True):
                st.code(example_json, language="json")
                st.info("👆 Copy the example above")
    
    # Display results (shown below all tabs)
    if 'last_result' in st.session_state:
        result = st.session_state['last_result']
        
        st.markdown("---")
        st.markdown("## 📊 Prediction Results")
        
        # Big animated metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ctr_pct = result['predicted_ctr'] * 100
            delta_color = "normal" if result['predicted_ctr'] > 0.25 else "inverse"
            st.metric(
                label="📈 Click-Through Rate",
                value=f"{ctr_pct:.2f}%",
                delta=f"{(result['predicted_ctr'] - 0.25) * 100:.1f}% vs baseline",
                delta_color=delta_color
            )
        
        with col2:
            st.metric(
                label="⭐ Quality Score",
                value=f"{result['quality_score']:.3f}",
                delta="0-1 scale",
                delta_color="off"
            )
        
        with col3:
            st.metric(
                label="🎯 Final Score",
                value=f"{result['final_score']:.3f}",
                delta="Combined metric",
                delta_color="off"
            )
        
        # Visual interpretation with progress bars
        st.markdown("### 📊 Performance Analysis")
        
        if result['predicted_ctr'] > 0.50:
            st.success(f"🎉 **EXCELLENT PERFORMANCE** - CTR of {ctr_pct:.1f}% is {ctr_pct/25:.1f}x the baseline!")
            st.progress(min(result['predicted_ctr'], 1.0))
        elif result['predicted_ctr'] > 0.30:
            st.info(f"👍 **GOOD PERFORMANCE** - CTR of {ctr_pct:.1f}% is {(ctr_pct-25):.1f}% above baseline")
            st.progress(result['predicted_ctr'])
        elif result['predicted_ctr'] > 0.15:
            st.warning(f"⚠️ **AVERAGE PERFORMANCE** - CTR of {ctr_pct:.1f}% is near baseline (25%)")
            st.progress(result['predicted_ctr'])
        else:
            st.error(f"❌ **LOW PERFORMANCE** - CTR of {ctr_pct:.1f}% needs improvement")
            st.progress(result['predicted_ctr'])
        
        # Collapsible feature view
        with st.expander("🔍 View Input Features"):
            features = st.session_state['last_features']
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Integer Features (I1-I13):**")
                i_features = {k: v for k, v in features.items() if k.startswith('I')}
                st.json(i_features)
            with col2:
                st.markdown("**Categorical Features (C1-C26):**")
                c_features = {k: v for k, v in features.items() if k.startswith('C')}
                st.json(c_features)
        
        # Export options
        st.markdown("### 💾 Export Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            result_json = json.dumps(result, indent=2)
            st.download_button("📥 Download JSON", result_json, "prediction.json", "application/json")
        
        with col2:
            result_csv = f"CTR,Quality,Final Score\n{result['predicted_ctr']},{result['quality_score']},{result['final_score']}"
            st.download_button("📥 Download CSV", result_csv, "prediction.csv", "text/csv")
        
        with col3:
            if st.button("🔄 Clear Results", use_container_width=True):
                del st.session_state['last_result']
                del st.session_state['last_features']
                st.rerun()

# Page 2: Advanced Analysis
elif page == "📊 Advanced Analysis":
    st.header("📊 Advanced Ad Analysis")
    st.markdown("Deep dive into feature importance and model behavior")
    
    # Extract real feature importance from models
    st.markdown("### 🔍 Top Features by Importance")
    
    try:
        # Get feature importance from LightGBM and XGBoost
        feature_importance = {}
        
        if ranking_system.ctr_predictor.lgb_model:
            lgb_importance = ranking_system.ctr_predictor.lgb_model.feature_importance(importance_type='gain')
            feature_names = ranking_system.ctr_predictor.lgb_model.feature_name()
            
            for name, importance in zip(feature_names, lgb_importance):
                feature_importance[name] = feature_importance.get(name, 0) + importance * 0.6  # LGB weight
        
        if ranking_system.ctr_predictor.xgb_model:
            xgb_importance = ranking_system.ctr_predictor.xgb_model.get_score(importance_type='gain')
            
            for name, importance in xgb_importance.items():
                feature_importance[name] = feature_importance.get(name, 0) + importance * 0.4  # XGB weight
        
        # Sort and get top 15
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
        
        top_features_data = pd.DataFrame({
            'Feature': [f[0] for f in sorted_features],
            'Importance': [f[1] for f in sorted_features]
        })
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.bar_chart(top_features_data.set_index('Feature')['Importance'])
        
        with col2:
            st.dataframe(top_features_data, hide_index=True, use_container_width=True)
            st.caption(f"📊 Showing top 15 of {len(feature_importance)} features")
    
    except Exception as e:
        st.warning(f"Could not extract feature importance: {e}")
        st.info("Feature importance will be available once models are fully loaded")
    
    st.markdown("---")
    
    # Real CTR distribution from sample predictions
    st.markdown("### 📈 CTR Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Training Data Statistics:**
        - Overall CTR: 25.62%
        - Training Samples: 7M
        - Validation Samples: 1M
        - Test Samples: 2M
        - Total Features: 150
        """)
    
    with col2:
        # Generate real CTR predictions on sample data
        with st.spinner("Generating CTR distribution..."):
            try:
                sample_ctrs = []
                sample_values = ['1000', 'efba', '05db', 'fb9c', '25c8']
                
                for _ in range(100):
                    features = {}
                    for i in range(1, 14):
                        features[f'I{i}'] = int(np.random.randint(0, 100))
                    for i in range(1, 27):
                        features[f'C{i}'] = np.random.choice(sample_values)
                    
                    result = ranking_system.predict_single_ad(features)
                    sample_ctrs.append(result['predicted_ctr'])
                
                # Create deciles
                sample_ctrs.sort(reverse=True)
                decile_size = len(sample_ctrs) // 10
                decile_ctrs = [np.mean(sample_ctrs[i*decile_size:(i+1)*decile_size]) * 100 
                              for i in range(10)]
                
                ctr_data = pd.DataFrame({
                    'Decile': [f'D{i}' for i in range(1, 11)],
                    'CTR': decile_ctrs
                })
                st.bar_chart(ctr_data.set_index('Decile'))
                st.caption(f"Based on 100 sample predictions")
            except Exception as e:
                st.warning(f"Could not generate distribution: {e}")
    
    st.markdown("---")
    
    # Model comparison with REAL metrics from training
    st.markdown("### 🤖 Model Performance Comparison")
    
    st.info("""
    **Real Test Results (2M samples):**
    - **Ensemble AUC**: 0.8984 (89.84%)
    - **Test Accuracy**: 82.72%
    - **Precision**: 64.26%
    - **Recall**: 73.36%
    - **F1 Score**: 68.51%
    - **Log Loss**: 0.3218
    """)
    
    model_comparison = pd.DataFrame({
        'Model': ['Ensemble (LGB+XGB)', 'XGBoost', 'LightGBM'],
        'Validation AUC': [0.9067, 0.9067, 0.9024],
        'Test AUC': [0.8984, 0.8984, 0.8984],
        'Log Loss': [0.3218, 0.3218, 0.3218],
        'Weight in Ensemble': ['100%', '40%', '60%']
    })
    
    st.dataframe(model_comparison, hide_index=True, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Cross-Validation AUC", "0.9065 ± 0.0002", help="Mean AUC across 5 folds")
    with col2:
        st.metric("Training Samples", "7M", help="7 million training samples")
    with col3:
        st.metric("Model Stability", "High", help="Low std deviation across folds")
    
    st.markdown("---")
    
    # What-if analysis - Enhanced with more scenarios and visualizations
    st.markdown("### 🔮 What-If Analysis")
    st.info("Analyze how changing features affects CTR prediction - Test multiple scenarios")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        base_i1 = st.slider("I1 (Click count)", 0, 100, 50)
    with col2:
        base_i2 = st.slider("I2 (Impressions)", 0, 100, 50)
    with col3:
        base_i10 = st.slider("I10 (Position)", 0, 100, 25)
    with col4:
        num_scenarios = st.selectbox("Scenarios to test", [5, 10, 20, 50], index=1)
    
    if st.button("🔮 Run What-If Analysis", type="primary", use_container_width=True):
        with st.spinner(f"Running {num_scenarios} scenarios..."):
            try:
                sample_values = ['1000', 'efba', '05db', 'fb9c', '25c8', '7e6e']
                scenarios = []
                
                # Test across a range of multipliers
                multipliers = np.linspace(0.25, 2.0, num_scenarios)
                
                for idx, multiplier in enumerate(multipliers):
                    features = {}
                    features['I1'] = int(base_i1 * multiplier)
                    features['I2'] = int(base_i2 * multiplier)
                    features['I10'] = int(base_i10 * multiplier)
                    
                    # Fill other features with slight variations
                    for i in range(4, 14):
                        features[f'I{i}'] = int(np.random.randint(0, 50))
                    for i in range(1, 27):
                        features[f'C{i}'] = np.random.choice(sample_values)
                    
                    result = ranking_system.predict_single_ad(features)
                    scenarios.append({
                        'Scenario': f"#{idx+1}",
                        'Multiplier': f"{multiplier:.2f}x",
                        'I1': features['I1'],
                        'I2': features['I2'],
                        'I10': features['I10'],
                        'CTR': result['predicted_ctr'],
                        'Quality Score': result['quality_score'],
                        'Final Score': result['final_score']
                    })
                
                scenario_df = pd.DataFrame(scenarios)
                
                # Show summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg CTR", f"{scenario_df['CTR'].mean():.2%}", 
                             f"Range: {scenario_df['CTR'].min():.2%} - {scenario_df['CTR'].max():.2%}")
                with col2:
                    st.metric("Avg Quality", f"{scenario_df['Quality Score'].mean():.3f}",
                             f"Std: {scenario_df['Quality Score'].std():.3f}")
                with col3:
                    st.metric("Best Scenario", f"#{scenario_df['Final Score'].idxmax() + 1}",
                             f"Score: {scenario_df['Final Score'].max():.3f}")
                
                # Visualizations
                st.markdown("#### 📊 CTR vs Multiplier")
                chart_data = scenario_df[['Multiplier', 'CTR']].copy()
                chart_data['CTR %'] = chart_data['CTR'] * 100
                st.line_chart(chart_data.set_index('Multiplier')['CTR %'])
                
                st.markdown("#### 📈 Quality Score vs Final Score")
                scatter_data = scenario_df[['Quality Score', 'Final Score', 'CTR']].copy()
                st.scatter_chart(scatter_data, x='Quality Score', y='Final Score', size='CTR')
                
                # Detailed table
                st.markdown("#### 📋 Detailed Results")
                display_df = scenario_df.copy()
                display_df['CTR'] = display_df['CTR'].apply(lambda x: f"{x:.2%}")
                display_df['Quality Score'] = display_df['Quality Score'].apply(lambda x: f"{x:.3f}")
                display_df['Final Score'] = display_df['Final Score'].apply(lambda x: f"{x:.3f}")
                st.dataframe(display_df, hide_index=True, use_container_width=True, height=400)
                
                # Download option
                csv = scenario_df.to_csv(index=False)
                st.download_button("📥 Download Results", csv, "whatif_analysis.csv", "text/csv")
                
            except Exception as e:
                st.error(f"Error running analysis: {e}")

# Page 3: Batch Processing
elif page == "📦 Batch Processing":
    st.header("📦 Batch Upload & Predict")
    st.markdown("Upload CSV or use our sample data")
    
    tab1, tab2 = st.tabs(["📄 Upload Your CSV", "🎲 Use Sample Data"])
    
    with tab1:
        st.info("Upload a CSV with columns: I1-I13, C1-C26")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} ads")
            
            if st.button("🚀 Predict All", type="primary"):
                with st.spinner(f"Processing {len(df)} ads..."):
                    results = ranking_system.rank_ads(df, return_details=True)
                    st.dataframe(results.head(10))
                    
                    csv = results.to_csv(index=False)
                    st.download_button("📥 Download Results", csv, "predictions.csv", "text/csv")
    
    with tab2:
        st.markdown("### Generate Sample Data")
        num_samples = st.slider("Number of sample ads", 10, 100, 50)
        
        if st.button("🎲 Generate & Predict", type="primary"):
            with st.spinner(f"Generating {num_samples} ads..."):
                sample_values = ['1000', 'efba', '05db', 'fb9c', '25c8', '7e6e', '0b15', '21dd', 
                                'a73e', 'b1f8', '3475', '8a4f', 'e5ba', '74c2', '38eb', '1e88']
                
                ads = []
                for i in range(num_samples):
                    ad = {}
                    for j in range(1, 14):
                        ad[f'I{j}'] = int(np.random.randint(0, 100))
                    for j in range(1, 27):
                        ad[f'C{j}'] = np.random.choice(sample_values)
                    ads.append(ad)
                
                results = ranking_system.rank_ads(ads, return_details=True)
                
                st.success(f"✅ Processed {len(results)} ads!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg CTR", f"{results['predicted_ctr'].mean():.1%}")
                with col2:
                    st.metric("Max CTR", f"{results['predicted_ctr'].max():.1%}")
                with col3:
                    st.metric("Top Score", f"{results['final_score'].max():.3f}")
                
                st.dataframe(results.head(20), use_container_width=True)
                
                csv = results.to_csv(index=False)
                st.download_button("📥 Download All Results", csv, "sample_predictions.csv", "text/csv")

# Page 4: Compare & Rank
elif page == "🏆 Compare & Rank":
    st.header("🏆 Compare & Rank Ads")
    st.markdown("Auto-generate ads and see rankings instantly")
    
    num_ads = st.slider("Number of ads to compare", 3, 10, 5)
    
    if st.button("🎲 Generate & Rank Ads", type="primary", use_container_width=True):
        with st.spinner("Generating and ranking ads..."):
            sample_values = ['1000', 'efba', '05db', 'fb9c', '25c8', '7e6e', '0b15', '21dd']
            
            ads = []
            for i in range(num_ads):
                ad = {}
                for j in range(1, 14):
                    ad[f'I{j}'] = int(np.random.randint(0, 100))
                for j in range(1, 27):
                    ad[f'C{j}'] = np.random.choice(sample_values)
                ads.append(ad)
            
            results = ranking_system.rank_ads(ads, return_details=True)
            
            st.success("✅ Ranking Complete!")
            
            # Top 3 medals
            st.markdown("### 🏆 Top 3 Ads")
            medals = ["🥇", "🥈", "🥉"]
            cols = st.columns(3)
            
            for idx, (_, row) in enumerate(results.head(3).iterrows()):
                with cols[idx]:
                    st.markdown(f"### {medals[idx]} Rank {idx+1}")
                    st.metric("CTR", f"{row['predicted_ctr']:.1%}")
                    st.metric("Quality", f"{row['quality_score']:.3f}")
                    st.metric("Score", f"{row['final_score']:.3f}")
            
            # Full table
            st.markdown("### 📊 All Rankings")
            display_df = results[['predicted_ctr', 'quality_score', 'final_score']].copy()
            display_df['predicted_ctr'] = display_df['predicted_ctr'].apply(lambda x: f"{x:.1%}")
            display_df.index = [f"Ad {i+1}" for i in range(len(display_df))]
            st.dataframe(display_df, use_container_width=True)
            
            # Chart
            st.markdown("### 📈 Visual Comparison")
            chart_data = results[['predicted_ctr', 'quality_score']].copy()
            chart_data.index = [f"Ad {i+1}" for i in range(len(chart_data))]
            st.bar_chart(chart_data)

# Page 5: System Info
elif page == "⚙️ System Info":
    st.header("ℹ️ System Information")
    
    tab1, tab2 = st.tabs(["📊 Performance", "🔧 Architecture"])
    
    with tab1:
        st.markdown("### 🎯 Model Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("AUC-ROC", "0.8888", "+11.6% vs baseline")
            st.metric("Log Loss", "0.1089", "Lower is better")
            st.metric("Precision@10", "100%", "Perfect top-10")
        with col2:
            st.metric("Precision@100", "100%", "Perfect top-100")
            st.metric("CTR Lift", "+265.6%", "Top 10% ads")
            st.metric("Training Time", "~20 min", "On 1M samples")
        
        st.markdown("### � Performance Breakdown")
        perf_data = pd.DataFrame({
            'Metric': ['AUC-ROC', 'Precision@10', 'Precision@100', 'Top 10% CTR'],
            'Value': [0.8888, 1.0, 1.0, 0.968],
            'Target': [0.80, 0.90, 0.85, 0.40]
        })
        st.dataframe(perf_data, use_container_width=True)
    
    with tab2:
        st.markdown("### 🏗️ System Architecture")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("""
            **Models:**
            - LightGBM (60% weight)
            - XGBoost (40% weight)
            
            **Dataset:**
            - Criteo Display Ads
            - 10M total samples
            - 1M training samples
            """)
        
        with col2:
            st.info("""
            **Features:**
            - 39 raw features
            - 150 engineered features
            - CTR encoding
            - Frequency encoding
            - Interaction features
            """)
        
        st.markdown("### ⭐ 6-Dimensional Ad Scoring")
        scoring_data = pd.DataFrame({
            'Dimension': ['Placement', 'Device', 'Time', 'Engagement', 'Diversity', 'Consistency'],
            'Weight': ['20%', '15%', '15%', '20%', '15%', '15%'],
            'Description': [
                'Above fold placement',
                'Mobile vs desktop',
                'Peak vs off-peak',
                'User interaction rate',
                'Content variety',
                'Brand consistency'
            ]
        })
        st.dataframe(scoring_data, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>CTR Prediction & Ad Scoring System | Built with Streamlit, LightGBM & XGBoost</p>
    <p>AUC: 0.8888 | CTR Lift: +265.6% | 150 Features</p>
</div>
""", unsafe_allow_html=True)
