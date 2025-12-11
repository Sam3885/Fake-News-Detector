import streamlit as st
import joblib
import os
import re

# ============ PAGE CONFIGURATION & STYLING ============
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Background */
    body {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        background-attachment: fixed;
    }
    
    .main {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        margin: 20px;
        padding: 30px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Title styling */
    h1 {
        text-align: center;
        color: #1f77e4;
        font-size: 3rem;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    
    /* Input area styling */
    .input-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        padding: 12px 20px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    
    /* Section headers */
    h2 {
        color: #1f77e4;
        border-bottom: 3px solid #1f77e4;
        padding-bottom: 10px;
        margin-top: 2rem;
    }
    
    /* Info boxes */
    .info-box {
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============ HEADER SECTION ============
st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1>📰 Fake News Detector</h1>
        <p class="subtitle">🔍 Advanced AI-Powered News Authenticity Analysis</p>
        <hr style="margin: 1rem 0;">
    </div>
""", unsafe_allow_html=True)

# ============ SIDEBAR ============
with st.sidebar:
    st.markdown("### ⚙️ About This App")
    st.info("""
    **Fake News Detector** uses Machine Learning to analyze news articles and determine their authenticity.
    
    **Features:**
    - 🎯 Prediction Confidence Score
    - 📊 Text Statistics Analysis
    - 🚩 Fake News Indicators Detection
    - 📈 Detailed Analytics
    """)
    
    st.markdown("---")
    st.markdown("### 📌 How to Use")
    st.write("""
    1. Paste your news article in the text area
    2. Click "Analyze News" button
    3. View detailed analysis results
    4. Check confidence scores and indicators
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.success("✓ Model Loaded Successfully")

# Check if files exist
if not os.path.exists("vectorizer.jb"):
    st.error("❌ vectorizer.jb not found!")
    st.stop()

if not os.path.exists("lr_model.jb"):
    st.error("❌ lr_model.jb not found!")
    st.stop()

# Load models
try:
    vectorizer = joblib.load("vectorizer.jb")
    model = joblib.load("lr_model.jb")
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    st.stop()

# ============ INPUT SECTION ============
st.markdown("### 📝 Enter News Article")
st.markdown('<div class="input-section">', unsafe_allow_html=True)
news_input = st.text_area(
    "Paste your news article here:",
    height=200,
    placeholder="Enter the news text you want to verify..."
)
st.markdown('</div>', unsafe_allow_html=True)

# ============ ANALYSIS BUTTON ============
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("🔍 Analyze News", use_container_width=True)

if analyze_button:
    if news_input.strip():
        transform_input = vectorizer.transform([news_input])
        prediction = model.predict(transform_input)
        prediction_proba = model.predict_proba(transform_input)

        # ============ PREDICTION RESULT ============
        st.markdown("---")
        
        is_real = prediction[0] == 1
        confidence = max(prediction_proba[0]) * 100
        
        if is_real:
            st.success("✅ VERDICT: This News Appears to be REAL", icon="✅")
            result_color = "#00ff00"
            result_text = "REAL NEWS"
        else:
            st.error("❌ VERDICT: This News Appears to be FAKE", icon="❌")
            result_color = "#ff0000"
            result_text = "FAKE NEWS"
        
        # ============ CONFIDENCE & METRICS ============
        st.markdown("### 📊 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "🎯 Confidence Score",
                f"{confidence:.1f}%",
                delta=f"{'High' if confidence > 75 else 'Medium' if confidence > 50 else 'Low'}"
            )
        
        with col2:
            st.metric(
                "📋 Verdict",
                result_text,
                delta="Likely Authentic" if is_real else "Likely False"
            )
        
        with col3:
            probability_fake = prediction_proba[0][0] * 100
            st.metric(
                "⚠️ Fake Probability",
                f"{probability_fake:.1f}%"
            )
        
        # Confidence bar
        st.markdown(f"""
        <div style="margin-top: 15px;">
            <p style="font-weight: bold; margin-bottom: 5px;">Confidence Level</p>
            <div style="width: 100%; height: 30px; background-color: #e0e0e0; border-radius: 15px; overflow: hidden;">
                <div style="width: {confidence}%; height: 100%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 12px;">
                    {confidence:.1f}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ============ TEXT STATISTICS ============
        st.markdown("---")
        st.markdown("### 📊 Text Statistics")
        
        word_count = len(news_input.split())
        char_count = len(news_input)
        sentence_count = len(re.split(r'[.!?]+', news_input)) - 1
        avg_word_length = char_count / word_count if word_count > 0 else 0
        reading_time = max(1, word_count // 200)
        
        stat_col1, stat_col2, stat_col3, stat_col4, stat_col5 = st.columns(5)
        with stat_col1:
            st.metric("📝 Words", word_count)
        with stat_col2:
            st.metric("🔤 Characters", char_count)
        with stat_col3:
            st.metric("✏️ Sentences", sentence_count)
        with stat_col4:
            st.metric("📏 Avg Word Length", f"{avg_word_length:.1f}")
        with stat_col5:
            st.metric("⏱️ Reading Time", f"{reading_time} min")
        
        # ============ FAKE NEWS INDICATORS ============
        st.markdown("---")
        st.markdown("### 🚩 Fake News Indicators Detection")
        
        indicators = []
        
        # Check for excessive ALL CAPS
        all_caps_words = len(re.findall(r'\b[A-Z]{2,}\b', news_input))
        if all_caps_words > word_count * 0.1:
            indicators.append(("⚠️ Excessive ALL CAPS", "high", f"{all_caps_words} words in caps"))
        
        # Check for excessive punctuation
        punctuation_count = len(re.findall(r'[!?]{2,}', news_input))
        if punctuation_count > 3:
            indicators.append(("⚠️ Excessive Punctuation", "medium", f"{punctuation_count} occurrences"))
        
        # Check for sensational words
        sensational_words = ['shocking', 'unbelievable', 'exclusive', 'breaking', 'revealed', 
                           'secret', 'scandal', 'bombshell', 'must-see', 'you wont believe']
        found_sensational = [word for word in sensational_words if word.lower() in news_input.lower()]
        if found_sensational:
            indicators.append((f"📢 Sensational Language", "medium", f"{len(found_sensational)} sensational words"))
        
        # Check for personal pronouns (emotional language)
        personal_pronouns = len(re.findall(r'\b(I|you|we|they)\b', news_input, re.IGNORECASE))
        if personal_pronouns > word_count * 0.05:
            indicators.append(("💬 High Emotional Language", "low", f"{personal_pronouns} emotional words"))
        
        # Check for question marks (clickbait pattern)
        question_count = news_input.count('?')
        if question_count > sentence_count * 0.5:
            indicators.append(("❓ Excessive Questions", "medium", f"{question_count} questions"))
        
        # Check for links/URLs
        if re.search(r'http|www|\.com', news_input):
            indicators.append(("🔗 Contains URLs", "low", "Links detected"))
        
        if indicators:
            for indicator, severity, detail in indicators:
                if severity == "high":
                    st.error(f"{indicator} - {detail}")
                elif severity == "medium":
                    st.warning(f"{indicator} - {detail}")
                else:
                    st.info(f"{indicator} - {detail}")
            st.markdown("⚠️ Multiple indicators detected - exercise caution with this article")
        else:
            st.success("✓ No major fake news indicators detected - Article appears legitimate")
        
        # ============ RECOMMENDATIONS ============
        st.markdown("---")
        st.markdown("### 💡 Recommendations")
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.markdown("""
            **Before sharing this news:**
            - ✓ Cross-reference with trusted sources
            - ✓ Check author credentials
            - ✓ Verify publication date
            - ✓ Look for supporting evidence
            """)
        
        with rec_col2:
            st.markdown("""
            **Red flags to watch for:**
            - ✗ Sensational headlines
            - ✗ Poor grammar/spelling
            - ✗ Missing source attribution
            - ✗ Emotional manipulation
            """)
        
    else:
        st.warning("⚠️ Please enter some text to analyze. Make sure the article has meaningful content.")