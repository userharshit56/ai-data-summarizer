import os
import streamlit as st
import pandas as pd
from google import genai

st.set_page_config(page_title="AI Data & Text Summarizer", page_icon="⚡", layout="wide")

st.title("⚡ AI-Powered Text & CSV Summarizer")
st.markdown("Extract structured insights using the Google Gemini API.")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    
    st.markdown("---")
    selected_model = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro"])

tab1, tab2 = st.tabs(["📝 Text Summarization", "📊 CSV Analytics"])

def get_gemini_client(key: str):
    if not key:
        st.error("Please provide a valid Gemini API Key in the sidebar.")
        return None
    return genai.Client(api_key=key)

# --- TAB 1: TEXT SUMMARIZATION ---
with tab1:
    st.subheader("Text Summarizer")
    user_text = st.text_area("Paste text or notes here:", height=200)
    summary_format = st.selectbox("Format", ["Key Bullet Points", "Executive Summary", "Action Items"])

    if st.button("Generate Summary", type="primary"):
        if not user_text.strip():
            st.warning("Please enter text to summarize.")
        else:
            client = get_gemini_client(api_key)
            if client:
                with st.spinner("Processing..."):
                    prompt = f"Provide a concise summary in {summary_format} format:\n\n{user_text}"
                    try:
                        response = client.models.generate_content(
                            model=selected_model,
                            contents=prompt
                        )
                        st.markdown("### Output")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"API Error: {e}")

# --- TAB 2: CSV ANALYTICS ---
with tab2:
    st.subheader("CSV Profiler & Insights")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("#### Data Preview", df.head())
        
        col1, col2 = st.columns(2)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])

        if st.button("Analyze CSV with AI"):
            client = get_gemini_client(api_key)
            if client:
                with st.spinner("Analyzing data trends..."):
                    sample_data = df.head(10).to_string()
                    prompt = f"Analyze this dataset sample and summarize key trends and structural insights:\n\n{sample_data}"
                    try:
                        response = client.models.generate_content(
                            model=selected_model,
                            contents=prompt
                        )
                        st.markdown("### Insights")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"API Error: {e}")