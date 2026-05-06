import streamlit as st
import pandas as pd
import plotly.express as px
from processor import extract_text, rank_resumes
import os

st.set_page_config(page_title="AI Resume Intelligence", layout="wide")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Automated Resume Intelligence Dashboard")
st.markdown("---")

# Sidebar - Smart Filters & Configuration
st.sidebar.header("Job Configuration")
jd_text = st.sidebar.text_area("Paste Job Description:", height=250, placeholder="Required: Python, NLP, SQL...")

st.sidebar.header("Threshold Settings")
min_score = st.sidebar.slider("Minimum Match Score (%)", 0, 100, 40)

# Main Dashboard Area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Candidate Ingestion")
    uploaded_files = st.file_uploader("Upload Resumes (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)

with col2:
    st.subheader("⚙️ Analysis Control")
    start_btn = st.button("Run Smart Screening")

if start_btn:
    if not jd_text or not uploaded_files:
        st.warning("Please provide both a Job Description and Resumes.")
    else:
        with st.spinner("Processing NLP Models..."):
            resumes_content = []
            filenames = []
            
            for file in uploaded_files:
                with open(file.name, "wb") as f:
                    f.write(file.getbuffer())
                content = extract_text(file.name)
                resumes_content.append(content)
                filenames.append(file.name)
                os.remove(file.name)
            
            scores = rank_resumes(jd_text, resumes_content)
            
            # Data Preparation
            df = pd.DataFrame({
                "Candidate": filenames,
                "Match Score": [round(s * 100, 2) for s in scores]
            }).sort_values(by="Match Score", ascending=False)

            # Filtering based on user threshold
            df["Status"] = df["Match Score"].apply(lambda x: "Shortlisted" if x >= min_score else "Rejected")

            # --- VISUALIZATION SECTION ---
            st.markdown("---")
            st.header("🎯 Screening Results & Insights")
            
            viz_col1, viz_col2 = st.columns([1.5, 1])
            
            with viz_col1:
                st.subheader("Candidate Rankings")
                st.dataframe(df.style.background_gradient(subset=['Match Score'], cmap='BuGn'), use_container_width=True)

            with viz_col2:
                st.subheader("Match Intelligence")
                # Horizontal Bar Chart like your screenshot
                fig = px.bar(df, x='Match Score', y='Candidate', orientation='h', 
                             color='Match Score', color_continuous_scale='Viridis',
                             text='Match Score')
                fig.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                st.plotly_chart(fig, use_container_width=True)

            # Summary Metrics
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Resumes", len(df))
            m2.metric("Shortlisted", len(df[df["Status"] == "Shortlisted"]))
            m3.metric("Avg Match Score", f"{round(df['Match Score'].mean(), 2)}%")