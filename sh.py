import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import plotly.express as px
from datetime import datetime

st.set_page_config(
    page_title="Job Market Analysis and Recommendation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 🎨 Colorful Professional CSS
st.markdown("""
<style>
/* Background & fonts */
body {
    background: linear-gradient(135deg, #0f172a, #1e293b); /* Navy gradient */
    color: #f9fafb;
    font-family: 'Poppins', sans-serif;
    margin: 0;
    padding: 0;
}

/* 🔴 Main heading */
h1 {
    font-weight: 900;
}

/* Navbar */
.navbar {
    background: linear-gradient(90deg, #ef4444, #f59e0b); /* Red → Yellow */
    padding: 1rem 2rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    border-radius: 10px;
}

.logo {
    font-size: 1.6rem;
    font-weight: 800;
    color: white;
    text-shadow: 1px 1px 6px rgba(0,0,0,0.6);
}

/* Navbar links */
.nav-link {
    cursor: pointer;
    color: #f1f5f9;
    font-weight: 600;
    transition: color 0.3s ease;
}
.nav-link:hover {
    color: #facc15; /* bright yellow */
}

/* Hero Section */
.hero-section {
    background: linear-gradient(135deg, #6366f1, #3b82f6); /* Purple → Blue */
    padding: 4rem 2rem;
    border-radius: 15px;
    text-align: center;
    margin: 2rem auto;
    max-width: 950px;
    color: white;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.45);
}
.hero-title {
    font-size: 3rem;
    font-weight: 900;
    text-shadow: 0px 2px 6px rgba(0,0,0,0.7);
}
.hero-subtitle {
    font-size: 1.3rem;
    font-weight: 400;
    color: #f4f4f5;
}

/* Stat cards */
.stat-card {
    background: linear-gradient(135deg, #f59e0b, #ef4444); /* Yellow-Red */
    border-radius: 15px;
    padding: 1.5rem 1rem;
    color: white;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.3);
}
.stat-number {
    font-size: 2.2rem;
    font-weight: bold;
}
.stat-label {
    color: rgba(255,255,255,0.85);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #06b6d4); /* Blue → Cyan */
    color: white;
    font-weight: 700;
    padding: 0.6rem 1.5rem;
    border: none;
    border-radius: 25px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.35);
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #0ea5e9, #38bdf8);
}

/* Job Card */
.job-card {
    background: linear-gradient(135deg, rgba(59,130,246,0.9), rgba(147,51,234,0.9));
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 6px 20px rgba(0,0,0,0.4);
}
.job-title {
    font-weight: 800;
    color: #fff;
    font-size: 1.4rem;
}
.job-meta {
    display: flex;
    gap: 1.5rem;
    font-weight: 500;
    color: #e0f2fe;
}

/* Apply button */
.apply-button {
    background: linear-gradient(90deg, #fbbf24, #f59e0b); /* Amber */
    color: black;
    padding: 0.5rem 1.4rem;
    border-radius: 8px;
    font-weight: 700;
    text-decoration: none;
    transition: 0.3s ease;
}
.apply-button:hover {
    background: linear-gradient(90deg, #f59e0b, #d97706);
    color: white;
}

/* Similarity badge */
.similarity-badge {
    background: linear-gradient(90deg, #22c55e, #16a34a); /* Green */
    color: white;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-weight: 700;
}

/* Footer */
.footer {
    background: #0f172a;
    padding: 2rem;
    color: rgba(255,255,255,0.75);
    border-radius: 15px;
    margin-top: 3rem;
}
.footer h3 {
    color: #60a5fa;
    font-weight: 800;
}
.footer-link {
    color: #f1f5f9;
    text-decoration: none;
    font-weight: 600;
}
.footer-link:hover { color: #f97316; } /* Orange hover */
</style>
""", unsafe_allow_html=True)


DATA_PATH = r"E:\NextHikes_Project8_job_market_analysis_and_recommendation_system\job_posting.xls"
MODEL_PATH = r"E:\NextHikes_Project8_job_market_analysis_and_recommendation_system\tfidf_vectorizer.1.pkl"

@st.cache_data
def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
        data.dropna(subset=["job_description"], inplace=True)
        data["job_description"] = data["job_description"].fillna("").astype(str)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        raise

@st.cache_resource
def load_model(filepath):
    try:
        with open(filepath, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise

def recommend_jobs(user_input, tfidf_matrix, vectorizer, data):
    data_copy = data.copy()
    user_query_tfidf = vectorizer.transform([user_input])
    cosine_sim = cosine_similarity(user_query_tfidf, tfidf_matrix).flatten()
    data_copy["similarity_score"] = cosine_sim
    recommended_jobs = (
        data_copy.sort_values(by="similarity_score", ascending=False)
        .head(5)
        .loc[:, ["Cleaned Job Title", "Category", "country", "average_hourly_rate", "link", "similarity_score"]]
    )
    return recommended_jobs

def main():
    if 'page' not in st.session_state:
        st.session_state.page = "home"

    # 🔴 Global Top Heading
    st.markdown("""
    <h1 style="text-align:center; color:#FF0000; font-weight:900; font-size:3rem;">
        Job Market Analysis and Recommendation System
    </h1>
    <hr style="border: 2px solid #FF0000; margin: 1rem 0 2rem 0;">
    """, unsafe_allow_html=True)

    # Navigation
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        if st.button("🏠 Home"):
            st.session_state.page = "home"
    with col2:
        if st.button("💼 Jobs"):
            st.session_state.page = "jobs"
    with col3:
        if st.button("📊 Analytics"):
            st.session_state.page = "analytics"
    with col4:
        if st.button("📞 Contact"):
            st.session_state.page = "contact"

    # Pages
    if st.session_state.page == "home":
        st.markdown("""
        <div class="hero-section animate-fade-in">
            <h1 class="hero-title">Find Your Dream Job</h1>
            <p class="hero-subtitle">Powered By NextHikes IT Solutions</p>
            <div style="margin-top: 2rem;">
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 25px; margin: 0 0.5rem; color: white;">🤖 AI Powered</span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 25px; margin: 0 0.5rem; color: white;">⚡ Real-time</span>
                <span style="background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 25px; margin: 0 0.5rem; color: white;">🎯 Personalized</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            data = load_data(DATA_PATH)
            avg_jobs = len(data)
            unique_categories = data['Category'].nunique() if 'Category' in data.columns else 0
            unique_countries = data['country'].nunique() if 'country' in data.columns else 0
            avg_salary = data['average_hourly_rate'].mean() if 'average_hourly_rate' in data.columns else 0

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"<div class='stat-card'><div class='stat-number'>{avg_jobs:,}</div><div class='stat-label'>Total Jobs</div></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div class='stat-card'><div class='stat-number'>{unique_categories}</div><div class='stat-label'>Categories</div></div>", unsafe_allow_html=True)
            with col3:
                st.markdown(f"<div class='stat-card'><div class='stat-number'>{unique_countries}</div><div class='stat-label'>Countries</div></div>", unsafe_allow_html=True)
            with col4:
                st.markdown(f"<div class='stat-card'><div class='stat-number'>${avg_salary:.0f}</div><div class='stat-label'>Avg Rate/hr</div></div>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Failed to load system components: {e}")

    elif st.session_state.page == "jobs":
        try:
            data = load_data(DATA_PATH)
            vectorizer = load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"Failed to load system components: {e}")
            return

        st.markdown("<h2 style='color:#60a5fa; text-align:center;'>Discover Your Perfect Match</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input("", placeholder="e.g., Senior Data Scientist...", label_visibility="collapsed")
        with col2:
            search_button = st.button("Search Jobs", use_container_width=True)

        if user_input or search_button:
            if user_input.strip():
                with st.spinner("Finding the best matches for you..."):
                    tfidf_matrix = vectorizer.transform(data["job_description"])
                    recommendations = recommend_jobs(user_input, tfidf_matrix, vectorizer, data)
                if not recommendations.empty:
                    st.subheader("🎯 Your Personalized Job Matches")
                    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
                        similarity_percentage = round(row['similarity_score'] * 100, 1)
                        st.markdown(f"""
                        <div class="job-card">
                            <div class="job-title">🏆 #{idx} {row['Cleaned Job Title']}</div>
                            <div class="job-meta">
                                <div>📂 {row['Category']}</div>
                                <div>🌍 {row['country']}</div>
                                <div>💰 ${row['average_hourly_rate']}/hour</div>
                            </div>
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 1rem;">
                                <a href="{row['link']}" target="_blank" class="apply-button">📄 View Job</a>
                                <div class="similarity-badge">🎯 {similarity_percentage}% Match</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    elif st.session_state.page == "analytics":
        try:
            data = load_data(DATA_PATH)
        except Exception as e:
            st.error(f"Failed to load analytics: {e}")
            return

        st.subheader("📈 Job Market Trends")
        if 'Category' in data.columns:
            cat_counts = data['Category'].value_counts().head(10)
            fig = px.bar(
                x=cat_counts.values,
                y=cat_counts.index,
                orientation='h',
                title="Top Job Categories",
                color=cat_counts.values,
                color_continuous_scale="Turbo"
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.page == "contact":
        st.markdown("""
        <div class="analysis-section" style="text-align: center;">
            <h2 class="section-title">📞 Contact Information</h2>
            <p style="font-size: 1.2rem; color: #000000;">
                <b>📧 Email:</b> ABC@example.com<br>
                <b>🏢 Address:</b> NextHikes IT Solutions, Gurugram, India
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        <div class="footer-content">
            <h3>Built By Ashish Pachauri</h3>
            <p>Powered by NextHikes IT Solution • Advanced AI Technology • Real-time Job Matching</p>
        </div>
        <div class="footer-links">
            <a href="#" class="footer-link">🏠 Home</a>
            <a href="#" class="footer-link">📧 Contact</a>
            <a href="https://yoursite.com/privacy" target="_blank" class="footer-link">🔒 Privacy</a>
            <a href="https://yoursite.com/terms" target="_blank" class="footer-link">📋 Terms</a>
        </div>
        <div style="margin-top: 2rem; color: rgba(255,255,255,0.7);">
            <p>© 2024 JobSeeker Pro. All rights reserved.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

