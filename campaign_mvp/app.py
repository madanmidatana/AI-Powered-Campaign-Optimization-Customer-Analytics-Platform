"""
AI Campaign Optimizer — MVP Streamlit Application
4-page dashboard: Overview · Segments · Predictions · Campaign Generator
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from pathlib import Path
from dotenv import load_dotenv

from modules.data_loader import load_data
from modules.segmentation import segment_customers, get_segment_summary
from modules.analytics import compute_kpis
from modules.prediction import train_model

# Load environment variables (from .env during local development)
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    try:
        st.sidebar.warning(
            "GEMINI_API_KEY not set. Create a local .env or set the environment variable. See .env.example."
        )
    except Exception:
        # If Streamlit isn't running yet, silently continue
        pass

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Campaign Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for a polished look
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea11, #764ba211);
        border: 1px solid #667eea33;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.08);
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.7;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* Main header */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        margin-bottom: 2rem;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #333;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Load & process data (cached)
# ──────────────────────────────────────────────
@st.cache_data
def prepare_data():
    df = load_data()
    df = segment_customers(df)
    df, metrics = train_model(df)
    return df, metrics


df, model_metrics = prepare_data()

# ──────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 Campaign AI")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Overview", "👥 Segments", "🎯 Predictions", "✉️ Campaign Generator"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(f"Dataset: {len(df)} customers")
    st.caption(f"Model accuracy: {model_metrics['accuracy']:.1%}")


# ══════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════
if page == "📊 Overview":
    st.markdown('<p class="main-header">Campaign Overview</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Key performance indicators and customer distribution at a glance.</p>', unsafe_allow_html=True)

    kpis = compute_kpis(df)

    # ── KPI cards ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{kpis['total_customers']:,}")
    c2.metric("Avg CLV", f"${kpis['avg_clv']:,.2f}")
    c3.metric("Top Segment", kpis["top_segment"], f"{kpis['top_segment_size']} customers")
    c4.metric("Conversion Rate", f"{kpis['conversion_rate']}%")

    st.markdown("---")

    # ── Charts row ──
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Segment Distribution")
        seg_counts = df["segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]
        colors = ["#667eea", "#764ba2", "#f093fb", "#4facfe"]
        fig_bar = px.bar(
            seg_counts, x="Segment", y="Count",
            color="Segment",
            color_discrete_sequence=colors,
            template="plotly_dark",
        )
        fig_bar.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin=dict(t=20, b=40),
            xaxis_title="",
            yaxis_title="Customers",
        )
        st.plotly_chart(fig_bar, width="stretch")

    with col_right:
        st.subheader("Purchases Over Time")
        if "last_purchase_date" in df.columns:
            time_df = (
                df.groupby(df["last_purchase_date"].dt.to_period("M"))
                .agg(Total_Purchases=("num_purchases", "sum"))
                .reset_index()
            )
            time_df["last_purchase_date"] = time_df["last_purchase_date"].dt.to_timestamp()
            fig_line = px.area(
                time_df, x="last_purchase_date", y="Total_Purchases",
                template="plotly_dark",
                color_discrete_sequence=["#667eea"],
            )
            fig_line.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=20, b=40),
                xaxis_title="Month",
                yaxis_title="Purchases",
            )
            st.plotly_chart(fig_line, width="stretch")
        else:
            st.info("No date column found for time-series chart.")


# ══════════════════════════════════════════════
# PAGE 2 — Segments
# ══════════════════════════════════════════════
elif page == "👥 Segments":
    st.markdown('<p class="main-header">Customer Segments</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">KMeans clustering on RFM features — explore each segment in detail.</p>', unsafe_allow_html=True)

    # ── Summary table ──
    summary = get_segment_summary(df)
    st.dataframe(summary, width="stretch", hide_index=True)

    st.markdown("---")

    # ── 3D scatter ──
    st.subheader("RFM Scatter Plot")
    colors_map = {
        "High Value": "#667eea",
        "Growth": "#4facfe",
        "At Risk": "#f093fb",
        "Dormant": "#fd746c",
    }
    fig_scatter = px.scatter_3d(
        df,
        x="recency_score", y="frequency_score", z="monetary_score",
        color="segment",
        color_discrete_map=colors_map,
        opacity=0.7,
        template="plotly_dark",
        hover_data=["customer_id", "total_spend"],
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=20),
        scene=dict(
            xaxis_title="Recency",
            yaxis_title="Frequency",
            zaxis_title="Monetary",
        ),
        height=550,
    )
    st.plotly_chart(fig_scatter, width="stretch")

    st.markdown("---")

    # ── Segment drill-down ──
    st.subheader("Segment Drill-Down")
    selected_segment = st.selectbox("Select a segment", sorted(df["segment"].unique()))
    filtered = df[df["segment"] == selected_segment][
        ["customer_id", "age", "gender", "total_spend", "num_purchases",
         "recency_score", "frequency_score", "monetary_score", "campaign_response"]
    ]
    st.dataframe(filtered, width="stretch", hide_index=True)
    st.caption(f"Showing {len(filtered)} customers in **{selected_segment}** segment.")


# ══════════════════════════════════════════════
# PAGE 3 — Predictions
# ══════════════════════════════════════════════
elif page == "🎯 Predictions":
    st.markdown('<p class="main-header">Response Predictions</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Logistic Regression model — identify the customers most likely to respond.</p>', unsafe_allow_html=True)

    # ── Model metrics ──
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{model_metrics['accuracy']:.1%}")
    m2.metric("AUC Score", f"{model_metrics['auc']:.3f}")
    m3.metric("Features Used", "4 (RFM + Email Opens)")

    st.markdown("---")

    # ── Probability filter ──
    col_slider, col_n = st.columns([3, 1])
    with col_slider:
        min_prob = st.slider(
            "Minimum response probability",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        )
    with col_n:
        top_n = st.number_input("Show top N", min_value=5, max_value=500, value=50, step=5)

    top_responders = (
        df[df["response_probability"] >= min_prob]
        .sort_values("response_probability", ascending=False)
        .head(top_n)
        [["customer_id", "segment", "email_opens", "total_spend",
          "response_probability", "campaign_response"]]
    )

    st.subheader(f"Top Predicted Responders (≥ {min_prob:.0%})")
    st.dataframe(top_responders, width="stretch", hide_index=True)
    st.caption(f"{len(top_responders)} customers match the filter.")

    st.markdown("---")

    # ── Distribution chart ──
    st.subheader("Probability Distribution")
    fig_hist = px.histogram(
        df, x="response_probability", nbins=30,
        color_discrete_sequence=["#667eea"],
        template="plotly_dark",
    )
    fig_hist.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=20, b=40),
        xaxis_title="Response Probability",
        yaxis_title="Count",
    )
    st.plotly_chart(fig_hist, width="stretch")


# ══════════════════════════════════════════════
# PAGE 4 — Campaign Generator
# ══════════════════════════════════════════════
elif page == "✉️ Campaign Generator":
    st.markdown('<p class="main-header">Campaign Generator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate personalized marketing messages with Google Gemini AI.</p>', unsafe_allow_html=True)

    col_config, col_preview = st.columns([1, 1.5])

    with col_config:
        st.subheader("Configure")

        segment = st.selectbox("Target Segment", sorted(df["segment"].unique()), key="gen_seg")
        channel = st.radio("Channel", ["Email", "SMS", "Push Notification"], horizontal=True)

        # Auto-generate top features from segment stats
        seg_data = df[df["segment"] == segment]
        avg_spend = seg_data["total_spend"].mean()
        avg_freq = seg_data["num_purchases"].mean()
        avg_recency = seg_data["days_since_last_purchase"].mean()
        auto_features = (
            f"avg spend ${avg_spend:.0f}, "
            f"avg {avg_freq:.0f} purchases, "
            f"avg {avg_recency:.0f} days since last purchase"
        )
        top_features = st.text_area(
            "Key traits (editable)",
            value=auto_features,
            height=80,
        )

        generate_btn = st.button("🤖 Generate with Gemini", width="stretch", type="primary")
        regenerate_btn = st.button("🔄 Regenerate", width="stretch")

    with col_preview:
        st.subheader("Generated Message")

        if generate_btn or regenerate_btn:
            with st.spinner("Generating with Gemini AI..."):
                try:
                    from modules.campaign_gen import generate_campaign
                    message = generate_campaign(segment, channel, top_features)
                    st.session_state["last_message"] = message
                except Exception as e:
                    st.error(f"Gemini API error: {e}")
                    st.info("Set `GEMINI_API_KEY` as an environment variable or in a local .env file.")
                    message = None

        # Display last generated message (persists across reruns)
        if "last_message" in st.session_state and st.session_state["last_message"]:
            st.text_area(
                "Message (editable — copy when ready)",
                value=st.session_state["last_message"],
                height=200,
                key="output_msg",
            )

            # Stats
            word_count = len(st.session_state["last_message"].split())
            char_count = len(st.session_state["last_message"])
            st.caption(f"📝 {word_count} words · {char_count} characters · Channel: {channel}")
        else:
            st.info("👆 Configure your campaign and click **Generate with Gemini** to create a message.")
