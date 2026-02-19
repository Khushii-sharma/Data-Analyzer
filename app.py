import os
import streamlit as st
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from openai import OpenAI

# ------------------- Setup -------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

st.set_page_config(page_title="AI Data Analyzer", layout="centered")

st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }

    /* Titles and Text */
    h1, h2, h3, p {
        color: #FFFFFF !important;
    }

    /* Glassmorphism for Dataframes and UI elements */
    .stDataFrame, div[data-testid="stExpander"], .stAlert {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(79, 172, 254, 0.3) !important;
        border-radius: 12px !important;
    }

    /* Button Styling (Neon Blue) */
    .stButton>button {
        background-color: transparent !important;
        color: #4FACFE !important;
        border: 2px solid #4FACFE !important;
        border-radius: 10px !important;
        font-weight: bold !important;
        width: 100%;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #4FACFE !important;
        color: #000000 !important;
        box-shadow: 0 0 15px rgba(79, 172, 254, 0.5);
    }

    /* Selectbox and Input styling */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1a1c24 !important;
        color: white !important;
    }

    /* File Uploader styling */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed #4FACFE !important;
        background: rgba(79, 172, 254, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------- Header -------------------
st.markdown("""
    <div style='text-align: center; padding: 10px;'>
        <h1 style='background: linear-gradient(90deg, #00F2FE, #4FACFE); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem;'> AI Data Analyzer</h1>
    </div>
""", unsafe_allow_html=True)

st.divider()

# ------------------- File Upload -------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # ------------------- Dataset Preview -------------------
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # ------------------- Basic Statistics -------------------
    st.subheader("Basic Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    # ------------------- Simple Visualization -------------------
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_columns) > 0:

        st.subheader("Data Visualization")

        selected_col = st.selectbox("Select a numeric column", numeric_columns)

        fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}", template="plotly_dark")
        fig.update_traces(marker_color='#4FACFE')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("No numeric columns found in dataset.")

    # ------------------- AI Insight -------------------
    st.subheader("AI Insight")

    if st.button("Generate AI Summary"):

        with st.spinner("Generating insight..."):

            sample_data = df.head(10).to_string()

            prompt = f"""
            You are a beginner-friendly data analyst.

            Here is a dataset sample:

            {sample_data}

            Provide a simple summary of what you observe.
            Keep explanation short and easy to understand.
            """

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You explain data in simple terms."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5
            )

            answer = response.choices[0].message.content

            st.success("AI Summary:")
            st.write(answer)

else:
    st.info("Please upload a CSV file to begin.")