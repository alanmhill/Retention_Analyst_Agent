import json
import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://app:8000")

st.set_page_config(page_title="Agentic Retention Analyst", layout="wide")
st.title("Agentic HR Retention Analyst")


# Sidebar
with st.sidebar:
    st.header("API Controls")
    if st.button("Check API Health"):
        try:
            r = requests.get(f"{API_BASE_URL}/health", timeout=10)
            st.json(r.json())
        except Exception as e:
            st.error(str(e))

st.divider()

# File Upload
st.header("1️⃣ Upload Dataset")
uploaded_file = st.file_uploader("Upload CSV or XLSX", type=["csv", "xlsx"])

if "dataset_id" not in st.session_state:
    st.session_state.dataset_id = None

if uploaded_file:
    if st.button("Upload to Agent"):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
        try:
            r = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=120)
            r.raise_for_status()
            data = r.json()
            st.session_state.dataset_id = data["dataset_id"]
            st.success("Upload successful")
            st.json(data)
        except Exception as e:
            st.error(str(e))

if st.session_state.dataset_id:
    st.info(f"Current dataset_id: {st.session_state.dataset_id}")

st.divider()

# Ask Section
st.header("2️⃣ Ask Question")

question = st.text_input(
    "Ask something about attrition:",
    value="Which department and job roles have the highest attrition rate? Provide the top 3 and include counts."
)

top_k = st.slider("RAG Context (top_k)", 3, 12, 6)

if st.button("Run Analysis"):
    if not st.session_state.dataset_id:
        st.error("Upload a dataset first.")
    else:
        payload = {
            "dataset_id": st.session_state.dataset_id,
            "question": question,
            "top_k": top_k
        }

        try:
            r = requests.post(
                f"{API_BASE_URL}/ask",
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=180
            )
            r.raise_for_status()
            result = r.json()

            analysis = result.get("analysis", {})

            # ================================
            # KPI CARDS
            # ================================
            overall = analysis.get("overall_attrition_rate", 0)

            dept_segments = analysis.get("segment_attrition", {}).get("Department", [])
            top_dept = dept_segments[0]["group"] if dept_segments else "N/A"

            col1, col2 = st.columns(2)
            col1.metric("Overall Attrition Rate", f"{overall*100:.1f}%")
            col2.metric("Highest Risk Department", top_dept)

            st.divider()

            # ================================
            # BAR CHART - Attrition by Dept
            # ================================
            import pandas as pd

            if dept_segments:
                df_dept = pd.DataFrame(dept_segments)
                st.subheader("Attrition by Department")
                st.bar_chart(
                    df_dept.set_index("group")["attrition_rate"]
                )

            st.divider()

            # ================================
            # EXECUTIVE SUMMARY
            # ================================
            st.subheader("📊 Executive Summary")
            st.write(result.get("executive_summary"))

            with st.expander("Structured Analysis"):
                st.json(analysis)

            with st.expander("RAG Retrieved Context"):
                st.json(result.get("retrieved"))

        except Exception as e:
            st.error(str(e))