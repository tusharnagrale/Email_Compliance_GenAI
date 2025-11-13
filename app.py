import os
import json
import streamlit as st
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ---------------- CONFIG ----------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"  # Change to gpt-4o or gpt-4.1 for more accuracy

llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0,
    api_key=OPENAI_API_KEY
)

CATEGORIES = [
    "Secrecy",
    "Market Manipulation/Misconduct",
    "Market Bribery",
    "Change in communication",
    "Complaints",
    "Employee ethics",
]

CATEGORY_WEIGHTS = {
    "Secrecy": 3.0,
    "Market Manipulation/Misconduct": 5.0,
    "Market Bribery": 5.0,
    "Change in communication": 2.0,
    "Complaints": 2.5,
    "Employee ethics": 3.0,
}


# ---------------- LANGCHAIN PROMPT TEMPLATE ----------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior bank compliance officer with 20+ years regulatory experience. "
        "Your job is to classify emails for financial regulatory risk, behavioural misconduct, "
        "client harm, market abuse, bribery, or ethics issues. You MUST return STRICT JSON only."
    ),
    (
        "user",
        """
Analyse the following email for potential non-compliance.

Classify it into one or more of these categories:
{categories}

THEN:
- Extract exact offending lines
- Give a short explanation
- Assign a risk score from **1 to 10**
- Assign risk level based on score:
    1-3    -> Low
    4-6    -> Medium
    7-10   -> High

Your JSON MUST follow this structure:

{{
  "categories": ["Category1", "Category2"],
  "offending_lines": ["line1", "line2"],
  "explanation": "Short explanation",
  "risk_score": 8,
  "risk_level": "High"
}}

EMAIL:
\"\"\"{email_text}\"\"\""""
    )
])


# ---------------- ANALYSIS FUNCTION ----------------

def analyze_email_langchain(email_text):
    formatted_prompt = prompt.format(
        categories=CATEGORIES,
        email_text=email_text
    )

    response = llm.invoke([{"role": "user", "content": formatted_prompt}])
    raw_output = response.content.strip()

    try:
        result = json.loads(raw_output)
    except Exception as e:
        result = {
            "categories": [],
            "offending_lines": [],
            "explanation": f"Could not parse JSON: {raw_output}",
            "risk_score": 0,
            "risk_level": "None"
        }

    return {
        "categories": result.get("categories", []),
        "offending_lines": result.get("offending_lines", []),
        "explanation": result.get("explanation", ""),
        "risk_score": result.get("risk_score", 0),
        "risk_level": result.get("risk_level", "None")
    }


# ---------------- STREAMLIT APP ----------------

st.set_page_config(page_title="Batch Email Compliance (LangChain)", layout="wide")

st.title("📧 AI Email Compliance Checker")
st.write("Upload an Excel/CSV file — the app will analyze every email and retain **all original columns** in the output.")

uploaded = st.file_uploader("Upload file (.xlsx/.csv)", type=["xlsx", "xls", "csv"])

if uploaded is None:
    st.stop()

# Load data
ext = uploaded.name.split(".")[-1].lower()
if ext in ["xlsx", "xls"]:
    df = pd.read_excel(uploaded)
    df = df.iloc[:10, :]
else:
    df = pd.read_csv(uploaded)

df.columns = [c.lower().strip() for c in df.columns]
if "id" not in df.columns:
    df["id"] = range(1, len(df) + 1)

if "body" not in df.columns:
    st.error("❌ Excel must contain a 'body' column for email text.")
    st.stop()

st.subheader("📂 Uploaded Data Preview")
st.dataframe(df.head(), width='stretch')


# ---------------- RUN BATCH ANALYSIS ----------------
if st.button("🚨 Analyse ALL Emails"):
    results = []
    with st.spinner("Analysing all emails..."):
        for _, row in df.iterrows():
            email_id = row["id"]
            body = str(row["body"])

            llm_result = analyze_email_langchain(body)

            results.append({
                "id": email_id,
                "categories": ", ".join(llm_result["categories"]),
                "offending_lines": " | ".join(llm_result["offending_lines"]),
                "explanation": llm_result["explanation"],
                "risk_score": llm_result["risk_score"],
                "risk_level": llm_result["risk_level"]
            })

    result_df = pd.DataFrame(results)

    # Merge back to keep ALL original columns
    final_df = pd.merge(df, result_df, on="id", how="left")

    st.subheader("🚩 Final Results (All Columns Retained)")
    st.dataframe(final_df.head(), width='stretch')

    # Create downloadable Excel
    output = BytesIO()
    final_df.to_excel(output, index=False, sheet_name="Results")
    output.seek(0)

    st.download_button(
        label="⬇️ Download Full Results Excel",
        data=output,
        file_name="compliance_results_full.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.success("✅ Done! All original columns + LLM results are included.")
