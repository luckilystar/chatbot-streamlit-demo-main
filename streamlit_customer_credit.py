import streamlit as st
import pandas as pd
import numpy as np
import os
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# --- 1. Data Generation Functions ---

def generate_dummy_data(n_rows=10000, random_seed=42):
    np.random.seed(random_seed)
    data = {}

    # Product Apply: 'CC' or 'PL'
    data['Product Apply'] = np.random.choice(['CC', 'PL'], size=n_rows)

    # App ID: e.g., CC0001 or PL0005
    app_ids = []
    cc_count = 1
    pl_count = 1
    for prod in data['Product Apply']:
        if prod == 'CC':
            app_ids.append(f'CC{cc_count:04d}')
            cc_count += 1
        else:
            app_ids.append(f'PL{pl_count:04d}')
            pl_count += 1
    data['App ID'] = app_ids

    # Segment: Payroll (60%), ETB (30%), NTB (5%), ETB Priority (5%)
    seg_choices = ['Payroll'] * 60 + ['ETB'] * 30 + ['NTB'] * 5 + ['ETB Priority'] * 5
    data['Segment'] = np.random.choice(seg_choices, size=n_rows)

    # Collectibility: 1 (80%), 2 (10%), 3 (10%)
    coll_choices = ['1'] * 80 + ['2'] * 10 + ['3'] * 10
    data['Collectibility'] = np.random.choice(coll_choices, size=n_rows)

    # AUM (IDR): 70% between 5,000,000 and 10,000,000, 30% elsewhere
    aum = []
    for _ in range(n_rows):
        if np.random.rand() < 0.7:
            aum.append(np.random.randint(5_000_000, 10_000_001))
        else:
            # Exclude 5,000,000 to 10,000,000
            if np.random.rand() < 0.5:
                aum.append(np.random.randint(100_000, 5_000_000))
            else:
                aum.append(np.random.randint(10_000_001, 1_000_000_000))
    data['AUM (IDR)'] = aum

    # MOB (Months): 0 to 120
    data['MOB (Months)'] = np.random.randint(0, 121, size=n_rows)

    # PL Balance: 50% zeros, 50% random in range
    pl_balance = []
    for _ in range(n_rows):
        if np.random.rand() < 0.5:
            pl_balance.append(0)
        else:
            pl_balance.append(np.random.randint(-10_000, 300_000_001))
    data['PL Balance'] = pl_balance

    # Payroll Credit M-1, M-2, M-3: only nonzero if segment is Payroll
    for m in ['M-1', 'M-2', 'M-3']:
        vals = []
        for seg in data['Segment']:
            if seg == 'Payroll':
                vals.append(np.random.randint(1, 100_000_001))
            else:
                vals.append(0)
        data[f'Payroll Credit {m}'] = vals

    # CC Bscore: 60% blank, 40% numeric (of which 30% > 410)
    cc_bscore = []
    for _ in range(n_rows):
        if np.random.rand() < 0.6:
            cc_bscore.append('')
        else:
            if np.random.rand() < 0.3:
                cc_bscore.append(np.random.randint(411, 801))
            else:
                cc_bscore.append(np.random.randint(50, 411))
    data['CC Bscore'] = cc_bscore

    # CC Block Code, CC Limit, CC Balance: only if CC Bscore is not blank
    cc_block_code = []
    cc_limit = []
    cc_balance = []
    for i in range(n_rows):
        if data['CC Bscore'][i] == '':
            cc_block_code.append('')
            cc_limit.append('')
            cc_balance.append('')
        else:
            cc_block_code.append(np.random.choice(['BA', 'OL', 'CP']))
            # CC Limit: 80% < 50,000,000, 20% 50,000,000-800,000,000, rounded in millions
            if np.random.rand() < 0.8:
                limit = np.random.randint(2_000_000, 50_000_001)
            else:
                limit = np.random.randint(50_000_001, 800_000_001)
            limit = int(round(limit, -6))  # round to millions
            cc_limit.append(limit)
            # CC Balance: -5000 up to CC Limit + 5%
            max_bal = int(limit * 1.05)
            cc_balance.append(np.random.randint(-5000, max_bal + 1))
    data['CC Block Code'] = cc_block_code
    data['CC Limit'] = cc_limit
    data['CC Balance'] = cc_balance

    df = pd.DataFrame(data)
    return df

# --- 2. LangChain Gemini LLM Setup ---

def get_gemini_llm(api_key: str):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2
    )

def clean_sql_query(sql_query: str) -> str:
    """
    Remove markdown/code block formatting from the SQL query.
    """
    import re
    # Remove triple backticks and any language hints (e.g., ```sql, ```sqlite)
    sql_query = re.sub(r"```[\w]*", "", sql_query)
    sql_query = sql_query.replace("```", "")
    return sql_query.strip()

def run_llm_query(llm, df: pd.DataFrame, user_query: str):
    """
    Use Gemini to generate a SQL query, execute it, and return results as a table and explanation.
    """
    # Prompt template for Gemini
    prompt = PromptTemplate(
        input_variables=["query", "columns"],
        template=(
            "You are a data assistant for a customer credit score dataset. "
            "The columns are: {columns}. "
            "Given the user query: '{query}', "
            "write a SQL query for SQLite to answer it. "
            "Assume the table is named 'df'. "
            "Return ONLY the SQL query, nothing else."
        )
    )
    columns = ', '.join(df.columns)
    # Step 1: Ask Gemini for the SQL query
    chain = LLMChain(llm=llm, prompt=prompt)
    sql_query = chain.run({"query": user_query, "columns": columns})

    # Clean the SQL query before execution
    sql_query_clean = clean_sql_query(sql_query)

    import pandasql
    try:
        result = pandasql.sqldf(sql_query_clean, {"df": df})
        # Step 3: Ask Gemini to explain the result
        explain_prompt = PromptTemplate(
            input_variables=["query", "result"],
            template=(
                "Given the user query: '{query}', and the result: '{result}', "
                "explain the result in a concise, user-friendly way."
            )
        )
        explain_chain = LLMChain(llm=llm, prompt=explain_prompt)
        # Show only first 5 rows for explanation to avoid flooding the LLM
        result_for_llm = result.head(5).to_markdown(index=False)
        explanation = explain_chain.run({"query": user_query, "result": result_for_llm})
        return sql_query, result, explanation
    except Exception as e:
        return sql_query, None, f"Sorry, there was an error executing the generated SQL: {e}"

# --- 3. Streamlit UI ---

st.set_page_config(page_title="Customer Credit Score Data Assistant", layout="wide")
st.title("💳 Customer Credit Score Data Assistant")
st.write("This app generates dummy customer data and lets you query it using natural language. Powered by Gemini 2.5 Flash and LangChain.")

# Sidebar for Gemini API Key
with st.sidebar:
    st.subheader("Settings")
    gemini_api_key = st.text_input("Google Gemini API Key", type="password")
    st.caption("Your API key is required to use Gemini 2.5 Flash.")

# Generate data on startup
@st.cache_data(show_spinner=True)
def get_data():
    return generate_dummy_data(10000)

df = get_data()

# Show first few rows with thousand separator formatting
st.subheader("Sample Data")
def format_thousands(x):
    if isinstance(x, (int, float, np.integer, np.floating)):
        return f"{x:,.0f}"
    return x

st.dataframe(df.head(20).applymap(format_thousands), use_container_width=True)

# User query input
st.subheader("Ask a question about the data")
user_query = st.text_input("Enter your query (e.g., 'What is the average AUM for Payroll segment?')", key="user_query")
submit = st.button("Submit", type="primary")

# Output area
if submit:
    if not gemini_api_key:
        st.warning("Please enter your Google Gemini API key in the sidebar.")
    else:
        with st.spinner("Processing your query..."):
            llm = get_gemini_llm(gemini_api_key)
            sql_query, result, explanation = run_llm_query(llm, df, user_query)
            st.markdown(f"**Generated SQL Query:**\n```sql\n{sql_query}\n```")
            if result is not None and not result.empty:
                # Format numeric columns with thousand separator
                result_fmt = result.copy()
                for col in result_fmt.select_dtypes(include=[np.number]).columns:
                    result_fmt[col] = result_fmt[col].apply(lambda x: f"{x:,.0f}")
                st.markdown("**Query Result:**")
                st.dataframe(result_fmt, use_container_width=True)
                st.markdown(f"**Explanation:**\n{explanation}")
            elif result is not None and result.empty:
                st.info("The query executed successfully, but returned no results.")
                st.markdown(f"**Explanation:**\n{explanation}")
            else:
                st.error(explanation)