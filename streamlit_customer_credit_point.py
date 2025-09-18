import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict

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
    sql_query = re.sub(r"```[\w]*", "", sql_query)
    sql_query = sql_query.replace("```", "")
    return sql_query.strip()

def generate_score_sql(llm, user_input: str, columns: List[str], table_name: str = "df") -> str:
    """
    Use Gemini to generate a single SQL query that calculates the score for each row
    based on multiple criteria, and orders the result by Score descending.
    """
    prompt = PromptTemplate(
        input_variables=["criteria_input", "columns", "table_name"],
        template=(
            "You are a data assistant for a customer credit score dataset. "
            "The columns are: {columns}. "
            "The user provides multiple lines of criteria, each line contains Product, Segment, Logic Description, and Point. "
            "For each row in the table '{table_name}', calculate a Score by summing the Point for each criteria that the row matches. "
            "If a field (like Product or Segment) contains multiple values separated by commas, treat each value as a valid match. "
            "If the Score is not matched by any criteria or not provided, it should be 1. "
            "If the criteria logic is complex, translate it into valid SQL syntax. "
            "Write a single SQL query for SQLite that returns all columns, adds a 'Score' column and treat each criteria as new column showing the value of True if matched the criteria and empty space if not, and orders the result by Score descending. "
            "Input criteria:\n{criteria_input}\n"
            "Return ONLY the SQL query, nothing else."
        )
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    sql_query = chain.run({"criteria_input": user_input, "columns": ", ".join(columns), "table_name": table_name})
    return clean_sql_query(sql_query)

# --- 3. Streamlit UI ---

st.set_page_config(page_title="Customer Credit Score Pointing", layout="wide")
st.title("🏆 Customer Credit Score Pointing System")
st.write(
    "Paste your scoring criteria below (one per line, e.g. `CC, Payroll, AUM (IDR) > 10,000,000 and PL Balance = 0, 10`).\n"
    "The app will use Gemini 2.5 Flash, LangChain, RAG, and ReAct to process and score the data."
)

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

# --- Criteria Input Section ---
st.subheader("Paste Scoring Criteria (one per line)")
criteria_input = st.text_area(
    "Example:\nCC, Payroll, AUM (IDR) > 10,000,000 and PL Balance = 0, 10\nPL, ETB, AUM (IDR) > 5,000,000, 5",
    height=200,
    help="Each line: Product, Segment, Logic Description, Point"
)

submit = st.button("Calculate Scores", type="primary")

if submit:
    if not gemini_api_key:
        st.warning("Please enter your Google Gemini API key in the sidebar.")
    elif not criteria_input.strip():
        st.warning("Please enter at least one criteria.")
    else:
        with st.spinner("Generating SQL and scoring with Gemini..."):
            llm = get_gemini_llm(gemini_api_key)
            columns = df.columns.tolist()
            # Use GenAI to generate a single SQL query for scoring
            sql_query = generate_score_sql(llm, criteria_input, columns, table_name="df")
            st.subheader("Generated SQL Query")
            st.code(sql_query, language="sql")
            # Execute the SQL query using pandasql
            import pandasql
            try:
                result = pandasql.sqldf(sql_query, {"df": df})
                # Format numeric columns with thousand separator, including CC Limit and CC Balance even if dtype is object
                result_fmt = result.copy()
                for col in result_fmt.columns:
                    if col in ["CC Limit", "CC Balance"]:
                        result_fmt[col] = pd.to_numeric(result_fmt[col], errors='coerce').apply(
                            lambda x: f"{x:,.0f}" if pd.notnull(x) else ""
                        )
                    elif pd.api.types.is_numeric_dtype(result_fmt[col]):
                        result_fmt[col] = result_fmt[col].apply(lambda x: f"{x:,.0f}")
                st.success("Scoring complete! Showing top 20 customers by score:")
                st.dataframe(result_fmt.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"Could not execute the generated SQL. Error: {e}")