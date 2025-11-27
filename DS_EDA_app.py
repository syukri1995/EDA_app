import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import numpy as np
import re
import difflib


# Docstring for DS_EDA_app
# a.) user can upload csv, excel 
# b.) basic eda --> show preview, info, describe, no. of missing values, no. of duplicate records
# c.) ask user to select columns [multiselect]
# d.) provide some diff diff graphs to the user
# e.) generate the graph using seaborn + matplotlib
# f.) ask user for some query, e.g., "top 5 categories" or "records where customer initiated more than 5 customer service calls"

st.set_page_config(layout="wide", page_title="Data Science EDA App")

st.title("📊 Data Science EDA Application")
st.markdown("---")
st.write("Upload your dataset (CSV or Excel) and perform basic Exploratory Data Analysis (EDA) in real-time.")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Read the uploaded file
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
        
    st.markdown("### 1. Basic Data Overview")
    
    # Dataset Preview
    head = st.slider("Number of rows to view", min_value=5, max_value=20, value=5)
    st.subheader("Dataset Preview (First 5 Rows)")
    st.dataframe(df.head(head))

    # Dataset Information
    st.subheader("Dataset Information (`df.info()`)")
    # Capture the output of df.info()
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    # Statistical Summary
    st.subheader("Statistical Summary (`df.describe()`)")
    st.dataframe(df.describe().style.format(precision=2))

    # Missing Values and Duplicates
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Missing Values")
        missing_values = df.isnull().sum()
        # Filter to show only columns with missing values
        missing_df = missing_values[missing_values > 0].reset_index()
        missing_df.columns = ['Feature', 'Missing Count']
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("🎉 No missing values found in the dataset!")

    with col2:
        st.subheader("Duplicate Records")
        duplicate_count = df.duplicated().sum()
        st.info(f"Total number of duplicate records: **{duplicate_count}**")
        if duplicate_count > 0:
            st.warning("Consider removing duplicates to ensure data quality.")

    st.markdown("---")

    st.markdown("### 2. Column Visualization")
    
    # Column Selection for Visualization
    selected_columns = st.multiselect(
        "Select one or more columns for visualization:", 
        df.columns.tolist(),
        help="Select a column to see its distribution (Histogram for numeric, Count Plot for categorical)."
    )

    if selected_columns:
        st.markdown("#### Generated Plots (Seaborn + Matplotlib)")
        
        # Determine the number of plots per row
        cols_per_row = 2 
        
        # Use st.columns to dynamically create plot layout
        plot_cols = st.columns(cols_per_row)
        
        for i, col in enumerate(selected_columns):
            # Place the plot in the next column in the cycle
            with plot_cols[i % cols_per_row]:
                
                # Check for numeric types
                if pd.api.types.is_numeric_dtype(df[col]):
                    fig, ax = plt.subplots(figsize=(7, 5))
                    sns.histplot(df[col].dropna(), kde=True, ax=ax, palette='viridis')
                    ax.set_title(f'Distribution (Histogram/KDE) of {col}', fontsize=14)
                    ax.set_xlabel(col)
                    st.pyplot(fig)
                    plt.close(fig) # Close figure to free memory
                    
                # Check for categorical/object types
                elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                    # Limit to top 20 unique values for readability in count plots
                    value_counts = df[col].value_counts().head(20)
                    
                    fig, ax = plt.subplots(figsize=(7, max(5, len(value_counts) * 0.4))) # Dynamic height
                    sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax, palette='mako')
                    ax.set_title(f'Count Plot of {col} (Top {len(value_counts)})', fontsize=14)
                    ax.set_xlabel('Count')
                    ax.set_ylabel(col)
                    st.pyplot(fig)
                    plt.close(fig) # Close figure to free memory

def best_match(user_col, df_cols):
    """Return closest column using fuzzy matching."""
    match = difflib.get_close_matches(user_col.lower(), [c.lower() for c in df_cols], n=1, cutoff=0.4)
    if match:
        for c in df_cols:
            if c.lower() == match[0]:
                return c
    return None

st.markdown("---")
st.markdown("### 3. Ask a Question About the Data")

# randomize column name sugestion
ran_col = np.random.choice(df.columns)
st.info(f"Try queries like: 'top 5 {ran_col}', '{ran_col } > 50', '{ran_col} contains abc'")
user_query = st.text_input(
    "Type your query",placeholder="e.g., top 5 categories"
)

if not user_query:
    st.stop()

query = user_query.strip().lower()
query_successful = False


# ===============================
# 0. Utility − detect column name
# ===============================
def detect_column(text):
    """Find best matching column for user input fragment."""
    # remove "top", "where", extra words
    text = text.replace("top", "").replace("where", "").strip()
    return best_match(text, df.columns)


# ===============================
# 1. TOP N queries
# "top 5 category", "top category", "top 10 items"
# ===============================
m = re.search(r"top\s*(\d*)\s*(.*)", query)
if m and not query_successful:
    raw_n = m.group(1)
    raw_col = m.group(2).strip()

    n = int(raw_n) if raw_n.isdigit() else 5

    col = detect_column(raw_col)
    if col:
        result = df[col].value_counts().head(n)
        st.success(f"Top **{n}** most frequent values in **{col}**")
        st.dataframe(result)
        query_successful = True


# ===============================
# 2. CONDITION queries
# "age > 30", "salary <= 5000", "score = 10"
# ===============================
if not query_successful:
    cond = re.search(r"(.+?)\s*(==|>=|<=|>|<|=)\s*(.+)", query)

    if cond:
        col_raw = cond.group(1).strip()
        op = cond.group(2).replace("=", "==")  # treat "=" as "=="
        val_raw = cond.group(3).strip()

        col = detect_column(col_raw)

        if col:
            # convert number if possible
            try: val = float(val_raw)
            except: val = val_raw

            try:
                # pandas query string
                expr = f"`{col}` {op} @val"
                result = df.query(expr)

                st.success(f"Filtered rows where **{col} {op} {val}**")
                st.dataframe(result)
                query_successful = True
            except Exception as e:
                st.error(f"Error evaluating condition: {e}")


# ===============================
# 3. CONTAINS search
# "name contains john", "email contain gmail"
# ===============================
if not query_successful:
    m = re.search(r"(.*)\s+contain[s]?\s+(.*)", query)
    if m:
        col_raw = m.group(1).strip()
        text = m.group(2).strip().strip('"').strip("'")

        col = detect_column(col_raw)

        if col:
            result = df[df[col].astype(str).str.contains(text, case=False, na=False)]
            st.success(f"Rows where **{col}** contains **'{text}'**")
            st.dataframe(result)
            query_successful = True


# ===============================
# 4. Select multiple columns
# "show age, salary, city", "select name, email"
# ===============================
if not query_successful and any(k in query for k in ["show", "select"]):
    parts = re.split(r",| and | ", query)
    parts = [p.strip() for p in parts if p.strip()]

    selected = []
    for p in parts:
        col = detect_column(p)
        if col and col not in selected:
            selected.append(col)

    if len(selected) >= 2:
        st.success("Showing selected columns:")
        st.dataframe(df[selected])
        query_successful = True


# ===============================
# 5. If no match
# ===============================
if not query_successful:
    st.error("❌ Query not understood. Try examples:\n- `age > 30`\n- `top 5 categories`\n- `name contains john`\n- `show age, salary`")
