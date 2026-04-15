## 2025-04-02 - Streamlit Rerun Optimization
**Learning:** Streamlit reruns the entire script on every user interaction (e.g., clicking a button, changing a slider, uploading a file). If file parsing (like `pd.read_csv`) isn't cached, the app will waste time and memory repeatedly loading the exact same file from scratch.
**Action:** Extract expensive operations like file loading into separate functions and decorate them with `@st.cache_data`. This caches the result based on the inputs (the uploaded file), so subsequent reruns bypass the expensive logic and fetch the dataframe instantly from memory.

## 2025-04-03 - Caching Computations Beyond I/O in Streamlit
**Learning:** Beyond caching I/O (like file loading), expensive dataframe computations (`df.describe()`, `df.info()`, `df.duplicated().sum()`) will severely degrade performance in Streamlit because they block the main thread on every rerun (e.g. when typing in a text field or adjusting a slider). Caching computations, not just data loading, is essential for UI responsiveness.
**Action:** Extract expensive EDA Pandas computations into a single helper function and decorate it with `@st.cache_data` so that it computes only once per dataset and immediately returns the cached metrics during interactive element state changes.

## 2024-05-18 - PyArrow CSV Engine Optimization
**Learning:** Pandas `read_csv` can be significantly sped up by using `engine='pyarrow'` which provides a multi-threaded C++ backend. This is especially useful for large files and since PyArrow is already a Streamlit dependency, it can be used safely without adding new external requirements.
**Action:** Use `engine='pyarrow'` in `pd.read_csv()` calls when performance is critical and PyArrow is available.

## 2026-04-15 - Seaborn KDE Performance in Streamlit
**Learning:** In Streamlit applications, Seaborn's Kernel Density Estimation (e.g., `sns.histplot` with `kde=True`) is computationally expensive and blocks the main thread, causing severe UI freezing for datasets with thousands of rows.
**Action:** Dynamically disable KDE for large datasets (e.g., >50,000 rows) using a length check (`len(df)`) to ensure the application remains responsive during interactive data exploration.
