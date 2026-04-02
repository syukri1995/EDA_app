## 2024-05-18 - Streamlit Re-rendering Bottleneck
**Learning:** Streamlit runs the entire script on every user interaction. Reading files without caching means data is re-parsed constantly, blocking the main thread.
**Action:** Always wrap expensive operations like `pd.read_csv` and `pd.read_excel` with `@st.cache_data` in Streamlit apps.
