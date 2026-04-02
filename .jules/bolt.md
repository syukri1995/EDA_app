## 2024-04-02 - Streamlit File Upload Caching
**Learning:** Uploading a large CSV/Excel file in Streamlit re-parses the file on every interaction (button click, text input, slider change) because Streamlit reruns the script top-to-bottom. Reading large datasets over and over is a massive bottleneck.
**Action:** Always wrap file loading functions (`pd.read_csv`, `pd.read_excel`) in `@st.cache_data`. This caches the parsed DataFrame in memory, making all subsequent interactions significantly faster by avoiding redundant I/O and parsing overhead.
