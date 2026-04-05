## 2025-04-02 - Streamlit Rerun Optimization
**Learning:** Streamlit reruns the entire script on every user interaction (e.g., clicking a button, changing a slider, uploading a file). If file parsing (like `pd.read_csv`) isn't cached, the app will waste time and memory repeatedly loading the exact same file from scratch.
**Action:** Extract expensive operations like file loading into separate functions and decorate them with `@st.cache_data`. This caches the result based on the inputs (the uploaded file), so subsequent reruns bypass the expensive logic and fetch the dataframe instantly from memory.

## 2025-04-03 - Caching Computations Beyond I/O in Streamlit
**Learning:** Beyond caching I/O (like file loading), expensive dataframe computations (`df.describe()`, `df.info()`, `df.duplicated().sum()`) will severely degrade performance in Streamlit because they block the main thread on every rerun (e.g. when typing in a text field or adjusting a slider). Caching computations, not just data loading, is essential for UI responsiveness.
**Action:** Extract expensive EDA Pandas computations into a single helper function and decorate it with `@st.cache_data` so that it computes only once per dataset and immediately returns the cached metrics during interactive element state changes.

## 2024-05-24 - Streamlit Matplotlib Caching
**Learning:** In Streamlit, un-cached `sns.histplot(kde=True)` or complex matplotlib figures inside an interaction loop cause severe main-thread lag when unrelated UI interactions (like text inputs) trigger a rerun. Streamlit 1.14+ supports caching UI elements. Calling `st.pyplot(fig)` inside a function decorated with `@st.cache_data` will replay the element safely, resolving the bottleneck without needing complex image buffer returns.
**Action:** Use `@st.cache_data` on functions that wrap the entire `plt.subplots()` -> `st.pyplot(fig)` -> `plt.close(fig)` sequence to prevent expensive plot recomputations on non-dependent reruns.
