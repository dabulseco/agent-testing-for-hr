import pandas as pd
import streamlit as st

import hr_core
import hr_ui

tool = st.segmented_control(
    "Tool", ["SQL query tool", "Policy search tool", "Python REPL tool"],
    default="SQL query tool", label_visibility="collapsed",
)

if tool == "SQL query tool":
    st.caption("Read-only, SELECT-only, and every query must filter by `:employee_id`. "
               "Try the notebook's three tests or write your own.")
    preset = st.selectbox("Preset", [*hr_core.SQL_TEST_PRESETS, "Custom query"])
    with st.form("sql_form"):
        sql = st.text_area("SQL", value=hr_core.SQL_TEST_PRESETS.get(preset, ""), height=150,
                           placeholder="SELECT ... WHERE employee_id = :employee_id")
        employee_id = st.text_input("Employee ID", value="EMP001")
        submitted = st.form_submit_button("Run SQL tool", icon=":material/play_arrow:", type="primary")
    if submitted:
        result = hr_ui.get_sql_tool().invoke({"query": sql, "employee_id": employee_id})
        if result.startswith("ERROR"):
            st.warning(result, icon=":material/block:")
        else:
            st.code(result, language=None, wrap_lines=True)
    with st.expander("Tool description the agent sees"):
        st.text(hr_ui.get_sql_tool().description)

elif tool == "Policy search tool":
    c = hr_ui.cfg()
    st.caption(f"Similarity search over the policy PDFs using `{st.session_state.embed_model}` embeddings, "
               f"returning the top {c['k']} chunks (change k in Agent settings).")
    with st.form("policy_form"):
        query = st.text_input("Search query", value="flexible working hours eligibility departments")
        submitted = st.form_submit_button("Search policies", icon=":material/search:", type="primary")
    if submitted:
        vs = hr_ui.get_vectorstore(st.session_state.embed_model, c["chunk_size"], c["chunk_overlap"])
        st.caption(f"Vector store contains {vs._collection.count()} chunks.")
        for rank, (doc, distance) in enumerate(vs.similarity_search_with_score(query, k=c["k"]), start=1):
            with st.container(border=True):
                st.markdown(f"**{rank}. {doc.metadata['document_name']}** · page {doc.metadata.get('page', 'N/A')} "
                            f"· distance {distance:.3f}")
                st.text(doc.page_content)
        with st.expander("Raw tool output (what the agent receives)"):
            st.text(hr_ui.get_policy_tool().invoke(query))

else:
    st.caption("Runs Python for calculations. The preset is the notebook's tenure calculation test.")
    with st.form("python_form"):
        code = st.text_area("Python code", value=hr_core.PYTHON_TEST_PRESET, height=180)
        submitted = st.form_submit_button("Run Python tool", icon=":material/play_arrow:", type="primary")
    if submitted:
        result = hr_ui.get_python_tool().invoke(code)
        st.code(result or "(no output: remember to print the result)", language=None, wrap_lines=True)

st.divider()
st.subheader("Summary of tools")
st.dataframe(pd.DataFrame([
    {"Tool": "sql_query_tool", "Purpose": "Query the HR database for employee-specific data",
     "Guardrails": "Read-only DB connection, SELECT-only check, must filter by employee_id"},
    {"Tool": "search_company_policies", "Purpose": "Retrieve relevant policy document sections",
     "Guardrails": "Returns only retrieved text; agent told not to invent numbers"},
    {"Tool": "Python_REPL", "Purpose": "Date math, balances, proration, salary calculations",
     "Guardrails": "Instructed not to access the database directly"},
]), hide_index=True)
