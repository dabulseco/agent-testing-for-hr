import streamlit as st

import hr_core
import hr_ui

gen_model, eval_model, embed_model = hr_ui.selected_models()

st.markdown(
    "TechCorp's HR team answers 80 to 100 routine employee queries a day: leave balances, policies, "
    "payroll, and performance. This app runs a single **ReAct agent** that answers those queries using three "
    "tools: a guarded read-only **SQL tool** over the HR database, a **policy search** (RAG) tool over four "
    "policy PDFs, and a **Python REPL** for calculations. An **LLM judge** scores answers against reference "
    "responses. All models run locally through Ollama."
)

with st.container(horizontal=True):
    st.metric("Agent LLM", gen_model, border=True)
    st.metric("Judge LLM", eval_model, border=True)
    st.metric("Embeddings", embed_model, border=True)

st.subheader("Connectivity checks")
st.caption("The same quick checks the notebook runs after configuring each model.")

with st.container(horizontal=True):
    test_llm = st.button("Test agent LLM", icon=":material/smart_toy:")
    test_judge = st.button("Test judge LLM", icon=":material/gavel:")
    test_embed = st.button("Test embeddings", icon=":material/scatter_plot:")

if test_llm:
    with st.spinner(f"Calling {gen_model}..."):
        reply = hr_core.make_llm(gen_model, hr_ui.cfg()["temperature"]).invoke("Say 'LLM is ready' and nothing else.")
    st.success(f"`{gen_model}` replied: {reply.content}", icon=":material/check_circle:")
if test_judge:
    with st.spinner(f"Calling {eval_model}..."):
        reply = hr_ui.get_judge_llm().invoke("Just say 'I am the judge' and nothing else.")
    st.success(f"`{eval_model}` replied: {reply.content}", icon=":material/check_circle:")
if test_embed:
    with st.spinner(f"Embedding with {embed_model}..."):
        vector = hr_core.make_embeddings(embed_model).embed_query("flexible working hours")
    st.success(f"`{embed_model}` returned a {len(vector)}-dimension vector.", icon=":material/check_circle:")

st.subheader("How to use this app")
st.markdown(
    """
- **Data explorer:** browse the five HR database tables, the policy documents, and the sample queries.
- **Tool lab:** test each tool on its own (the notebook's SQL, policy search, and Python REPL tests, plus your own inputs).
- **Ask the agent:** ask any question as any employee and watch the agent reason step by step.
- **Agent settings:** edit the system prompt, judge prompt, retrieval, and evaluation parameters.
- **Validation set / Test set:** run and judge the 8 sample queries and the 20 held-out test queries,
  individually or all at once, and see pass/fail results by category.
"""
)
st.caption(f"Data folder: `{hr_core.DATA_DIR}`")
