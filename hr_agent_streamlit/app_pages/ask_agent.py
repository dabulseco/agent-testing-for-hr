import streamlit as st

import hr_core
import hr_ui

employees = hr_ui.load_table("employees")
emp_names = dict(zip(employees["employee_id"], employees["name"]))
QUERY_SETS = {"Validation set": "sample_data.csv", "Test set": "test.csv"}

st.session_state.setdefault("ask_employee", employees["employee_id"].iloc[0])
st.session_state.setdefault("ask_query", "")
st.session_state.setdefault("ask_expected", "")


def load_example():
    df = hr_ui.load_query_set(QUERY_SETS[st.session_state.ask_set])
    row = df.loc[st.session_state.ask_example]
    st.session_state.ask_employee = row["Employee Id"]
    st.session_state.ask_query = row["Query"]
    st.session_state.ask_expected = row["Response"]


with st.expander("Load a query from the validation or test set", icon=":material/upload:"):
    query_set = st.segmented_control("Query set", list(QUERY_SETS), default="Validation set", key="ask_set")
    if query_set:
        df_set = hr_ui.load_query_set(QUERY_SETS[query_set])
        st.selectbox(
            "Query", df_set.index, key="ask_example",
            format_func=lambda i: f"#{i + 1} · {df_set.loc[i, 'Category']} · {df_set.loc[i, 'Query'][:90]}",
        )
        st.button("Load query", icon=":material/download:", on_click=load_example)

st.selectbox("Authenticated employee", list(emp_names), key="ask_employee",
             format_func=lambda e: f"{e} · {emp_names[e]}")
st.text_area("Question", key="ask_query", height=100,
             placeholder="How many casual leaves do I have left this year?")
st.text_area("Expected response (optional, enables judging)", key="ask_expected", height=80)

ask = st.button("Ask the agent", icon=":material/send:", type="primary",
                disabled=not st.session_state.ask_query.strip())

if ask:
    status = st.status(f"Agent `{st.session_state.gen_model}` is thinking...", expanded=True)
    rec = hr_core.run_agent_query(
        hr_ui.get_agent_executor(), st.session_state.ask_employee, st.session_state.ask_query,
        callbacks=[hr_ui.LiveTraceHandler(status)],
    )
    status.update(label=f"Finished in {rec['num_steps']} steps ({rec['seconds']}s).",
                  state="error" if rec["error"] else "complete", expanded=False)
    rec.update({"employee_id": st.session_state.ask_employee, "query": st.session_state.ask_query,
                "gen_model": st.session_state.gen_model, "score": None, "reasoning": None})
    st.session_state.ask_result = rec

rec = st.session_state.get("ask_result")
if rec:
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f"**Final answer** for {rec['employee_id']}")
            hr_ui.score_badge(rec["score"], hr_ui.cfg()["pass_threshold"])
            st.caption(f"{rec['num_steps']} steps · {rec['seconds']}s · `{rec['gen_model']}`")
        st.caption(f"Q: {rec['query']}")
        if rec["error"]:
            st.error(rec["agent_answer"], icon=":material/error:")
        else:
            st.markdown(rec["agent_answer"])

    st.markdown("**Agent reasoning steps**")
    hr_ui.render_steps(rec["steps"])

    expected = st.session_state.ask_expected.strip()
    if st.button("Judge this answer", icon=":material/gavel:", disabled=not expected,
                 help=None if expected else "Add an expected response to enable judging."):
        with st.spinner(f"Judging with {st.session_state.eval_model}..."):
            rec["score"], rec["reasoning"] = hr_core.judge_response(
                hr_ui.get_judge_llm(), hr_ui.cfg()["judge_prompt"], rec["query"], expected, rec["agent_answer"]
            )
        st.rerun()
    if not expected:
        st.caption("To judge this answer, enter an expected response above (then click outside the box), "
                   "or load a query from the validation or test set, which includes one.")
    if rec["reasoning"]:
        st.info(f"**Judge ({st.session_state.eval_model}):** {rec['score']:.2f}. {rec['reasoning']}",
                icon=":material/gavel:")
