import streamlit as st

import hr_core
import hr_ui

c = hr_ui.cfg()

st.caption("Refine the agent the way the notebook suggests: adjust the system prompt, retrieval, and "
           "evaluation settings, then re-run the validation set to see the effect. Changes apply to this "
           "browser session only.")

with st.form("settings_form"):
    st.subheader("Agent")
    system_prompt = st.text_area("System prompt", value=c["system_prompt"], height=420)
    with st.container(horizontal=True):
        temperature = st.number_input("Agent temperature", 0.0, 2.0, value=c["temperature"], step=0.1)
        max_iterations = st.number_input("Max iterations", 1, 50, value=c["max_iterations"])

    st.subheader("Policy retrieval (RAG)")
    with st.container(horizontal=True):
        chunk_size = st.number_input("Chunk size (characters)", 200, 4000, value=c["chunk_size"], step=100)
        chunk_overlap = st.number_input("Chunk overlap", 0, 1000, value=c["chunk_overlap"], step=50)
        k = st.number_input("Chunks retrieved per search (k)", 1, 20, value=c["k"])

    st.subheader("Evaluation")
    judge_prompt = st.text_area(
        "Judge prompt", value=c["judge_prompt"], height=420,
        help="Must keep the {query}, {expected}, and {agent_response} placeholders. Use {{ }} for literal braces.",
    )
    with st.container(horizontal=True):
        eval_temperature = st.number_input("Judge temperature", 0.0, 2.0, value=c["eval_temperature"], step=0.1)
        pass_threshold = st.number_input("Pass threshold (per query)", 0.0, 1.0, value=c["pass_threshold"], step=0.05)
        overall_pass_rate = st.number_input("Required pass rate (test set)", 0.0, 1.0,
                                            value=c["overall_pass_rate"], step=0.05)

    saved = st.form_submit_button("Save settings", icon=":material/save:", type="primary")

if saved:
    missing = [p for p in ("{query}", "{expected}", "{agent_response}") if p not in judge_prompt]
    if chunk_overlap >= chunk_size:
        st.error("Chunk overlap must be smaller than chunk size.", icon=":material/error:")
    elif missing:
        st.error(f"The judge prompt is missing: {', '.join(missing)}", icon=":material/error:")
    else:
        c.update({
            "system_prompt": system_prompt, "temperature": temperature, "max_iterations": int(max_iterations),
            "chunk_size": int(chunk_size), "chunk_overlap": int(chunk_overlap), "k": int(k),
            "judge_prompt": judge_prompt, "eval_temperature": eval_temperature,
            "pass_threshold": pass_threshold, "overall_pass_rate": overall_pass_rate,
        })
        st.toast("Settings saved.", icon=":material/check_circle:")


def reset():
    st.session_state.cfg = dict(hr_ui.DEFAULT_CONFIG)
    st.session_state.cfg["system_prompt"] = hr_core.default_system_prompt()
    st.session_state.cfg["judge_prompt"] = hr_core.DEFAULT_JUDGE_PROMPT


st.button("Reset to notebook defaults", icon=":material/restart_alt:", type="tertiary", on_click=reset)
