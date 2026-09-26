"""Streamlit-side helpers shared by the pages: state, cached resources, and the query-set runner."""

import pandas as pd
import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler

import hr_core

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "temperature": 0.0,
    "eval_temperature": 0.0,
    "max_iterations": 15,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "k": 4,
    "pass_threshold": 0.7,
    "overall_pass_rate": 0.60,
}


def init_state():
    """Initialize per-user state once. Called from streamlit_app.py before every page."""
    if "cfg" not in st.session_state:
        st.session_state.cfg = dict(DEFAULT_CONFIG)
        st.session_state.cfg["system_prompt"] = hr_core.default_system_prompt()
        st.session_state.cfg["judge_prompt"] = hr_core.DEFAULT_JUDGE_PROMPT
    for key in ("results_validation", "results_test"):
        st.session_state.setdefault(key, {})


def cfg():
    return st.session_state.cfg


def selected_models():
    return st.session_state.gen_model, st.session_state.eval_model, st.session_state.embed_model


# ---------------------------------------------------------------------------
# Cached data and resources
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner="Asking Ollama for installed models...")
def installed_models():
    return hr_core.list_ollama_models()


@st.cache_data(show_spinner=False)
def load_table(name):
    return hr_core.load_table(name)


@st.cache_data(show_spinner=False)
def load_query_set(filename):
    return hr_core.load_query_set(filename)


@st.cache_data(show_spinner=False)
def pdf_overview():
    return hr_core.pdf_overview()


@st.cache_data(show_spinner=False)
def chunk_documents(chunk_size, chunk_overlap):
    docs, chunks = hr_core.load_and_chunk_documents(chunk_size, chunk_overlap)
    return len(docs), [(c.page_content, c.metadata) for c in chunks]


@st.cache_resource(max_entries=4, show_spinner="Embedding policy documents with Ollama...")
def get_vectorstore(embed_model, chunk_size, chunk_overlap):
    return hr_core.build_vectorstore(embed_model, chunk_size, chunk_overlap)


@st.cache_resource(show_spinner=False)
def get_sql_tool():
    return hr_core.make_sql_tool()


@st.cache_resource(show_spinner=False)
def get_python_tool():
    return hr_core.make_python_tool()


def get_policy_tool():
    c = cfg()
    vs = get_vectorstore(st.session_state.embed_model, c["chunk_size"], c["chunk_overlap"])
    return hr_core.make_policy_tool(vs, c["k"])


def get_tools():
    return [get_sql_tool(), get_policy_tool(), get_python_tool()]


def get_agent_executor():
    c = cfg()
    llm = hr_core.make_llm(st.session_state.gen_model, c["temperature"])
    return hr_core.build_agent_executor(llm, get_tools(), c["system_prompt"], c["max_iterations"])


def get_judge_llm():
    return hr_core.make_llm(st.session_state.eval_model, cfg()["eval_temperature"])


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def render_steps(steps):
    """Show each tool call the agent made, like run_agent_query's verbose output in the notebook."""
    if not steps:
        st.caption("The agent answered without calling any tools.")
    for i, step in enumerate(steps, start=1):
        with st.expander(f"Step {i}: `{step['tool']}`"):
            st.markdown("**Input**")
            tool_input = step["input"]
            if isinstance(tool_input, dict) and "query" in tool_input and step["tool"] == "sql_query_tool":
                st.code(tool_input["query"], language="sql")
                st.caption(f"employee_id = {tool_input.get('employee_id')}")
            elif isinstance(tool_input, dict) and "query" in tool_input and step["tool"] == "Python_REPL":
                st.code(tool_input["query"], language="python")
            else:
                st.code(str(tool_input), language=None, wrap_lines=True)
            st.markdown("**Output**")
            st.code(step["output"], language=None, wrap_lines=True)


class LiveTraceHandler(BaseCallbackHandler):
    """Writes each tool call into a status container while the agent runs."""

    def __init__(self, status):
        self.status = status
        self.step = 0

    def on_tool_start(self, serialized, input_str, **kwargs):
        self.step += 1
        name = (serialized or {}).get("name", "tool")
        self.status.update(label=f"Step {self.step}: calling `{name}`...")
        self.status.markdown(f"**Step {self.step}: `{name}`**")
        self.status.code(input_str, language=None, wrap_lines=True)

    def on_tool_end(self, output, **kwargs):
        text = str(getattr(output, "content", output))
        self.status.caption(text[:500] + ("..." if len(text) > 500 else ""))
        self.status.update(label=f"Step {self.step} done. Thinking...")


def score_badge(score, threshold):
    if score is None:
        return
    if score >= threshold:
        st.badge(f"Pass · {score:.2f}", icon=":material/check_circle:", color="green")
    else:
        st.badge(f"Fail · {score:.2f}", icon=":material/cancel:", color="red")


# ---------------------------------------------------------------------------
# Query-set runner (validation and test pages)
# ---------------------------------------------------------------------------

def _run_one(executor, row):
    rec = hr_core.run_agent_query(executor, row["Employee Id"], row["Query"])
    rec.update({
        "category": row["Category"],
        "employee_id": row["Employee Id"],
        "query": row["Query"],
        "expected": row["Response"],
        "gen_model": st.session_state.gen_model,
        "score": None,
        "reasoning": None,
        "eval_model": None,
    })
    return rec


def _judge_one(judge_llm, rec):
    score, reasoning = hr_core.judge_response(
        judge_llm, cfg()["judge_prompt"], rec["query"], rec["expected"], rec["agent_answer"]
    )
    rec.update({"score": score, "reasoning": reasoning, "eval_model": st.session_state.eval_model})


def _execute(df, results, indices, run_agent, judge):
    """Run the agent and/or judge on the given row indices, storing each result as soon as it's ready."""
    executor = get_agent_executor() if run_agent else None
    judge_llm = get_judge_llm() if judge else None
    verb = "Running and judging" if run_agent and judge else ("Running" if run_agent else "Judging")
    with st.status(f"{verb} {len(indices)} queries...", expanded=True) as status:
        bar = st.progress(0.0)
        for n, idx in enumerate(indices, start=1):
            row = df.loc[idx]
            st.write(f"**Query {idx + 1}** · {row['Category']} · {row['Employee Id']}: {row['Query'][:90]}...")
            if run_agent:
                results[idx] = _run_one(executor, row)
                rec = results[idx]
                st.caption(f"Agent answered in {rec['num_steps']} steps ({rec['seconds']}s).")
            if judge:
                if idx not in results:
                    st.caption("Skipped judging: no agent answer yet.")
                else:
                    _judge_one(judge_llm, results[idx])
                    st.caption(f"Score: {results[idx]['score']:.2f} · {results[idx]['reasoning']}")
            bar.progress(n / len(indices))
        status.update(label=f"{verb} complete for {len(indices)} queries.", state="complete", expanded=False)


def render_query_set(set_key, filename, show_verdict):
    """Table of queries with run/judge controls, summary metrics, and per-query drill-down."""
    df = load_query_set(filename)
    results = st.session_state[f"results_{set_key}"]
    c = cfg()
    table_key = f"table_{set_key}"

    selection = st.session_state.get(table_key)
    selected = list(selection.selection.rows) if selection else []

    gen_model, eval_model, _ = selected_models()
    st.caption(f"Agent: `{gen_model}` · Judge: `{eval_model}` · Select rows in the table to run or judge "
               "individual queries, or use the run-all buttons.")

    with st.container(horizontal=True, vertical_alignment="center"):
        run_sel = st.button(f"Run selected ({len(selected)})", icon=":material/play_arrow:",
                            type="primary", disabled=not selected, key=f"{set_key}_run_sel")
        run_all = st.button(f"Run all ({len(df)})", icon=":material/playlist_play:", key=f"{set_key}_run_all")
        judge_sel = st.button("Judge selected", icon=":material/gavel:", disabled=not selected,
                              key=f"{set_key}_judge_sel")
        judge_all = st.button("Judge all answered", icon=":material/balance:", disabled=not results,
                              key=f"{set_key}_judge_all")
        auto_judge = st.toggle("Judge after running", value=True, key=f"{set_key}_auto_judge")
        if st.button("Clear results", icon=":material/delete:", type="tertiary", disabled=not results,
                     key=f"{set_key}_clear"):
            results.clear()
            st.rerun()

    if run_sel or run_all:
        _execute(df, results, selected if run_sel else list(df.index), run_agent=True, judge=auto_judge)
    elif judge_sel or judge_all:
        _execute(df, results, selected if judge_sel else sorted(results), run_agent=False, judge=True)

    # Overview table with current status of every query
    table = pd.DataFrame({
        "#": df.index + 1,
        "Category": df["Category"],
        "Employee": df["Employee Id"],
        "Query": df["Query"],
        "Steps": [results[i]["num_steps"] if i in results else None for i in df.index],
        "Score": [results[i]["score"] if i in results else None for i in df.index],
        "Result": [
            None if i not in results or results[i]["score"] is None
            else ("Pass" if results[i]["score"] >= c["pass_threshold"] else "Fail")
            for i in df.index
        ],
    })
    st.dataframe(
        table, hide_index=True, on_select="rerun", selection_mode="multi-row", key=table_key,
        column_config={
            "#": st.column_config.NumberColumn(width="small"),
            "Query": st.column_config.TextColumn(width="large"),
            "Score": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f"),
        },
    )

    judged = {i: r for i, r in results.items() if r["score"] is not None}
    if judged:
        _render_summary(judged, len(df), show_verdict)

    if results:
        _render_detail(set_key, df, results)


def _render_summary(judged, total, show_verdict):
    c = cfg()
    df_res = pd.DataFrame(judged.values())
    df_res["passed"] = df_res["score"] >= c["pass_threshold"]
    n, passed = len(df_res), int(df_res["passed"].sum())
    pass_rate = passed / n

    st.subheader("Evaluation results")
    with st.container(horizontal=True):
        st.metric("Judged", f"{n} / {total}", border=True)
        st.metric(f"Passed (≥ {c['pass_threshold']})", passed, border=True)
        st.metric("Failed", n - passed, border=True)
        st.metric("Pass rate", f"{pass_rate:.1%}", border=True)
        st.metric("Average score", f"{df_res['score'].mean():.2f}", border=True)

    if show_verdict:
        required = c["overall_pass_rate"]
        partial = " (partial: not every query has been judged)" if n < total else ""
        if pass_rate >= required:
            st.success(f"**Verdict: PASSED.** Pass rate {pass_rate:.1%} meets the required {required:.0%}{partial}.",
                       icon=":material/verified:")
        else:
            st.error(f"**Verdict: FAILED.** Pass rate {pass_rate:.1%} is below the required {required:.0%}{partial}.",
                     icon=":material/error:")

    cat = df_res.groupby("category")["score"].agg(["mean", "count"]).rename(
        columns={"mean": "avg_score", "count": "num_queries"})
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Average score by category**")
        st.bar_chart(cat, y="avg_score", horizontal=True, y_label="", x_label="Average score")
    with right:
        st.dataframe(cat, column_config={
            "avg_score": st.column_config.NumberColumn("Avg score", format="%.2f"),
            "num_queries": st.column_config.NumberColumn("Queries"),
        })

    with st.expander("Detailed scores by query"):
        detail = df_res[["category", "employee_id", "query", "score", "passed", "reasoning"]]
        st.dataframe(detail, hide_index=True, column_config={
            "score": st.column_config.NumberColumn(format="%.2f"),
            "query": st.column_config.TextColumn(width="medium"),
            "reasoning": st.column_config.TextColumn(width="large"),
        })
        export = pd.DataFrame(judged.values()).drop(columns=["steps"])
        st.download_button("Download results as CSV", export.to_csv(index=False), file_name="evaluation_results.csv",
                           mime="text/csv", icon=":material/download:")


def _render_detail(set_key, df, results):
    st.subheader("Inspect a query")
    options = sorted(results)
    idx = st.selectbox(
        "Query", options, key=f"{set_key}_detail",
        format_func=lambda i: f"#{i + 1} · {df.loc[i, 'Category']} · {df.loc[i, 'Query'][:80]}",
    )
    rec = results[idx]
    c = cfg()

    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.markdown(f"**#{idx + 1} · {rec['category']} · {rec['employee_id']}**")
            score_badge(rec["score"], c["pass_threshold"])
            st.caption(f"{rec['num_steps']} steps · {rec['seconds']}s · agent `{rec['gen_model']}`"
                       + (f" · judge `{rec['eval_model']}`" if rec["eval_model"] else ""))
        st.markdown(f"**Query:** {rec['query']}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Agent answer**")
            with st.container(border=True):
                st.markdown(rec["agent_answer"])
        with right:
            st.markdown("**Expected response**")
            with st.container(border=True):
                st.markdown(rec["expected"])

        if rec["reasoning"]:
            st.markdown(f"**Judge reasoning:** {rec['reasoning']}")

        st.markdown("**Agent reasoning steps**")
        render_steps(rec["steps"])

        with st.container(horizontal=True):
            rerun = st.button("Re-run this query", icon=":material/replay:", key=f"{set_key}_rerun_one")
            rejudge = st.button("Re-judge this query", icon=":material/gavel:", key=f"{set_key}_rejudge_one")

    if rerun:
        with st.spinner("Running the agent..."):
            results[idx] = _run_one(get_agent_executor(), df.loc[idx])
            if st.session_state.get(f"{set_key}_auto_judge", True):
                _judge_one(get_judge_llm(), results[idx])
        st.rerun()
    if rejudge:
        with st.spinner("Judging..."):
            _judge_one(get_judge_llm(), rec)
        st.rerun()
