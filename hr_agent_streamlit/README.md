# TechCorp HR agent (Streamlit)

A Streamlit version of `MLS_Notebook_HR_Agent_Notebook (2).ipynb`. It runs the same ReAct HR agent
(SQL tool, policy search RAG tool, Python REPL tool) and the same LLM-as-a-judge evaluation, using
models served by a local Ollama instance.

**New to the app? Read the [user manual](USER_MANUAL.md)** for a page-by-page guide, common workflows,
and troubleshooting.

## Run

```bash
conda create -n hr_agent python=3.12
conda activate hr_agent
cd hr_agent_streamlit
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Ollama must be running (`ollama serve` or the desktop app). The app reads the `Datasets/` folder next to
this one; set `HR_DATA_DIR` to use a different location and `OLLAMA_HOST` for a non-default Ollama address.

## Models

The sidebar lists only models installed in Ollama, filtered by capability:

| Picker | Shows | Default |
|---|---|---|
| Agent LLM | models that support tool calling | `gemma4:31b-cloud` |
| Judge LLM | any chat model | `gemma4:31b-cloud` |
| Embeddings | embedding models | `nomic-embed-text:latest` |

## Pages

| Page | Notebook equivalent |
|---|---|
| Overview | LLM, judge, and embedding connectivity tests |
| Data explorer | Database table exploration, policy document overview, chunking, sample queries |
| Tool lab | SQL tool tests 1 to 3, policy search test, Python REPL test (plus custom inputs) |
| Ask the agent | `run_agent_query` for any employee and question, with the reasoning steps |
| Agent settings | Editing the system prompt, judge prompt, chunking, k, and thresholds |
| Validation set | Running and judging the 8 sample queries |
| Test set | Running and judging the 20 test queries, pass/fail verdict, scores by category |

On the validation and test pages you can run or judge selected rows, run everything, and re-run or
re-judge a single query from the "Inspect a query" panel. Results are kept for the browser session.
Starting another action while a batch is running stops the batch; finished queries are kept.

## Files

- `streamlit_app.py`: entry point, navigation, and sidebar model pickers
- `hr_core.py`: all agent, tool, RAG, and judge logic (no Streamlit)
- `hr_ui.py`: session state, caching, and the shared query-set runner
- `app_pages/`: one file per page
