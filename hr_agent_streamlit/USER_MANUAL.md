# TechCorp HR agent: user manual

This manual explains how to use the TechCorp HR agent app, page by page. For installation, see the
[README](README.md).

## Contents

1. [What the app does](#what-the-app-does)
2. [Starting the app](#starting-the-app)
3. [Choosing models (sidebar)](#choosing-models-sidebar)
4. [Overview](#overview)
5. [Data explorer](#data-explorer)
6. [Tool lab](#tool-lab)
7. [Ask the agent](#ask-the-agent)
8. [Agent settings](#agent-settings)
9. [Validation set and Test set](#validation-set-and-test-set)
10. [Typical workflows](#typical-workflows)
11. [Understanding scores](#understanding-scores)
12. [Troubleshooting](#troubleshooting)

---

## What the app does

The app is an HR assistant for TechCorp employees. You ask a question as a specific employee (for example,
"How many casual leaves do I have left?"), and an AI agent answers by using three tools:

| Tool | What it does |
|---|---|
| **SQL query tool** | Looks up the employee's own records in the HR database (profile, leave, attendance, payroll, reviews). It is read-only and can only see the authenticated employee's data. |
| **Policy search tool** | Searches the four company policy PDFs for rules, limits, and deadlines. |
| **Python REPL tool** | Does calculations such as date differences, leave balances, and salary estimates. |

A second AI model, the **judge**, can score the agent's answers against reference answers, so you can
measure how well the agent performs.

All models run on your computer (or through your Ollama account for `-cloud` models) via Ollama. No API
keys are needed.

## Starting the app

1. Make sure **Ollama is running** (the Ollama desktop app, or `ollama serve` in a terminal).
2. In a terminal:

   ```bash
   conda activate hr_agent
   cd hr_agent_streamlit
   streamlit run streamlit_app.py
   ```

3. Your browser opens at `http://localhost:8501`. If it doesn't, open that address yourself.

To stop the app, press `Ctrl+C` in the terminal.

## Choosing models (sidebar)

The sidebar on the left has the page menu and three model pickers. Each picker lists **only models
installed in your Ollama** that can do that job:

| Picker | Used for | Lists | Default |
|---|---|---|---|
| **Agent LLM** | Answering questions | Models that support tool calling | `gemma4:31b-cloud` |
| **Judge LLM** | Scoring answers | Any chat model | `gemma4:31b-cloud` |
| **Embeddings** | Searching the policy documents | Embedding models | `nomic-embed-text:latest` |

- A new choice takes effect on your next action. You don't need to restart anything.
- If you pull a new model with `ollama pull`, click **Refresh model list** to see it.
- Changing the **Embeddings** model means the policy documents are re-indexed the next time they're
  searched. You'll see "Embedding policy documents with Ollama..." for a few seconds.
- **Tip:** for a fairer evaluation, pick a different (ideally stronger) model for the judge than for the
  agent, so the judge isn't favouring its own style of answer.

## Overview

The starting page. It summarizes the app and shows the three models currently selected.

Under **Connectivity checks**, use the buttons to confirm each model responds:

- **Test agent LLM**: should reply "LLM is ready".
- **Test judge LLM**: should reply "I am the judge".
- **Test embeddings**: should report a vector size (768 for `nomic-embed-text`).

Run these first whenever you change models or something seems wrong.

## Data explorer

Browse the data the agent works with. Choose a section at the top:

**Database tables**: pick one of the five tables (`employees`, `leave_records`, `attendance_logs`,
`payroll`, `performance_reviews`) to see its row and column counts, key facts (such as leave types or the
date range), and the full table. Click a column header to sort, and hover over the table to search or
download it.

**Policy documents**: each of the four PDFs with its page count, a description, and **Show extracted
text** to read the full text. The **Chunking preview** below shows how the documents are split into
chunks for searching; change **Chunk number** to step through them.

**Sample queries**: the 8 example questions with their expected answers. Click one to expand it.

## Tool lab

Test each of the agent's tools directly, without the agent. Choose a tool at the top.

### SQL query tool

1. Pick a **Preset**:
   - *Test 1: employee profile lookup*: returns the employee's name, department, level, and employment type.
   - *Test 2: approved sick leave count*: totals the employee's approved sick days.
   - *Test 3: blocked DELETE attempt*: shows the safety check refusing a destructive query.
   - *Custom query*: write your own.
2. Edit the **SQL** and **Employee ID** if you like, then click **Run SQL tool**.

Rules the tool enforces (you'll see an `ERROR:` message if a query breaks one):

- Only a single `SELECT` statement; no `INSERT`, `UPDATE`, `DELETE`, `DROP`, comments, or `;`.
- The query must filter with `:employee_id` (for example `WHERE employee_id = :employee_id`). Don't type
  the ID into the SQL; it comes from the Employee ID box.

Expand **Tool description the agent sees** to read the instructions the agent gets for this tool.

### Policy search tool

Type a **Search query** (for example "notice period for resignation") and click **Search policies**. You
get the top matching chunks with their document, page, and distance (lower means a closer match).
**Raw tool output** shows exactly what the agent would receive. The number of chunks returned is the
**k** setting on the Agent settings page.

### Python REPL tool

Edit the **Python code** (the preset calculates an employee's tenure) and click **Run Python tool**.
Always `print()` the result; the tool only returns printed output.

> This runs real Python code on your computer. Only run code you understand.

## Ask the agent

Ask any question as any employee and watch the agent work.

1. **Authenticated employee**: the employee who is asking. The agent can only see this employee's
   records and will refuse to reveal anyone else's.
2. **Question**: type your question in plain English.
3. **Expected response** (optional): a reference answer. Only needed if you want the judge to score the
   answer.
4. Click **Ask the agent**.

While the agent works, a status box shows each tool call as it happens. When it finishes you see:

- **Final answer**, with the number of steps, the time taken, and the model used.
- **Agent reasoning steps**: expand any step to see the exact tool input (for example the SQL query) and
  what the tool returned. This is the best way to check how the agent reached its answer.

### Loading a ready-made question

Open **Load a query from the validation or test set**, choose the set and the query, and click **Load
query**. This fills in the employee, the question, and the expected response.

### Judging the answer

**Judge this answer** is available only when there is an expected response. Either load a query from a
set (it includes one), or type your own expected response. Streamlit registers typed text when you click
outside the box or press `Cmd+Enter` (`Ctrl+Enter` on Windows). You can add the expected response after
the agent has answered; you don't need to ask again.

The judge's score (0 to 1) and reasoning appear below the button, and a Pass or Fail badge appears next
to the final answer.

## Agent settings

Change how the agent and the judge behave. Edit any fields, then click **Save settings**. Nothing changes
until you save.

| Setting | What it does |
|---|---|
| **System prompt** | The agent's instructions: its rules, guardrails, and how to use the tools. The main thing to edit when improving the agent. |
| **Agent temperature** | Randomness of the agent's answers. 0 gives the most consistent answers. |
| **Max iterations** | Maximum tool calls the agent can make per question before stopping. |
| **Chunk size / Chunk overlap** | How the policy PDFs are split for searching. Changing these re-indexes the documents. Overlap must be smaller than size. |
| **Chunks retrieved per search (k)** | How many policy passages each search returns. |
| **Judge prompt** | The judge's scoring instructions. It must keep the `{query}`, `{expected}` and `{agent_response}` placeholders. |
| **Judge temperature** | Randomness of the judge. Keep at 0 for consistent scoring. |
| **Pass threshold (per query)** | The minimum score for a query to count as passed (default 0.7). |
| **Required pass rate (test set)** | The share of test queries that must pass for the agent to pass overall (default 60%). |

**Reset to notebook defaults** restores every setting to its original value.

Settings last until you close or reload the browser tab. To keep an edited prompt, copy it somewhere
before closing.

## Validation set and Test set

These two pages work the same way:

- **Validation set**: the 8 sample queries. Use it while improving the agent.
- **Test set**: the 20 held-out queries. Use it for the final measurement. This page also gives an
  overall **Verdict** (passed or failed against the required pass rate).

### Running queries

The table lists every query with its category, employee, and question. After a run it also shows the
number of **Steps**, the judge's **Score**, and the **Result** (Pass or Fail).

- **Run all**: runs the agent on every query.
- **Run selected**: tick the checkboxes at the left of the table rows first, then click it. Useful for
  re-trying just the queries that failed.
- **Judge after running** (on by default): scores each answer as soon as it's produced. Turn it off to
  run the agent now and judge later.
- **Judge selected** / **Judge all answered**: score answers without re-running the agent. Use this after
  changing the judge model or judge prompt.
- **Clear results**: removes all results from this page.

A progress box shows each query as it's processed. Running all 20 test queries takes a few minutes.

> **Don't click anything else while a batch is running.** Any click restarts the page and stops the batch.
> Queries that already finished keep their results, so you can use **Run selected** to finish the rest.

### Evaluation results

Once at least one answer has been judged you'll see:

- **Metrics**: queries judged, passed, failed, pass rate, and average score.
- **Verdict** (Test set only): green if the pass rate meets the required rate, red if not. It notes when
  not every query has been judged yet.
- **Average score by category**: a chart and table showing which kinds of questions the agent handles
  best and worst.
- **Detailed scores by query**: every query with its score and the judge's reasoning, plus **Download
  results as CSV** to save the results.

### Inspecting a single query

Under **Inspect a query**, pick any query that has been run to see:

- the agent's answer and the expected response side by side,
- the judge's reasoning,
- every reasoning step the agent took.

**Re-run this query** runs the agent on just that query again (and judges it if **Judge after running**
is on). **Re-judge this query** scores the existing answer again.

## Typical workflows

### First-time check

1. **Overview**: click all three connectivity checks.
2. **Tool lab**: run the three SQL presets, one policy search, and the Python preset.
3. **Ask the agent**: load a validation query, ask, and judge it.

### Improving the agent

1. **Validation set**: click **Run all** and note which queries fail and why (use **Inspect a query**).
2. **Agent settings**: adjust the system prompt (for example, add a rule addressing a failure you saw)
   or retrieval settings, then **Save settings**.
3. **Validation set**: select the failed rows and click **Run selected**. Repeat until you're satisfied.
4. **Test set**: click **Run all** for the final measurement.

### Comparing models

1. Run the validation or test set with one Agent LLM.
2. Download the results as CSV.
3. Change the **Agent LLM** in the sidebar, click **Clear results**, and run again.

Answers vary a little between runs even at temperature 0, so run important comparisons more than once.

## Understanding scores

The judge compares each answer with the reference answer on correctness, completeness, guardrail
compliance (refusing to share other employees' data, not inventing facts), and tone.

| Score | Meaning |
|---|---|
| 1.0 | Matches the reference in all key facts |
| 0.8 to 0.9 | Essentially correct, minor omissions |
| 0.5 to 0.7 | Partly correct, missing important details or some errors |
| 0.2 to 0.4 | Significant errors or mostly missing |
| 0.0 to 0.1 | Wrong, invented, or a guardrail violation |

A query passes at or above the pass threshold (default 0.7). If the judge's reply can't be read, the
query scores 0 and the reasoning starts with "Evaluation failed"; use **Re-judge this query**.

## Troubleshooting

| Problem | What to do |
|---|---|
| Sidebar says "Could not reach Ollama" | Start Ollama, then click **Retry**. |
| "No installed Ollama models are suitable for …" | Install a suitable model, e.g. `ollama pull gemma4:31b-cloud` (agent/judge) or `ollama pull nomic-embed-text` (embeddings), then **Refresh model list**. |
| A model I just pulled isn't listed | Click **Refresh model list** (the list is otherwise refreshed every minute). |
| Agent LLM list is shorter than `ollama list` | Only models that support tool calling are listed, because the agent needs tools. |
| **Judge this answer** is greyed out | Add an expected response (click outside the box after typing) or load a query from a set. |
| A batch run stopped part way | Something was clicked during the run. Select the unfinished rows and click **Run selected**. |
| Answer says "AGENT ERROR" | Run the connectivity checks on the Overview page. Cloud models (`-cloud`) need an internet connection and a signed-in Ollama account. |
| My settings or results disappeared | They're kept per browser tab and are lost on reload. Download results as CSV to keep them. |
| Policy search looks wrong after changing embeddings | Expected: different embedding models find different passages. Compare results on the **Tool lab** policy search. |
