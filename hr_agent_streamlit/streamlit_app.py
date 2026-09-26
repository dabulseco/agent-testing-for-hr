import warnings

import streamlit as st

import hr_core
import hr_ui

warnings.filterwarnings("ignore", category=DeprecationWarning)

st.set_page_config(page_title="TechCorp HR agent", page_icon=":material/support_agent:", layout="wide")

hr_ui.init_state()

page = st.navigation(
    {
        "": [
            st.Page("app_pages/overview.py", title="Overview", icon=":material/home:", default=True),
            st.Page("app_pages/data_explorer.py", title="Data explorer", icon=":material/database:"),
        ],
        "Build and test": [
            st.Page("app_pages/tool_lab.py", title="Tool lab", icon=":material/build:"),
            st.Page("app_pages/ask_agent.py", title="Ask the agent", icon=":material/forum:"),
            st.Page("app_pages/agent_settings.py", title="Agent settings", icon=":material/tune:"),
        ],
        "Evaluate": [
            st.Page("app_pages/validation.py", title="Validation set", icon=":material/fact_check:"),
            st.Page("app_pages/test_set.py", title="Test set", icon=":material/assignment_turned_in:"),
        ],
    },
    position="sidebar",
)

# ---------------------------------------------------------------------------
# Sidebar: model selection, limited to models installed in Ollama
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Ollama models", divider="gray")
    try:
        models = hr_ui.installed_models()
    except RuntimeError as e:
        st.error(str(e), icon=":material/cloud_off:")
        st.button("Retry", icon=":material/refresh:", on_click=hr_ui.installed_models.clear)
        st.stop()

    pickers = [
        ("gen_model", "Agent LLM", "tools", hr_core.DEFAULT_GEN_MODEL, "Installed models that support tool calling."),
        ("eval_model", "Judge LLM", "completion", hr_core.DEFAULT_EVAL_MODEL, "Any installed chat model."),
        ("embed_model", "Embeddings", "embedding", hr_core.DEFAULT_EMBED_MODEL, "Installed embedding models."),
    ]
    for key, label, capability, default, help_text in pickers:
        options = hr_core.models_with(models, capability)
        if not options:
            st.error(f"No installed Ollama models are suitable for: {label}.")
            st.stop()
        # Fall back to the default (or first option) if the current choice is no longer installed
        if st.session_state.get(key) not in options:
            st.session_state[key] = default if default in options else options[0]
        st.selectbox(label, options, key=key, help=help_text)

    st.caption(f"{len(models)} models installed at `{hr_core.OLLAMA_BASE_URL}`")
    st.button("Refresh model list", icon=":material/refresh:", type="tertiary",
              on_click=hr_ui.installed_models.clear)

st.title(page.title)
page.run()
