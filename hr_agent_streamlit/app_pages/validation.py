import streamlit as st

import hr_ui

st.markdown("Run the agent on the 8 sample queries from `sample_data.csv`, examine which tools it called, "
            "and score each answer with the LLM judge. Use this to refine the agent before the final test.")

hr_ui.render_query_set("validation", "sample_data.csv", show_verdict=False)
