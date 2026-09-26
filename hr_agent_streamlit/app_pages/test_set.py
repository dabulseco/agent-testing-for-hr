import streamlit as st

import hr_ui

c = hr_ui.cfg()
st.markdown(f"The final evaluation on the 20 held-out queries in `test.csv`. A query passes with a judge score of "
            f"at least **{c['pass_threshold']}**, and the agent passes with a pass rate of at least "
            f"**{c['overall_pass_rate']:.0%}**.")
st.caption("Results vary between runs because the model is non-deterministic, so consider running the test "
           "more than once.")

hr_ui.render_query_set("test", "test.csv", show_verdict=True)
