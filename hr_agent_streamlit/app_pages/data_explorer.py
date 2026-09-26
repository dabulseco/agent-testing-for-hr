import streamlit as st

import hr_ui

TABLE_NOTES = {
    "employees": "Master employee profiles. Almost every query begins with a lookup in this table.",
    "leave_records": "Every leave application (casual, sick, earned) with dates, days, approval status, and approver.",
    "attendance_logs": "Daily 2026 attendance with check-in/out times, status, and hours. Used for WFH tracking.",
    "payroll": "Monthly salary records (January to August 2026) broken down into components.",
    "performance_reviews": "Half-yearly reviews (H2 2025, H1 2026) with rating, promotion flag, and training hours.",
}

# Per-table facts the notebook prints for each table
TABLE_FACTS = {
    "employees": lambda df: {
        "Levels": sorted(df["level"].unique()),
        "Employment types": sorted(df["employment_type"].unique()),
        "Departments": sorted(df["department"].unique()),
    },
    "leave_records": lambda df: {
        "Leave types": sorted(df["leave_type"].unique()),
        "Statuses": sorted(df["status"].unique()),
    },
    "attendance_logs": lambda df: {
        "Statuses": sorted(df["status"].unique()),
        "Date range": f"{df['date'].min()} to {df['date'].max()}",
    },
    "payroll": lambda df: {
        "Months covered": sorted(df["month"].unique().tolist()),
        "Year": df["year"].unique().tolist(),
    },
    "performance_reviews": lambda df: {
        "Review cycles": sorted(df["review_cycle"].unique()),
        "Rating range": f"{df['rating'].min()} to {df['rating'].max()}",
    },
}

section = st.segmented_control(
    "Data source", ["Database tables", "Policy documents", "Sample queries"],
    default="Database tables", label_visibility="collapsed",
)

if section == "Database tables":
    table = st.segmented_control("Table", list(TABLE_NOTES), default="employees")
    if table:
        df = hr_ui.load_table(table)
        st.caption(TABLE_NOTES[table])
        with st.container(horizontal=True):
            st.metric("Rows", f"{df.shape[0]:,}", border=True)
            st.metric("Columns", df.shape[1], border=True)
        for label, value in TABLE_FACTS[table](df).items():
            st.markdown(f"**{label}:** {', '.join(map(str, value)) if isinstance(value, list) else value}")
        st.dataframe(df, hide_index=True)

elif section == "Policy documents":
    docs = hr_ui.pdf_overview()
    for doc in docs:
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.markdown(f"**{doc['document']}**")
                st.caption(f"{doc['pages']} pages · ~{doc['characters']:,} characters")
            st.write(doc["content"])
            with st.expander("Show extracted text"):
                st.text(doc["text"])

    st.subheader("Chunking preview")
    c = hr_ui.cfg()
    n_pages, chunks = hr_ui.chunk_documents(c["chunk_size"], c["chunk_overlap"])
    st.caption(f"{n_pages} pages split into **{len(chunks)} chunks** "
               f"(size {c['chunk_size']}, overlap {c['chunk_overlap']}; change these in Agent settings).")
    i = st.number_input("Chunk number", 0, len(chunks) - 1, value=min(5, len(chunks) - 1))
    text, meta = chunks[i]
    st.caption(f"Source: {meta['document_name']} · page {meta.get('page', 'N/A')} · {len(text)} characters")
    st.text(text)

else:
    df = hr_ui.load_query_set("sample_data.csv")
    st.caption(f"{len(df)} sample queries across {df['Category'].nunique()} categories, used for validation.")
    for idx, row in df.iterrows():
        with st.expander(f"Query {idx + 1} · {row['Category']} · {row['Employee Id']}"):
            st.markdown(f"**Q:** {row['Query']}")
            st.markdown("**Expected response:**")
            st.markdown(row["Response"])
