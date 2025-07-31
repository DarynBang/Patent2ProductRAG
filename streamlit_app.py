from InternshipRAG_pipeline import InternshipRAG_Pipeline
import streamlit as st
import pandas as pd
from config.rag_config import firm_config
from config.agent_config import agent_config
import ast

# Patent collection: patent_text_index
# Firm collection: company_summary_index

def main():
    st.set_page_config(layout="wide")

    st.title("Patent to Product RAG Demo")

    # firm_df = pd.read_csv(firm_config.get("firm_csv"))

    INDEX_DIR = r"C:\Users\Daryn Bang\Desktop\Internship\RAG_experiments\RAG_INDEX"

    # Initialize Pipeline
    pipeline = InternshipRAG_Pipeline(
        index_dir=INDEX_DIR,
        agent_config=agent_config,
        firm_config=firm_config,
        ingest_only=False
    )

    # Sidebar: configuration
    st.sidebar.header("Config")
    top_k = st.sidebar.number_input(
        "Top K results", min_value=1, max_value=20, value=5, step=1
    )

    # New sidebar buttons
    use_planning = st.sidebar.checkbox("Optimize the Query", value=False)
    display_market = st.sidebar.checkbox("Display Market Analysis", value=False)

    # Detect toggle change for 'use_planning'
    if 'prev_use_planning' not in st.session_state:
        st.session_state['prev_use_planning'] = False

    # If the toggle changed, clear the previous results
    if use_planning != st.session_state['prev_use_planning']:
        st.session_state['result'] = None
        st.session_state['last_query'] = ""
        st.session_state['prev_use_planning'] = use_planning

    st.write("---")
    query = st.text_input("Enter your query:", key='input_query')

    # Perform search and cache results in session_state
    if st.button("Search") and query:
        st.session_state['last_query'] = query
        with st.spinner("Processing Retrieval of relevant firms..."):
            res = pipeline.process_query(
                query,
                top_k=top_k,
                planning=use_planning
            )
        # Store results
        st.session_state['result'] = res

    # If we have a previous result, display it
    result = st.session_state.get('result')
    if result:
        st.markdown(f"**Input Patent Abstract:** {st.session_state.get('last_query', '')}")

        firm_results = result.get('retrieved_firms', [])
        market_response = result.get('market_analysis', '')
        product_suggestions = result.get('product_suggestions', {})
        firm_used_text = result.get('firm_used_text', {})


        # Display planning query if used
        if use_planning:
            new_query = result.get("query", "")
            st.markdown(f"**New Query after Planning Agent:** {new_query}")

        # Display firm-level results
        st.header("Firm & Product Suggestions")
        for ctx in firm_results:
            c_id = ctx["company_id"]
            hightech_status = ctx.get('hightechflag', False)
            if hightech_status:
                st.markdown(
                    f"**Rank {ctx['rank']}** | Company: `{ctx['company_name']}` (HighTech: {hightech_status}) (Company ID: {c_id})| Score: {ctx['score']:.3f}"
                )
            else:
                st.markdown(
                    f"**Rank {ctx['rank']}** | Company: `{ctx['company_name']}` (Company ID: {c_id}) | Score: {ctx['score']:.3f}"
                )
            # st.write("**Firm summary:**", ctx['chunk'])
            st.write(f"**Keywords:** {ctx['company_keywords']}")

            if firm_used_text[c_id]:
                st.write("**For Debugging:** Used Text + Keywords")
            else:
                st.write("**For Debugging:** Used Keywords Only")
            st.write(f"**Products:** \n {product_suggestions[c_id]}")

            st.markdown(
                f"**Webpages:** {ctx['webpages']}"
            )
            st.write("---")

        # if display_market:
        #     st.header("Market Analysis")
        #     if market_response:
        #         st.write(market_response)
        #     else:
        #         st.write("No market analysis available.")


if __name__ == "__main__":
    main()
