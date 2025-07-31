"""
streamlit_app.py

Patent to Product RAG Demo - Streamlit Web Application

This module provides a comprehensive web interface for the Patent to Product RAG system,
allowing users to find relevant firms and market opportunities based on patent abstracts.

Key Features:
- Interactive search interface with patent abstract input
- Multi-tab interface (Main Search, Testing, Documentation, Settings)
- Export functionality (TXT and JSON formats)
- Market analysis display toggle
- Query optimization with planning agent
- Example patent abstracts for quick testing
- System diagnostics and debugging tools

Main Components:
- main(): Entry point that sets up the Streamlit interface with tabs
- main_search_interface(): Core search functionality and results display
- display_results(): Renders search results with firm information and product suggestions
- export_results_txt(): Exports search results to formatted text files
- export_results_json(): Exports search results to JSON format
- testing_interface(): Debugging and system testing tools
- documentation_interface(): Comprehensive system documentation
- settings_interface(): Configuration management and file maintenance

Usage:
    streamlit run streamlit_app.py

Dependencies:
    - InternshipRAG_pipeline: Core RAG processing pipeline
    - streamlit: Web application framework
    - pandas: Data manipulation
    - json: JSON handling for exports
    - datetime: Timestamp generation
    - os: File system operations
"""

from InternshipRAG_pipeline import InternshipRAG_Pipeline
import streamlit as st
import pandas as pd
from config.rag_config import firm_config
from config.agent_config import agent_config
import json
from datetime import datetime
import os

def main():
    """
    Main entry point for the Streamlit application.
    
    Sets up the page configuration and creates a tabbed interface with:
    - Main Search: Primary patent-to-product search functionality
    - Testing & Debug: System diagnostics and component testing
    - Documentation: Comprehensive system documentation
    - Settings: Configuration management and maintenance
    """
    st.set_page_config(
        page_title="Patent to Product RAG Demo",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="🔬"
    )

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Main Search", "🧪 Testing & Debug", "📚 Documentation", "⚙️ Settings"])
    
    with tab1:
        main_search_interface()
    
    with tab2:
        testing_interface()
    
    with tab3:
        documentation_interface()
    
    with tab4:
        settings_interface()

def main_search_interface():
    """
    Main search interface for patent-to-product matching.
    
    Provides:
    - Patent abstract input with example queries
    - Configuration options (top-k results, query optimization, market analysis)
    - Search execution and results display
    - Export functionality for search results
    
    The interface handles pipeline initialization, query processing, and result presentation
    with proper error handling and user feedback.
    """
    st.title("🔬 Patent to Product RAG Demo")
    st.markdown("*Find relevant firms and market opportunities based on patent abstracts*")

    INDEX_DIR = r"RAG_INDEX"

    # Initialize Pipeline with error handling
    try:
        pipeline = InternshipRAG_Pipeline(
            index_dir=INDEX_DIR,
            agent_config=agent_config,
            firm_config=firm_config,
            ingest_only=False
        )
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {str(e)}")
        st.stop()

    # Sidebar Configuration
    st.sidebar.header("🔧 Configuration")
    top_k = st.sidebar.number_input(
        "Top K results", min_value=1, max_value=20, value=5, step=1
    )

    use_planning = st.sidebar.checkbox("🧠 Optimize the Query", value=False)
    display_market = st.sidebar.checkbox("📈 Display Market Analysis", value=False)

    # Handle state changes
    if 'prev_use_planning' not in st.session_state:
        st.session_state['prev_use_planning'] = False

    if use_planning != st.session_state['prev_use_planning']:
        st.session_state['result'] = None
        st.session_state['last_query'] = ""
        st.session_state['prev_use_planning'] = use_planning

    # Query Interface
    st.markdown("---")
    
    # Example queries dropdown (placed before the text area)
    example_queries = {
        "Medical Diagnosis": "An apparatus and a method for diagnosis are provided. The apparatus for diagnosis lesion include: a model generation unit configured to categorize learning data into one or more categories and to generate one or more categorized diagnostic models based on the categorized learning data...",
        "AI/ML Technology": "A machine learning system for automated data processing and pattern recognition in large datasets...",
        "IoT Device": "An Internet of Things device comprising sensors, wireless communication modules, and data processing capabilities..."
    }
    
    selected_example = st.selectbox("💡 Try an example:", [""] + list(example_queries.keys()))
    
    # Get the default value for the text area
    default_query = ""
    if selected_example and selected_example in example_queries:
        default_query = example_queries[selected_example]
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_area(
            "📝 Enter your patent abstract:", 
            value=default_query,
            height=150,
            placeholder="Paste your patent abstract here..."
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_button = st.button("🚀 Search", type="primary", use_container_width=True)

    # Search Processing
    if search_button and query:
        st.session_state['last_query'] = query
        with st.spinner("🔄 Processing retrieval of relevant firms..."):
            try:
                res = pipeline.process_query(query, top_k=top_k, planning=use_planning)
                st.session_state['result'] = res
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                return

    # Display Results
    result = st.session_state.get('result')
    if result:
        display_results(result, display_market, use_planning)

def display_results(result, display_market, use_planning):
    """
    Display search results in a structured format.
    
    Args:
        result (dict): Search results from the RAG pipeline containing:
            - retrieved_firms: List of relevant firms with scores and metadata
            - product_suggestions: AI-generated product recommendations per firm
            - firm_used_text: Flags indicating retrieval method used
            - market_analysis: Market analysis data (if available)
            - query: Optimized query (if planning was used)
        display_market (bool): Whether to show market analysis section
        use_planning (bool): Whether query planning was used
    
    Features:
    - Export buttons for TXT and JSON formats
    - Expandable firm cards with detailed information
    - Market analysis section (toggleable)
    - Product suggestions with clean formatting
    - Relevance scores and ranking information
    """
    st.markdown("---")
    st.markdown("## 📊 Search Results")
    
    # Export functionality
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("💾 Export to TXT", type="secondary"):
            export_results_txt(result, use_planning)
    with col2:
        if st.button("📄 Export to JSON", type="secondary"):
            export_results_json(result, use_planning)
    
    st.markdown(f"**📝 Input Patent Abstract:** {st.session_state.get('last_query', '')}")
    
    # Display planning query if used
    if use_planning:
        new_query = result.get("query", "")
        st.markdown(f"**🧠 Optimized Query:** {new_query}")
    
    firm_results = result.get('retrieved_firms', [])
    product_suggestions = result.get('product_suggestions', {})
    firm_used_text = result.get('firm_used_text', {})
    market_analysis = result.get('market_analysis')
    
    if firm_results:
        st.markdown(f"**🏢 Found {len(firm_results)} Relevant Firms:**")
        
        for ctx in firm_results:
            c_id = ctx["company_id"]
            hightech_status = ctx.get('hightechflag', False)
            
            with st.expander(f"🏢 Rank #{ctx['rank']}: {ctx['company_name']} (Score: {ctx['score']:.3f})", expanded=ctx['rank']<=3):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Company ID:** {c_id}")
                    if hightech_status:
                        st.markdown("**Status:** 🔬 HighTech Company")
                    st.markdown(f"**Keywords:** {ctx['company_keywords']}")
                    st.markdown(f"**Webpages:** {ctx.get('webpages', 'N/A')}")
                    
                    # Debug info
                    debug_info = "Used Text + Keywords" if firm_used_text.get(c_id) else "Used Keywords Only"
                    st.markdown(f"**Debug Info:** {debug_info}")
                
                with col2:
                    st.metric("Relevance Score", f"{ctx['score']:.3f}")
                    st.metric("Rank", ctx['rank'])
                
                # Product suggestions
                if c_id in product_suggestions:
                    st.markdown("**💡 Product Suggestions:**")
                    st.write(product_suggestions[c_id])
    else:
        st.warning("⚠️ No relevant firms found. Try adjusting your query or search parameters.")
    
    # Display Market Analysis at the bottom if enabled
    if display_market:
        st.markdown("---")
        st.markdown("## 📈 Market Analysis")
        
        if market_analysis:
            with st.container():
                st.write(market_analysis)
        else:
            st.info("ℹ️ No market analysis data available for this search.")

def export_results_txt(result, use_planning):
    """
    Export search results to a formatted TXT file.
    
    Args:
        result (dict): Search results from the RAG pipeline
        use_planning (bool): Whether query planning was used
    
    Creates a comprehensive text export including:
    - Header with timestamp and query information
    - Firm details with rankings, scores, and metadata
    - Keywords formatted for readability (10 per line)
    - Webpages as bulleted lists
    - Product suggestions with cleaned formatting (markdown removed)
    - Debug information about retrieval methods
    - Market analysis (if available) at the bottom
    
    The export is saved to the 'exports/' directory with timestamp-based filename
    and provides both file storage and direct download capability.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"search_results_{timestamp}.txt"
    
    def clean_product_suggestions(text):
        """Clean product suggestions text from markdown formatting"""
        if not text:
            return ""
        
        # Remove markdown formatting
        import re
        # Remove ** bold formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        # Remove * italic formatting
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        # Remove numbered list formatting and replace with better formatting
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Handle numbered items (1. 2. 3. etc.)
            if re.match(r'^\d+\.\s+', line):
                # Extract the title and description
                match = re.match(r'^(\d+)\.\s+(.*)', line)
                if match:
                    num, content = match.groups()
                    cleaned_lines.append(f"\n{num}. {content}")
            else:
                # Regular content lines - add proper indentation
                cleaned_lines.append(f"   {line}")
        
        return '\n'.join(cleaned_lines)
    
    content = []
    content.append("=" * 70)
    content.append("PATENT TO PRODUCT RAG SEARCH RESULTS")
    content.append("=" * 70)
    content.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    content.append(f"Original Query: {st.session_state.get('last_query', '')}")
    
    if use_planning:
        content.append(f"Optimized Query: {result.get('query', '')}")
    
    content.append("=" * 70)
    
    firm_results = result.get('retrieved_firms', [])
    product_suggestions = result.get('product_suggestions', {})
    firm_used_text = result.get('firm_used_text', {})
    
    content.append(f"Number of firms found: {len(firm_results)}")
    content.append("")
    
    for ctx in firm_results:
        c_id = ctx["company_id"]
        content.append(f"FIRM #{ctx['rank']}")
        content.append("-" * 40)
        content.append(f"Company Name: {ctx['company_name']}")
        content.append(f"Company ID: {c_id}")
        content.append(f"Relevance Score: {ctx['score']:.3f}")
        content.append(f"HighTech Status: {ctx.get('hightechflag', False)}")
        content.append("")
        content.append("Keywords:")
        keywords = ctx['company_keywords']
        if isinstance(keywords, list):
            # Format keywords in multiple lines, 10 per line
            for i in range(0, len(keywords), 10):
                keyword_line = ", ".join(keywords[i:i+10])
                content.append(f"  {keyword_line}")
        else:
            content.append(f"  {keywords}")
        content.append("")
        
        webpages = ctx.get('webpages', 'N/A')
        content.append("Webpages:")
        if isinstance(webpages, list):
            for webpage in webpages:
                content.append(f"  - {webpage}")
        else:
            content.append(f"  {webpages}")
        content.append("")
        
        content.append(f"Debug Info: {'Used Text + Keywords' if firm_used_text.get(c_id) else 'Used Keywords Only'}")
        content.append("")
        
        if c_id in product_suggestions:
            content.append("Product Suggestions:")
            cleaned_suggestions = clean_product_suggestions(product_suggestions[c_id])
            content.append(cleaned_suggestions)
        
        content.append("")
        content.append("=" * 40)
        content.append("")
    
    # Market analysis is not included in exports per user request
    
    # Create exports directory if it doesn't exist
    os.makedirs("exports", exist_ok=True)
    filepath = os.path.join("exports", filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(content))
    
    st.success(f"✅ Results exported to: {filepath}")
    
    # Provide download button
    with open(filepath, 'r', encoding='utf-8') as f:
        st.download_button(
            label="⬇️ Download TXT File",
            data=f.read(),
            file_name=filename,
            mime='text/plain'
        )

def export_results_json(result, use_planning):
    """
    Export search results to a structured JSON file.
    
    Args:
        result (dict): Search results from the RAG pipeline
        use_planning (bool): Whether query planning was used
    
    Creates a JSON export containing:
    - Metadata (timestamp, original query, optimized query)
    - Complete results data structure
    - Configuration flags and settings
    
    The JSON format preserves the original data structure for programmatic access
    and further processing. Files are saved with timestamp-based naming and provide
    both local storage and direct download options.
    
    Note: Market analysis is excluded from exports per user requirements.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"search_results_{timestamp}.json"
    
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "original_query": st.session_state.get('last_query', ''),
        "optimized_query": result.get('query', '') if use_planning else None,
        "use_planning": use_planning,
        "results": result  # Market analysis excluded from exports per user request
    }
    
    os.makedirs("exports", exist_ok=True)
    filepath = os.path.join("exports", filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    st.success(f"✅ Results exported to: {filepath}")
    
    # Provide download button
    st.download_button(
        label="⬇️ Download JSON File",
        data=json.dumps(export_data, indent=2, ensure_ascii=False),
        file_name=filename,
        mime='application/json'
    )

def testing_interface():
    """
    Testing and debugging interface for system diagnostics.
    
    Provides tools for:
    - Pipeline initialization testing
    - Index directory status checking  
    - Configuration validation
    - System information display
    - Export history review
    
    Helps developers and users diagnose issues, verify system status,
    and understand the current configuration state.
    """
    st.header("🧪 Testing & Debug Interface")
    st.markdown("Test individual components and debug the system")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Component Testing")
        
        if st.button("Test Pipeline Initialization"):
            try:
                pipeline = InternshipRAG_Pipeline(
                    index_dir="RAG_INDEX",
                    agent_config=agent_config,
                    firm_config=firm_config,
                    ingest_only=False
                )
                st.success("✅ Pipeline initialized successfully")
            except Exception as e:
                st.error(f"❌ Pipeline initialization failed: {str(e)}")
        
        if st.button("Test Index Status"):
            index_dir = "RAG_INDEX"
            if os.path.exists(index_dir):
                st.success(f"✅ Index directory exists at: {index_dir}")
                contents = os.listdir(index_dir)
                st.write("Contents:", contents)
            else:
                st.warning(f"⚠️ Index directory not found at: {index_dir}")
        
        if st.button("Test Configuration"):
            st.write("**Firm Config:**", firm_config)
            st.write("**Agent Config:**", agent_config)
    
    with col2:
        st.subheader("📊 System Information")
        st.write("**Current Working Directory:**", os.getcwd())
        st.write("**Available Exports:**")
        
        if os.path.exists("exports"):
            exports = os.listdir("exports")
            if exports:
                for export_file in exports[-5:]:  # Show last 5 exports
                    st.write(f"- {export_file}")
            else:
                st.write("No exports found")
        else:
            st.write("No exports directory found")

def documentation_interface():
    """
    Comprehensive documentation interface for the RAG system.
    
    Provides detailed information about:
    - System architecture and components
    - Technologies and frameworks used
    - Usage guidelines and best practices
    - Search methods and retrieval strategies
    - Data sources and system requirements
    - Troubleshooting guides and common issues
    - Result interpretation guidelines
    
    Serves as a complete reference for users and developers
    working with the Patent to Product RAG system.
    """
    st.header("📚 Documentation")
    
    st.markdown("""
    ## 🔬 Patent to Product RAG System
    
    This application uses Retrieval-Augmented Generation (RAG) to help identify relevant firms and product opportunities based on patent abstracts.
    
    ### 🏗️ System Architecture
    
    #### Components:
    - **RAG Pipeline**: Core retrieval and generation system
    - **Multi-Agent System**: Planning, market analysis, and product suggestion agents
    - **Vector Database**: ChromaDB for semantic search
    - **Embedding Models**: Sentence transformers for text embedding
    
    #### Technologies Used:
    - **LangChain**: Framework for LLM applications
    - **ChromaDB**: Vector database for similarity search
    - **Sentence Transformers**: For text embeddings
    - **Streamlit**: Web interface
    - **Multiple LLM Providers**: OpenAI, Google Gemini, Qwen
    
    ### 📖 Usage Guidelines
    
    #### 1. Basic Search:
    1. Enter your patent abstract in the text area
    2. Adjust the "Top K results" to control how many firms to retrieve
    3. Click "Search" to find relevant firms
    
    #### 2. Advanced Features:
    - **Query Optimization**: Enable "Optimize the Query" to use AI planning for better search queries
    - **Market Analysis**: Enable to get market insights (when available)
    - **Export Results**: Save your search results as TXT or JSON files
    
    #### 3. Configuration:
    - **Top K Results**: Number of most relevant firms to return (1-20)
    - **Query Optimization**: Uses planning agent to refine your search query
    - **Market Analysis**: Provides market context for found firms
    
    ### 🔍 Search Methods
    
    The system uses multiple retrieval strategies:
    - **Semantic Search**: Using sentence transformers for meaning-based matching
    - **Keyword Matching**: Traditional keyword-based search
    - **Mixed Retrieval**: Combination of semantic and keyword approaches
    
    ### 💾 Data Sources
    
    - **Firm Database**: Company information with keywords and summaries
    - **Patent Data**: Patent abstracts for training and testing
    - **Market Data**: Business intelligence for market analysis
    
    ### 🔧 Troubleshooting
    
    #### Common Issues:
    1. **Pipeline Initialization Failed**: Check if index directory exists
    2. **No Results Found**: Try broader keywords or check query formatting
    3. **Export Failed**: Ensure write permissions in the application directory
    
    #### Debug Features:
    - Use the "Testing & Debug" tab to diagnose issues
    - Check system information and component status
    - Review configuration settings
    
    ### 📊 Result Interpretation
    
    #### Relevance Scores:
    - **0.8-1.0**: Highly relevant firms
    - **0.6-0.8**: Moderately relevant firms
    - **0.4-0.6**: Potentially relevant firms
    - **<0.4**: Weakly relevant firms
    
    #### Company Information:
    - **HighTech Flag**: Indicates if company is in high-tech sector
    - **Keywords**: Key business areas and technologies
    - **Product Suggestions**: AI-generated product recommendations
    - **Debug Info**: Shows which retrieval method was used
    
    ### 🚀 Best Practices
    
    1. **Query Writing**:
       - Include technical details and specific terminology
       - Mention application domains and use cases
       - Specify technologies and methods used
    
    2. **Result Analysis**:
       - Focus on firms with scores > 0.6 for best matches
       - Review product suggestions for innovation opportunities
       - Check company websites for detailed information
    
    3. **Export Usage**:
       - Use TXT format for reports and documentation
       - Use JSON format for further data processing
       - Regular exports help track search history
    """)

def settings_interface():
    """
    Settings and configuration management interface.
    
    Features:
    - Current configuration display (firm and agent configs)
    - Export file management and history
    - Maintenance tools (cleanup old exports)
    - System file information
    
    Allows users to view current settings, manage exported files,
    and perform basic maintenance tasks on the system.
    """
    st.header("⚙️ Settings & Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Current Configuration")
        
        st.markdown("**Firm Configuration:**")
        for key, value in firm_config.items():
            st.write(f"- **{key}**: {value}")
        
        st.markdown("**Agent Configuration:**")
        for key, value in agent_config.items():
            st.write(f"- **{key}**: {value}")
    
    with col2:
        st.subheader("📁 File Management")
        
        # Show export directory contents
        if os.path.exists("exports"):
            st.markdown("**Recent Exports:**")
            exports = sorted(os.listdir("exports"), reverse=True)[:10]
            for export_file in exports:
                filepath = os.path.join("exports", export_file)
                file_size = os.path.getsize(filepath)
                st.write(f"📄 {export_file} ({file_size} bytes)")
        
        # Cleanup options
        st.markdown("**Maintenance:**")
        if st.button("🧹 Clear All Exports", type="secondary"):
            if os.path.exists("exports"):
                deleted_count = 0
                
                for filename in os.listdir("exports"):
                    filepath = os.path.join("exports", filename)
                    try:
                        os.remove(filepath)
                        deleted_count += 1
                    except Exception as e:
                        st.error(f"Failed to delete {filename}: {e}")
                
                if deleted_count > 0:
                    st.success(f"✅ Deleted {deleted_count} export files")
                else:
                    st.info("No export files found to delete")
            else:
                st.info("No exports directory found")

if __name__ == "__main__":
    main()
