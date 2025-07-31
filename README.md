# 🔬 Patent2ProductRAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?logo=database&logoColor=white)](https://docs.trychroma.com/)

**An Intelligent Agentic RAG System for Patent-to-Product Innovation Matching**

Patent2ProductRAG is a sophisticated Retrieval-Augmented Generation (RAG) system that bridges the gap between patent innovations and commercial opportunities. Using multi-agent architecture and advanced semantic search, it identifies relevant firms and generates actionable product suggestions based on patent abstracts.

---

## ✨ Key Features

- 🤖 **Multi-Agent Architecture**: Specialized agents for query planning, product suggestions, and market analysis
- 🔍 **Hybrid Search**: Combines semantic embeddings with keyword-based retrieval (BM25)
- 💡 **Smart Product Suggestions**: AI-powered recommendations tailored to firm capabilities
- 📊 **Market Analysis**: Comprehensive market opportunity identification
- 🌐 **Interactive Web Interface**: User-friendly Streamlit application
- 📁 **Export Capabilities**: Results export in TXT and JSON formats
- 🔧 **Configurable Backend**: Support for OpenAI, Google Gemini, and Qwen models

---

## 🏗️ System Architecture

<details>
<summary>Click to view detailed architecture</summary>

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Patent Input  │───▶│  Planning Agent │───▶│  Query Optimizer│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  ChromaDB       │◄───│   RAG System    │───▶│  Firm Retrieval │
│  Vector Store   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Product Suggest │◄───│ Multi-Agent     │───▶│ Market Analysis │
│ Agent           │    │ Runner          │    │ Agent           │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │ Streamlit UI    │
                       │ & Export System │
                       └─────────────────┘
```

### Core Components:
- **FirmSummaryRAG**: Manages firm data indexing and semantic retrieval
- **MultiAgentRunner**: Orchestrates the agent workflow and data flow
- **PlanningAgent**: Optimizes search queries from patent abstracts
- **ProductSuggestionAgent**: Generates business recommendations
- **MarketAnalysisAgent**: Identifies market opportunities and insights

</details>

---

## 📁 Project Structure

<details>
<summary>Click to view project structure</summary>

```
Patent2ProductRAG/
├── main.py                    # Clean CLI entry point (107 lines only!)
├── streamlit_app.py          # Web interface application
├── InternshipRAG_pipeline.py # Core RAG pipeline
├── firm_summary_rag.py       # Firm data processing
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE                   # MIT license
│
├── agents/                   # Multi-agent system
│   ├── __init__.py
│   ├── base.py              # Base agent class
│   ├── multi_agent_runner.py # Agent orchestration
│   ├── planning_agent.py    # Query optimization
│   ├── product_suggestion_agent.py # Product recommendations
│   ├── market_analysis_agent.py    # Market insights
│   └── registry.py          # Agent registration
│
├── config/                   # Configuration modules
│   ├── __init__.py
│   ├── agent_config.py      # Agent configurations
│   ├── logging_config.py    # Centralized logging setup
│   ├── prompt.py            # LLM prompts and templates
│   └── rag_config.py        # RAG system settings
│
├── utils/                    # Utility modules (refactored)
│   ├── __init__.py
│   ├── mode_utils.py        # Mode functions (test, chat, batch, ingest)
│   ├── export_utils.py      # TXT/JSON export functionality
│   ├── display_utils.py     # Console display and formatting
│   ├── cli_utils.py         # Command-line utilities
│   ├── cluster_utils.py     # Text clustering utilities
│   └── model_utils.py       # Model loading and management
│
├── query_generation/         # Query processing modules
│   ├── __init__.py
│   ├── base_runner.py       # Base query runner
│   ├── openai_runner.py     # OpenAI implementation
│   ├── gemini_runner.py     # Google Gemini implementation
│   └── qwen_runner.py       # Qwen model implementation
│
├── product_suggestion/       # Product suggestion modules
│   ├── __init__.py
│   ├── base_runner.py       # Base suggestion runner
│   ├── openai_runner.py     # OpenAI implementation
│   ├── gemini_runner.py     # Google Gemini implementation
│   └── qwen_runner.py       # Qwen model implementation
│
├── exports/                  # Auto-generated export files
│   ├── search_results_test_*.txt    # Human-readable results
│   ├── search_results_test_*.json   # Structured data results
│   ├── search_results_chat_*.txt    # Chat mode exports
│   └── search_results_chat_*.json   # Chat mode structured data
│
├── logs/                     # Auto-generated log files
│   └── patent2product_rag_*.log     # System logs
│
├── RAG_INDEX/               # ChromaDB vector database
│   ├── firm_data/           # Firm vector embeddings
│   └── firm_summary_index/  # Search indices
│
└── myenv/                   # Python virtual environment (user-created)
    └── ...
```

### 🏗️ Architecture Highlights

**Clean Separation of Concerns:**
- 🎯 **main.py**: Ultra-clean entry point (only 107 lines!)
- 🔧 **utils/mode_utils.py**: All mode operations (test, chat, batch, ingest)
- 📤 **utils/export_utils.py**: Automatic TXT/JSON export for every query
- 🖥️ **utils/display_utils.py**: Console formatting and display
- ⚙️ **utils/cli_utils.py**: Command-line argument handling
- 🤖 **agents/**: Specialized AI agents for different tasks
- ⚡ **Guaranteed Exports**: Every test/chat query creates 2 files automatically

</details>

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (recommended for optimal performance)
- GPU support (optional, for Qwen models)

### Installation

<details>
<summary>Click to view installation steps</summary>

1. **Clone the repository**
   ```bash
   git clone https://github.com/DarynBang/Patent2ProductRAG.git
   cd Patent2ProductRAG
   ```

2. **Create virtual environment**
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   GENAI_API_KEY=your_google_genai_api_key_here
   ```

5. **Download required data files**
   Download the CSV data files from Google Drive:
   ```
   🔗 https://drive.google.com/drive/folders/1DvbE5Izw8UloboOCa76hDZGQVSfmomPl?usp=sharing
   ```
   
   Place these files in the project root directory:
   - `firms_summary_keywords_qwen.csv`
   - `firms_summary_keywords_gpt.csv`
   - `firm_id_to_text_mapping.csv`

6. **Initialize the system (First-time setup)**
   Run data ingestion to build search indices:
   ```bash
   python main.py --mode ingest
   ```
   
   This will:
   - Process the CSV files
   - Create vector embeddings
   - Build ChromaDB search indices
   - Set up the RAG system for queries

</details>

### Usage

<details>
<summary>Click to view usage instructions</summary>

#### 🚀 First-Time Users - Quick Start

**Before using the system, you MUST run data ingestion:**

1. **Ensure data files are in place** (from step 5 above):
   - ✅ `firms_summary_keywords_qwen.csv`
   - ✅ `firms_summary_keywords_gpt.csv`
   - ✅ `firm_id_to_text_mapping.csv`

2. **Run initial data ingestion** (required once):
   ```bash
   python main.py --mode ingest
   ```
   
   This process will:
   - 📊 Load firm data from CSV files
   - 🔍 Create vector embeddings for semantic search
   - 💾 Build ChromaDB indices
   - ⚡ Prepare the system for fast queries
   
   **Note**: This step takes 5-15 minutes depending on your hardware.

3. **Test the system**:
   ```bash
   python main.py --mode test
   ```

#### Option 1: Web Interface (Recommended)
```bash
streamlit run streamlit_app.py
```
Navigate to `http://localhost:8501` in your browser.

#### Option 2: Command Line Interface

The system provides a comprehensive CLI with multiple operation modes:

**🧪 Test Mode (Default)**
```bash
python main.py --mode test
python main.py --mode test --top-k 10 --planning --market-analysis
```

**💬 Interactive Chat Mode**
```bash
python main.py --mode chat
python main.py --mode chat --top-k 7 --market-analysis
```

**📊 Batch Processing Mode**
```bash
python main.py --mode batch --input patents.txt --output results.json
```

**📥 Data Ingestion Mode**
```bash
python main.py --mode ingest --force-reindex
```

**Available Arguments:**
- `--mode`: Operation mode (`test`, `chat`, `ingest`, `batch`)
- `--input`: Input file for batch processing
- `--output`: Output file for results
- `--top-k`: Number of results (default: 5)
- `--planning`: Enable query optimization
- `--market-analysis`: Show market insights
- `--force-reindex`: Rebuild search indices

#### Basic Workflow:
1. **Data Ingestion**: Load and index firm data (run once)
2. **Query Processing**: Input patent abstracts for analysis
3. **Results Review**: Examine firm matches and product suggestions
4. **Export**: Save results for further analysis

</details>

---

## 📊 Data Sources

**📥 Data Download**: Get the required CSV files from [Google Drive](https://drive.google.com/drive/folders/1DvbE5Izw8UloboOCa76hDZGQVSfmomPl?usp=sharing)

The system processes several types of data:

<details>
<summary>Click to view data structure details</summary>

- **Firm Data** (`firms_summary_keywords_*.csv`):
  - Company profiles and summaries
  - Business keywords and categories
  - High-tech classification flags
  - Webpage URLs and contact information

- **Text Mappings** (`firm_id_to_text_mapping.csv`):
  - Extracted webpage content
  - Company descriptions and capabilities
  - Product and service information

- **Patent Abstracts**:
  - Technical innovation descriptions
  - Problem statements and solutions
  - Key technological components

**Required Files** (place in project root):
- `firms_summary_keywords_qwen.csv` - Main firm dataset processed with Qwen LLM
- `firms_summary_keywords_gpt.csv` - Alternative firm dataset processed with GPT
- `firm_id_to_text_mapping.csv` - Webpage text content for each firm

</details>

---

## ⚙️ Configuration

### RAG Configuration

<details>
<summary>Click to view RAG settings</summary>

Located in `config/rag_config.py`:

```python
firm_config = {
    "firm_csv": "firms_summary_keywords_qwen.csv",
    "firm_id_to_text_mapping": "firm_id_to_text_mapping.csv",
    "embed_model": "sentence-transformers/all-MiniLM-L12-v2",
    "top_k": 5,
    "retrieval_mode": "mixed",  # Options: "semantic", "keyword", "mixed"
    "force_reindex": False
}
```

</details>

### Agent Configuration

<details>
<summary>Click to view agent settings</summary>

Located in `config/agent_config.py`:

```python
agent_config = {
    "qa_planning_agent": "gemini",        # Query optimization
    "qa_product_suggestion_agent": "gemini",  # Product recommendations  
    "qa_market_analyst_agent": "gemini"   # Market analysis
}
```

Supported models: `"openai"`, `"gemini"`, `"qwen"`

</details>

---

## 🎯 Use Cases

- **Innovation Scouting**: Identify firms with relevant capabilities for patent commercialization
- **Technology Transfer**: Match university patents with industry partners
- **Competitive Analysis**: Discover market players in specific technology domains
- **Investment Research**: Evaluate commercial potential of patent portfolios
- **Product Development**: Generate innovation-driven product concepts

---

## 🔧 Advanced Features

### Retrieval Strategies

<details>
<summary>Click to view retrieval methods</summary>

- **Semantic Search**: Uses sentence transformers for meaning-based matching
- **Keyword Search**: BM25-based traditional text retrieval
- **Mixed Mode**: Intelligent combination of both approaches for optimal results

</details>

### Export Formats

<details>
<summary>Click to view export options</summary>

- **TXT Format**: Human-readable reports with structured information
- **JSON Format**: Machine-readable data for programmatic access
- **Timestamp-based naming**: Automatic file organization
- **Comprehensive data**: Includes queries, results, scores, and metadata

</details>

### System Diagnostics

<details>
<summary>Click to view diagnostic features</summary>

- **Pipeline Testing**: Verify component initialization and functionality
- **Index Status**: Check database state and contents
- **Configuration Validation**: Ensure proper system setup
- **Performance Monitoring**: Track processing times and resource usage

</details>

---

## 📚 API Reference

### Core Classes

<details>
<summary>Click to view API documentation</summary>

#### `InternshipRAG_Pipeline`
Main pipeline orchestrator for the entire system.

```python
pipeline = InternshipRAG_Pipeline(
    index_dir="RAG_INDEX",
    agent_config=agent_config,
    firm_config=firm_config
)

results = pipeline.process_query(
    query="patent abstract text",
    top_k=5,
    planning=True
)
```

#### `FirmSummaryRAG`
Handles firm data indexing and retrieval operations.

```python
rag = FirmSummaryRAG(
    df=firm_dataframe,
    index_dir="RAG_INDEX",
    config=firm_config
)

matches = rag.retrieve_firm_contexts(query, top_k=10)
```

#### `MultiAgentRunner`
Manages the multi-agent workflow and coordination.

```python
runner = MultiAgentRunner(firm_summary_rag=rag_system)
runner.register_agent("PlanningAgent", qa_model="gemini")
results = runner.run(initial_input, planning=True, top_k=5)
```

</details>

---

## 🧪 Testing

<details>
<summary>Click to view testing information</summary>

The system includes built-in testing capabilities:

- **Component Tests**: Verify individual module functionality
- **Integration Tests**: End-to-end pipeline validation  
- **Performance Tests**: Benchmark retrieval and generation speed
- **Data Quality Tests**: Validate index integrity and search results

Run tests through the Streamlit interface under the "Testing & Debug" tab.

</details>

---

## 🛠️ Troubleshooting

<details>
<summary>Click to view common issues and solutions</summary>

### Common Issues

**Pipeline Initialization Failed**
- Ensure all dependencies are installed correctly
- Check that data files exist in the specified locations
- Verify API keys are properly configured

**No Search Results**
- Try broader search terms or different keywords
- Check if the index was built successfully
- Verify firm data is properly loaded

**Memory Issues**
- Reduce batch sizes in configuration
- Close other memory-intensive applications
- Consider using quantized models for GPU operations

**API Rate Limits**
- Implement request delays between calls
- Use local models (Qwen) as alternatives
- Monitor API usage and quotas

</details>

---

## 🤝 Contributing

<details>
<summary>Click to view Contributing Guidelines</summary>

We welcome contributions to Patent2ProductRAG! Here's how you can help:

### Getting Started
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Contribution Areas
- **Core Features**: Enhance RAG algorithms and agent capabilities
- **Documentation**: Improve guides, examples, and API documentation
- **Testing**: Add test cases and improve coverage
- **Performance**: Optimize search and generation speed
- **UI/UX**: Enhance the Streamlit interface
- **Integrations**: Add support for new LLM providers

### Code Standards
- Follow PEP 8 style guidelines
- Include comprehensive docstrings
- Add type hints where appropriate
- Write unit tests for new features
- Update documentation for changes

</details>

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👥 Authors

This project was developed by:

**[Nguyen Quang Phu (pdz1804)](https://github.com/pdz1804)** and **[Tieu Tri Bang (DarynBang)](https://github.com/DarynBang)**

---

## 📞 Support

For questions, issues, or contributions:

- 🐛 **Issues**: [GitHub Issues](https://github.com/DarynBang/Patent2ProductRAG/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/DarynBang/Patent2ProductRAG/discussions)
- 📧 **Email**: quangphunguyen1804@gmail.com

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by the Patent2ProductRAG team

</div>
