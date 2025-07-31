"""
utils/display_utils.py

Display utilities for Patent2ProductRAG system.
Handles formatting and displaying search results in the console.
"""

from config.logging_config import get_logger

logger = get_logger(__name__)

def display_results(results, show_market_analysis=False):
    """
    Display search results in a formatted way.
    
    Args:
        results (dict): Results from pipeline processing
        show_market_analysis (bool): Whether to display market analysis
    """
    print("\n" + "="*60)
    print("📊 SEARCH RESULTS")
    print("="*60)
    
    if 'query' in results:
        print(f"🧠 Optimized Query: {results['query']}")
    
    firms = results.get('retrieved_firms', [])
    product_suggestions = results.get('product_suggestions', {})
    market_analysis = results.get('market_analysis')
    
    print(f"\n🏢 Found {len(firms)} relevant firms:")
    
    for firm in firms:
        print(f"\n📈 Rank #{firm['rank']}: {firm['company_name']}")
        print(f"   Score: {firm['score']:.3f}")
        print(f"   Company ID: {firm['company_id']}")
        print(f"   Keywords: {firm['company_keywords'][:100]}...")
        
        if firm['company_id'] in product_suggestions:
            suggestions = product_suggestions[firm['company_id']]
            print(f"   💡 Products: {suggestions[:150]}...")
    
    if show_market_analysis and market_analysis:
        print(f"\n📈 MARKET ANALYSIS:")
        print("-" * 40)
        print(market_analysis)
    elif show_market_analysis:
        print(f"\n📈 MARKET ANALYSIS: Not available")

def show_help():
    """Display help information."""
    print("""
📚 HELP - Patent2ProductRAG CLI

Available Commands:
- Enter any patent abstract to search for relevant firms
- 'quit' or 'q': Exit the application
- 'help': Show this help message

Tips:
- Provide detailed patent abstracts for better results
- Include technical keywords and application domains
- Be specific about the innovation and its purpose

Example patent abstract:
"A machine learning system for automated medical diagnosis using 
convolutional neural networks to analyze medical imaging data..."
    """)

def display_mode_header(mode_name, description=""):
    """
    Display a formatted header for different modes.
    
    Args:
        mode_name (str): Name of the mode
        description (str): Optional description
    """
    print("\n" + "="*60)
    print(f"🎯 {mode_name.upper()}")
    print("="*60)
    if description:
        print(description)
        print("="*60)

def display_progress(current, total, operation="Processing"):
    """
    Display progress information.
    
    Args:
        current (int): Current item number
        total (int): Total number of items
        operation (str): Operation being performed
    """
    percentage = (current / total) * 100 if total > 0 else 0
    print(f"🔄 {operation} {current}/{total} ({percentage:.1f}%)")

def display_configuration(config_dict):
    """
    Display configuration settings in a formatted way.
    
    Args:
        config_dict (dict): Configuration parameters to display
    """
    print("🔧 Configuration:")
    for key, value in config_dict.items():
        print(f"   {key}: {value}")

def display_performance_summary(metrics_dict):
    """
    Display performance metrics summary.
    
    Args:
        metrics_dict (dict): Performance metrics to display
    """
    print("\n⚡ Performance Summary:")
    for metric, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.2f}s")
        else:
            print(f"   {metric}: {value}")

def display_export_summary(txt_file, json_file):
    """
    Display export summary information.
    
    Args:
        txt_file (str): TXT filename
        json_file (str): JSON filename
    """
    if txt_file and json_file:
        print(f"\n📄 Results exported to:")
        print(f"   📝 TXT: {txt_file}")
        print(f"   📊 JSON: {json_file}")
    else:
        print(f"\n❌ Export failed")
