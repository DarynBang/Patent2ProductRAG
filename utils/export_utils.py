"""
utils/export_utils.py

Export utilities for Patent2ProductRAG system.
Handles exporting search results to TXT and JSON formats.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from config.logging_config import get_logger

logger = get_logger(__name__)

def export_results_to_files(results, query_text="", mode="search"):
    """
    Export results to both TXT and JSON files automatically.
    
    Args:
        results (dict): Results from pipeline processing
        query_text (str): The original patent abstract query
        mode (str): The mode (test/chat/search) used for filename
    
    Returns:
        tuple: (txt_filename, json_filename) of created files
    """
    # Create exports directory if it doesn't exist
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate filenames
    txt_filename = f"search_results_{mode}_{timestamp}.txt"
    json_filename = f"search_results_{mode}_{timestamp}.json"
    
    txt_path = exports_dir / txt_filename
    json_path = exports_dir / json_filename
    
    try:
        # Export TXT format
        _export_txt_format(txt_path, results, query_text, mode)
        
        # Export JSON format
        _export_json_format(json_path, results, query_text, mode)
        
        logger.info(f"Results exported to: {txt_filename} and {json_filename}")
        print(f"📄 Results exported to:")
        print(f"   📝 TXT: {txt_filename}")
        print(f"   📊 JSON: {json_filename}")
        
        return txt_filename, json_filename
        
    except Exception as e:
        logger.error(f"Error exporting results: {e}", exc_info=True)
        print(f"❌ Export failed: {e}")
        return None, None

def _export_txt_format(file_path, results, query_text, mode):
    """Export results in human-readable TXT format matching Streamlit app style."""
    
    def clean_product_suggestions(text):
        """Clean product suggestions text from markdown formatting"""
        if not text:
            return ""
        
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

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("PATENT TO PRODUCT RAG SEARCH RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Original Query: {query_text}\n")
        
        if 'query' in results:
            f.write(f"Optimized Query: {results['query']}\n")
        
        f.write("=" * 70 + "\n")
        
        firms = results.get('retrieved_firms', [])
        product_suggestions = results.get('product_suggestions', {})
        firm_used_text = results.get('firm_used_text', {})
        market_analysis = results.get('market_analysis')
        
        f.write(f"Number of firms found: {len(firms)}\n")
        f.write("\n")
        
        for firm in firms:
            c_id = firm["company_id"]
            f.write(f"FIRM #{firm['rank']}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Company Name: {firm['company_name']}\n")
            f.write(f"Company ID: {c_id}\n")
            f.write(f"Relevance Score: {firm['score']:.3f}\n")
            f.write(f"HighTech Status: {firm.get('hightechflag', False)}\n")
            f.write("\n")
            
            # Keywords formatting
            f.write("Keywords:\n")
            keywords = firm['company_keywords']
            if isinstance(keywords, list):
                # Format keywords in multiple lines, 10 per line
                for i in range(0, len(keywords), 10):
                    keyword_line = ", ".join(keywords[i:i+10])
                    f.write(f"  {keyword_line}\n")
            else:
                f.write(f"  {keywords}\n")
            f.write("\n")
            
            # Webpages formatting
            webpages = firm.get('webpages', 'N/A')
            f.write("Webpages:\n")
            if isinstance(webpages, list):
                for webpage in webpages:
                    f.write(f"  - {webpage}\n")
            else:
                f.write(f"  {webpages}\n")
            f.write("\n")
            
            # Debug info
            debug_info = "Used Text + Keywords" if firm_used_text.get(c_id) else "Used Keywords Only"
            f.write(f"Debug Info: {debug_info}\n")
            f.write("\n")
            
            # Product suggestions with clean formatting
            if c_id in product_suggestions:
                f.write("Product Suggestions:\n")
                cleaned_suggestions = clean_product_suggestions(product_suggestions[c_id])
                f.write(cleaned_suggestions + "\n")
            
            f.write("\n")
            f.write("=" * 40 + "\n")
            f.write("\n")
        
        if market_analysis:
            f.write("MARKET ANALYSIS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"{market_analysis}\n")

def _export_json_format(file_path, results, query_text, mode):
    """Export results in structured JSON format."""
    export_data = {
        "metadata": {
            "mode": mode,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "query_text": query_text,
            "query_length": len(query_text)
        },
        "results": results
    }
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

def cleanup_old_exports(days_old=7):
    """
    Clean up export files older than specified days.
    
    Args:
        days_old (int): Number of days after which files should be deleted
    
    Returns:
        int: Number of files deleted
    """
    exports_dir = Path("exports")
    if not exports_dir.exists():
        return 0
    
    cutoff_time = datetime.now().timestamp() - (days_old * 24 * 3600)
    deleted_count = 0
    
    try:
        for file_path in exports_dir.glob("search_results_*.txt"):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                deleted_count += 1
        
        for file_path in exports_dir.glob("search_results_*.json"):
            if file_path.stat().st_mtime < cutoff_time:
                file_path.unlink()
                deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old export files")
            
    except Exception as e:
        logger.error(f"Error cleaning up exports: {e}")
    
    return deleted_count
