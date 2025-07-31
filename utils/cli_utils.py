"""
utils/cli_utils.py

Command-line interface utilities for Patent2ProductRAG system.
Handles argument parsing and CLI-specific functionality.
"""

import argparse
from config.logging_config import get_logger

logger = get_logger(__name__)

def create_argument_parser():
    """
    Create and configure the command-line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description='Patent2ProductRAG - Find relevant firms and products for patents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode test
  python main.py --mode chat --planning --market-analysis
  python main.py --mode batch --input patents.txt --output results.json
  python main.py --mode ingest --force-reindex

Modes:
  test         Run with predefined patent abstract for testing
  chat         Interactive mode for custom patent abstracts
  batch        Process multiple patent abstracts from file
  ingest       Data ingestion and indexing only

For more information, visit: https://github.com/DarynBang/Patent2ProductRAG
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['test', 'chat', 'ingest', 'batch'],
        required=True,
        help='Operation mode'
    )
    
    parser.add_argument(
        '--input', 
        type=str,
        help='Input file for batch mode'
    )
    
    parser.add_argument(
        '--output', 
        type=str,
        help='Output file for results'
    )
    
    parser.add_argument(
        '--top-k', 
        type=int, 
        default=5,
        help='Number of top results to retrieve (default: 5)'
    )
    
    parser.add_argument(
        '--planning', 
        action='store_true',
        help='Enable query planning optimization'
    )
    
    parser.add_argument(
        '--market-analysis', 
        action='store_true',
        help='Enable market analysis display'
    )
    
    parser.add_argument(
        '--force-reindex', 
        action='store_true',
        help='Force reindexing of data'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--export-only',
        action='store_true',
        help='Export results without displaying them (useful for batch processing)'
    )
    
    return parser

def validate_arguments(args):
    """
    Validate command-line arguments and check for required combinations.
    
    Args:
        args: Parsed arguments from argparse
    
    Returns:
        bool: True if arguments are valid, False otherwise
    """
    if args.mode == 'batch' and not args.input:
        print("❌ Batch mode requires --input file")
        return False
    
    if args.input and args.mode != 'batch':
        print("⚠️ --input is only used with batch mode")
    
    if args.force_reindex and args.mode != 'ingest':
        print("⚠️ --force-reindex is only used with ingest mode")
    
    return True

def get_user_input_with_commands(prompt="Enter your input"):
    """
    Get user input with support for special commands.
    
    Args:
        prompt (str): Input prompt message
    
    Returns:
        tuple: (user_input, command_type) where command_type is 'quit', 'help', or 'input'
    """
    try:
        user_input = input(f"{prompt}: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            return user_input, 'quit'
        elif user_input.lower() == 'help':
            return user_input, 'help'
        elif not user_input:
            return user_input, 'empty'
        else:
            return user_input, 'input'
            
    except KeyboardInterrupt:
        return "", 'quit'
    except EOFError:
        return "", 'quit'

def format_config_display(args):
    """
    Format command-line arguments for display.
    
    Args:
        args: Parsed arguments from argparse
    
    Returns:
        dict: Formatted configuration dictionary
    """
    config = {
        'Mode': args.mode,
        'Top-K Results': args.top_k,
        'Query Planning': 'Enabled' if args.planning else 'Disabled',
        'Market Analysis': 'Enabled' if args.market_analysis else 'Disabled',
    }
    
    if hasattr(args, 'input') and args.input:
        config['Input File'] = args.input
    
    if hasattr(args, 'output') and args.output:
        config['Output File'] = args.output
    
    if hasattr(args, 'force_reindex') and args.force_reindex:
        config['Force Reindex'] = 'Yes'
    
    return config
