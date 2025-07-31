"""
main.py

Command-line interface for the Patent2ProductRAG System

This module provides a clean entry point for the Patent to Product RAG system,
supporting different modes of operation including testing, interactive chat, and 
batch processing.

Modes:
- test: Run with a predefined patent abstract for testing
- chat: Interactive mode for querying with custom patent abstracts
- ingest: Data ingestion and indexing only
- batch: Process multiple patent abstracts from a file

Usage Examples:
    python main.py --mode test
    python main.py --mode chat
    python main.py --mode ingest --force-reindex
    python main.py --mode batch --input patents.txt

Dependencies:
    - InternshipRAG_pipeline: Core RAG functionality
    - utils modules: Mode operations, CLI, display, and export utilities
    - config modules: Configuration and logging setup
"""

import logging
import sys
from datetime import datetime
from InternshipRAG_pipeline import InternshipRAG_Pipeline
from config.rag_config import firm_config
from config.agent_config import agent_config
from config.logging_config import setup_logging, get_logger, log_system_info, log_performance

# Import utility functions
from utils.cli_utils import create_argument_parser, validate_arguments, format_config_display
from utils.display_utils import display_configuration
from utils.mode_utils import test_mode, chat_mode, batch_mode, ingest_mode

# Set up centralized logging
setup_logging(
    log_level=logging.INFO,
    log_dir="logs",
    console_output=True
)
logger = get_logger(__name__)

def main():
    """Main entry point for the Patent2ProductRAG CLI."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not validate_arguments(args):
        sys.exit(1)
    
    # Log system information
    log_system_info()
    logger.info(f"Starting Patent2ProductRAG in {args.mode} mode")
    
    # Display configuration
    config = format_config_display(args)
    display_configuration(config)
    
    # Initialize pipeline
    try:
        logger.info("Initializing Patent2ProductRAG Pipeline...")
        start_time = datetime.now()
        
        pipeline = InternshipRAG_Pipeline(
            index_dir="RAG_INDEX",
            agent_config=agent_config,
            firm_config=firm_config,
            ingest_only=(args.mode == 'ingest')
        )
        
        end_time = datetime.now()
        log_performance("Pipeline Initialization", start_time.timestamp(), end_time.timestamp())
        logger.info("Pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
        print(f"❌ Pipeline initialization failed: {e}")
        sys.exit(1)
    
    # Run based on mode
    try:
        if args.mode == 'test':
            test_mode(pipeline, args.top_k, args.planning, args.market_analysis)
        elif args.mode == 'chat':
            chat_mode(pipeline, args.top_k, args.planning, args.market_analysis)
        elif args.mode == 'ingest':
            ingest_mode(pipeline, args.force_reindex)
        elif args.mode == 'batch':
            batch_mode(pipeline, args.input, args.output, args.top_k, args.planning)
    
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled by user")
    except Exception as e:
        logger.error(f"Error in {args.mode} mode: {e}", exc_info=True)
        print(f"❌ {args.mode.title()} mode failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
