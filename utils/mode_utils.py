"""
utils/mode_utils.py

Mode utilities for Patent2ProductRAG system.
Contains all the different operation modes: test, chat, batch, and ingest.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from config.logging_config import get_logger, log_performance
from utils.export_utils import export_results_to_files
from utils.display_utils import display_results, show_help, display_mode_header, display_configuration
from utils.cli_utils import get_user_input_with_commands

logger = get_logger(__name__)

def test_mode(pipeline, top_k=5, planning=False, market_analysis=False):
    """
    Test mode with predefined patent abstract.
    
    Args:
        pipeline: Initialized RAG pipeline
        top_k (int): Number of results to retrieve
        planning (bool): Enable query planning
        market_analysis (bool): Enable market analysis display
    """
    display_mode_header("TEST MODE", "Running with sample patent abstract for testing")
    
    test_abstract = """An apparatus and a method for diagnosis are provided.
The apparatus for diagnosis lesion include: a model generation unit configured to categorize learning data into one or more categories and to generate
one or more categorized diagnostic models based on the categorized learning data, a model selection unit configured to select one or more diagnostic model
for diagnosing a lesion from the categorized diagnostic models, and a diagnosis unit configured to diagnose the lesion based on image data of the lesion
and the selected one or more diagnostic model.
"""

    print(f"📝 Patent Abstract: {test_abstract[:100]}...")
    display_configuration({
        'Top-K Results': top_k,
        'Query Planning': planning,
        'Market Analysis': market_analysis
    })
    print("\n🔄 Processing...")
    
    try:
        logger.info("Starting test mode processing")
        start_time = datetime.now()
        
        results = pipeline.process_query(
            test_abstract, 
            top_k=top_k, 
            planning=planning
        )
        
        end_time = datetime.now()
        log_performance("Test Query Processing", start_time.timestamp(), end_time.timestamp(), 
                       top_k=top_k, planning=planning, query_length=len(test_abstract))
        
        display_results(results, market_analysis)
        
        # Auto-export results
        export_results_to_files(results, test_abstract, "test")
        
    except Exception as e:
        logger.error(f"Error in test mode: {e}", exc_info=True)
        print(f"❌ Test failed: {e}")

def chat_mode(pipeline, top_k=5, planning=False, market_analysis=False):
    """
    Interactive chat mode for custom queries.
    
    Args:
        pipeline: Initialized RAG pipeline
        top_k (int): Number of results to retrieve
        planning (bool): Enable query planning
        market_analysis (bool): Enable market analysis display
    """
    display_mode_header("INTERACTIVE CHAT MODE", 
                       "Enter patent abstracts to find relevant firms and products.\n" +
                       "Commands: 'quit' to exit, 'help' for assistance")
    
    while True:
        try:
            user_input, command_type = get_user_input_with_commands("📝 Enter your patent abstract (or 'quit' to exit)")
            
            if command_type == 'quit':
                print("👋 Goodbye!")
                break
            elif command_type == 'help':
                show_help()
                continue
            elif command_type == 'empty':
                print("⚠️ Please enter a patent abstract or 'quit' to exit.")
                continue
            
            print(f"\n🔄 Processing query (top_k={top_k}, planning={planning})...")
            
            start_time = datetime.now()
            results = pipeline.process_query(
                user_input, 
                top_k=top_k, 
                planning=planning
            )
            end_time = datetime.now()
            log_performance("Chat Query Processing", start_time.timestamp(), end_time.timestamp(), 
                           top_k=top_k, planning=planning, query_length=len(user_input))
            
            display_results(results, market_analysis)
            
            # Auto-export results for each chat query
            export_results_to_files(results, user_input, "chat")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in chat mode: {e}", exc_info=True)
            print(f"❌ Error processing query: {e}")

def batch_mode(pipeline, input_file, output_file=None, top_k=5, planning=False):
    """
    Batch processing mode for multiple patent abstracts.
    
    Args:
        pipeline: Initialized RAG pipeline
        input_file (str): Path to input file with patent abstracts
        output_file (str): Optional output file for results
        top_k (int): Number of results to retrieve
        planning (bool): Enable query planning
    """
    display_mode_header("BATCH PROCESSING MODE", f"Processing multiple abstracts from: {input_file}")
    
    # Default output filename if not provided
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"batch_results_{timestamp}.json"
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            abstracts = [line.strip() for line in f if line.strip()]
        
        print(f"📁 Loaded {len(abstracts)} patent abstracts from {input_file}")
        
        all_results = []
        for i, abstract in enumerate(abstracts, 1):
            print(f"\n🔄 Processing {i}/{len(abstracts)}: {abstract[:50]}...")
            
            try:
                start_time = datetime.now()
                results = pipeline.process_query(abstract, top_k=top_k, planning=planning)
                end_time = datetime.now()
                
                log_performance(f"Batch Query {i}", start_time.timestamp(), end_time.timestamp(), 
                               top_k=top_k, planning=planning, query_length=len(abstract))
                
                all_results.append({
                    "query_id": i,
                    "patent_abstract": abstract,
                    "results": results
                })
                
                # Auto-export individual results
                export_results_to_files(results, abstract, f"batch_{i}")
                
            except Exception as e:
                logger.error(f"Error processing query {i}: {e}")
                all_results.append({
                    "query_id": i,
                    "patent_abstract": abstract,
                    "error": str(e)
                })
        
        # Save batch results
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"✅ Batch results saved to {output_file}")
        
        print(f"✅ Batch processing completed: {len(all_results)} queries processed")
        
    except FileNotFoundError:
        print(f"❌ Input file not found: {input_file}")
    except Exception as e:
        logger.error(f"Error in batch mode: {e}", exc_info=True)
        print(f"❌ Batch processing failed: {e}")

def ingest_mode(pipeline, force_reindex=False):
    """
    Data ingestion mode.
    
    Args:
        pipeline: Initialized RAG pipeline
        force_reindex (bool): Whether to force reindexing
    """
    display_mode_header("DATA INGESTION MODE", "Building search indices and processing firm data")
    
    try:
        print("🔄 Starting data ingestion...")
        start_time = datetime.now()
        
        pipeline.ingest_firm(force_reindex=force_reindex)
        
        end_time = datetime.now()
        log_performance("Data Ingestion", start_time.timestamp(), end_time.timestamp())
        print("✅ Data ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in ingestion: {e}", exc_info=True)
        print(f"❌ Data ingestion failed: {e}")
