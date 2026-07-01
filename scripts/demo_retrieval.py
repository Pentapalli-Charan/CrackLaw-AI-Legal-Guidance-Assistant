import os
import json
import logging
from src.config import Config
from src.retrieval.search_types import SearchMode
from src.retrieval.retrieval_service import RetrievalService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CrackLaw.Retrieval.Demo")

def main():
    config = Config()
    
    # 1. Create a dummy chunk file to simulate parsing output
    os.makedirs(config.chunks_dir, exist_ok=True)
    doc_id = "demo_environmental_act"
    chunk_file = os.path.join(config.chunks_dir, f"{doc_id}_chunks.json")
    
    dummy_chunks = [
        {
            "chunk_id": "env_chunk_1",
            "text": "THE ENVIRONMENTAL PROTECTION DIRECTIVE 2026. Section 5: Standard emissions. All industrial zones must restrict nitrogen emissions to under 50ppm.",
            "metadata": {
                "act": "Environmental Protection Directive 2026",
                "chapter": "Chapter II",
                "section": "Section 5",
                "language": "en",
                "source": "Ministry of Environment"
            }
        },
        {
            "chunk_id": "env_chunk_2",
            "text": "Section 12: Penalties and enforcement. Violation of the nitrogen emissions limit will result in fines up to 5,000,000 INR per day.",
            "metadata": {
                "act": "Environmental Protection Directive 2026",
                "chapter": "Chapter III",
                "section": "Section 12",
                "language": "en",
                "source": "Ministry of Environment"
            }
        },
        {
            "chunk_id": "contract_chunk_1",
            "text": "NDA Non-Disclosure Agreement. Section 8: Confidentiality term. The receiving party agrees to maintain confidentiality of all proprietary information for a period of 5 years.",
            "metadata": {
                "act": "Confidentiality Agreement",
                "section": "Section 8",
                "language": "en",
                "source": "Corporate Legal Department"
            }
        }
    ]
    
    logger.info("Writing dummy chunks to %s", chunk_file)
    with open(chunk_file, "w", encoding="utf-8") as f:
        json.dump(dummy_chunks, f, indent=2)
        
    # 2. Instantiate Retrieval Service
    logger.info("Initializing RetrievalService...")
    service = RetrievalService(config)
    
    # 3. Index Document
    logger.info("Indexing document chunks into the VectorStore...")
    service.index_document(doc_id)
    
    # 4. Perform Retrieval Queries
    queries = [
        ("nitrogen emissions in industrial zones", SearchMode.SEMANTIC),
        ("confidentiality term nda agreement", SearchMode.KEYWORD),
        ("nitrogen violation penalties", SearchMode.HYBRID)
    ]
    
    print("\n" + "="*50)
    print("           CRACKLAW SEARCH RESULTS DEMO           ")
    print("="*50)
    
    for query_text, search_mode in queries:
        print(f"\nQuery:  '{query_text}'")
        print(f"Mode:   {search_mode.value.upper()}")
        print("-" * 50)
        
        response = service.retrieve(
            query=query_text,
            mode=search_mode,
            top_k=2,
            min_similarity=0.1
        )
        
        print(f"Latency: {response.latency_ms:.2f} ms")
        print(f"Matching Chunks: {len(response.results)}")
        for idx, result in enumerate(response.results):
            print(f"  [{idx + 1}] Chunk ID: {result.chunk_id} | Score: {result.score:.4f}")
            print(f"      Citation: {result.citation}")
            print(f"      Text:     {result.text[:120]}...")
            
        print("\nAssembled Prompt Context:")
        print(response.context)
        print("=" * 50)

    # Cleanup mock files
    try:
        os.remove(chunk_file)
        logger.info("Cleaned up dummy chunks file.")
    except Exception:
        pass

if __name__ == "__main__":
    main()
