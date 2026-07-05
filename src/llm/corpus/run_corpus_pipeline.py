import logging
import time
from src.llm.corpus.config import CorpusConfig
from src.llm.corpus.corpus_builder import CorpusBuilder
from src.llm.corpus.corpus_cleaner import CorpusCleaner
from src.llm.corpus.corpus_validator import CorpusValidator
from src.llm.corpus.corpus_statistics import CorpusStatistics
from src.llm.corpus.corpus_exporter import CorpusExporter

# Configure logging for the standalone script
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("CrackLaw.LLM.CorpusPipeline")

def main():
    t_start = time.time()
    logger.info("Initializing Legal Corpus Preparation Pipeline...")
    
    config = CorpusConfig()
    
    # 1. Build
    builder = CorpusBuilder(config)
    logger.info("Reading processed documents and building corpus instances...")
    raw_corpus = builder.build_corpus()
    
    # 2. Clean
    cleaner = CorpusCleaner(config)
    logger.info("Cleaning corpus text (OCR artifacts, pagination, whitespace)...")
    cleaned_corpus = cleaner.process(raw_corpus)
    
    # 3. Validate
    validator = CorpusValidator(config)
    logger.info("Validating corpus (dropping empty, filtering duplicates)...")
    valid_corpus = validator.process(cleaned_corpus)
    
    # 4. Statistics
    stats_engine = CorpusStatistics()
    logger.info("Generating corpus statistics...")
    stats = stats_engine.generate_report(valid_corpus)
    stats_engine.print_report(stats)
    
    # 5. Export
    exporter = CorpusExporter(config)
    logger.info("Exporting finalized corpus to disk...")
    exporter.export_all(valid_corpus, base_filename="cracklaw_corpus")
    
    elapsed = time.time() - t_start
    logger.info(f"Corpus pipeline completed successfully in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
