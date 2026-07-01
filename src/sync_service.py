import os
import shutil
import logging
import concurrent.futures
from typing import Optional, Dict, Any, List
from src.config import Config
from src.metadata import MetadataManager
from src.registry import KnowledgeRegistry
from src.verifier import KnowledgeVerifier
from src.classification import RuleBasedClassifier
from src.adapters import AdapterFactory
from src.parsers import ParserFactory

logger = logging.getLogger("CrackLaw.SyncService")

class SyncService:
    """Coordinates data acquisition, integrity verification, document classification, and metadata registration."""

    def __init__(
        self,
        config: Optional[Config] = None,
        metadata_manager: Optional[MetadataManager] = None,
        knowledge_registry: Optional[KnowledgeRegistry] = None,
        verifier: Optional[KnowledgeVerifier] = None,
        classifier: Optional[RuleBasedClassifier] = None
    ):
        self.config = config or Config()
        self.metadata_mgr = metadata_manager or MetadataManager(self.config)
        self.knowledge_reg = knowledge_registry or KnowledgeRegistry(self.config)
        self.verifier = verifier or KnowledgeVerifier(self.config, self.metadata_mgr)
        self.classifier = classifier or RuleBasedClassifier(self.config)
        
        self.settings = self.config.get("knowledge_acquisition", {})
        self.retry_count = self.settings.get("retry_count", 3)
        self.parallel = self.settings.get("parallel_downloads", False)

    def check_for_updates(self) -> List[Dict[str, Any]]:
        """Scans pre-configured datasets and returns a list of datasets with pending updates."""
        datasets_config = self.settings.get("datasets", [])
        pending_updates = []
        
        for dataset_cfg in datasets_config:
            name = dataset_cfg.get("name")
            source_type = dataset_cfg.get("source_type")
            if not name or not source_type:
                continue
                
            try:
                adapter = AdapterFactory.get_adapter(source_type)
                has_update = adapter.check_updates(dataset_cfg)
                if has_update:
                    pending_updates.append(dataset_cfg)
                    logger.info("Update detected for dataset: %s (%s)", name, source_type)
            except Exception as e:
                logger.error("Failed to check updates for %s: %s", name, str(e))
                
        return pending_updates

    def sync_dataset(self, dataset_config: Dict[str, Any]) -> bool:
        """Syncs a single dataset config: registers, downloads, verifies, classifies, and registers files."""
        name = dataset_config.get("name")
        source_type = dataset_config.get("source_type")
        if not name or not source_type:
            logger.error("Dataset config missing name or source_type: %s", dataset_config)
            return False

        logger.info("Initializing sync for dataset: %s (Source: %s)", name, source_type)
        
        # 1. Register dataset in knowledge registry
        self.knowledge_reg.register_dataset(dataset_config)
        self.knowledge_reg.update_status(name, download_status="syncing")

        # 2. Setup download staging directory
        downloads_dir = os.path.normpath(os.path.join(self.config.downloads_dir, name))
        os.makedirs(downloads_dir, exist_ok=True)

        try:
            # 3. Download via Adapter
            adapter = AdapterFactory.get_adapter(source_type)
            
            # Retry mechanism
            synced_paths: List[str] = []
            retries = 0
            while retries < self.retry_count:
                try:
                    synced_paths = adapter.sync(dataset_config, downloads_dir)
                    break
                except Exception as e:
                    retries += 1
                    logger.warning("Sync attempt %d failed for %s: %s", retries, name, str(e))
                    if retries >= self.retry_count:
                        raise e
                    # Wait briefly before retry
                    import time
                    time.sleep(1)

            # 4. Verify and Classify each synced file
            successful_imports = 0
            for file_path in synced_paths:
                if not os.path.exists(file_path):
                    logger.error("Adapter reported synced file but it was not found: %s", file_path)
                    continue

                # Run integrity verification
                verify_report = self.verifier.verify_file(file_path)
                
                # Check for duplicates or corruption
                if not verify_report["is_valid"]:
                    if verify_report["duplicate_doc_id"]:
                        logger.warning("Skipping duplicate file: %s (Registered ID: %s)", file_path, verify_report["duplicate_doc_id"])
                        # Delete file from download staging area to avoid pollution
                        try:
                            os.remove(file_path)
                        except OSError:
                            pass
                        continue
                    else:
                        logger.error("Integrity check failed for file %s: %s", file_path, verify_report["errors"])
                        continue

                # Classify document
                # Get text sample to classify (try parsing)
                text_sample = ""
                try:
                    parser = ParserFactory.get_parser(file_path)
                    parsed = parser.parse(file_path)
                    text_sample = parsed.get("text", "")[:5000] # First 5000 chars are sufficient for keywords
                except Exception as e:
                    logger.warning("Could not parse file %s for classification sample: %s. Using filename hints.", file_path, str(e))

                category = self.classifier.classify(text_sample, os.path.basename(file_path))
                
                # Fallback to config default document types if unknown
                if category == "unknown":
                    doc_types = dataset_config.get("document_types", [])
                    category = doc_types[0] if doc_types else "miscellaneous"

                # 5. Move file to target raw directory
                dest_dir = os.path.join(self.config.raw_dir, category)
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.normpath(os.path.join(dest_dir, os.path.basename(file_path)))
                
                # Relocate file
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                shutil.move(file_path, dest_path)
                logger.info("Relocated file to raw database: %s -> %s", file_path, dest_path)

                # Register document in main registry
                self.metadata_mgr.register_document(
                    file_path=dest_path,
                    source=name,
                    doc_type=category,
                    language=dataset_config.get("supported_languages", ["en"])[0]
                )
                successful_imports += 1

            # Update registry status
            status = "completed" if successful_imports > 0 or not synced_paths else "failed"
            self.knowledge_reg.update_status(name, download_status=status, processing_status="raw")
            logger.info("Dataset sync completed for %s. Successfully imported %d documents.", name, successful_imports)
            return status == "completed"

        except Exception as e:
            logger.error("Failed to sync dataset %s: %s", name, str(e), exc_info=True)
            self.knowledge_reg.update_status(name, download_status="failed")
            return False

    def sync_all(self) -> Dict[str, int]:
        """Synchronizes all datasets defined in the system configurations."""
        datasets_config = self.settings.get("datasets", [])
        logger.info("Starting synchronization of %d configured datasets.", len(datasets_config))
        
        results = {"success": 0, "failed": 0}
        
        if self.parallel and len(datasets_config) > 1:
            logger.info("Running parallel synchronizations.")
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(self.sync_dataset, d): d["name"] for d in datasets_config}
                for future in concurrent.futures.as_completed(futures):
                    dataset_name = futures[future]
                    try:
                        success = future.result()
                        if success:
                            results["success"] += 1
                        else:
                            results["failed"] += 1
                    except Exception as e:
                        logger.error("Parallel sync thread threw exception for %s: %s", dataset_name, str(e))
                        results["failed"] += 1
        else:
            logger.info("Running sequential synchronizations.")
            for dataset_cfg in datasets_config:
                success = self.sync_dataset(dataset_cfg)
                if success:
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    
        logger.info("Global synchronization complete: %s", results)
        return results

    def import_local_folder(self, folder_path: str, default_category: str) -> Dict[str, Any]:
        """Imports documents from a local folder, validating, classifying and moving them into raw/."""
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Source folder does not exist: {folder_path}")

        logger.info("Manual import initiated for folder: %s", folder_path)
        
        files_to_import = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                files_to_import.append(os.path.join(root, file))

        import_results = {
            "total_files": len(files_to_import),
            "imported": 0,
            "skipped_duplicates": 0,
            "failed_verification": 0,
            "details": []
        }

        for file_path in files_to_import:
            verify_report = self.verifier.verify_file(file_path)
            
            if not verify_report["is_valid"]:
                if verify_report["duplicate_doc_id"]:
                    import_results["skipped_duplicates"] += 1
                    import_results["details"].append({
                        "file": os.path.basename(file_path),
                        "status": "skipped",
                        "reason": f"Duplicate of ID {verify_report['duplicate_doc_id']}"
                    })
                else:
                    import_results["failed_verification"] += 1
                    import_results["details"].append({
                        "file": os.path.basename(file_path),
                        "status": "failed",
                        "reason": ", ".join(verify_report["errors"])
                    })
                continue

            # Classify
            text_sample = ""
            try:
                parser = ParserFactory.get_parser(file_path)
                parsed = parser.parse(file_path)
                text_sample = parsed.get("text", "")[:5000]
            except Exception:
                pass

            category = self.classifier.classify(text_sample, os.path.basename(file_path))
            if category == "unknown":
                category = default_category

            # Move and register
            dest_dir = os.path.join(self.config.raw_dir, category)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.normpath(os.path.join(dest_dir, os.path.basename(file_path)))
            
            try:
                shutil.copy2(file_path, dest_path)
                self.metadata_mgr.register_document(
                    file_path=dest_path,
                    source="manual_import",
                    doc_type=category
                )
                import_results["imported"] += 1
                import_results["details"].append({
                    "file": os.path.basename(file_path),
                    "status": "imported",
                    "category": category,
                    "destination": dest_path
                })
            except Exception as e:
                import_results["failed_verification"] += 1
                import_results["details"].append({
                    "file": os.path.basename(file_path),
                    "status": "failed",
                    "reason": f"File copy error: {str(e)}"
                })

        logger.info("Manual folder import completed: %s", import_results)
        return import_results
