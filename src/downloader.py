import os
import time
import shutil
import logging
from typing import Optional, Dict, Any
import requests
from src.config import Config
from src.metadata import MetadataManager
from src.utils import calculate_checksum

logger = logging.getLogger("CrackLaw.Downloader")

class DownloadManager:
    """Handles downloading datasets from URLs, Hugging Face, and Kaggle with integrity checking and resuming."""

    def __init__(self, config: Optional[Config] = None, metadata_manager: Optional[MetadataManager] = None):
        self.config = config or Config()
        self.metadata_mgr = metadata_manager or MetadataManager(self.config)
        self.settings = self.config.downloader_settings
        self.headers = {"User-Agent": self.settings.get("user-agent", "CrackLawDownloader/1.0")}

    def verify_file_integrity(self, file_path: str, expected_checksum: Optional[str] = None) -> bool:
        """Verifies file integrity. If an expected checksum is provided, validates it."""
        if not os.path.exists(file_path):
            return False
        
        try:
            checksum = calculate_checksum(file_path)
            if expected_checksum:
                return checksum.lower() == expected_checksum.lower()
            return True
        except Exception as e:
            logger.error("Failed to verify integrity for %s: %s", file_path, str(e))
            return False

    def download_from_url(
        self,
        url: str,
        dest_category: str,
        filename: Optional[str] = None,
        overwrite: bool = False
    ) -> str:
        """Downloads a file from a URL to the appropriate raw directory with resuming support."""
        if not filename:
            filename = url.split("/")[-1].split("?")[0]
            if not filename:
                filename = f"download_{int(time.time())}"

        dest_dir = os.path.join(self.config.raw_dir, dest_category)
        os.makedirs(dest_dir, exist_ok=True)
        final_path = os.path.normpath(os.path.join(dest_dir, filename))
        part_path = final_path + ".part"

        logger.info("Starting download from URL: %s", url)

        # Check registry for duplicates by filename or registry check
        # But we do this after downloading or if file exists
        if os.path.exists(final_path) and not overwrite:
            logger.info("File already exists at %s. Skipping download.", final_path)
            # Register it if not already registered
            self.metadata_mgr.register_document(final_path, source=url, doc_type=dest_category)
            return final_path

        # Determine download resume byte position if file exists
        resume_byte = 0
        write_mode = "wb"
        if self.settings.get("auto_resume", True) and os.path.exists(part_path):
            resume_byte = os.path.getsize(part_path)
            write_mode = "ab"
            logger.info("Found partial download. Resuming from byte %d", resume_byte)

        # Set up request headers for range download
        headers = self.headers.copy()
        if resume_byte > 0:
            headers["Range"] = f"bytes={resume_byte}-"

        try:
            verify_ssl = self.settings.get("verify_ssl", True)
            response = requests.get(url, headers=headers, stream=True, timeout=30, verify=verify_ssl)
            
            # Handle range request status code
            if resume_byte > 0 and response.status_code == 416:
                logger.warning("Range not satisfiable. Restarting download.")
                resume_byte = 0
                write_mode = "wb"
                response = requests.get(url, headers=self.headers, stream=True, timeout=30, verify=verify_ssl)
            elif resume_byte > 0 and response.status_code != 206:
                logger.warning("Server does not support resume. Restarting download.")
                resume_byte = 0
                write_mode = "wb"
                response = requests.get(url, headers=self.headers, stream=True, timeout=30, verify=verify_ssl)

            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0)) + resume_byte

            with open(part_path, write_mode) as f:
                downloaded = resume_byte
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
            # Download complete, rename part file
            if os.path.exists(final_path):
                os.remove(final_path)
            os.rename(part_path, final_path)
            logger.info("Download completed and saved to %s", final_path)

            # Check if this file is a duplicate of a previously registered file
            duplicate_doc_id = self.metadata_mgr.check_duplicate(final_path)
            if duplicate_doc_id:
                logger.warning("Downloaded file is a duplicate of registered doc %s. Deleting duplicate file.", duplicate_doc_id)
                os.remove(final_path)
                # Return the existing duplicate file path
                existing_meta = self.metadata_mgr.get_document_metadata(duplicate_doc_id)
                if existing_meta:
                    return existing_meta["file_path"]

            # Register document in metadata
            self.metadata_mgr.register_document(final_path, source=url, doc_type=dest_category)
            return final_path

        except Exception as e:
            logger.error("Download failed for %s: %s", url, str(e))
            raise e

    def download_from_hf(
        self,
        repo_id: str,
        filename: str,
        dest_category: str,
        repo_type: str = "dataset"
    ) -> str:
        """Downloads a dataset file from Hugging Face Hub."""
        logger.info("Starting HF download. Repo: %s, File: %s", repo_id, filename)
        try:
            from huggingface_hub import hf_hub_download
            
            dest_dir = os.path.join(self.config.raw_dir, dest_category)
            os.makedirs(dest_dir, exist_ok=True)
            
            # Downloads to local cache and returns path
            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type=repo_type,
                cache_dir=self.config.cache_dir
            )
            
            final_path = os.path.normpath(os.path.join(dest_dir, filename))
            
            # Copy from cache to target raw directory
            shutil.copy2(cached_path, final_path)
            logger.info("HF Download successful. Saved to %s", final_path)
            
            # Check duplicate
            duplicate_doc_id = self.metadata_mgr.check_duplicate(final_path)
            if duplicate_doc_id:
                logger.warning("HF file is a duplicate of registered doc %s. Deleting duplicate.", duplicate_doc_id)
                os.remove(final_path)
                existing_meta = self.metadata_mgr.get_document_metadata(duplicate_doc_id)
                if existing_meta:
                    return existing_meta["file_path"]

            self.metadata_mgr.register_document(final_path, source=f"hf://{repo_id}/{filename}", doc_type=dest_category)
            return final_path
            
        except ImportError:
            logger.error("huggingface_hub is not installed. Please install it using pip.")
            raise ImportError("huggingface_hub package required for HF downloads.")
        except Exception as e:
            logger.error("HF download failed: %s", str(e))
            raise e

    def download_from_kaggle(
        self,
        dataset_slug: str,
        filename: str,
        dest_category: str
    ) -> str:
        """Downloads a dataset file from Kaggle Hub."""
        logger.info("Starting Kaggle download. Dataset: %s, File: %s", dataset_slug, filename)
        try:
            # kaggle requires authentication config. Ensure we log it
            os.environ["KAGGLE_CONFIG_DIR"] = self.config.cache_dir
            import kaggle
            
            dest_dir = os.path.join(self.config.raw_dir, dest_category)
            os.makedirs(dest_dir, exist_ok=True)
            
            kaggle.api.dataset_download_file(
                dataset=dataset_slug,
                file_name=filename,
                path=dest_dir,
                force=False,
                quiet=False
            )
            
            final_path = os.path.normpath(os.path.join(dest_dir, filename))
            
            # Kaggle api downloads files zipped sometimes, check if zipped or direct
            # If downloaded file is zip, extract it. Otherwise, use direct path.
            # Usually dataset_download_file returns the file directly or handles zip.
            if not os.path.exists(final_path):
                # Check if it was zipped, e.g. filename.zip
                zip_path = final_path + ".zip"
                if os.path.exists(zip_path):
                    logger.info("Kaggle download was zipped. Extracting...")
                    shutil.unpack_archive(zip_path, dest_dir)
                    os.remove(zip_path)
            
            if not os.path.exists(final_path):
                raise FileNotFoundError(f"Could not locate downloaded Kaggle file at: {final_path}")
                
            logger.info("Kaggle Download successful. Saved to %s", final_path)
            
            # Check duplicate
            duplicate_doc_id = self.metadata_mgr.check_duplicate(final_path)
            if duplicate_doc_id:
                logger.warning("Kaggle file is a duplicate of registered doc %s. Deleting duplicate.", duplicate_doc_id)
                os.remove(final_path)
                existing_meta = self.metadata_mgr.get_document_metadata(duplicate_doc_id)
                if existing_meta:
                    return existing_meta["file_path"]

            self.metadata_mgr.register_document(final_path, source=f"kaggle://{dataset_slug}/{filename}", doc_type=dest_category)
            return final_path
            
        except ImportError:
            logger.error("kaggle is not installed. Please install it using pip.")
            raise ImportError("kaggle package required for Kaggle downloads.")
        except Exception as e:
            logger.error("Kaggle download failed: %s. Ensure kaggle.json exists in %s or ~/.kaggle/", str(e), self.config.cache_dir)
            raise e

    def add_manual_file(
        self,
        src_path: str,
        dest_category: str,
        overwrite: bool = False
    ) -> str:
        """Registers a manually added file by copying it to the appropriate raw directory."""
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source file does not exist: {src_path}")
            
        filename = os.path.basename(src_path)
        dest_dir = os.path.join(self.config.raw_dir, dest_category)
        os.makedirs(dest_dir, exist_ok=True)
        final_path = os.path.normpath(os.path.join(dest_dir, filename))

        logger.info("Registering manual file: %s to category: %s", src_path, dest_category)

        if os.path.exists(final_path) and not overwrite:
            logger.info("File already exists at target location %s. Using existing.", final_path)
            self.metadata_mgr.register_document(final_path, source="manual", doc_type=dest_category)
            return final_path

        # Check duplicate before copying
        duplicate_doc_id = self.metadata_mgr.check_duplicate(src_path)
        if duplicate_doc_id:
            logger.warning("Manual file is a duplicate of registered doc %s.", duplicate_doc_id)
            existing_meta = self.metadata_mgr.get_document_metadata(duplicate_doc_id)
            if existing_meta:
                return existing_meta["file_path"]

        # Copy to raw/
        shutil.copy2(src_path, final_path)
        logger.info("Manual file copied and saved to %s", final_path)
        
        self.metadata_mgr.register_document(final_path, source="manual", doc_type=dest_category)
        return final_path
