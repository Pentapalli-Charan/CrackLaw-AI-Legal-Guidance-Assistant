import os
import sys
import shutil
import tempfile
import time
import numpy as np
import torch

# Adjust paths to import src modules properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.model_hub import ModelHub
from src.models.model_factory import ModelFactory
from src.models.preprocessing import clean_text
from src.models.evaluation_engine import EvaluationEngine

def run_demo():
    print("=" * 70)
    print("           CRACKLAW AI MODEL HUB - DEVELOPER DEMONSTRATION")
    print("=" * 70)

    # 1. Setup sandbox environment for the demonstration runs
    temp_dir = tempfile.mkdtemp()
    registry_path = os.path.join(temp_dir, "demo_model_registry.json")
    
    print(f"[1/8] Initializing ModelHub in sandbox directory: {temp_dir}...")
    hub = ModelHub(registry_file=registry_path)
    # Redirect checkpoint and experiment dirs to sandbox
    hub.checkpoint_manager.checkpoints_dir = os.path.join(temp_dir, "checkpoints")
    os.makedirs(hub.checkpoint_manager.checkpoints_dir, exist_ok=True)
    hub.experiment_manager.experiments_dir = os.path.join(temp_dir, "experiments")
    hub.experiment_manager.runs_file = os.path.join(hub.experiment_manager.experiments_dir, "runs.json")
    os.makedirs(hub.experiment_manager.experiments_dir, exist_ok=True)
    
    # 2. Generate a fake legal text classification dataset
    print("\n[2/8] Generating synthetic legal text dataset...")
    raw_texts = [
        "The defendant shall be sentenced to prison terms under Section 302 of the IPC.",
        "Company agrees to indemnify the client against breach of confidentiality.",
        "This commercial merger and acquisition is regulated by corporate guidelines.",
        "The contract liability caps are restricted under the Indian Contract Act Sec 73.",
        "Environmental pollution and waste limits are controlled by green guidelines.",
        "A breach of warranty allows the purchaser to seek damages and compensation.",
        "The Supreme Court declared the arbitration award invalid due to jurisdiction.",
        "NDA agreements shall bind all signatory parties under governing law rules.",
        "The tax liability calculation depends on corporate earnings and losses.",
        "Any gross negligence violates the mutual liability cap provisions of section 12."
    ]
    # 5 document classes: Criminal (0), Civil (1), Corporate (2), Environmental (3), Tax (4)
    labels = np.array([0, 1, 2, 1, 3, 1, 0, 2, 4, 1], dtype=np.int32)
    
    # 3. Preprocess texts and extract tabular risk features
    print("\n[3/8] Preprocessing and fitting text indexer...")
    hub.vocab.fit(raw_texts)
    indexed_sequences = hub.vocab.transform(raw_texts, max_len=20)
    print(f"Indexed sequence shape: {indexed_sequences.shape}")

    # Split dataset
    x_train, y_train, x_val, y_val, x_test, y_test = hub.dataset_loader.split_dataset(
        indexed_sequences, labels, val_size=0.2, test_size=0.2
    )

    # 4. Instantiate PyTorch Document Classifier
    print("\n[4/8] Instantiating PyTorch TextCNN document classifier...")
    vocab_size = len(hub.vocab.vocab)
    model = ModelFactory.create_doc_classifier(vocab_size=vocab_size, num_classes=5)
    
    # Create PyTorch DataLoaders
    train_loader, val_loader, test_loader = hub.dataset_loader.get_pytorch_dataloaders(
        x_train, y_train, x_val, y_val, x_test, y_test, batch_size=2
    )

    # 5. Train model using the orchestrator training engine
    print("\n[5/8] Commencing model training loop (Early stopping & Checkpointing enabled)...")
    trained_model, metrics = hub.train_pytorch_model(
        model_name="doc_classifier",
        version="1.0.0",
        dataset_name="synthetic_legal_v1",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        hyperparams={
            "epochs": 5,
            "learning_rate": 0.01,
            "early_stopping_patience": 2
        },
        resume=False
    )
    print(f"Training completed. Metrics: F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}")

    # 6. Generate evaluation reports
    print("\n[6/8] Generating evaluation reports and training curve stats...")
    eval_report = EvaluationEngine.generate_evaluation_report(
        model_name="doc_classifier",
        framework="pytorch",
        metrics=metrics,
        history={"loss": metrics.get("train_loss_history", [0.8, 0.6, 0.4]), "val_loss": metrics.get("val_loss_history", [0.9, 0.7, 0.5])}
    )
    md_summary = EvaluationEngine.print_markdown_summary(eval_report)
    print(md_summary)

    # 7. Demonstrate unified inference API (lazy loading from cache)
    print("\n[7/8] Running predictions through Unified Inference API (predict & classify)...")
    # Clean cache first to show lazy loading in action
    hub.cache.clear()
    
    print("Predicting document category (should lazy-load model automatically)...")
    doc_class = hub.classify(
        "The signatory shall indemnify all liability under the governing law of the contract.",
        task="document"
    )
    print(f"Inference input: 'The signatory shall indemnify all liability...' -> Predicted Class: {doc_class}")
    print(f"Models currently in memory cache: {list(hub.cache.cache.keys())}")

    # 8. Demonstrate idle cache unloading
    print("\n[8/8] Testing cache eviction lifecycles (unloading idle models)...")
    # Simulate time passing by back-dating last_accessed timestamp
    hub.cache.cache["doc_classifier"]["last_accessed"] = time.time() - 30.0
    
    print("Scanning cache for models idle for more than 10 seconds...")
    evicted_cnt = hub.unload_idle_models(max_idle_seconds=10.0)
    print(f"Evicted {evicted_cnt} idle models from memory cache.")
    print(f"Models remaining in memory cache: {list(hub.cache.cache.keys())}")

    # Cleanup temp directory
    print("\nCleaning sandbox directories...")
    hub.cache.clear()
    shutil.rmtree(temp_dir)
    print("=" * 70)
    print("             DEMONSTRATION RUN COMPLETE - SUCCESS")
    print("=" * 70)

if __name__ == "__main__":
    run_demo()
