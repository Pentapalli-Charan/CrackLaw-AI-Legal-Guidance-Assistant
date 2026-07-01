import unittest
import os
import shutil
import tempfile
import time
import numpy as np
import torch
import torch.nn as nn
from src.models.exceptions import ModelHubError, ModelNotFoundError
from src.models.preprocessing import clean_text, VocabularyIndexer, NumericalScaler
from src.models.augmentation import DataAugmenter, synonym_replacement
from src.models.feature_engineering import LegalFeatureExtractor
from src.models.dataset_loader import ModelDatasetLoader
from src.models.model_factory import ModelFactory
from src.models.model_registry import ModelRegistry
from src.models.model_cache import ModelCache
from src.models.checkpoint_manager import CheckpointManager
from src.models.experiment_manager import ExperimentManager
from src.models.metrics import MetricCalculator
from src.models.training_engine import ModelTrainingEngine, EarlyStopping
from src.models.evaluation_engine import EvaluationEngine
from src.models.model_hub import ModelHub

class TestModelHub(unittest.TestCase):

    def setUp(self):
        # Create unique sandbox folders to prevent WinError 32 PermissionError during cleanup
        self.temp_dir = tempfile.mkdtemp()
        self.registry_file = os.path.join(self.temp_dir, "test_registry.json")
        self.hub = ModelHub(registry_file=self.registry_file)
        
        # Override paths for sandbox safety
        self.hub.checkpoint_manager.checkpoints_dir = os.path.join(self.temp_dir, "checkpoints")
        os.makedirs(self.hub.checkpoint_manager.checkpoints_dir, exist_ok=True)
        self.hub.experiment_manager.experiments_dir = os.path.join(self.temp_dir, "experiments")
        self.hub.experiment_manager.runs_file = os.path.join(self.hub.experiment_manager.experiments_dir, "runs.json")
        os.makedirs(self.hub.experiment_manager.experiments_dir, exist_ok=True)

    def tearDown(self):
        # Clean cache and registry references
        self.hub.cache.clear()
        
        # Safely shut down and release any lock handles
        try:
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

    def test_text_cleaning_and_preprocessing(self):
        """Tests text cleaning, VocabularyIndexer token mappings, and NumericalScalers."""
        raw_text = "  Please review the NDA, shall  we proceed?! "
        self.assertEqual(clean_text(raw_text), "please review the nda, shall we proceed?!")

        # Fit Vocab indexer
        texts = ["contract shall be signed", "agreement between the parties", "arbitration and governing law clause"]
        indexer = VocabularyIndexer(max_vocab_size=50)
        indexer.fit(texts)
        self.assertTrue(indexer.is_fitted)
        self.assertIn("shall", indexer.vocab)
        self.assertEqual(indexer.vocab["<PAD>"], 0)
        self.assertEqual(indexer.vocab["<UNK>"], 1)

        # Transform to sequence
        seq = indexer.transform(["contract shall sign"], max_len=5)
        self.assertEqual(seq.shape, (1, 5))
        self.assertEqual(seq[0, 0], indexer.vocab["contract"])
        self.assertEqual(seq[0, 1], indexer.vocab["shall"])
        self.assertEqual(seq[0, 2], indexer.vocab["<UNK>"])  # "sign" not in vocab
        self.assertEqual(seq[0, 3], 0)  # PAD
        self.assertEqual(seq[0, 4], 0)  # PAD

        # Scale numerical values
        scaler = NumericalScaler()
        arr = np.array([[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]], dtype=np.float32)
        scaler.fit(arr)
        scaled = scaler.transform(np.array([[20.0, 2.0]], dtype=np.float32))
        self.assertAlmostEqual(scaled[0, 0], 0.0, places=4)
        self.assertAlmostEqual(scaled[0, 1], 0.0, places=4)

    def test_data_augmentation(self):
        """Tests text and tabular data augmentation noise pipelines."""
        original = "The contract shall require a compensation of damages."
        syn_swapped = synonym_replacement(original, probability=1.0)
        self.assertNotEqual(original, syn_swapped)
        self.assertIn("agreement", syn_swapped.lower())  # "contract" -> "agreement"
        self.assertIn("must", syn_swapped.lower())      # "shall" -> "must"

        # Tabular Gaussian noise
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        augmenter = DataAugmenter(num_noise_std=0.01)
        augmented = augmenter.augment_numerical(data)
        self.assertEqual(data.shape, augmented.shape)
        self.assertFalse(np.array_equal(data, augmented))

    def test_legal_feature_extraction(self):
        """Tests keyword counts and structural feature engineering."""
        text = "This Agreement shall hold harmless and indemnify the party in case of breach."
        extractor = LegalFeatureExtractor()
        features = extractor.extract_features_from_text(text)
        
        self.assertEqual(features["count_shall"], 1.0)
        self.assertEqual(features["count_indemnify"], 1.0)
        self.assertEqual(features["count_breach"], 1.0)
        self.assertTrue(features["char_length"] > 0.0)
        self.assertTrue(features["word_count"] > 0.0)

        # Batch transform
        matrix = extractor.transform_to_array([text, "shall breach"])
        self.assertEqual(matrix.shape[0], 2)
        self.assertEqual(matrix.shape[1], len(extractor.get_feature_names()))

    def test_model_factory_architectures(self):
        """Tests model factory instantiation and emulator compilations."""
        # Sklearn Recommendation Engine
        rec_engine = ModelFactory.create_recommendation_engine()
        embeddings = np.array([
            [0.1, 0.2, 0.3], # case_1
            [0.9, 0.8, 0.7], # case_2
            [0.12, 0.18, 0.32] # case_3 (close to case_1)
        ], dtype=np.float32)
        rec_engine.fit(["case_1", "case_2", "case_3"], embeddings)
        
        recs = rec_engine.recommend(["case_1"], top_n=1)
        self.assertEqual(recs, ["case_3"])

        # PyTorch TextCNN
        doc_classifier = ModelFactory.create_doc_classifier(vocab_size=100, num_classes=5)
        inputs = torch.randint(0, 100, (2, 20))
        outputs = doc_classifier(inputs)
        self.assertEqual(outputs.shape, (2, 5))

        # TensorFlow Keras Emulator Regression MLP
        risk_predictor = ModelFactory.create_legal_risk_predictor(num_features=5)
        risk_predictor.compile(optimizer_name="adam", loss_name="mean_squared_error")
        x = np.random.randn(10, 5).astype(np.float32)
        y = np.random.randn(10, 1).astype(np.float32)
        
        history = risk_predictor.fit(x, y, epochs=2, batch_size=4)
        self.assertEqual(len(history["loss"]), 2)
        preds = risk_predictor.predict(x)
        self.assertEqual(preds.shape, (10, 1))

    def test_model_registry_operations(self):
        """Tests model registry logging and version tracking."""
        registry = ModelRegistry(self.registry_file)
        
        # Register a version
        registry.register_model(
            model_name="doc_classifier",
            version="1.0.0",
            framework="pytorch",
            dataset_used="legal_docs_v1",
            metrics={"f1": 0.85, "accuracy": 0.86},
            checkpoint_path=os.path.join(self.temp_dir, "doc_v1.pt"),
            status="active"
        )

        # Register version 2
        registry.register_model(
            model_name="doc_classifier",
            version="2.0.0",
            framework="pytorch",
            dataset_used="legal_docs_v2",
            metrics={"f1": 0.91, "accuracy": 0.92},
            checkpoint_path=os.path.join(self.temp_dir, "doc_v2.pt"),
            status="active"
        )

        active = registry.get_active_model("doc_classifier")
        self.assertIsNotNone(active)
        self.assertEqual(active["version"], "2.0.0")

        # Verify previous model is inactive
        v1_meta = registry.get_model_metadata("doc_classifier", "1.0.0")
        self.assertEqual(v1_meta["status"], "inactive")

    def test_model_cache_lazy_loading(self):
        """Tests memory cache lazy loading and idle eviction."""
        cache = ModelCache()
        
        # Save a mock sklearn model
        sklearn_path = os.path.join(self.temp_dir, "mock_sklearn.pkl")
        clf = ModelFactory.create_baseline_classifier("logistic_regression")
        # Dummy fit so it can be pickled
        x = np.random.randn(5, 2)
        y = np.array([0, 1, 0, 1, 0])
        clf.fit(x, y)
        with open(sklearn_path, "wb") as f:
            import pickle
            pickle.dump(clf, f)

        # Load into cache
        model1 = cache.get_model("baseline_classifier", "sklearn", sklearn_path)
        self.assertIsNotNone(model1)
        self.assertIn("baseline_classifier", cache.cache)

        # Cache hit
        model2 = cache.get_model("baseline_classifier", "sklearn", sklearn_path)
        self.assertIs(model1, model2)

        # Idle eviction
        # Override access time to simulate 10 seconds ago
        cache.cache["baseline_classifier"]["last_accessed"] = time.time() - 10.0
        evicted = cache.unload_idle_models(max_idle_seconds=5.0)
        self.assertEqual(evicted, 1)
        self.assertNotIn("baseline_classifier", cache.cache)

    def test_training_and_checkpoints(self):
        """Tests early stopping behavior, checkpoint saving, and automatic resume scans."""
        # Early stopping test
        es = EarlyStopping(patience=2, min_delta=0.01)
        self.assertFalse(es(0.50)) # epoch 1
        self.assertFalse(es(0.48)) # epoch 2 (improved by > 0.01)
        self.assertFalse(es(0.48)) # epoch 3 (no change, counter = 1)
        self.assertTrue(es(0.48))  # epoch 4 (no change, counter = 2 -> stops!)

        # Checkpoint resume scan
        checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        cm = CheckpointManager(checkpoints_dir=checkpoint_dir)
        
        # Save epoch 2 and epoch 5 checkpoints
        model = ModelFactory.create_ner_extractor(vocab_size=10, num_tags=3)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        cm.save_checkpoint("ner_extractor", "pytorch", model, opt, 2, {"loss": 0.1}, {})
        cm.save_checkpoint("ner_extractor", "pytorch", model, opt, 5, {"loss": 0.05}, {})

        latest = cm.find_latest_checkpoint("ner_extractor")
        self.assertIsNotNone(latest)
        self.assertTrue(latest.endswith("ner_extractor_epoch_5.pt"))

        ckpt = cm.load_checkpoint(latest)
        self.assertEqual(ckpt["epoch"], 5)

    def test_evaluation_engine_metrics_report(self):
        """Tests classification score aggregations and plottable loss coordinates."""
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 2, 1]
        
        metrics = MetricCalculator.calculate_classification_metrics(y_true, y_pred)
        self.assertAlmostEqual(metrics["accuracy"], 0.6667, places=3)
        self.assertIn("confusion_matrix", metrics)

        report = EvaluationEngine.generate_evaluation_report(
            model_name="doc_classifier",
            framework="pytorch",
            metrics=metrics,
            history={"loss": [0.9, 0.6, 0.4], "val_loss": [1.0, 0.7, 0.5]}
        )
        self.assertEqual(report["loss_curves"]["epochs"], [1, 2, 3])
        self.assertEqual(report["loss_curves"]["train_loss"], [0.9, 0.6, 0.4])

        md = EvaluationEngine.print_markdown_summary(report)
        self.assertIn("# Model Evaluation Summary: `doc_classifier`", md)
        self.assertIn("Framework**: pytorch", md)

    def test_end_to_end_model_hub_inference(self):
        """Tests E2E pipeline registry fittings and risk/classification inferences."""
        # 1. Setup mock datasets and fit preprocessors
        texts = ["shall violate contract terms", "agreement breach risk indemnity", "this is an NDA representation"]
        self.hub.vocab.fit(texts)
        
        features = np.random.randn(10, len(self.hub.extractor.get_feature_names()))
        self.hub.scaler.fit(features)

        # 2. Register mock active models
        # A. Intent Classifier
        intent_path = os.path.join(self.temp_dir, "intent_classifier.pt")
        intent_model = ModelFactory.create_intent_classifier(vocab_size=len(self.hub.vocab.vocab), num_classes=8)
        intent_model.save(intent_path)
        self.hub.registry.register_model(
            model_name="intent_classifier",
            version="1.0.0",
            framework="tensorflow",
            dataset_used="legal_dataset_v1",
            metrics={"accuracy": 0.9},
            checkpoint_path=intent_path,
            status="active"
        )

        # B. NER model
        ner_path = os.path.join(self.temp_dir, "ner_extractor.pt")
        ner_model = ModelFactory.create_ner_extractor(vocab_size=len(self.hub.vocab.vocab), num_tags=9)
        torch.save(ner_model.state_dict(), ner_path)
        self.hub.registry.register_model(
            model_name="ner_extractor",
            version="1.0.0",
            framework="pytorch",
            dataset_used="legal_ner_v1",
            metrics={"f1": 0.88},
            checkpoint_path=ner_path,
            status="active"
        )

        # C. Recommendation Engine
        rec_path = os.path.join(self.temp_dir, "recommendation_engine.pkl")
        rec_model = ModelFactory.create_recommendation_engine()
        rec_model.fit(["case_A", "case_B", "case_C"], np.random.randn(3, 4))
        with open(rec_path, "wb") as f:
            import pickle
            pickle.dump(rec_model, f)
        self.hub.registry.register_model(
            model_name="recommendation_engine",
            version="1.0.0",
            framework="sklearn",
            dataset_used="case_index_v1",
            metrics={"coverage": 1.0},
            checkpoint_path=rec_path,
            status="active"
        )

        # D. Legal Risk MLP
        risk_path = os.path.join(self.temp_dir, "legal_risk_predictor.pt")
        risk_model = ModelFactory.create_legal_risk_predictor(num_features=len(self.hub.extractor.get_feature_names()))
        risk_model.save(risk_path)
        self.hub.registry.register_model(
            model_name="legal_risk_predictor",
            version="1.0.0",
            framework="tensorflow",
            dataset_used="contracts_v1",
            metrics={"mse": 0.02},
            checkpoint_path=risk_path,
            status="active"
        )

        # 3. Test unified API inference returns
        intent = self.hub.classify("Please review the NDA indemnity breach", task="intent")
        self.assertIsInstance(intent, str)

        entities = self.hub.extract_entities("breach of agreement contract Section 73")
        self.assertIsInstance(entities, list)

        recommendations = self.hub.recommend(user_id="user_1", user_history=["case_A"])
        self.assertIsInstance(recommendations, list)

        risk = self.hub.calculate_risk("The signer shall indemnify all breach liability and damages.")
        self.assertIn("risk_score", risk)
        self.assertIn("risk_level", risk)
        self.assertIn("engineered_features", risk)
        self.assertIsInstance(risk["risk_score"], float)

if __name__ == "__main__":
    unittest.main()
