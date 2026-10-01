"""
ML-based Prompt Injection Classifier
Uses TF-IDF + Naive Bayes for text classification
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_loader import DatasetLoader
from utils.logger import setup_logger

logger = setup_logger("ml_classifier")

class MLPromptClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.classifier = MultinomialNB()
        self.is_trained = False
    
    def train(self, texts, labels):
        """Train the classifier"""
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_trained = True
        logger.info(f"Model trained on {len(texts)} samples")
    
    def predict(self, text):
        """Predict if text is injection"""
        if not self.is_trained:
            return {"safe": True, "score": 0.0, "reason": "Model not trained"}
        
        X = self.vectorizer.transform([text])
        prediction = self.classifier.predict(X)[0]
        probability = self.classifier.predict_proba(X)[0]
        
        score = probability[1] if len(probability) > 1 else 0.0
        safe = prediction == 0
        
        return {"safe": safe, "score": score, "reason": "ML classifier"}
    
    def evaluate(self, texts, labels):
        """Evaluate classifier performance"""
        X = self.vectorizer.transform(texts)
        predictions = self.classifier.predict(X)
        
        accuracy = accuracy_score(labels, predictions)
        report = classification_report(labels, predictions, output_dict=True)
        
        return {
            "accuracy": accuracy,
            "precision": report["1"]["precision"] if "1" in report else 0,
            "recall": report["1"]["recall"] if "1" in report else 0,
            "f1": report["1"]["f1-score"] if "1" in report else 0
        }
    
    def save_model(self, path="ml_classifier.pkl"):
        """Save trained model"""
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer, "classifier": self.classifier}, f)
    
    def load_model(self, path="ml_classifier.pkl"):
        """Load trained model"""
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.classifier = data["classifier"]
            self.is_trained = True

def train_ml_classifier(dataset_names=None):
    """Train ML classifier on datasets via DatasetLoader"""
    print("=" * 60)
    print("TRAINING ML-BASED PROMPT INJECTION CLASSIFIER")
    print("=" * 60)

    loader = DatasetLoader()

    if dataset_names is None:
        dataset_names = [
            "command_injection",
            "harmful_behaviors",
            "harmful_strings",
            "transfer_behaviors",
            "shieldlm_pi",
            "neuralchemy_pi",
        ]

    print(f"\nLoading {len(dataset_names)} dataset(s)...")
    all_data = loader.get_training_data(names=dataset_names)

    if len(all_data) == 0:
        print("No training data available via DatasetLoader. Falling back to original paths.")
        return _train_fallback()

    print(f"\nLoaded {len(all_data)} total samples")
    print(f"  Positive (injection): {all_data['label'].sum()}")
    print(f"  Negative (clean):     {len(all_data) - all_data['label'].sum()}")

    all_texts = all_data["text"].tolist()
    all_labels = all_data["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set:      {len(X_test)} samples")

    print("\nTraining ML classifier...")
    classifier = MLPromptClassifier()
    classifier.train(X_train, y_train)

    print("\nEvaluating on test set...")
    metrics = classifier.evaluate(X_test, y_test)

    print(f"\nML Classifier Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"  F1-Score:  {metrics['f1']:.4f} ({metrics['f1'] * 100:.2f}%)")

    classifier.save_model("ml_classifier.pkl")
    print("\nModel saved to ml_classifier.pkl")

    return classifier, metrics


def _train_fallback():
    """Fallback training using original hardcoded paths"""
    import warnings
    warnings.warn("Using fallback training with original hardcoded paths")

    df_cmd = pd.read_csv("PromptInjection-to-CommandInjection - CommandInjection.csv")
    df_harm = pd.read_csv("harmful_behaviors.csv")
    df_strings = pd.read_csv("harmful_strings.csv")

    texts = df_cmd["Prompt"].tolist() + df_harm["goal"].tolist() + df_strings["target"].tolist()
    labels = df_cmd["Label"].tolist() + [1] * len(df_harm) + [1] * len(df_strings)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    classifier = MLPromptClassifier()
    classifier.train(X_train, y_train)
    metrics = classifier.evaluate(X_test, y_test)

    print(f"\nML Classifier Results (fallback):")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"  F1-Score:  {metrics['f1']:.4f} ({metrics['f1'] * 100:.2f}%)")

    classifier.save_model("ml_classifier.pkl")
    return classifier, metrics

if __name__ == "__main__":
    from utils.logger import setup_logger
    logger = setup_logger("ml_classifier")
    
    classifier, metrics = train_ml_classifier()
