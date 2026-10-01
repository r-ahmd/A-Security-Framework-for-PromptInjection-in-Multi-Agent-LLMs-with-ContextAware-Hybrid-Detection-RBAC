"""
Transformer-based Prompt Injection Classifier
Uses DistilBERT for sequence classification
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.logger import setup_logger

logger = setup_logger("transformer_classifier")


class PromptDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class TransformerClassifier:
    def __init__(self, model_name="distilbert-base-uncased", max_length=128):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.is_trained = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    def train(
        self,
        texts,
        labels,
        epochs=3,
        batch_size=16,
        learning_rate=2e-5,
        warmup_steps=0,
        max_samples=None,
    ):
        if max_samples and len(texts) > max_samples:
            logger.info(f"Subsampling to {max_samples} samples for training")
            indices = np.random.RandomState(42).choice(
                len(texts), max_samples, replace=False
            )
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]

        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.1, random_state=42, stratify=labels
        )

        logger.info(
            f"Train: {len(X_train)} samples, Val: {len(X_val)} samples"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        ).to(self.device)

        train_dataset = PromptDataset(
            X_train, y_train, self.tokenizer, self.max_length
        )
        val_dataset = PromptDataset(
            X_val, y_val, self.tokenizer, self.max_length
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate
        )

        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        class_weights = self._compute_class_weights(y_train)
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(self.device)
        )

        best_val_acc = 0.0
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_batch = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = loss_fn(outputs.logits, labels_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            val_acc, val_report = self._evaluate_loader(val_loader)
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        self.is_trained = True
        logger.info(
            f"Training complete. Best validation accuracy: {best_val_acc:.4f}"
        )

    def _compute_class_weights(self, labels):
        labels_t = torch.tensor(labels)
        class_counts = torch.bincount(labels_t)
        total = len(labels_t)
        weights = total / (len(class_counts) * class_counts.float())
        return weights

    def _evaluate_loader(self, loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_batch = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                preds = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        report = classification_report(
            all_labels, all_preds, output_dict=True, zero_division=0
        )
        return acc, report

    def predict(self, text):
        if not self.is_trained or self.model is None:
            return {"safe": True, "score": 0.0, "reason": "Model not trained"}

        self.model.eval()
        encodings = self.tokenizer(
            [text],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            probs = torch.softmax(outputs.logits, dim=1)[0]
            prediction = torch.argmax(outputs.logits, dim=1).item()

        score = probs[1].item()
        safe = prediction == 0

        return {"safe": safe, "score": score, "reason": "Transformer (DistilBERT)"}

    def predict_batch(self, texts, batch_size=32):
        if not self.is_trained or self.model is None:
            return [{"safe": True, "score": 0.0, "reason": "Model not trained"} for _ in texts]

        self.model.eval()
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            encodings = self.tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.no_grad():
                input_ids = encodings["input_ids"].to(self.device)
                attention_mask = encodings["attention_mask"].to(self.device)
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                probs = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(outputs.logits, dim=1)

            for j in range(len(batch)):
                results.append({
                    "safe": preds[j].item() == 0,
                    "score": probs[j][1].item(),
                    "reason": "Transformer (DistilBERT)",
                })
        return results

    def evaluate(self, texts, labels):
        if not self.is_trained or self.model is None:
            return {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
            }

        dataset = PromptDataset(
            texts, labels, self.tokenizer, self.max_length
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        acc, report = self._evaluate_loader(loader)

        return {
            "accuracy": acc,
            "precision": report["1"]["precision"] if "1" in report else 0,
            "recall": report["1"]["recall"] if "1" in report else 0,
            "f1": report["1"]["f1-score"] if "1" in report else 0,
        }

    def save_model(self, path="transformer_classifier"):
        if self.model is None:
            logger.error("No model to save")
            return
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        torch.save({"is_trained": self.is_trained}, os.path.join(path, "state.pt"))
        logger.info(f"Model saved to {path}")

    def load_model(self, path="transformer_classifier"):
        if not os.path.exists(path):
            logger.error(f"Model path not found: {path}")
            return
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(
            self.device
        )
        state_path = os.path.join(path, "state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.is_trained = state.get("is_trained", True)
        else:
            self.is_trained = True
        logger.info(f"Model loaded from {path}")


def train_transformer_classifier(dataset_names=None):
    """Train transformer classifier on datasets via DatasetLoader"""
    print("=" * 60)
    print("TRAINING TRANSFORMER-BASED PROMPT INJECTION CLASSIFIER")
    print("=" * 60)

    from dataset_loader import DatasetLoader

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
        print("No training data available.")
        return None, None

    print(f"\nLoaded {len(all_data)} total samples")
    print(f"  Positive (injection): {all_data['label'].sum()}")
    print(f"  Negative (clean):     {len(all_data) - all_data['label'].sum()}")

    all_texts = all_data["text"].tolist()
    all_labels = all_data["label"].tolist()

    print(f"\nInitializing transformer classifier...")
    classifier = TransformerClassifier()

    print(f"\nTraining transformer model...")
    classifier.train(
        all_texts,
        all_labels,
        epochs=1,
        batch_size=32,
        max_samples=1500,
    )

    print("\nEvaluating...")
    _, X_test, _, y_test = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    metrics = classifier.evaluate(X_test, y_test)

    print(f"\nTransformer Classifier Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"  F1-Score:  {metrics['f1']:.4f} ({metrics['f1'] * 100:.2f}%)")

    classifier.save_model("transformer_classifier")
    print("\nModel saved to transformer_classifier/")

    return classifier, metrics


if __name__ == "__main__":
    classifier, metrics = train_transformer_classifier()
