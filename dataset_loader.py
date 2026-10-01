"""
dataset_loader.py - Unified dataset loading for LLM security evaluation
Loads both local CSV datasets and HuggingFace datasets, normalizing to a standard format.
"""

import pandas as pd
import os
import logging
from typing import Dict, List, Optional

logger = logging.getLogger("dataset_loader")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
EXISTING_DIR = os.path.join(DATA_DIR, "existing")
EXTERNAL_DIR = os.path.join(DATA_DIR, "external")


class DatasetLoader:
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = data_dir or DATA_DIR
        self.existing_dir = os.path.join(self.data_dir, "existing")
        self.external_dir = os.path.join(self.data_dir, "external")

        os.makedirs(self.existing_dir, exist_ok=True)
        os.makedirs(self.external_dir, exist_ok=True)

        self._registry = self._build_registry()

    def _build_registry(self) -> dict:
        return {
            "command_injection": {
                "type": "local",
                "path": os.path.join(self.existing_dir, "PromptInjection-to-CommandInjection - CommandInjection.csv"),
                "columns": {"text": "Prompt", "label": "Label"},
                "description": "2627 prompt-to-command injection samples (Label: 0=clean, 1=injection)",
                "source": "CommandInjection Dataset",
            },
            "harmful_behaviors": {
                "type": "local",
                "path": os.path.join(self.existing_dir, "harmful_behaviors.csv"),
                "columns": {"text": "goal"},
                "default_label": 1,
                "description": "521 harmful behavior prompts from AdvBench (all attacks)",
                "source": "AdvBench (Harmful Behaviors)",
            },
            "harmful_strings": {
                "type": "local",
                "path": os.path.join(self.existing_dir, "harmful_strings.csv"),
                "columns": {"text": "target"},
                "default_label": 1,
                "description": "575 harmful strings from AdvBench (all attacks)",
                "source": "AdvBench (Harmful Strings)",
            },
            "transfer_behaviors": {
                "type": "local",
                "path": os.path.join(self.existing_dir, "transfer_expriment_behaviors.csv"),
                "columns": {"text": "goals"},
                "default_label": 1,
                "header": 0,
                "description": "Transfer experiment behaviors (all attacks)",
                "source": "Transfer Behaviors",
            },
            "jailbreak_bench": {
                "type": "huggingface",
                "path": "JailbreakBench/JBB-Behaviors",
                "config": "behaviors",
                "columns": {"text": "Behavior"},
                "use_all_splits": True,
                "source_labels": {"harmful": 1, "benign": 0},
                "description": "200 balanced behaviors (100 harmful + 100 benign)",
                "source": "JailbreakBench (JBB-Behaviors)",
            },
            "shieldlm_pi": {
                "type": "huggingface",
                "path": "Abdennebi/shieldlm-prompt-injection",
                "config": "default",
                "columns": {"text": "text", "label": "label_binary"},
                "description": "54K unified prompt injection samples across 8 languages",
                "source": "ShieldLM Prompt Injection",
            },

            "harmbench": {
                "type": "huggingface",
                "path": "walledai/HarmBench",
                "config": "standard",
                "columns": {"text": "prompt"},
                "default_label": 1,
                "description": "200 standardized harmful behaviors",
                "source": "HarmBench",
            },
            "neuralchemy_pi": {
                "type": "huggingface",
                "path": "enieva/Prompt-injection-dataset",
                "config": "core",
                "columns": {"text": "text", "label": "label"},
                "description": "16.9K debiased prompt injection samples (zero data leakage)",
                "source": "NeurAlchemy PI Dataset",
            },
        }

    def list_available(self) -> List[str]:
        available = []
        for name, info in self._registry.items():
            if info["type"] == "local":
                if os.path.exists(info["path"]):
                    available.append(name)
                else:
                    logger.debug(f"Local dataset not found: {info['path']}")
            elif info["type"] == "huggingface":
                available.append(name)
        return available

    def get_info(self, name: str) -> dict:
        info = self._registry.get(name)
        if info:
            return {
                "name": name,
                "source": info.get("source", name),
                "description": info.get("description", ""),
                "type": info["type"],
                "available": name in self.list_available(),
            }
        return {"name": name, "error": f"Unknown dataset: {name}"}

    def list_all_info(self) -> List[dict]:
        return [self.get_info(name) for name in self._registry]

    def load(self, name: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        info = self._registry.get(name)
        if info is None:
            logger.error(f"Unknown dataset: {name}")
            return pd.DataFrame(columns=["text", "label", "source"])

        try:
            if info["type"] == "local":
                df = self._load_local(info)
            elif info["type"] == "huggingface":
                df = self._load_huggingface(info)
            else:
                logger.error(f"Unknown type for {name}: {info['type']}")
                return pd.DataFrame(columns=["text", "label", "source"])

            if sample_size and len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)

            logger.info(f"Loaded '{name}': {len(df)} samples ({df['label'].sum()} attacks, {len(df) - df['label'].sum()} clean)")
            return df

        except Exception as e:
            logger.error(f"Failed to load dataset '{name}': {e}")
            return pd.DataFrame(columns=["text", "label", "source"])

    def _load_local(self, info: dict) -> pd.DataFrame:
        columns = info["columns"]
        text_col = columns.get("text")
        label_col = columns.get("label")

        if text_col == "goals":
            lines = []
            with open(info["path"], "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if line and line != "goals":
                        lines.append(line)
            text = pd.Series(lines, dtype=str)
        else:
            read_kwargs = {}
            if "header" in info:
                read_kwargs["header"] = info["header"]
            else:
                read_kwargs["header"] = 0

            df = pd.read_csv(info["path"], **read_kwargs)

            if isinstance(text_col, int):
                text = df.iloc[:, text_col].astype(str)
            else:
                text = df[text_col].astype(str)

        if label_col is not None:
            df = pd.read_csv(info["path"])
            label = df[label_col].astype(int)
        elif "default_label" in info:
            label = pd.Series([info["default_label"]] * len(text), dtype=int)
        else:
            label = pd.Series([0] * len(text), dtype=int)

        result = pd.DataFrame({
            "text": text,
            "label": label,
            "source": info.get("source", "local"),
        })
        return result

    def _load_huggingface(self, info: dict) -> pd.DataFrame:
        try:
            from datasets import load_dataset as hf_load
        except ImportError:
            logger.error("datasets library not installed. Run: pip install datasets")
            return pd.DataFrame(columns=["text", "label", "source"])

        if info.get("requires_auth"):
            logger.warning(f"Dataset '{info['path']}' requires HuggingFace authentication. Skipping.")
            return pd.DataFrame(columns=["text", "label", "source"])

        load_kwargs = {}
        if "config" in info:
            load_kwargs["name"] = info["config"]

        dataset = hf_load(info["path"], **load_kwargs)
        split_names = list(dataset.keys())

        columns = info["columns"]
        text_col = columns.get("text")
        label_col = columns.get("label")

        if info.get("use_all_splits"):
            source_labels = info.get("source_labels", {})
            frames = []
            for split_name in split_names:
                df_split = dataset[split_name].to_pandas()
                text = (df_split.iloc[:, text_col] if isinstance(text_col, int) else df_split[text_col]).astype(str)
                label_val = source_labels.get(split_name, 0)
                label = pd.Series([label_val] * len(df_split), dtype=int)
                frames.append(pd.DataFrame({"text": text, "label": label, "source": info.get("source", info["path"])}))
            result = pd.concat(frames, ignore_index=True)
        else:
            df = dataset[split_names[0]].to_pandas()

            if isinstance(text_col, int):
                text = df.iloc[:, text_col].astype(str)
            else:
                text = df[text_col].astype(str)

            if label_col is not None:
                label = df[label_col].astype(int)
            elif "label_fn" in info:
                label = info["label_fn"](df).astype(int)
            elif "default_label" in info:
                label = pd.Series([info["default_label"]] * len(df), dtype=int)
            else:
                label = pd.Series([0] * len(df), dtype=int)

            result = pd.DataFrame({
                "text": text,
                "label": label,
                "source": info.get("source", info["path"]),
            })

        return result

    def load_all(self, sample_size: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        result = {}
        for name in self.list_available():
            df = self.load(name, sample_size=sample_size)
            if len(df) > 0:
                result[name] = df
        return result

    def get_training_data(self, names: Optional[List[str]] = None) -> pd.DataFrame:
        if names:
            datasets = {name: self.load(name) for name in names}
        else:
            datasets = self.load_all()

        dfs = [df for df in datasets.values() if len(df) > 0]
        if not dfs:
            logger.warning("No training data available")
            return pd.DataFrame(columns=["text", "label", "source"])
        combined = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined training data: {len(combined)} samples ({combined['label'].sum()} attacks, {len(combined) - combined['label'].sum()} clean)")
        return combined

    def get_benchmark_data(self, names: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        if names:
            return {name: self.load(name) for name in names if name in self._registry}
        return self.load_all()

    def print_summary(self):
        print("=" * 70)
        print("DATASET LOADER SUMMARY")
        print("=" * 70)
        print(f"{'Dataset':<25} {'Samples':<10} {'Attacks':<10} {'Clean':<10} {'Available':<10}")
        print("-" * 70)
        for name in self._registry:
            info = self._registry[name]
            info_display = self.get_info(name)
            available = info_display["available"]
            if available:
                df = self.load(name)
                attacks = df["label"].sum()
                clean = len(df) - attacks
                print(f"{name:<25} {len(df):<10} {attacks:<10} {clean:<10} {'YES':<10}")
            else:
                print(f"{name:<25} {'-':<10} {'-':<10} {'-':<10} {'NO (not found)':<10}")
        print("-" * 70)
