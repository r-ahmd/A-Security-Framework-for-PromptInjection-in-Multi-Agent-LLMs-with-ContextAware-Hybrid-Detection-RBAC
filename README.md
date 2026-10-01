# A Unified Security Framework against Prompt Injections in Multi-Agent LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Springer Nature](https://img.shields.io/badge/Paper-Springer%20Cybersecurity-red.svg)](#citation)

Official implementation of the paper:  
**"A Unified Security Framework against Prompt Injections in Multi-Agent LLMs: Integrating Context-Aware Hybrid Detection and Role-Based Access Control"**  
*Under review at Cybersecurity (Springer Nature, SCIE Q1).*

---

## 📌 Overview

Multi-Agent Systems (MAS) powered by Large Language Models (LLMs) allow autonomous agents to execute shell commands, read/write files, and query databases. However, because LLMs process instructions and untrusted user data in the same natural-language stream, adversarial prompt injections can hijack agent workflows and cause privilege escalation.

This repository provides an enterprise-ready, **Two-Gate Defense-in-Depth** framework:
- **Gate 1 (Threat Detection Layer):** 4-tier detection cascade:
  1. *Pattern Heuristic* (0.20 ms CPU) — 20+ regexes and 40+ synonym mappings.
  2. *Statistical ML* (1.48 ms CPU) — Sublinear TF-IDF + Multinomial Naive Bayes.
  3. *Cascaded Hybrid* (0.35 ms avg CPU) — Fast-path fallback filter.
  4. *Fine-tuned DistilBERT* (32.40 ms CPU) — Contextual transformer achieving **92.0% accuracy** across 13,700 held-out samples.
  5. *Stateful ContextDetector ($k=3$)* — Sliding window yielding **+2.9% accuracy gain** on multi-turn distributed attacks.
- **Gate 2 (Role-Based Access Control):** Enforces least privilege across 4 agent roles (`orchestrator`, `worker`, `monitor`, `guest`) and 11 granular action types.
- **Local On-Premise Execution:** Containerized FastAPI REST service connected to local **Llama 3** via Ollama (`localhost:11434`), delivering a **100.2× latency speedup** over commercial cloud firewalls (e.g., Lakera Guard) with zero cloud data transmission.

---

## 📂 Repository Structure

```
prompt-injection-defense/
├── app.py                      # FastAPI REST microservice (/chat, /detect, /authorize)
├── dataset_loader.py           # Unified loader for all 8 benchmark corpora (46,813 samples)
├── evaluate.py                 # Evaluation suite to reproduce Table 3, Table 4, and Table 7
├── requirements.txt            # Python dependencies
├── detectors/
│   ├── pattern_detector.py     # Heuristic regex & weighted keyword scanner
│   ├── ml_classifier.py        # TF-IDF + Multinomial Naive Bayes classifier
│   ├── distilbert_detector.py  # Fine-tuned DistilBERT inference wrapper
│   └── hybrid_cascade.py       # Two-stage fast-exit cascade
├── context/
│   └── context_detector.py     # Stateful sliding-window history buffer (k=3)
├── rbac/
│   └── rbac_engine.py          # 4-role / 11-action policy enforcement engine
├── agents/
│   ├── orchestrator.py         # Multi-agent coordinator
│   ├── worker_agent.py         # Computational & tool execution agent
│   ├── monitor_agent.py        # Auditing & memory inspection agent
│   └── guest_agent.py          # Read-only public interface agent
└── tests/
    ├── test_detectors.py       # Unit tests for Gate 1
    ├── test_rbac.py            # Unit tests for Gate 2
    └── test_integration.py     # 36 automated unit & pipeline integration tests
```

---

## 📊 Benchmark Datasets (46,813 Total Samples)

All datasets are normalized to the schema `(text, label, source)` where `label ∈ {0 (SAFE), 1 (UNSAFE)}`:

| Dataset Name | Source / Provider | Total Samples | Attack Samples | Characteristics |
|---|---|---|---|---|
| **Command Injection** | Local Benchmark | 2,627 | 1,275 | Shell injection, Base64 payloads |
| **Harmful Behaviors** | AdvBench | 520 | 520 | Direct malicious directives |
| **Harmful Strings** | AdvBench | 574 | 574 | Target trigger tokens |
| **Transfer Behaviors** | AdvBench | 388 | 388 | Cross-model transferable exploits |
| **JailbreakBench** | HuggingFace Hub | 200 | 100 | Equal-split class-imbalance probe |
| **ShieldLM PI** | HuggingFace Hub | 37,913 | 13,275 | Multilingual attacks (8 languages) |
| **NeurAlchemy PI** | HuggingFace Hub | 4,391 | 2,650 | Debiased real/synthetic injections |
| **HarmBench** | HuggingFace Hub | 200 | 200 | Automated red-teaming benchmarks |

---

## ⚡ Quickstart

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running locally:
  ```bash
  ollama run llama3
  ```

### 2. Installation
```bash
git clone https://github.com/tanvirahmd/prompt-injection-defense.git
cd prompt-injection-defense
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Production API
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
Access the interactive Swagger UI at: `http://localhost:8000/docs`

---

## 🔬 Reproducing Paper Results

To evaluate all 4 detector tiers across the held-out test split (13,700 samples):
```bash
python evaluate.py --all-detectors --datasets all
```

To run the full unit and integration test suite (36 tests):
```bash
pytest tests/ -v
```

---

## 📝 Citation

If you use this codebase or framework in your research, please cite our manuscript:

```bibtex
@article{ahmed2026unified,
  title={A Unified Security Framework against Prompt Injections in Multi-Agent LLMs: Integrating Context-Aware Hybrid Detection and Role-Based Access Control},
  author={Ahmed, Mohammad Tanvir and Hasan, Samiul and Biplob, Md. Badiuzzaman},
  journal={Cybersecurity},
  year={2026},
  publisher={Springer Nature}
}
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
