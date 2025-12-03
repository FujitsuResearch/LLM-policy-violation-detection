
# LLM Policy Violation Detection – Whitening Evaluation (DynaBench)

This repository provides a CLI tool to evaluate policy-violation detection performance using whitening-based OOD detection on the **DynaBench** dataset.

🔗 Dataset link:
https://huggingface.co/datasets/montehoover/DynaBench

It supports the following models:

- **Meta-Llama-3.1-8B-Instruct**
- **Qwen2.5-7B-Instruct**

## License

This repository is released under the **CC BY-NC 4.0 license**.  
You may use the code for research and academic purposes only.  
Commercial use is strictly prohibited. Patent rights are fully reserved.

Copyright © 2025  
**Fujitsu Research of Europe (FRE)**  
All rights reserved.

---

# 🚀 1. Environment Setup

### Create and activate virtual environment

```
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# or
.\.venv\Scripts\activate       # Windows
```

### Install dependencies from requirements.txt

```
pip install -r requirements.txt
```

## 🔥 Install PyTorch (Manual Step)

PyTorch must be installed manually according to your system specs.

👉 Official installation guide: https://pytorch.org/get-started/locally/

---

# 🔑 2. Hugging Face Authentication (Required)

This project uses gated Hugging Face models:

- `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`

Steps:

### 1. Create Hugging Face Access Token
https://huggingface.co/settings/tokens  
Create a token with **read** permissions.

### 2. Accept model licenses

```
https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct
https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
```

### 3. Login

```
huggingface-cli login
```

### 4. Verify

```
huggingface-cli whoami
```

---

# 🧪 3. Running the Evaluation Script

### Llama:

```
python main.py --model llama-3.1-8b-instruct
```

### Qwen:

```
python main.py --model Qwen2.5-7B-Instruct
```

### Slice:

```
python main.py --model llama-3.1-8b-instruct --slice 200
```

---

# 📁 4. Project Structure

```
.
├── precomputed_stats/
├── runs/
├── whitening_eval_dynabench.py
├── main.py
├── utils.py
├── requirements.txt
├── README.md
└── .venv/
```
## 📦 Precomputed Statistics Directory (```precomputed_stats/```)
The ```precomputed_stats/``` directory contains all whitening-based statistical artifacts required for evaluation.
These are precomputed offline so that the evaluation script can run without any training or recalculation.

This directory is structured per model and per rule category.
```
precomputed_stats/
│
├── llama-3.1-8b-instruct/
│   ├── age_appropriate/
│   │   ├── means/
│   │   │   └── layer_XX_mean.npy
│   │   ├── White_Matrices/
│   │   │   └── layer_XX_last_token_W_White.npy
│   ├── harmful_content/
│   ├── personal_data/
│   ├── ...
│   (total: 12 categories)
│
└── Qwen2.5-7B-Instruct/
    ├── age_appropriate/
    ├── harmful_content/
    ├── ...
    (same structure as above)
```
Each model directory inside precomputed_stats/ also contains a guards.json file.
This file stores the automatically selected best-performing layer for each policy guard (rule category) based on AUC performance on a calibration set.

## Output Files & Results

Each evaluation run writes a timestamped .jsonl file into the runs/ directory:
```
runs/dynabench_whitening_results_{model_name}_20250324_184512.jsonl
```
### 📄 What’s inside the JSONL file?

Each line corresponds to a single evaluated DynaBench sample:

```
{
  "index": 42,
  "guard": "age_appropriate",
  "similarity": 0.91,
  "pred": 1,
  "threshold": 3.53,
  "score": 5.77,
  "label": 1
}
```

Where:

index – original dataset index

guard – guard category with highest similarity

similarity – cosine similarity to the guard’s mean

pred – predicted policy violation (0=in-policy, 1=out-of-policy)

threshold – guard decision threshold

score – whitened projection norm

label – ground-truth DynaBench label

## 📊 5.  F1 Score Summary


After each evaluation run, the script automatically computes and prints a clean F1 score report:
```
──────────────────────────────────────── 
  DynaBench Evaluation Summary
────────────────────────────────────────
• Model:          llama-3.1-8b-instruct
• File:           runs/dynabench_whitening_20250324_184512.jsonl
• Samples:        543
• F1 Score:       0.743
────────────────────────────────────────
```

You can recompute the F1 score for any saved run using:

```
from utils import compute_f1
compute_f1("runs/your_file.jsonl", "model_name")
```