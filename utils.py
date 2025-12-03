import json
import numpy as np
from sklearn.metrics import f1_score


def compute_f1(jsonl_path, model_name):
    preds, labels = [], []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                preds.append(obj["pred"])
                labels.append(obj["label"])

    f1 = f1_score(labels, preds)

    print("────────────────────────────────────────")
    print("      DynaBench Evaluation Summary      ")
    print("────────────────────────────────────────")
    print(f"• Model:          {model_name}")
    print(f"• File:           {jsonl_path}")
    print(f"• Samples:        {len(labels)}")
    print(f"• F1 Score:       {f1:.4f}")
    print("────────────────────────────────────────")

    return f1


def create_system_prompt_from_policy(policy: str):
    return (
        "You are a helpful assistant. You are provided with a set of rules:\n\n"
        f"{policy}\n"
    )


def numeric_label(label):
    if label == "PASS":
        return 0
    if label == "FAIL":
        return 1
    raise ValueError(f"Unknown label '{label}'")


def classify_sample(x: np.ndarray, W: np.ndarray, mean: np.ndarray, threshold: float):
    x_centered = x - mean
    proj = x_centered @ W
    score = np.linalg.norm(proj)
    return int(score > threshold), score
