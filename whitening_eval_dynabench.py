import os
import json
import time
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    compute_f1,
    classify_sample,
    numeric_label,
    create_system_prompt_from_policy,
)


class DynabenchWhiteningEval:
    def __init__(self, model, dataset_slice):
        """
        Initialize evaluator: load dataset, configure device, load model.
        """

        self.model_name = model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"[INFO] Using device: {self.device}")

        # ------------------------------
        # Dataset loading
        # ------------------------------
        print("[INFO] Loading DynaBench dataset (split='test')...")
        self.ds = load_dataset("tomg-group-umd/DynaBench", split="test")

        if dataset_slice is not None:
            print(f"[INFO] Taking first {dataset_slice} samples from dataset")
            self.ds = self.ds.select(range(dataset_slice))

        self.ds = self.ds.add_column(
            "orig_idx",
            list(range(len(self.ds))),
        )
        print(
            f"[INFO] Dataset loaded successfully — total rows: {len(self.ds)}"
        )

        # ------------------------------
        # Model & tokenizer loading
        # ------------------------------
        model_map = {
            "llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
        }

        if self.model_name not in model_map:
            raise ValueError(
                f"[ERROR] Unknown model '{self.model_name}'. "
                f"Options: {list(model_map.keys())}"
            )

        hf_model_id = model_map[self.model_name]
        print(f"[INFO] Loading model & tokenizer: {hf_model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            output_hidden_states=True,
        ).to(self.device)

        self.model.eval()
        print(f"[INFO] Model '{self.model_name}' loaded and moved to device.")

    @torch.no_grad()
    def embed_last_token_for_all_layer(self, system_prompt, transcript):
        """
        Process (system_prompt + transcript) and return last-token embeddings
        for all layers. Handles malformed chat structures gracefully.
        """

        messages = []

        if system_prompt:
            messages.append(
                {"role": "system", "content": system_prompt.strip()}
            )

        for line in transcript.splitlines():
            text = line.strip()
            if not text:
                continue

            low = text.lower()
            if low.startswith("user:"):
                messages.append(
                    {"role": "user", "content": text.split(":", 1)[1].strip()}
                )
            elif low.startswith("agent:"):
                messages.append(
                    {
                        "role": "assistant",
                        "content": text.split(":", 1)[1].strip(),
                    }
                )
            else:
                messages.append({"role": "user", "content": text})

        cleaned = []
        last_role = None
        for msg in messages:
            if msg["role"] == last_role:
                continue
            cleaned.append(msg)
            last_role = msg["role"]

        if cleaned and cleaned[0]["role"] == "assistant":
            cleaned.insert(0, {"role": "user", "content": "Hello"})

        try:
            text = self.tokenizer.apply_chat_template(
                cleaned,
                tokenize=False,
                add_generation_prompt=False,
            )
        # pylint: disable=broad-exception-caught
        except Exception as exc:
            print(f"[WARN] Failed to apply chat template: {exc}")
            return None

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[1:]
        embeddings = [h[0, -1, :].cpu().numpy() for h in hidden_states]

        return np.stack(embeddings)

    def load_guardrails(self, guard_path: str) -> dict:
        """
        Load whitening matrices and mean vectors.
        """

        with open(guard_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        base = os.path.join(os.getcwd(), "precomputed_stats", self.model_name)

        for guard in data:
            layer = data[guard]["layer"]
            mean_path = os.path.join(
                base,
                guard,
                "means",
                f"layer_{layer}_mean.npy",
            )
            W_path = os.path.join(
                base,
                guard,
                "White_Matrices",
                f"layer_{layer}_last_token_W_White.npy",
            )

            data[guard]["mean"] = np.load(mean_path)
            data[guard]["W"] = np.load(W_path)

        return data

    def run_evaluation(self):
        """
        Evaluate the model on every sample in the dataset.
        """

        cwd = os.getcwd()
        guards_path = os.path.join(
            cwd,
            "precomputed_stats",
            self.model_name,
            "guards.json",
        )
        guardrails = self.load_guardrails(guards_path)

        results = []

        for _, row in tqdm(
            enumerate(self.ds),
            total=len(self.ds),
            desc="Evaluating samples",
        ):
            label_raw = row.get("label", None)
            best_sim = -float("inf")
            best_pred = None
            best_score = None
            best_guard = None
            best_threshold = None

            sys_prompt = create_system_prompt_from_policy(
                row.get("policy", "")
            )
            transcript = row.get("transcript", "")

            sample_emb = self.embed_last_token_for_all_layer(
                sys_prompt,
                transcript,
            )
            if sample_emb is None:
                continue

            # Evaluate each guard
            for guard in guardrails:
                layer = max(0, guardrails[guard]["layer"] - 1)
                mean_vec = guardrails[guard]["mean"]

                sim = np.dot(
                    sample_emb[layer],
                    mean_vec,
                ) / (
                    np.linalg.norm(sample_emb[layer])
                    * np.linalg.norm(mean_vec)
                )

                if sim > best_sim:
                    best_sim = sim
                    pred, score = classify_sample(
                        sample_emb[layer],
                        guardrails[guard]["W"],
                        guardrails[guard]["mean"],
                        guardrails[guard]["threshold"],
                    )
                    best_guard = guard
                    best_pred = pred
                    best_score = score
                    best_threshold = guardrails[guard]["threshold"]

            results.append(
                {
                    "index": int(row["orig_idx"]),
                    "guard": best_guard,
                    "similarity": float(best_sim),
                    "pred": int(best_pred),
                    "threshold": float(best_threshold),
                    "score": float(best_score),
                    "label": int(numeric_label(label_raw)),
                }
            )

        runs_path = os.path.join(cwd, "runs")
        os.makedirs(runs_path, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = (
            f"dynabench_whitening_results_"
            f"{self.model_name}_{timestamp}.jsonl"
        )
        out_path = os.path.join(runs_path, filename)

        with open(out_path, "w", encoding="utf-8") as f:
            for row_result in results:
                f.write(json.dumps(row_result) + "\n")

        compute_f1(out_path, self.model_name)
