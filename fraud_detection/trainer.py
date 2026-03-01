import pathlib

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    Trainer,
    TrainingArguments,
)

from fraud_detection.dataset import FinancialFraudDataset

_PACKAGE_DIR = pathlib.Path(__file__).parent.resolve()
_DEFAULT_DATA_PATH = _PACKAGE_DIR.parent / "data" / "fraud_detection_sample.csv"
_DEFAULT_MODEL_DIR = pathlib.Path("fraud_bert_model")
_DEFAULT_PRETRAINED = "hfl/chinese-roberta-wwm-ext"


class FinancialFraudTrainer:
    def __init__(self):
        self.data_path = None
        self.train_texts = None
        self.val_texts = None
        self.train_labels = None
        self.val_labels = None
        self.tokenizer = None
        self.train_dataset = None
        self.val_dataset = None
        self.model = None

    # ── 資料前處理 ──────────────────────────────────────────────────────────

    def prepare_dataset(self, file_path=None):
        """讀取 CSV 檔案並分割為訓練集與驗證集。"""
        self.data_path = pathlib.Path(file_path) if file_path else _DEFAULT_DATA_PATH
        df = pd.read_csv(self.data_path, encoding="utf-8")
        (
            self.train_texts,
            self.val_texts,
            self.train_labels,
            self.val_labels,
        ) = train_test_split(
            df["text"].tolist(),
            df["label"].tolist(),
            test_size=0.2,
            random_state=42,
        )

    # ── 文字編碼 ────────────────────────────────────────────────────────────

    def tokenize_data(self, pretrained=_DEFAULT_PRETRAINED):
        """載入 tokenizer 並將訓練/驗證文本編碼。"""
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        train_encodings = self.tokenizer(
            self.train_texts, truncation=True, padding=True, max_length=128
        )
        val_encodings = self.tokenizer(
            self.val_texts, truncation=True, padding=True, max_length=128
        )
        self.train_dataset = FinancialFraudDataset(train_encodings, self.train_labels)
        self.val_dataset = FinancialFraudDataset(val_encodings, self.val_labels)

    # ── 載入預訓練模型 ──────────────────────────────────────────────────────

    def load_model(self, pretrained=_DEFAULT_PRETRAINED):
        """載入中文 RoBERTa 分類模型，分類數為 2（合法/詐騙）。"""
        self.model = BertForSequenceClassification.from_pretrained(
            pretrained, num_labels=2
        )

    # ── 訓練 ────────────────────────────────────────────────────────────────

    def train_model(self, output_dir="./results"):
        """執行模型訓練。"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=20,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            report_to="none",
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()

    def compute_metrics(self, pred):
        """計算 accuracy、precision、recall、F1 分數。"""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    # ── 儲存 / 載入 ─────────────────────────────────────────────────────────

    def save_model(self, model_dir=_DEFAULT_MODEL_DIR):
        """儲存模型與 tokenizer。"""
        model_dir = pathlib.Path(model_dir)
        self.model.save_pretrained(str(model_dir))
        self.tokenizer.save_pretrained(str(model_dir))

    def load_saved_model(self, model_dir=_DEFAULT_MODEL_DIR):
        """重新載入已儲存的模型與 tokenizer，供推論使用。"""
        model_dir = pathlib.Path(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(str(model_dir))
        self.tokenizer = BertTokenizer.from_pretrained(str(model_dir))
        self.model.eval()

    def model_already_trained(self, model_dir=_DEFAULT_MODEL_DIR):
        """檢查模型是否已經訓練過。"""
        model_dir = pathlib.Path(model_dir)
        model_path_bin = model_dir / "pytorch_model.bin"
        model_path_safe = model_dir / "model.safetensors"

        if model_path_bin.exists() or model_path_safe.exists():
            print("✅ Model already trained.")
            return True

        print("❌ Model not trained yet.")
        return False

    # ── 推論（核心：供 LINE Bot 呼叫）───────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        回傳 dict：
          {
            "label":       "scam" | "legitimate" | "error",
            "confidence":  float (0.0 ~ 1.0),
            "emoji_label": str (顯示用文字)
          }
        """
        try:
            self.model.eval()
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probs, dim=1).item()
                confidence = probs[0][prediction].item()

            if prediction == 0:
                return {
                    "label": "legitimate",
                    "confidence": confidence,
                    "emoji_label": "✅ 正常訊息",
                }
            else:
                return {
                    "label": "scam",
                    "confidence": confidence,
                    "emoji_label": "⚠️ 詐騙訊息",
                }
        except Exception as e:
            return {
                "label": "error",
                "confidence": 0.0,
                "emoji_label": f"❌ 發生錯誤：{e}",
            }

    # ── Gradio 用的字串輸出 ─────────────────────────────────────────────────

    def predict_transaction(self, text: str) -> str:
        """Gradio 介面用，回傳純字串。"""
        result = self.predict(text)
        if result["label"] == "error":
            return result["emoji_label"]
        return f"{result['emoji_label']}  (Confidence: {result['confidence']:.2f})"
