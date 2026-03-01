import torch
from torch.utils.data import Dataset


class FinancialFraudDataset(Dataset):
    """
    自定義 Dataset 類別，將文本和標籤轉換為 PyTorch 能處理的格式。
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
