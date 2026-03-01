import pytest
from transformers import BertTokenizer, BertForSequenceClassification

from fraud_detection import FinancialFraudTrainer


@pytest.fixture(scope="session")
def trainer():
    t = FinancialFraudTrainer()
    t.tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    t.model = BertForSequenceClassification.from_pretrained(
        "hfl/chinese-roberta-wwm-ext", num_labels=2
    )
    t.model.eval()
    return t
