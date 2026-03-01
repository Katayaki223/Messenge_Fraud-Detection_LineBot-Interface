from fraud_detection import FinancialFraudTrainer


def test_predict_chinese(trainer: FinancialFraudTrainer):
    result = trainer.predict_transaction("您已中獎，請點選連結領取獎金")
    assert "正常訊息" in result or "詐騙訊息" in result or "錯誤" in result


def test_predict_english(trainer: FinancialFraudTrainer):
    result = trainer.predict_transaction("Congratulations! You won a prize.")
    assert "正常訊息" in result or "詐騙訊息" in result or "錯誤" in result


def test_predict_dict_output(trainer: FinancialFraudTrainer):
    """測試 predict() 回傳 dict 格式（供 LINE Bot 使用）。"""
    result = trainer.predict("您已中獎，請點選連結領取獎金")
    assert isinstance(result, dict)
    assert "label" in result
    assert "confidence" in result
    assert "emoji_label" in result
    assert result["label"] in ("scam", "legitimate", "error")
    assert 0.0 <= result["confidence"] <= 1.0
