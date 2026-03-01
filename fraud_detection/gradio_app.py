"""Gradio 介面入口。

啟動方式：python -m fraud_detection.gradio_app
"""

import gradio as gr

from fraud_detection.trainer import FinancialFraudTrainer


def main():
    trainer = FinancialFraudTrainer()

    if not trainer.model_already_trained():
        trainer.prepare_dataset()
        trainer.tokenize_data()
        trainer.load_model()
        trainer.train_model()
        trainer.save_model()

    trainer.load_saved_model()

    gr.Interface(
        fn=trainer.predict_transaction,
        inputs=gr.Textbox(lines=3, placeholder="輸入交易簡訊..."),
        outputs="text",
        title="💳 中英文詐騙簡訊判斷器",
        description="輸入交易相關訊息，判斷是否為詐騙訊息（支援中文與英文）。",
    ).launch(share=False)


if __name__ == "__main__":
    main()
