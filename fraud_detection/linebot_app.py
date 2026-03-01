"""Flask + LINE Bot webhook 入口。

啟動方式：
  開發：python -m fraud_detection.linebot_app
  正式：gunicorn -w 1 -b 0.0.0.0:5000 fraud_detection.linebot_app:app
"""

import os

from dotenv import load_dotenv
from flask import Flask, abort, request
from linebot.v3 import WebhookHandler
from linebot.v3.exceptions import InvalidSignatureError
from linebot.v3.messaging import (
    ApiClient,
    Configuration,
    MessagingApi,
    ReplyMessageRequest,
    TextMessage,
)
from linebot.v3.webhooks import MessageEvent, TextMessageContent

from fraud_detection.trainer import FinancialFraudTrainer

load_dotenv()

# ── 全域：初始化模型 ──────────────────────────────────────────────────────
fraud_trainer = FinancialFraudTrainer()


def _init_model():
    """確保模型已訓練並載入。"""
    if not fraud_trainer.model_already_trained():
        print("🚀 開始訓練模型，請稍候...")
        fraud_trainer.prepare_dataset()
        fraud_trainer.tokenize_data()
        fraud_trainer.load_model()
        fraud_trainer.train_model()
        fraud_trainer.save_model()
        print("✅ 訓練完成，模型已儲存。")
    fraud_trainer.load_saved_model()
    print("✅ 模型已載入，LINE Bot 準備就緒。")


# ── Flask App ──────────────────────────────────────────────────────────────
app = Flask(__name__)

configuration = Configuration(
    access_token=os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
)
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET", ""))


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint，確認伺服器是否正常運行。"""
    return {"status": "ok", "model_loaded": fraud_trainer.model is not None}, 200


@app.route("/callback", methods=["POST"])
def callback():
    """LINE Platform 將用戶訊息 POST 到此端點。"""
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"


@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event):
    """
    收到使用者的文字訊息後：
      1. 呼叫 predict() 判斷詐騙機率
      2. 組合回覆文字
      3. 透過 ReplyMessage API 回傳給使用者
    """
    user_text = event.message.text.strip()

    # ── 特殊指令：說明 ──────────────────────────────────────────────────────
    if user_text in ("help", "說明", "使用說明", "/help"):
        reply_text = (
            "💳 詐騙簡訊偵測機器人\n\n"
            "📌 使用方式：\n"
            "直接將可疑的簡訊或訊息貼上傳送給我，\n"
            "我將自動判斷是否為詐騙內容。\n\n"
            "📊 回覆格式：\n"
            "• ✅ 正常訊息 → 低風險\n"
            "• ⚠️ 詐騙訊息 → 請提高警覺！\n\n"
            "⚠️ 本工具僅供參考，請自行判斷。"
        )
    else:
        # ── 呼叫模型推論 ────────────────────────────────────────────────────
        result = fraud_trainer.predict(user_text)

        if result["label"] == "error":
            reply_text = result["emoji_label"]
        elif result["label"] == "scam":
            reply_text = (
                f"🚨 偵測結果：{result['emoji_label']}\n"
                f"信心度：{result['confidence']:.1%}\n\n"
                "⚠️ 此訊息疑似詐騙！\n"
                "請勿點擊任何連結、提供個人資料或進行轉帳。\n"
                "如有疑慮，請撥打 165 反詐騙專線。"
            )
        else:
            reply_text = (
                f"偵測結果：{result['emoji_label']}\n"
                f"信心度：{result['confidence']:.1%}\n\n"
                "目前未偵測到詐騙特徵，但仍請保持謹慎。"
            )

    # ── 回覆 LINE 使用者 ────────────────────────────────────────────────────
    with ApiClient(configuration) as api_client:
        line_bot_api = MessagingApi(api_client)
        line_bot_api.reply_message(
            ReplyMessageRequest(
                reply_token=event.reply_token,
                messages=[TextMessage(text=reply_text)],
            )
        )


if __name__ == "__main__":
    _init_model()
    print("🤖 LINE Bot 伺服器啟動中，監聽 port 5000...")
    app.run(host="0.0.0.0", port=5000, debug=False)
