# 🛡️ Fraud Detection LINE Bot

> 📘 基於 BERT 深度學習的中英文詐騙訊息偵測系統，支援 Gradio 網頁介面與 LINE Bot 聊天機器人。

## 👨‍💻 ccClub Student Project

---

## 🔍 專案簡介

本專案開發一套基於 Python 與深度學習語言模型的詐騙訊息偵測系統，能夠有效處理繁體中文與英文雙語資料。使用者可透過 **LINE Bot** 直接傳送可疑訊息，系統將自動辨識是否為詐騙內容並回覆判斷結果；也可透過 **Gradio** 網頁介面進行測試。

---

## 📑 計畫書連結

👉 [查看完整計畫書（Google Drive）](https://drive.google.com/file/d/1oROgam9Gi4sWE_0txIXegUWiMfho0Kko/view?usp=drive_link)

---

## 🧠 使用技術與模型

- `Python 3.10+`
- `PyTorch` 深度學習框架
- `Transformers` 套件（Hugging Face）
- `Flask` + `LINE Bot SDK v3` 聊天機器人
- `Gradio` 網頁測試介面
- `scikit-learn` 資料集分切、精準度分數
- 模型基底：[`hfl/chinese-roberta-wwm-ext`](https://huggingface.co/hfl/chinese-roberta-wwm-ext)

---

## 🗂️ 專案結構

```
.
├── .github/workflows/python-ci.yml   # GitHub Actions CI 流程
├── fraud_detection/                   # 主套件
│   ├── __init__.py                    # 套件匯出
│   ├── dataset.py                     # PyTorch Dataset 類別
│   ├── trainer.py                     # 核心 ML 訓練/推論邏輯
│   ├── gradio_app.py                  # Gradio 網頁介面入口
│   └── linebot_app.py                # Flask + LINE Bot webhook 入口
├── tests/                             # 測試
│   ├── conftest.py                    # pytest 共用 fixture
│   ├── test_fraud_predict.py          # 推論功能測試
│   └── test_linebot_app.py           # LINE Bot endpoint 測試
├── data/
│   └── fraud_detection_sample.csv     # 訓練資料集
├── .env.example                       # 環境變數範本
├── .gitignore
├── README.md
├── requirements.txt                   # 套件需求
└── setup.py                           # 安裝設定
```

---

## 🛠️ 環境建置

### 建議 Python 版本

Python 3.10+

### 安裝依賴

```bash
# 建立虛擬環境
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# 方法一：透過 setup.py 安裝（推薦）
pip install .

# 方法二：使用 requirements.txt
pip install -r requirements.txt
```

### LINE Bot 設定

1. 複製環境變數範本：
   ```bash
   cp .env.example .env
   ```

2. 在 `.env` 中填入你的 LINE Bot 憑證：
   ```
   LINE_CHANNEL_SECRET=your_channel_secret_here
   LINE_CHANNEL_ACCESS_TOKEN=your_channel_access_token_here
   ```

3. 到 [LINE Developers Console](https://developers.line.biz/) 設定 Webhook URL 為 `https://your-domain/callback`

---

## 🚀 啟動方式

### LINE Bot（開發模式）

```bash
python -m fraud_detection.linebot_app
```

### LINE Bot（正式部署）

```bash
gunicorn -w 1 -b 0.0.0.0:5000 fraud_detection.linebot_app:app
```

### Gradio 網頁介面

```bash
python -m fraud_detection.gradio_app
```

---

## 🧪 測試

```bash
pytest tests/ -v
```

---

## 📚 Reference 資源與致謝

> 本專案靈感與部分程式碼來自以下開源項目與模型，特此致謝：

- 📌 **模型：**
  - [`hfl/chinese-roberta-wwm-ext`](https://huggingface.co/hfl/chinese-roberta-wwm-ext) – 哈工大與科大訊飛聯合實驗室發布的全詞遮罩中文語言模型
  - [GitHub 原始碼](https://github.com/ymcui/Chinese-BERT-wwm) – 由 Yiming Cui 團隊維護的 BERT-WWM 中文模型系列
  - [iFLYTEK HFL-Anthology 模型總覽](https://github.com/iflytek/HFL-Anthology?tab=readme-ov-file#Pre-trained-Language-Model)

- 💻 **程式碼參考與基礎模板：**
  - [alamin19/fraud-detection-bert](https://github.com/alamin19/fraud-detection-bert) – 英文詐騙訊息分類的 BERT 模型範例

- 📘 **文件與教學：**
  - [Gradio 官方文檔](https://www.gradio.app/guides)
  - [Hugging Face Transformers Docs](https://huggingface.co/docs/transformers/index)
  - [LINE Messaging API](https://developers.line.biz/en/docs/messaging-api/)
