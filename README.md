yaml
---
title: IMDB Sentiment Classifier
emoji: 💬
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: 4.24.0
python_version: 3.9
app_file: app.py
pinned: false
license: apache-2.0
---

A simple Gradio app using a DistilBERT model fine-tuned on the IMDB dataset to classify movie reviews as Positive or Negative.
# 🎬 IMDB Sentiment Analysis App  

This is a sentiment analysis app built with Hugging Face Spaces and Gradio.  

[![Try it on Hugging Face](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-blue)](https://yomex139-imdb-sentiment-app.hf.space)

---

## 🚀 Features
- Predicts sentiment (Positive / Negative) from IMDB reviews
- Built with **Gradio** for interactive UI
- Deployed on **Hugging Face Spaces**

## 🛠️ Installation
```bash
git clone https://github.com/Yomex139/imdb-sentiment-app.git
cd imdb-sentiment-app
pip install -r requirements.txt
