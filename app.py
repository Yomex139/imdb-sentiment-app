# app.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr

# Set your model repo on the Hub
MODEL_ID = "Yomex139/imdb-sentiment-model"

# make CPU inference a bit more stable on Spaces
torch.set_num_threads(1)

# Load once at startup (cached by HF Spaces)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

LABELS = {0: "Negative 😡", 1: "Positive 😀"}

def predict(text):
    if not text or not text.strip():
        return {"label": "—", "confidence": 0.0}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs))
        conf = float(probs[pred_id])
    return {"label": LABELS[pred_id], "confidence": round(conf, 4)}

def ui_predict(text):
    out = predict(text)
    return f"{out['label']}  (confidence: {out['confidence']:.2f})"

examples = [
    "This movie was absolutely wonderful, I loved every second of it!",
    "The film was terrible, boring and a complete waste of time.",
    "Not the best, not the worst. Just okay.",
]

demo = gr.Interface(
    fn=ui_predict,
    inputs=gr.Textbox(lines=4, label="Enter a movie review"),
    outputs=gr.Textbox(label="Prediction"),
    title="IMDB Sentiment Classifier",
    description="DistilBERT fine-tuned on IMDB. Type a review and get Positive/Negative with confidence.",
    examples=[[e] for e in examples],
)

if __name__ == "__main__":
    demo.launch()