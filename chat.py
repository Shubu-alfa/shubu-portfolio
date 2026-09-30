import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

load_dotenv()

# Local Flan-Alpaca model (HF serverless no longer supports this model family)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "flan-alpaca-base")

bot_name = "Shubham"
_tokenizer = None
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model():
    global _tokenizer, _model
    if _model is not None:
        return

    print(f"Loading local model from {MODEL_PATH} on {_device}...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    _model = _model.to(_device)
    _model.eval()
    print("Model loaded.")


def get_response(msg):
    print(msg)
    try:
        _load_model()
    except Exception as exc:
        print(exc)
        return "Sorry, the local chatbot model failed to load."

    input_text = f"question: {msg}"
    features = _tokenizer([input_text], return_tensors="pt")
    input_ids = features["input_ids"].to(_device)
    attention_mask = features["attention_mask"].to(_device)

    with torch.no_grad():
        out = _model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
        )

    text = _tokenizer.decode(out[0], skip_special_tokens=True).strip()
    return text or "I do not understand..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break
        print(get_response(sentence))
