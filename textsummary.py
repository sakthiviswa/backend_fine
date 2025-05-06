from flask import Flask, request, jsonify
from transformers import pipeline
import torch
import os

app = Flask(__name__)

# Set device
device = 0 if torch.cuda.is_available() else -1

# Load a faster summarization model
# PEGASUS-XSum is optimized for shorter summaries and faster inference
text_summary = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=device
)

# Warm-up the model with a dummy run to reduce first inference delay
text_summary("This is a warm-up request to load the model and reduce latency.")

@app.route("/")
def home():
    return "Welcome to the Optimized Text Summarization API!"

@app.route("/summarize", methods=["POST"])
def summarize_text():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    input_text = data["text"]

    # Limit input length to 512 words for performance
    if len(input_text.split()) > 512:
        return jsonify({"error": "Input text too long (maximum 512 words allowed)."}), 400

    try:
        output = text_summary(input_text)
        summary = output[0]["summary_text"]
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": f"Error generating summary: {str(e)}"}), 500

if __name__ == "__main__":
    # For development/testing only. Use gunicorn in production.
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
