from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS  # <-- Import CORS
import os

app = Flask(__name__)
CORS(app)
summarizer = None  # Lazy-loaded to reduce memory use

@app.route("/")
def home():
    return "Welcome to the Lightweight Text Summarizer API!"

@app.route("/summarize", methods=["POST"])
def summarize_text():
    global summarizer

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    input_text = data["text"]

    # Limit input length to 512 words
    if len(input_text.split()) > 512:
        return jsonify({"error": "Input text too long (maximum 512 words allowed)."}), 400

    try:
        if summarizer is None:
            summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6",  # Smaller and faster than BART-large
                device=-1  # Force CPU to save memory (no GPU on Render Free)
            )
        summary = summarizer(input_text, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": f"Error generating summary: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
