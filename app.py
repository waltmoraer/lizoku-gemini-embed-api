from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# 🔑 Hardcode for testing only (move to env vars later)
GEMINI_API_KEY = "AIzaSyDaLZ20PGzudiXNr1qnueqB8mIaYnOUXo4"
GEMINI_EMBEDDING_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"

MAX_BATCH_SIZE = 50
MAX_CHARS = 2000


def chunk_text(text, max_chars=MAX_CHARS):
    """Split oversized text into safe sub-chunks."""
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def embed_batch(texts, task_type):
    """Send a batch of texts to Gemini API and return embeddings."""
    embeddings = []
    for t in texts:
        payload = {
            "model": "models/gemini-embedding-001",
            "content": {"parts": [{"text": t}]},
            "taskType": task_type
        }

        response = requests.post(
            GEMINI_EMBEDDING_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(f"Gemini API error: {response.text}")

        result = response.json()
        if "embedding" not in result:
            raise Exception("No embedding returned for input.")

        embeddings.append(result["embedding"]["values"])
    return embeddings


@app.route('/embed', methods=['POST'])
def embed():
    if request.method != "POST":
        return jsonify({"error": "Only POST method is allowed"}), 405

    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        texts = data.get("texts")
        task_type = data.get("taskType")

        if not texts or not isinstance(texts, list):
            return jsonify({"error": "Invalid or missing 'texts'. Must be an array of strings."}), 400
        if task_type not in ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]:
            return jsonify({"error": "Invalid or missing 'taskType'."}), 400

        safe_texts = []
        for t in texts:
            if len(t) > MAX_CHARS:
                safe_texts.extend(chunk_text(t, MAX_CHARS))
            else:
                safe_texts.append(t)

        all_embeddings = []
        for i in range(0, len(safe_texts), MAX_BATCH_SIZE):
            batch = safe_texts[i:i + MAX_BATCH_SIZE]
            all_embeddings.extend(embed_batch(batch, task_type))

        return jsonify({"embeddings": all_embeddings}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to generate embeddings: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
