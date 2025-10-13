from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'status': 'ok',
        'message': 'lizoku-gemini-embed-api is running'
    })

@app.route('/embed', methods=['POST'])
def embed():
    """Simple placeholder endpoint that expects JSON with a 'text' field
    and returns a fake embedding (list of floats) for demonstration.
    """
    data = request.get_json(silent=True) or {}
    text = data.get('text', '')
    # Deterministic fake embedding: length and ords
    embedding = [float(len(text))]
    embedding += [float(ord(c) % 100) / 100.0 for c in text[:10]]
    return jsonify({
        'input_text': text,
        'embedding': embedding
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
