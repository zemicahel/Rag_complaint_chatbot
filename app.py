from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_pipeline import generate_rag_response

app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"})

@app.route("/query", methods=["POST"])
def query():
    data = request.get_json()

    if not data or "question" not in data:
        return jsonify({"error": "Provide 'question' field"}), 400

    result = generate_rag_response(data["question"])
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
