import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setAnswer("");
    setSources([]);
    setError("");

    try {
      const response = await axios.post(
        "http://127.0.0.1:5000/query",
        { question: question },
        { headers: { "Content-Type": "application/json" } }
      );

      setAnswer(response.data.answer);
      setSources(response.data.sources || []);
    } catch (err) {
      setError("Unable to reach the server. Please ensure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setQuestion("");
    setAnswer("");
    setSources([]);
    setError("");
  };

  return (
    <div className="container">
      <header>
        <h1>RAG Complaint Assistant</h1>
        <p className="subtitle">Retrieve insights from your document database</p>
      </header>

      <div className="input-section">
        <textarea
          placeholder="e.g., What are the common complaints regarding credit cards in California?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
        />
        <div className="button-group">
          <button 
            className="primary-btn" 
            onClick={handleSubmit} 
            disabled={loading || !question.trim()}
          >
            {loading ? "Analyzing..." : "Ask Question"}
          </button>
          <button className="secondary-btn" onClick={handleClear}>
            Clear
          </button>
        </div>
      </div>

      {error && <div className="error-message">{error}</div>}

      {answer && (
        <div className="results-container">
          <section className="answer-section">
            <h2>AI Generated Answer</h2>
            <div className="answer-text">
              <p>{answer}</p>
            </div>
          </section>

          {sources.length > 0 && (
            <section className="sources-section">
              <h3>Supporting Sources</h3>
              <div className="sources-grid">
                {sources.map((src, index) => (
                  <div key={index} className="source-card">
                    <span className="source-index">Source #{index + 1}</span>
                    <p><strong>Product:</strong> {src.product || "N/A"}</p>
                    <p><strong>Company:</strong> {src.company || "N/A"}</p>
                    <p><strong>State:</strong> {src.state || "N/A"}</p>
                    <div className="chunk-tag">Chunk Index: {src.chunk_index}</div>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>
      )}
    </div>
  );
}

export default App;