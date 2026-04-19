import { useState } from "react";
import MockNotice from "../components/MockNotice.jsx";
import { predictReview } from "../services/api.js";

export default function PredictReview() {
  const [reviewText, setReviewText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handlePredict(event) {
    event.preventDefault();
    if (!reviewText.trim()) {
      setError("Please enter a review.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      setResult(await predictReview(reviewText));
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="page-section">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Predict review</p>
          <h2>Try a new Amazon review</h2>
        </div>
      </div>

      <form className="predict-form" onSubmit={handlePredict}>
        <textarea
          value={reviewText}
          onChange={(event) => setReviewText(event.target.value)}
          placeholder="Type an English product review..."
          rows={7}
        />
        <button type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>

      {error ? <p className="error-text">{error}</p> : null}

      {result ? (
        <section className="prediction-result">
          <MockNotice
            active={result.is_mock}
            message="Backend predict chua ket noi, nen ket qua nay dang dung bo du doan demo."
          />
          <p>Predicted sentiment</p>
          <strong>{result.predicted_label}</strong>
          <span>Model: {result.model_name}</span>
          {result.probabilities ? (
            <div className="probability-list">
              {Object.entries(result.probabilities).map(([label, value]) => (
                <div key={label}>
                  <span>{label}</span>
                  <meter min="0" max="1" value={value} />
                  <b>{(value * 100).toFixed(1)}%</b>
                </div>
              ))}
            </div>
          ) : null}
        </section>
      ) : null}
    </section>
  );
}
