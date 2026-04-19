import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import EmptyState from "../components/EmptyState.jsx";
import Loading from "../components/Loading.jsx";
import MetricCard from "../components/MetricCard.jsx";
import MockNotice from "../components/MockNotice.jsx";
import { getOverview } from "../services/api.js";

export default function OverviewDashboard() {
  const [overview, setOverview] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadOverview() {
      try {
        setError("");
        setOverview(await getOverview());
      } catch (err) {
        setError(err.message);
      }
    }
    loadOverview();
  }, []);

  if (error) {
    return <EmptyState title="Backend is not ready" message={error} />;
  }

  if (!overview) {
    return <Loading />;
  }

  const sentimentData = overview.sentiment_distribution || [];

  return (
    <section className="page-section">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Overview dashboard</p>
          <h2>Review and sentiment snapshot</h2>
        </div>
      </div>
      <MockNotice active={overview.is_mock} />

      <div className="metric-grid">
        <MetricCard label="Total reviews" value={overview.total_reviews.toLocaleString()} />
        <MetricCard label="Total products" value={overview.total_products.toLocaleString()} />
        <MetricCard
          label="Sentiment labels"
          value={sentimentData.reduce((sum, item) => sum + item.count, 0).toLocaleString()}
          note="Negative, Neutral, Positive"
        />
      </div>

      <section className="chart-panel">
        <h3>Sentiment distribution</h3>
        {sentimentData.length > 0 ? (
          <div className="chart-box">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={sentimentData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="label" />
                <YAxis allowDecimals={false} />
                <Tooltip />
                <Bar dataKey="count" fill="#2f8f83" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <EmptyState
            title="No sentiment data"
            message="Run the labeling pipeline or connect a CSV with sentiment labels."
          />
        )}
      </section>
    </section>
  );
}
