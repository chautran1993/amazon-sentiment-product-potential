import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import EmptyState from "../components/EmptyState.jsx";
import MetricCard from "../components/MetricCard.jsx";
import MockNotice from "../components/MockNotice.jsx";

export default function ProductDetail({ asin, onAsinChange, product, error }) {
  const sentimentData = product?.sentiment_distribution || [];

  return (
    <section className="page-section">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Product detail</p>
          <h2>Sentiment by ASIN</h2>
        </div>
        <form className="asin-form" onSubmit={(event) => event.preventDefault()}>
          <input
            value={asin}
            onChange={(event) => onAsinChange(event.target.value)}
            placeholder="Enter ASIN"
          />
        </form>
      </div>

      {!asin ? (
        <EmptyState title="Choose a product" message="Select a product from ranking or enter an ASIN." />
      ) : error ? (
        <EmptyState
          title="Product not found"
          message="The current dataset may not include asin. Generate product_ranking.csv first."
        />
      ) : product ? (
        <>
          <MockNotice active={product.is_mock} />
          <div className="metric-grid">
            <MetricCard label="ASIN" value={product.asin} />
            <MetricCard label="Review count" value={product.review_count?.toLocaleString() || "0"} />
            <MetricCard
              label="Average sentiment score"
              value={product.sentiment_score_mean == null ? "-" : Number(product.sentiment_score_mean).toFixed(3)}
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
                    <Bar dataKey="count" fill="#e4a11b" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <EmptyState title="No sentiment distribution" message="This product has no review-level sentiment rows." />
            )}
          </section>

          {product.sample_reviews?.length > 0 ? (
            <section className="reviews-panel">
              <h3>Sample reviews</h3>
              {product.sample_reviews.map((review, index) => (
                <p key={`${product.asin}-${index}`}>{review}</p>
              ))}
            </section>
          ) : null}
        </>
      ) : (
        <EmptyState title="Loading product" message="Fetching product detail." />
      )}
    </section>
  );
}
