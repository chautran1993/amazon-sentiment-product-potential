import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import EmptyState from "../components/EmptyState.jsx";
import Loading from "../components/Loading.jsx";
import MockNotice from "../components/MockNotice.jsx";
import { getTopRanking } from "../services/api.js";

export default function TopProductRanking({ onSelectProduct }) {
  const [products, setProducts] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    async function loadRanking() {
      try {
        setError("");
        setProducts(await getTopRanking(10));
      } catch (err) {
        setError(err.message);
      }
    }
    loadRanking();
  }, []);

  if (error) {
    return <EmptyState title="Ranking unavailable" message={error} />;
  }

  if (!products) {
    return <Loading />;
  }

  if (products.length === 0) {
    return (
      <EmptyState
        title="No product ranking yet"
        message="Create outputs/ranking/product_ranking.csv from a dataset that includes asin."
      />
    );
  }

  return (
    <section className="page-section">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Top product ranking</p>
          <h2>Top 10 potential products</h2>
        </div>
      </div>
      <MockNotice active={products.some((product) => product.is_mock)} />

      <section className="chart-panel">
        <h3>Potential score</h3>
        <div className="chart-box">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={products}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="asin" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="product_potential_score" fill="#d45d4c" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>ASIN</th>
              <th>Reviews</th>
              <th>Avg sentiment</th>
              <th>Positive</th>
              <th>Negative</th>
              <th>Potential</th>
            </tr>
          </thead>
          <tbody>
            {products.map((product) => (
              <tr key={product.asin} onClick={() => onSelectProduct(product.asin)}>
                <td>{product.asin}</td>
                <td>{product.review_count}</td>
                <td>{formatNumber(product.sentiment_score_mean)}</td>
                <td>{formatPercent(product.positive_ratio)}</td>
                <td>{formatPercent(product.negative_ratio)}</td>
                <td>{formatNumber(product.product_potential_score)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function formatNumber(value) {
  return value == null ? "-" : Number(value).toFixed(3);
}

function formatPercent(value) {
  return value == null ? "-" : `${(Number(value) * 100).toFixed(1)}%`;
}
