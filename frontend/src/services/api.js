import {
  getMockOverview,
  getMockPrediction,
  getMockProductDetail,
  getMockProducts,
  getMockTopRanking,
} from "./mockData.js";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const ENABLE_MOCK_FALLBACK = import.meta.env.VITE_ENABLE_MOCK_FALLBACK !== "false";

async function request(path, options = {}) {
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Request failed: ${response.status}`);
    }

    return response.json();
  } catch (error) {
    throw new Error(`Backend unavailable at ${API_BASE_URL}. ${error.message}`);
  }
}

function withMockMeta(data) {
  if (Array.isArray(data)) {
    return data.map((item) => ({ ...item, is_mock: false }));
  }
  return { ...data, is_mock: false };
}

function shouldUseFallback(data) {
  if (!ENABLE_MOCK_FALLBACK) {
    return false;
  }
  if (Array.isArray(data)) {
    return data.length === 0;
  }
  if (data?.items) {
    return data.items.length === 0;
  }
  return false;
}

export function getOverview() {
  return request("/overview")
    .then(withMockMeta)
    .catch(() => getMockOverview());
}

export function getProducts(limit = 20, offset = 0) {
  return request(`/products?limit=${limit}&offset=${offset}`)
    .then((data) => (shouldUseFallback(data) ? getMockProducts(limit, offset) : withMockMeta(data)))
    .catch(() => getMockProducts(limit, offset));
}

export function getTopRanking(limit = 10) {
  return request(`/products/top-ranking?limit=${limit}`)
    .then((data) => (shouldUseFallback(data) ? getMockTopRanking(limit) : withMockMeta(data)))
    .catch(() => getMockTopRanking(limit));
}

export function getProductDetail(asin) {
  return request(`/products/${encodeURIComponent(asin)}`)
    .then(withMockMeta)
    .catch(() => getMockProductDetail(asin));
}

export function predictReview(reviewText) {
  return request("/predict-review", {
    method: "POST",
    body: JSON.stringify({ review_text: reviewText }),
  })
    .then(withMockMeta)
    .catch(() => getMockPrediction(reviewText));
}
