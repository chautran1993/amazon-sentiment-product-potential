export const mockSentimentDistribution = [
  { label: "Negative", count: 5727 },
  { label: "Neutral", count: 2810 },
  { label: "Positive", count: 4984 },
];

export const mockProducts = [
  {
    asin: "B00DEMO001",
    review_count: 1240,
    sentiment_score_mean: 2.82,
    positive_ratio: 0.78,
    negative_ratio: 0.08,
    product_potential_score: 0.94,
    sentiment_distribution: [
      { label: "Negative", count: 99 },
      { label: "Neutral", count: 174 },
      { label: "Positive", count: 967 },
    ],
    sample_reviews: [
      "Works perfectly and feels reliable for everyday use.",
      "Great value for the price and the setup was simple.",
      "Battery life is better than expected.",
    ],
  },
  {
    asin: "B00DEMO002",
    review_count: 980,
    sentiment_score_mean: 2.74,
    positive_ratio: 0.71,
    negative_ratio: 0.10,
    product_potential_score: 0.86,
    sentiment_distribution: [
      { label: "Negative", count: 98 },
      { label: "Neutral", count: 186 },
      { label: "Positive", count: 696 },
    ],
    sample_reviews: [
      "Good sound quality and easy controls.",
      "The product arrived quickly and worked out of the box.",
    ],
  },
  {
    asin: "B00DEMO003",
    review_count: 765,
    sentiment_score_mean: 2.69,
    positive_ratio: 0.66,
    negative_ratio: 0.09,
    product_potential_score: 0.79,
    sentiment_distribution: [
      { label: "Negative", count: 69 },
      { label: "Neutral", count: 191 },
      { label: "Positive", count: 505 },
    ],
    sample_reviews: [
      "Solid build and useful features.",
      "A little expensive, but the performance is good.",
    ],
  },
  {
    asin: "B00DEMO004",
    review_count: 640,
    sentiment_score_mean: 2.61,
    positive_ratio: 0.61,
    negative_ratio: 0.13,
    product_potential_score: 0.70,
    sentiment_distribution: [
      { label: "Negative", count: 83 },
      { label: "Neutral", count: 166 },
      { label: "Positive", count: 391 },
    ],
    sample_reviews: [
      "Useful product, but the instructions could be clearer.",
      "Works well after a few setup steps.",
    ],
  },
  {
    asin: "B00DEMO005",
    review_count: 510,
    sentiment_score_mean: 2.54,
    positive_ratio: 0.56,
    negative_ratio: 0.14,
    product_potential_score: 0.62,
    sentiment_distribution: [
      { label: "Negative", count: 71 },
      { label: "Neutral", count: 153 },
      { label: "Positive", count: 286 },
    ],
    sample_reviews: [
      "Decent quality for the price.",
      "Not perfect, but it does the job.",
    ],
  },
];

export function getMockOverview() {
  return {
    total_reviews: 13521,
    total_products: mockProducts.length,
    sentiment_distribution: mockSentimentDistribution,
    is_mock: true,
  };
}

export function getMockTopRanking(limit = 10) {
  return mockProducts.slice(0, limit).map(stripDetailFields);
}

export function getMockProducts(limit = 20, offset = 0) {
  const items = mockProducts.slice(offset, offset + limit).map(stripDetailFields);
  return {
    total: mockProducts.length,
    limit,
    offset,
    items,
    is_mock: true,
  };
}

export function getMockProductDetail(asin) {
  const product = mockProducts.find((item) => item.asin === asin) || mockProducts[0];
  return { ...product, is_mock: true };
}

export function getMockPrediction(reviewText) {
  const lowerText = reviewText.toLowerCase();
  const positiveWords = ["great", "good", "excellent", "love", "perfect", "works", "amazing"];
  const negativeWords = ["bad", "poor", "broken", "worst", "hate", "return", "defective"];
  const positiveHits = positiveWords.filter((word) => lowerText.includes(word)).length;
  const negativeHits = negativeWords.filter((word) => lowerText.includes(word)).length;

  const scores = {
    Negative: 1 + negativeHits,
    Neutral: 1,
    Positive: 1 + positiveHits,
  };
  const total = scores.Negative + scores.Neutral + scores.Positive;
  const probabilities = Object.fromEntries(
    Object.entries(scores).map(([label, value]) => [label, value / total]),
  );
  const predicted_label = Object.entries(probabilities).sort((a, b) => b[1] - a[1])[0][0];

  return {
    predicted_label,
    probabilities,
    model_name: "mock_keyword_fallback",
    is_mock: true,
  };
}

function stripDetailFields(product) {
  const { sentiment_distribution, sample_reviews, ...rankingItem } = product;
  return rankingItem;
}
