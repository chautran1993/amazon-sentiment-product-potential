export default function Header() {
  return (
    <header className="header">
      <div>
        <p className="eyebrow">Amazon sentiment analysis</p>
        <h1>Product Potential Evaluation</h1>
        <p className="subtitle">
          Review sentiment, product ranking, and live sentiment prediction for demo workflows.
        </p>
      </div>
      <img
        className="header-image"
        src="https://images.unsplash.com/photo-1586880244386-8b3e34c8382c?auto=format&fit=crop&w=600&q=70"
        alt="Packed parcels in a warehouse"
      />
    </header>
  );
}
