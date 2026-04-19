export default function MetricCard({ label, value, note }) {
  return (
    <section className="metric-card">
      <p>{label}</p>
      <strong>{value}</strong>
      {note ? <span>{note}</span> : null}
    </section>
  );
}
