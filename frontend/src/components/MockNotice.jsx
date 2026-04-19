const DEFAULT_MESSAGE =
  "Dang dung du lieu demo vi backend chua san sang hoac file du lieu rong.";

export default function MockNotice({ active, message = DEFAULT_MESSAGE }) {
  if (!active) {
    return null;
  }

  return <div className="mock-notice">{message}</div>;
}
