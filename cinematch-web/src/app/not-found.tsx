import Link from "next/link";

export default function NotFound() {
  return (
    <div style={{
      minHeight: "100vh",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      padding: "24px",
      textAlign: "center",
      background: "var(--color-bg)",
    }}>
      <h1 style={{ fontSize: "64px", fontWeight: 800, margin: 0, opacity: 0.1, position: "absolute", zIndex: 0 }}>404</h1>
      <div style={{ zIndex: 1, position: "relative", display: "flex", flexDirection: "column", alignItems: "center" }}>
        <h2 style={{ fontSize: "28px", fontWeight: 700, marginBottom: "16px" }}>Page Not Found</h2>
        <p style={{ color: "var(--color-text-muted)", marginBottom: "32px", maxWidth: "400px" }}>
          We couldn't find the page you're looking for. It might have been moved or deleted.
        </p>
        <Link href="/dashboard" style={{
            padding: "12px 28px",
            borderRadius: "100px",
            background: "#fff",
            color: "#000",
            fontWeight: 700,
            textDecoration: "none",
            display: "inline-block",
            transition: "transform 0.2s ease"
          }}>
          Return Home
        </Link>
      </div>
    </div>
  );
}
