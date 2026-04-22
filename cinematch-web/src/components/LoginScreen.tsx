"use client";

import { useState, useRef } from "react";
import { motion } from "framer-motion";
import { apiLogin, type UserSession } from "@/lib/api";

const ALL_POSTERS = [
  "nMKdUUepR0i5zn0y1T4CsSB5chy.jpg", // The Dark Knight
  "j0BB9DoqobGvRqKVeAveP70hWi2.jpg", // Interstellar
  "7RyHsO4yDXtBv1zUU3mTpHeQ0d5.jpg", // Avengers: Infinity War
  "oMsxZEvz9a708d49b6UdZK1KAo5.jpg", // The Matrix
  "hZkgoQYus5vegHoetLkCJzb17zJ.jpg", // Fight Club
  "nMKdUUepR0i5zn0y1T4CsSB5chy.jpg", // Joker
  "14QbnygCuTO0vl7CAFmPf1fgZfV.jpg", // Spider-Man: No Way Home
  "CpLAfXgSNeNRRbRzPrTuzKmIHO.jpg", // Dune
  "8b8R8l88Qje9dn9OE8PY05Nxl1X.jpg", // Dune: Part Two
  "62HCnUTziyWcpDaBO2i1DX17ljH.jpg", // Top Gun: Maverick
  "8tZYtuWezp8JbcsvHYO0O46tFbo.jpg", // Mad Max: Fury Road
  "q6y0Go1tsGEsmtFryDOJo3dEmqu.jpg", // The Shawshank Redemption
  "d5iIlFn5s0ImszYzBPb8JPIfbXD.jpg", // Pulp Fiction
  "7IiTTgloJzvGI1TAYymCfbfl3vT.jpg", // Parasite
  "rCzpDGLbOoPwLjy3OAm5NUPOTrC.jpg", // Venom
  "uxzzxijgPIY7slzFvMotPv8wjKA.jpg", // Black Panther
  "5Av0uOmsji4IZWrUHa6CnVZBULL.jpg", // Gravity
  "uzr65Z3xDCjuMW7fw8Whm5curr7.jpg", // 1917
  "iwjyATmaI498r7sSBc4HZPp1ven.jpg", // Oppenheimer
  "8Mv3ldkLTo9w7cHvG1fwZxsVaAw.jpg", // Barbie
  "3cdfnihGSrMiQWzmVPaEs3p2Mp1.jpg", // Your Name.
  "39wmItIWsg5sZMyRUHLkWBcuVCM.jpg", // Spirited Away
  "gl0jzn4BupSbL2qMVeqrjKkF9Js.jpg", // Princess Mononoke
  "nv5wwZou159v5OC61i4ElR7OqyY.jpg", // Howl's Moving Castle
  "vp7ZXmEHxb8rI6LWguqHQzkRJ8L.jpg", // Suzume
  "fK40VGYIm7hmKrLJ26fgPQU0qRG.jpg", // The Boy and the Heron
  "w1oD1MzHjnBJc5snKupIQaSBLIh.jpg", // Akira
  "uhUO7vQQKvCTfQWubOt5MAKokbL.jpg", // Nausicaä of the Valley of the Wind
  "j2ZvLJyz163MlmBFsoaDYOwxgws.jpg", // Castle in the Sky
  "fxYazFVeOCHpHwuqGuiqcCTw162.jpg", // My Neighbor Totoro
  "gMcIbTJ5bFZuyMOYhlG8uquqjIz.jpg", // Train to Busan
  "sdwjQEM869JFwMytTmvr6ggvaUl.jpg", // Oldboy
  "tllOU4ZGsucCiGF7n9RgRBWoZ3Z.jpg", // The Wailing
  "wyv2Y9vXYJwJzF6cNVuVwVOsOUj.jpg", // Along With the Gods: The Two Worlds
  "pMBKzEotLUzk3NuQiyVjZEDlH70.jpg", // Burning
  "mKDFcIkvvCmwKIwlO1J3EmvbQM9.jpg", // RRR
  "k38sLjhQsdLRJZCu3hIL7RcGo3A.jpg", // Sita Ramam
  "iy9uFMJvzlDC3kMFPI2Fk2HJZ2x.jpg", // Ala Vaikunthapurramuloo
  "cthkOu8gxDoCg0OcRHrTFu3v3xm.jpg", // Arjun Reddy
  "2pJ9xW1mA7zRfrg9On9e8AekrQM.jpg", // Geetha Govindam
  "s5QSniG6P7mtytKApSQUr3XlEbg.jpg", // Saaho
  "tUKAFRfIqhLXRPPvjDvPulOx47j.jpg", // Dhruva
  "nH6hPhJq3EEv9CnBZgXU3IQnpJo.jpg", // Venky Mama
  "u7kuUaySqXBVAtqEl9vkTkAzHV9.jpg", // 3 Idiots
  "x7Sz339F2oC8mBf0DHCQpKizXaL.jpg", // PK
  "fJ3k4ctIvIyeQxyNhUWZaLwKeP5.jpg", // Dil Chahta Hai
  "z4k7b66jAHP8sQbEahxss6Ct8BW.jpg", // Zindagi Na Milegi Dobara
  "42vFebJ0VRnwdemaUOIr7c6Tjo1.jpg", // Kabhi Khushi Kabhie Gham...
  "8aYAfAPolsRFrHbP1rafeSgg2Ew.jpg", // Sholay
  "u4YATs3X5PLcwCb4j4M6xJcgbty.jpg", // Lagaan: Once Upon a Time in India
  "dE4dYJHGHoNzdxyNJZi2TJXhSOs.jpg", // Gangs of Wasseypur
  "xkgp35nyquBbMPb0ICJUF188vPG.jpg", // Mughal-E-Azam
  "hh2bBmgqzJEY1J7enMKqTsziVRO.jpg", // Baasha
  "izN82Ub6Bv6AEStUwGrWFX8V1JL.jpg", // Enthiran
  "sCOosltSrgTlBFVbZX3iYvB1kF9.jpg", // Kabali
  "r8pUcom5Mw8igtBpj3AHSAUvH0R.jpg", // Master
  "9TEUJy5aRP7LaM1LKTfcxVK34JK.jpg", // Amélie
  "bGksau9GGu0uJ8DJQ8DYc9JW5LM.jpg", // The Intouchables
  "jRJrQ72VLyEnVsvwfep8Xjlvu8c.jpg", // Léon: The Professional
  "jzT1mA2q4cN3VZXejnuINrwS57q.jpg", // The Artist
  "o0IWa75BXyXryNDVqw0xoXZzy1J.jpg", // Pan's Labyrinth
  "dQL2wJZo05GDd21VgOacMeCuyZy.jpg", // Roma
  "fNHCogWhABNAmzk0IFdzQP5XLYj.jpg", // The Secret in Their Eyes
  "ylZ06kRUF2JKkrCG2E3qn5D9w8L.jpg", // Crouching Tiger, Hidden Dragon
  "wgvc3PmjQGtYYDWaeuV867mnFDs.jpg", // Hero
  "3jKynKnUtRERxBFAcvZ8AvkTo4c.jpg", // The Wandering Earth
  "sztvp3gX6wxy3X85yH0kq82QCJw.jpg", // Ne Zha
  "69QJSMkm5Hh0x9fLElZ8hjv5N4P.jpg", // Ip Man
  "zoVeIgKzGJzpdG6Gwnr7iOYfIMU.jpg", // Cinema Paradiso
  "gavyCu1UaTaTNPsVaGXT6pe5u24.jpg", // Life Is Beautiful
  "pHyxb2RV5wLlboAwm9ZJ9qTVEDw.jpg", // Chainsaw Man
  "sUsVimPdA1l162FvdBIlmKBlWHx.jpg", // Demon Slayer
  "21sC2assImQIYCEDA84Qh9d1RsK.jpg", // Baahubali
  "nlu9WbcetNFRGXXPWITr30ob7W6.jpg", // Salaar
  "lQfuaXjANoTsdx5iS0gCXlK9D2L.jpg", // Devara
  "uDg52hGwy4Dm8hGGVYK3PHQzsKc.jpg", // OG
];

// Shuffle and pick a display set on every mount
function shuffled<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const BASE = "/login-posters/";

interface Props {
  onLogin: (session: UserSession) => void;
}

export default function LoginScreen({ onLogin }: Props) {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  // Shuffle once on mount so every visit shows a different arrangement
  const [posters] = useState(() => shuffled(ALL_POSTERS));
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = email.trim();
    if (!trimmed) {
      setError("Please enter your email.");
      inputRef.current?.focus();
      return;
    }
    setError("");
    setLoading(true);
    try {
      const session = await apiLogin(trimmed);
      onLogin(session);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Connection failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        minHeight: "100dvh",
        width: "100%",
        padding: "0 24px",
        fontFamily: "var(--font-sans)",
      }}
    >
      {/* Background Mosaic (Optimized) */}
      <div style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0, overflow: "hidden", zIndex: 0, background: "#0a0a12" }}>
        {/* Simplified scrim to reduce compositing costs. Avoid backdropFilter blur on large moving areas! */}
        <div style={{ position: "absolute", inset: 0, background: "radial-gradient(ellipse at 50% 50%, rgba(10,10,18,0.15) 0%, rgba(0,0,0,0.85) 100%)", zIndex: 3, pointerEvents: "none" }} />
        <div style={{ position: "absolute", inset: "-20%", width: "140%", height: "140%", display: "flex", flexWrap: "wrap", gap: "16px", transform: "rotate(-10deg) scale(1.2)", opacity: 0.55, zIndex: 1, willChange: "transform" }}>
          <motion.div
            animate={{
              y: [0, -1000],
            }}
            transition={{
              repeat: Infinity,
              duration: 60,
              ease: "linear",
            }}
            style={{ display: "flex", flexWrap: "wrap", gap: "16px", justifyContent: "center", willChange: "transform" }}
          >
            {/* Restored full visual density, optimized via async decoding, lazy loading, and no-blur */}
            {[...posters, ...posters, ...posters].map((path, i) => (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                key={i}
                src={`${BASE}${path}`}
                alt=""
                loading="lazy"
                decoding="async"
                onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
                style={{
                  width: "130px",
                  height: "195px",
                  borderRadius: "12px",
                  objectFit: "cover",
                  flexShrink: 0,
                  opacity: 1,
                }}
              />
            ))}
          </motion.div>
        </div>
      </div>

      <div style={{ zIndex: 10, display: "flex", flexDirection: "column", alignItems: "center", width: "100%" }}>
      {/* Brand */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.1, ease: [0.25, 0.1, 0.25, 1] }}
        style={{
          textAlign: "center",
          marginBottom: "5vh",
        }}
      >
        <style>{`
          @keyframes title-shimmer {
            0% { background-position: 200% center; }
            100% { background-position: -200% center; }
          }
        `}</style>
        <h1
          className="heading-display"
          style={{
            fontSize: "clamp(3.5rem, 11vw, 6.5rem)",
            lineHeight: 0.95,
            fontWeight: 700,
            letterSpacing: "-0.055em",
            background: "linear-gradient(90deg, #888892 0%, #ffffff 30%, #d8d8e0 55%, #ffffff 70%, #888892 100%)",
            backgroundSize: "200% auto",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            backgroundClip: "text",
            margin: 0,
            animation: "title-shimmer 6s linear infinite",
          }}
        >
          CineMatch
        </h1>
        <p
          style={{
            marginTop: "14px",
            fontSize: "clamp(0.95rem, 2vw, 1.2rem)",
            color: "var(--color-text-secondary)",
            fontWeight: 400,
            letterSpacing: "-0.005em",
          }}
        >
          Discover movies you&apos;ll love.
        </p>
      </motion.div>

      {/* Email form */}
      <motion.form
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, delay: 0.25, ease: [0.25, 0.1, 0.25, 1] }}
        onSubmit={handleSubmit}
        style={{ width: "100%", maxWidth: "420px" }}
      >
        <div style={{ position: "relative" }}>
          <input
            ref={inputRef}
            id="email-input"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Enter your email"
            autoComplete="email"
            autoFocus
            disabled={loading}
            style={{
              width: "100%",
              padding: "16px 0",
              background: "transparent",
              border: "none",
              borderBottom: "1px solid rgba(255,255,255,0.2)",
              fontSize: "clamp(1.1rem, 2.5vw, 1.4rem)",
              fontWeight: 300,
              color: "var(--color-text-primary)",
              outline: "none",
              fontFamily: "inherit",
              opacity: loading ? 0.4 : 1,
              transition: "border-color 0.3s ease",
            }}
            onFocus={(e) => { e.currentTarget.style.borderBottomColor = "rgba(255,255,255,0.7)"; }}
            onBlur={(e) => { e.currentTarget.style.borderBottomColor = "rgba(255,255,255,0.2)"; }}
          />
        </div>

        {error && (
          <motion.p
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            style={{ marginTop: "12px", fontSize: "13px", color: "var(--color-danger)", fontWeight: 400 }}
          >
            {error}
          </motion.p>
        )}

        <motion.button
          type="submit"
          disabled={loading}
          whileTap={{ scale: 0.98 }}
          className="primary-button"
          style={{
            marginTop: "36px",
            width: "100%",
            padding: "15px 0",
            fontSize: "15px",
            cursor: loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? (
            <span style={{ display: "inline-flex", alignItems: "center", gap: "8px" }}>
              <svg
                style={{ animation: "spin 1s linear infinite", height: "16px", width: "16px" }}
                viewBox="0 0 24 24"
                fill="none"
              >
                <circle style={{ opacity: 0.25 }} cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
                <path style={{ opacity: 0.75 }} fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Connecting
            </span>
          ) : (
            "Login"
          )}
        </motion.button>
      </motion.form>
      </div>

    </div>
  );
}
