"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";

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

// Shuffle and pick a display set each visit.
function shuffled<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const BASE = "/login-posters/";

/**
 * Animated cinematic poster-wall background. Fills its nearest positioned
 * ancestor (or the viewport) and is purely decorative (aria-hidden). Shared by
 * the login screen and the public landing page.
 *
 * The shuffle happens AFTER mount, not during render, so the server-rendered
 * order matches the first client render — important because the landing page is
 * statically prerendered and would otherwise throw a hydration mismatch.
 */
export default function PosterMosaic() {
  const [posters, setPosters] = useState(ALL_POSTERS);
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- cosmetic shuffle, client-only post-hydration
    setPosters(shuffled(ALL_POSTERS));
  }, []);

  return (
    <div aria-hidden style={{ position: "absolute", inset: 0, overflow: "hidden", zIndex: 0, background: "#0a0a12" }}>
      {/* Scrim */}
      <div style={{ position: "absolute", inset: 0, background: "radial-gradient(ellipse at 50% 50%, rgba(10,10,18,0.15) 0%, rgba(0,0,0,0.85) 100%)", zIndex: 3, pointerEvents: "none" }} />
      <div style={{ position: "absolute", inset: "-20%", width: "140%", height: "140%", display: "flex", flexWrap: "wrap", gap: "16px", transform: "rotate(-10deg) scale(1.2)", opacity: 0.55, zIndex: 1, willChange: "transform" }}>
        <motion.div
          animate={{ y: [0, -1000] }}
          transition={{ repeat: Infinity, duration: 60, ease: "linear" }}
          style={{ display: "flex", flexWrap: "wrap", gap: "16px", justifyContent: "center", willChange: "transform" }}
        >
          {[...posters, ...posters, ...posters].map((path, i) => (
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
  );
}
