console.log("📦 KaTeX math.js loaded");

document$.subscribe(() => {
  console.log("✅ KaTeX rendering triggered");

  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false }
    ],
    throwOnError: false,
    macros: {
      "\\RR": "\\mathbb{R}"
    }
  });
});
