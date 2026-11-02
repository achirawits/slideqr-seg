// Footer: <body data-section="..." data-note="..."> ; page number from the file name (NN-name.html).
(function () {
  const TOTAL = 19;
  const n = parseInt(location.pathname.split("/").pop(), 10);
  const b = document.body;
  const f = document.createElement("div");
  f.className = "foot";
  f.innerHTML = `<span>${b.dataset.note || ""}</span><span class="pg"><b>${n} / ${TOTAL}</b></span>`;
  b.appendChild(f);
})();
