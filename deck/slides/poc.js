// Read rate per px/module bin, YOLO11n, rig test set (SYNTHETIC), from docs/results_synthetic.md.
const POC = {
  x: [2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75],
  series: [
    { key: "full", name: "ป้อนทั้งภาพ", y: [70.7, 70.9, 73.6, 64.1, 73.4, 79.4, 83.1, 85.4, 94.2, 100, 94.4, 96.3] },
    { key: "crop", name: "ตัดตามกรอบ (baseline)", y: [75.6, 82.0, 92.1, 88.5, 92.7, 96.4, 93.8, 91.1, 90.3, 94.8, 86.1, 86.7] },
    { key: "hybrid", name: "ดัดภาพ + ตัดกรอบสำรอง (ของเรา)", y: [79.5, 85.5, 97.8, 95.7, 97.4, 99.4, 99.4, 98.7, 97.4, 100, 96.5, 96.3] },
  ],
  bands: [{ lo: 3.5, hi: 4.0, label: "ประมาณ 4" }],
};

// theme: {font, ink, muted, grid, band, bandText, accent, others:{key:{color,dash,width}}, labelSize, tickSize}
function drawPoc(svg, W, H, t) {
  const NS = "http://www.w3.org/2000/svg";
  const m = { l: 110, r: 30, t: 30, b: 100 };
  const X = v => m.l + (v - 2) / 6 * (W - m.l - m.r);
  const Y = v => m.t + (100 - v) / 50 * (H - m.t - m.b);
  const el = (n, a, txt) => {
    const e = document.createElementNS(NS, n);
    for (const k in a) e.setAttribute(k, a[k]);
    if (txt !== undefined) e.textContent = txt;
    svg.appendChild(e);
    return e;
  };
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.style.fontFamily = t.font;
  for (const b of POC.bands) {
    el("rect", { x: X(b.lo), y: m.t, width: X(b.hi) - X(b.lo), height: H - m.t - m.b, fill: t.band });
    el("text", { x: (X(b.lo) + X(b.hi)) / 2, y: H - m.b - 18, "text-anchor": "middle", "font-size": t.tickSize, fill: t.bandText, "font-weight": 600 }, b.label);
  }
  for (let v = 50; v <= 100; v += 10) {
    el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), stroke: t.grid, "stroke-width": 1 });
    el("text", { x: m.l - 14, y: Y(v) + 7, "text-anchor": "end", "font-size": t.tickSize, fill: t.muted }, v);
  }
  for (let v = 2; v <= 8; v += 1) {
    el("text", { x: X(v), y: H - m.b + 36, "text-anchor": "middle", "font-size": t.tickSize, fill: t.muted }, v);
  }
  el("text", { x: (m.l + W - m.r) / 2, y: H - 18, "text-anchor": "middle", "font-size": t.labelSize, fill: t.ink }, "ความละเอียดของรหัส (พิกเซลต่อ module, px/module)");
  el("text", { x: 0, y: 0, transform: `translate(30 ${(m.t + H - m.b) / 2}) rotate(-90)`, "text-anchor": "middle", "font-size": t.labelSize, fill: t.ink }, "อัตราอ่านได้ถูกต้อง (%)");
  for (const s of POC.series) {
    const st = s.key === "hybrid" ? { color: t.accent, dash: "", width: 6 } : t.others[s.key];
    const d = s.y.map((v, i) => `${i ? "L" : "M"}${X(POC.x[i]).toFixed(1)},${Y(v).toFixed(1)}`).join(" ");
    el("path", { d, fill: "none", stroke: st.color, "stroke-width": st.width, "stroke-dasharray": st.dash, "stroke-linejoin": "round", "stroke-linecap": "round" });
  }
  // the values at about 4 px/module (x = 3.75)
  const at = [["hybrid", 95.7, t.accent, -18], ["crop", 88.5, t.others.crop.color, 34], ["full", 64.1, t.others.full.color, 34]];
  for (const [k, v, c, dy] of at) {
    el("circle", { cx: X(3.75), cy: Y(v), r: 8, fill: c, stroke: t.bg || "#fff", "stroke-width": 3 });
    el("text", { x: X(3.75) - 16, y: Y(v) + dy, "text-anchor": "end", "font-size": t.tickSize, "font-weight": 700, fill: c, stroke: t.bg || "#fff", "stroke-width": 8, "paint-order": "stroke", "stroke-linejoin": "round" }, v.toFixed(1) + "%");
  }
}

function pocLegend(container, t) {
  for (const s of POC.series.slice().reverse()) {
    const st = s.key === "hybrid" ? { color: t.accent, dash: "", width: 6 } : t.others[s.key];
    const row = document.createElement("div");
    row.className = "lg-row" + (s.key === "hybrid" ? " lg-main" : "");
    row.innerHTML = `<svg width="56" height="16"><line x1="2" x2="54" y1="8" y2="8" stroke="${st.color}" stroke-width="${st.width}" stroke-dasharray="${st.dash}" stroke-linecap="round"/></svg><span>${s.name}</span>`;
    container.appendChild(row);
  }
}
