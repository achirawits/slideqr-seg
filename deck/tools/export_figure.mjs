// Render one fixed-size HTML figure to a vector PDF and a PNG, using the installed Edge.
//   node export_figure.mjs ../figures/poc_chart.html ../../proposal/figures/read_rate_yolo11n_rig 1600 720
import { chromium } from "playwright";
import path from "path";
import { pathToFileURL } from "url";

const [src, out, w = "1600", h = "720"] = process.argv.slice(2);
const width = parseInt(w, 10), height = parseInt(h, 10);
const browser = await chromium.launch({ channel: "msedge" });
const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 2 });
await page.goto(pathToFileURL(path.resolve(src)).href, { waitUntil: "networkidle" });
await page.evaluate(() => document.fonts.ready);
await page.pdf({ path: out + ".pdf", width: width + "px", height: height + "px", printBackground: true, pageRanges: "1" });
await page.screenshot({ path: out + ".png" });
await browser.close();
console.log(out + ".pdf", out + ".png");
