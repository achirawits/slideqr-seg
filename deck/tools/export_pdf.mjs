// Export deck/slides/NN-*.html to one PDF (vector text), using the installed Edge.
//   node export_pdf.mjs ../slides ../SlideQR_RM.pdf
import { chromium } from "playwright";
import { PDFDocument } from "pdf-lib";
import fs from "fs/promises";
import path from "path";
import { pathToFileURL } from "url";

const [dir = "../slides", out = "../SlideQR_RM.pdf"] = process.argv.slice(2);
const files = (await fs.readdir(dir)).filter(f => /^\d\d-.*\.html$/.test(f)).sort();
const browser = await chromium.launch({ channel: "msedge" });
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
const merged = await PDFDocument.create();
for (const f of files) {
  await page.goto(pathToFileURL(path.resolve(dir, f)).href, { waitUntil: "networkidle" });
  await page.evaluate(() => document.fonts.ready);
  const buf = await page.pdf({ width: "1920px", height: "1080px", printBackground: true, pageRanges: "1" });
  const doc = await PDFDocument.load(buf);
  const [p] = await merged.copyPages(doc, [0]);
  merged.addPage(p);
  console.log(f);
}
await fs.writeFile(out, await merged.save());
await browser.close();
console.log("->", out, files.length, "pages");
