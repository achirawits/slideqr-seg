// Print one HTML file to an A4 PDF with the installed Edge.
//   node html_to_pdf.mjs input.html output.pdf
import { chromium } from "playwright";
import path from "path";
import { pathToFileURL } from "url";

const [input, output] = process.argv.slice(2);
const browser = await chromium.launch({ channel: "msedge" });
const page = await browser.newPage();
await page.goto(pathToFileURL(path.resolve(input)).href, { waitUntil: "networkidle" });
await page.evaluate(() => document.fonts.ready);
await page.pdf({ path: output, format: "A4", preferCSSPageSize: true, printBackground: true });
await browser.close();
console.log("->", output);
