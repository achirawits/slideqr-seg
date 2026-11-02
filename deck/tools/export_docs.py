"""Build the rehearsal and paper-reading PDFs from the markdown docs.

  python export_docs.py <work_dir>
  node html_to_pdf.mjs <work_dir>/rehearsal.html ../SlideQR_ซ้อมบทพูด.pdf
  node html_to_pdf.mjs <work_dir>/papers.html ../SlideQR_คู่มืออ่าน_paper.pdf
  node html_to_pdf.mjs <work_dir>/explain.html ../SlideQR_อธิบายตั้งแต่ศูนย์.pdf

Run from deck/tools after `sh deck/shoot.sh` so deck/shots holds every page. Needs `markdown`.
"""
import glob
import os
import re
import sys

import markdown

work = sys.argv[1]
root = os.path.abspath('../..')
deck = os.path.join(root, 'deck')
url = lambda p: 'file:///' + os.path.abspath(p).replace(os.sep, '/')
CSS = '''
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;600;700&family=Noto+Serif+Thai:wght@600;700&display=swap');
@page { size: A4; margin: 16mm 15mm; }
body { font-family: 'IBM Plex Sans Thai', sans-serif; font-size: 11.5pt; line-height: 1.6; color: #1d1d1b; }
h1, h2, h3 { font-family: 'Noto Serif Thai', serif; line-height: 1.3; }
h1 { font-size: 22pt; margin: 0 0 8pt; }
h2 { font-size: 16pt; color: #a8361f; margin: 18pt 0 6pt; break-after: avoid; }
h3 { font-size: 13pt; margin: 14pt 0 4pt; break-after: avoid; }
ul, ol { margin: 4pt 0 6pt; padding-left: 18pt; }
li { margin: 2pt 0; }
table { border-collapse: collapse; width: 100%; font-size: 10pt; margin: 6pt 0; }
th, td { border-bottom: 1px solid #ccc; padding: 3pt 5pt; text-align: left; vertical-align: top; }
blockquote { margin: 6pt 0; padding: 6pt 10pt; background: #f4e8e4; border: 0; }
p + p { margin-top: 5pt; }
hr { border: 0; border-top: 1px solid #ccc; margin: 10pt 0; }
figure { margin: 6pt 0 8pt; break-inside: avoid; }
figure img { max-width: 100%; max-height: 62mm; border: 1px solid #ddd; }
figure img[src*='shots'] { max-height: 115mm; }
figcaption { font-size: 9.5pt; color: #5c5c58; }
.slide { break-inside: avoid; margin-bottom: 8mm; }
.slide img { width: 100%; border: 1px solid #ccc; }
.slide h3 { margin-top: 0; }
.say { font-size: 13pt; line-height: 1.65; margin: 5pt 0; }
.kw { color: #a8361f; font-weight: 600; font-size: 11pt; }
.time { color: #5c5c58; font-size: 10pt; }
.cover { margin-bottom: 10mm; }
'''


def page(title, body):
    return f'<!doctype html><html lang="th"><head><meta charset="utf-8"><title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>'


def md(text):
    # python-markdown needs 4-space nesting and a blank line before a list
    out = []
    for line in text.split('\n'):
        indent = len(line) - len(line.lstrip(' '))
        item = re.match(r' *(- |\d+\. )', line)
        if item and indent:
            line = ' ' * (indent * 2) + line.lstrip(' ')
        elif item and out and out[-1].strip() and not re.match(r' *(- |\d+\. )', out[-1]):
            out.append('')
        out.append(line)
    return markdown.markdown('\n'.join(out), extensions=['tables', 'sane_lists'])

# rehearsal book: slide picture, spoken text, keywords, then the timing table and Q&A
src = open(os.path.join(root, 'docs/rm/06_script_final.md'), encoding='utf-8').read().replace('\r\n', '\n')
slides = [os.path.basename(p)[:-5] for p in glob.glob(os.path.join(deck, 'slides', '[0-9][0-9]-*.html'))]
shots = {n[:2]: os.path.join(deck, 'shots', n + '.png') for n in slides}
times = {m[0]: (m[1], m[2]) for m in re.findall(r'^\| (\d+) \| .*? \| \d+ \| ([\d:]+) \| ([\d:]+) \|$', src, re.M)}
head = src[:src.index('## Research Problem')]
tail = src[src.index('## ประมาณเวลา'):]
body = f'<div class="cover">{md(head)}</div>'
for k, title, text in re.findall(r'### หน้า (\d+) · (.*)\n((?:.*\n)*?)(?=### หน้า|---\n|## )', src[:src.index('## ประมาณเวลา')]):
    say = re.search(r'\*\*บทพูด:\*\* (.*)', text)
    kw = re.search(r'\*\*คำหลัก:\*\* (.*)', text)
    img = f'<img src="{url(shots[k.zfill(2)])}">' if k.zfill(2) in shots else ''
    t = times.get(k)
    body += f'<div class="slide"><h3>หน้า {k} · {title.strip()}</h3>{img}'
    body += f'<p class="say">{say.group(1) if say else text.strip()}</p>'
    if kw:
        body += f'<p class="kw">{kw.group(1)}</p>'
    if t:
        body += f'<p class="time">ถึงท้ายหน้านี้: {t[0]} ถึง {t[1]}</p>'
    body += '</div>'
body += md(tail)
open(os.path.join(work, 'rehearsal.html'), 'w', encoding='utf-8').write(page('ซ้อมบทพูด', body))

# paper guide with the marked tables from each paper
src = open(os.path.join(root, 'docs/rm/07_paper_study_guide.md'), encoding='utf-8').read().replace('\r\n', '\n')


def fig(m):
    return f'<figure><img src="{url(os.path.join(deck, "assets", m.group(1)))}"><figcaption>{m.group(2)}</figcaption></figure>'


html = md(re.sub(r'\{\{fig:(.*?)\|(.*?)\}\}', fig, src))
open(os.path.join(work, 'papers.html'), 'w', encoding='utf-8').write(page('คู่มืออ่าน paper', html))

# beginner explanation of the whole project
src = open(os.path.join(root, 'docs/rm/08_explain_from_zero.md'), encoding='utf-8').read().replace('\r\n', '\n')
html = md(re.sub(r'\{\{fig:(.*?)\|(.*?)\}\}', fig, src))
open(os.path.join(work, 'explain.html'), 'w', encoding='utf-8').write(page('อธิบายงานตั้งแต่ศูนย์', html))
print('ok', len(shots))
