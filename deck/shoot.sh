#!/bin/sh
# usage: sh shoot.sh 01 02 ...   (screenshots slides/NN-*.html into shots/)
cd "$(dirname "$0")"
for n in "$@"; do
  for f in slides/$n-*.html; do
    b=$(basename "$f" .html)
    npx -y playwright@1.49.1 screenshot --channel msedge --viewport-size=1920,1080 --wait-for-timeout=2500 \
      "file:///C:/Users/chi/Desktop/chichi/slide%20desktop%20scanner/deck/$f" "shots/$b.png" >/dev/null 2>&1 && echo "$b"
  done
done
