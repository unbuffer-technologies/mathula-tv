#!/bin/bash
# Weekly: crawl the new Isolezwe articles, then rebuild the web index (FineWeb + every crawl file) into a new folder and swap it in, so a running
# dub never reads a half-written index.  The crawl is polite and resumable (see mathula_tv/isolezwe_crawler.py); the section sitemaps list about
# two weeks, so a weekly run overlaps and misses nothing.  Install with:  0 3 * * 0 ~/mathula-tv/scripts/weekly_crawl.sh >> ~/mathula-tv/logs/weekly_crawl.log 2>&1
set -u
cd "$(dirname "$0")/.." || exit 1
exec 9>/tmp/mathula_weekly_crawl.lock
flock -n 9 || { echo "another weekly crawl is running"; exit 0; }
set -a; . ./.env; set +a
PY=.venv/bin/python
RES=working/resources/attestation
echo "== $(date -Is) crawl"
if ! PYTHONPATH=src $PY -m mathula_tv.isolezwe_crawler; then
    echo "crawl failed"; exit 1
fi
echo "== $(date -Is) rebuild"
rm -rf "$RES/web.new"
PYTHONPATH=src $PY -c "
from pathlib import Path
from mathula_tv.attestation import build_web_index
print(build_web_index([Path.home() / 'sadilar/fineweb2_zul/zu_docs.tsv', Path('$RES/crawl/isolezwe.tsv')], Path('$RES/web.new')))
" || { echo "rebuild failed"; exit 1; }
rm -rf "$RES/web.old"
mv "$RES/web" "$RES/web.old" && mv "$RES/web.new" "$RES/web"
echo "== $(date -Is) done"
