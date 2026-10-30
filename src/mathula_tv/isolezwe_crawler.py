"""A polite crawler for Isolezwe, so the attestation index can follow the Zulu the audience will read when they follow a story up.

Rules it keeps: it identifies itself (user agent with a contact address), obeys robots.txt and any crawl delay, waits at least ``MIN_DELAY`` seconds
between requests, reads the sitemaps the site publishes (they list roughly the last two weeks per section), skips what it already has, and stops
when the site pushes back (403/429 or repeated errors).  The text is kept for the local word-count index only; nothing is republished.

The site's terms prohibit harvesting data for commercial gain and ask for written permission for commercial use; run it only once that is settled.
Output: ``<work_dir>/resources/attestation/crawl/isolezwe.tsv`` with ``host<TAB>date<TAB>text`` lines (what ``attestation build-web`` reads) and
``isolezwe.urls`` (one fetched URL per line, for resuming).
"""
from __future__ import annotations

import html
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

HOST = "isolezwe.co.za"
ROBOTS = f"https://{HOST}/robots.txt"
SITEMAP_INDEX = f"https://{HOST}/sitemap/"
USER_AGENT = "MathulaTV-research/0.1 (+mailto:streamspn@gmail.com)"
MIN_DELAY = 2.0
STOP_AFTER_FAILURES = 3
MIN_PARAGRAPH = 25

Fetch = Callable[[str], str]


def http_fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Language": "zu,en;q=0.5"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", errors="replace")


class Robots:
    """robots.txt for one user agent with the standard's precedence: the longest matching rule wins, allow wins a tie, ``*`` and ``$`` work.  (The
    standard library parser takes the first match, so a leading ``Allow: /`` would hide every ``Disallow`` after it.)"""

    def __init__(self, text: str, agent: str) -> None:
        token = agent.split("/")[0].split()[0].lower()
        groups: list[tuple[list[str], list[tuple[bool, str]], list[float]]] = []
        collecting = False
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if ":" not in line:
                continue
            field, value = (part.strip() for part in line.split(":", 1))
            field = field.lower()
            if field == "user-agent":
                if not collecting:
                    groups.append(([], [], []))
                groups[-1][0].append(value.lower())
                collecting = True
                continue
            collecting = False
            if not groups:
                continue
            if field in ("allow", "disallow") and value:
                groups[-1][1].append((field == "allow", value))
            elif field == "crawl-delay":
                try:
                    groups[-1][2].append(float(value))
                except ValueError:
                    pass
        named = [g for g in groups if any(a != "*" and a in token for a in g[0])]
        chosen = named or [g for g in groups if "*" in g[0]]
        self._rules = [(allow, self._pattern(path)) for g in chosen for allow, path in g[1]]
        self.crawl_delay = max((d for g in chosen for d in g[2]), default=0.0)

    @staticmethod
    def _pattern(path: str) -> tuple[int, "re.Pattern[str]"]:
        body = re.escape(path.rstrip("$")).replace(r"\*", ".*")
        return len(path.replace("*", "")), re.compile(body + ("$" if path.endswith("$") else ""))

    def can_fetch(self, url: str) -> bool:
        parsed = urlparse(url)
        path = (parsed.path or "/") + ("?" + parsed.query if parsed.query else "")
        best_length, allowed = -1, True
        for allow, (length, pattern) in self._rules:
            if pattern.match(path) and (length > best_length or (length == best_length and allow)):
                best_length, allowed = length, allow
        return allowed


def sitemap_urls(xml: str) -> list[str]:
    return re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", xml)


def article_urls(index_xml: str, fetch: Fetch, allowed: Callable[[str], bool], pause: Callable[[], None]) -> list[str]:
    """The article URLs of the Zulu news sections (``/isolezwe/<section>/`` sitemaps), newest first as the site lists them."""
    found: list[str] = []
    for sitemap in sitemap_urls(index_xml):
        if "/sitemap/isolezwe/" not in sitemap or not allowed(sitemap):
            continue
        pause()
        for url in sitemap_urls(fetch(sitemap)):
            if urlparse(url).netloc == HOST and "/isolezwe/" in url and url not in found:
                found.append(url)
    return found


def article_text(page: str) -> tuple[str, str]:
    """(publication date, text) of an article page: the headline and description from its JSON-LD, then the paragraphs of the body."""
    date, headline, description = "", "", ""
    for block in re.findall(r'<script type="application/ld\+json"[^>]*>(.*?)</script>', page, re.S):
        try:
            data = json.loads(block)
        except ValueError:
            continue
        for item in data if isinstance(data, list) else data.get("@graph", [data]):
            if isinstance(item, dict) and item.get("@type") in ("NewsArticle", "Article"):
                date = str(item.get("datePublished", ""))[:10]
                headline, description = str(item.get("headline", "")), str(item.get("description", ""))
    paragraphs: list[str] = []
    for raw in re.findall(r"<p[^>]*>(.*?)</p>", page, re.S):
        text = " ".join(html.unescape(re.sub(r"<[^>]+>", "", raw)).split())
        if len(text) >= MIN_PARAGRAPH and '{"@context"' not in text and not text.startswith(("Image:", "Picture:", "Photo:")):
            paragraphs.append(text)
    return date, " ".join(part for part in [headline, description, *paragraphs] if part)


def crawl(out_dir: Path, *, fetch: Fetch = http_fetch, sleep: Callable[[float], None] = time.sleep, limit: int | None = None,
          user_agent: str = USER_AGENT) -> dict[str, int]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_file, text_file = out_dir / "isolezwe.urls", out_dir / "isolezwe.tsv"
    seen = set(seen_file.read_text(encoding="utf-8").split()) if seen_file.is_file() else set()
    robots = Robots(fetch(ROBOTS), user_agent)
    delay = max(MIN_DELAY, robots.crawl_delay)
    allowed = robots.can_fetch
    pause = lambda: sleep(delay)                                                              # noqa: E731
    stats = {"fetched": 0, "skipped_seen": 0, "refused_by_robots": 0, "failed": 0, "stopped": 0}
    if not allowed(SITEMAP_INDEX):
        stats["refused_by_robots"] += 1
        return stats
    urls = article_urls(fetch(SITEMAP_INDEX), fetch, allowed, pause)
    failures = 0
    with seen_file.open("a", encoding="utf-8", newline="\n") as seen_out, text_file.open("a", encoding="utf-8", newline="\n") as text_out:
        for url in urls:
            if limit is not None and stats["fetched"] >= limit:
                break
            if url in seen:
                stats["skipped_seen"] += 1
                continue
            if not allowed(url):
                stats["refused_by_robots"] += 1
                continue
            pause()
            try:
                page = fetch(url)
            except urllib.error.HTTPError as error:
                failures += 1
                stats["failed"] += 1
                if error.code in (403, 429) or failures >= STOP_AFTER_FAILURES:
                    stats["stopped"] = 1
                    break
                continue
            except (urllib.error.URLError, TimeoutError, OSError):
                failures += 1
                stats["failed"] += 1
                if failures >= STOP_AFTER_FAILURES:
                    stats["stopped"] = 1
                    break
                continue
            failures = 0
            date, text = article_text(page)
            seen_out.write(url + "\n")
            seen_out.flush()
            if text:
                text_out.write(f"{HOST}\t{date}\t{text}\n")
                text_out.flush()
            stats["fetched"] += 1
    return stats


def main(argv: list[str] | None = None) -> int:
    import argparse

    from .attestation import resources_dir
    from .config import load_settings

    parser = argparse.ArgumentParser(prog="mathula_tv.isolezwe_crawler")
    parser.add_argument("--limit", type=int, default=None, help="fetch at most this many new articles")
    args = parser.parse_args(argv)
    print(crawl(resources_dir(load_settings().work_dir) / "crawl", limit=args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
