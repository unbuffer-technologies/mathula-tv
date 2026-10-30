import json
import urllib.error

from mathula_tv import isolezwe_crawler as crawler

ROBOTS = "User-agent: *\nAllow: /\nDisallow: /api/\nDisallow: /isolezwe/private/\n\nUser-agent: GPTBot\nDisallow: /\n"
INDEX = ("<sitemapindex><sitemap><loc>https://isolezwe.co.za/sitemap/video/</loc></sitemap>"
         "<sitemap><loc>https://isolezwe.co.za/sitemap/isolezwe/izindaba/</loc></sitemap></sitemapindex>")
SECTION = ("<urlset><url><loc>https://isolezwe.co.za/isolezwe/izindaba/2026-10-30-a/</loc></url>"
           "<url><loc>https://isolezwe.co.za/isolezwe/izindaba/2026-10-29-b/</loc></url>"
           "<url><loc>https://isolezwe.co.za/isolezwe/private/secret/</loc></url></urlset>")


def page(headline, paragraphs):
    ld = json.dumps({"@type": "NewsArticle", "headline": headline, "description": "Isifinyezo sendaba", "datePublished": "2026-10-30T10:00:00Z"})
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    return (f'<html><script type="application/ld+json">{ld}</script><p>Izindaba Ezemidlalo <script>{{"@context":"x"}}</script></p>'
            f"<p>Image: Photographer/ANA</p>{body}</html>")


def fake(pages, calls):
    def fetch(url):
        calls.append(url)
        if url == crawler.ROBOTS:
            return ROBOTS
        if url == crawler.SITEMAP_INDEX:
            return INDEX
        if url.endswith("/sitemap/isolezwe/izindaba/"):
            return SECTION
        if url in pages:
            return pages[url]
        raise urllib.error.HTTPError(url, 404, "not found", None, None)
    return fetch


def test_an_article_is_its_headline_description_and_body_paragraphs_without_navigation_or_captions():
    date, text = crawler.article_text(page("UJohnson noKhan", ["Bobabili babhekene necala lokushaya ngenhloso yokulimaza.", "Short ."]))
    assert date == "2026-10-30"
    assert text.startswith("UJohnson noKhan Isifinyezo sendaba Bobabili")
    assert "Photographer" not in text and "@context" not in text and "Short" not in text


def test_the_crawler_reads_the_section_sitemaps_obeys_robots_waits_between_requests_and_resumes(tmp_path):
    pages = {"https://isolezwe.co.za/isolezwe/izindaba/2026-10-30-a/": page("A", ["Lesi isihloko sokuqala sinezinhlamvu ezanele ."]),
             "https://isolezwe.co.za/isolezwe/izindaba/2026-10-29-b/": page("B", ["Lesi isihloko sesibili sinezinhlamvu ezanele ."])}
    calls, naps = [], []
    stats = crawler.crawl(tmp_path, fetch=fake(pages, calls), sleep=naps.append)
    assert stats["fetched"] == 2 and stats["refused_by_robots"] == 1
    assert "https://isolezwe.co.za/isolezwe/private/secret/" not in calls and "https://isolezwe.co.za/sitemap/video/" not in calls
    assert naps and min(naps) >= crawler.MIN_DELAY
    lines = (tmp_path / "isolezwe.tsv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and lines[0].startswith("isolezwe.co.za\t2026-10-30\tA ")
    again = crawler.crawl(tmp_path, fetch=fake(pages, []), sleep=lambda s: None)
    assert again["fetched"] == 0 and again["skipped_seen"] == 2


def test_the_crawler_stops_when_the_site_pushes_back(tmp_path):
    def fetch(url):
        if url == crawler.ROBOTS:
            return ROBOTS
        if url == crawler.SITEMAP_INDEX:
            return INDEX
        if url.endswith("/sitemap/isolezwe/izindaba/"):
            return SECTION
        raise urllib.error.HTTPError(url, 429, "slow down", None, None)
    stats = crawler.crawl(tmp_path, fetch=fetch, sleep=lambda s: None)
    assert stats["stopped"] == 1 and stats["failed"] == 1 and stats["fetched"] == 0


def test_robots_uses_the_longest_matching_rule_so_a_leading_allow_does_not_hide_a_disallow():
    text = ("User-agent: *\nAllow: /\nAllow: /_next/static/\nDisallow: /_next/\nDisallow: /api/\nDisallow: /search?q=*\n\n"
            "User-agent: GPTBot\nUser-agent: ClaudeBot\nDisallow: /\n")
    ours = crawler.Robots(text, crawler.USER_AGENT)
    assert ours.can_fetch("https://isolezwe.co.za/isolezwe/izindaba/2026-10-30-a/")
    assert ours.can_fetch("https://isolezwe.co.za/_next/static/app.js")
    assert not ours.can_fetch("https://isolezwe.co.za/api/x") and not ours.can_fetch("https://isolezwe.co.za/_next/data/y.json")
    assert not ours.can_fetch("https://isolezwe.co.za/search?q=umbuthano") and ours.can_fetch("https://isolezwe.co.za/search")
    assert not crawler.Robots(text, "GPTBot/1.0").can_fetch("https://isolezwe.co.za/isolezwe/izindaba/2026-10-30-a/")
    assert crawler.Robots("User-agent: *\nCrawl-delay: 7\n", crawler.USER_AGENT).crawl_delay == 7
