# utils/academic_search.py
"""
Real academic search across Semantic Scholar, ArXiv, and Papers With Code.
All APIs are free and require no API key.

Includes persistent caching to avoid redundant API calls.
"""
import time
import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional
from urllib.parse import quote_plus

# Rate limiting: minimum 5s between requests (Semantic Scholar is very strict)
_last_request_time: float = 0.0
_RATE_LIMIT_SECONDS: float = 5.0

# Retry configuration for rate limiting (429 errors)
_MAX_RETRIES: int = 3
_BASE_BACKOFF_SECONDS: float = 5.0

# ══════════════════════════════════════════════════════════════════════
# PERSISTENT CACHE (prevents redundant API calls across sessions)
# ══════════════════════════════════════════════════════════════════════

_cache: Dict[str, List[Dict]] = {}
_cache_file: Optional[Path] = None


def set_cache_file(filepath: Path):
    """Set the cache file path and load existing cache."""
    global _cache, _cache_file
    _cache_file = Path(filepath)
    load_cache()


def load_cache():
    """Load cache from disk."""
    global _cache
    if _cache_file and _cache_file.exists():
        try:
            with open(_cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                _cache = data.get("queries", {})
                print(f"[AcademicSearch] Loaded {len(_cache)} cached queries from {_cache_file}")
        except Exception as e:
            print(f"[AcademicSearch] Failed to load cache: {e}")
            _cache = {}


def save_cache():
    """Save cache to disk."""
    if _cache_file:
        try:
            with open(_cache_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "queries": _cache,
                    "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[AcademicSearch] Failed to save cache: {e}")


def get_cached_results(query: str) -> Optional[List[Dict]]:
    """Get cached results for a query."""
    normalized = query.lower().strip()
    return _cache.get(normalized)


def cache_results(query: str, results: List[Dict]):
    """Cache results for a query and save to disk."""
    normalized = query.lower().strip()
    _cache[normalized] = results
    save_cache()


def _rate_limit():
    """Enforce global rate limit of 1 request per second."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _RATE_LIMIT_SECONDS:
        time.sleep(_RATE_LIMIT_SECONDS - elapsed)
    _last_request_time = time.time()


def _safe_get(url: str, timeout: int = 10, headers: Optional[Dict] = None) -> Optional[object]:
    """
    Perform a GET request with exponential backoff for rate limiting.

    Handles 429 (Too Many Requests) errors by waiting and retrying with
    exponential backoff + jitter to avoid thundering herd.

    Returns response object or None on failure.
    """
    import requests
    import random

    for attempt in range(_MAX_RETRIES):
        try:
            _rate_limit()
            resp = requests.get(url, timeout=timeout, headers=headers or {})

            # Handle rate limiting with exponential backoff
            if resp.status_code == 429:
                if attempt < _MAX_RETRIES - 1:
                    # Exponential backoff with jitter: base * 2^attempt + random(0, 1)
                    wait_time = _BASE_BACKOFF_SECONDS * (2 ** attempt) + random.uniform(0, 1)
                    print(f"  [AcademicSearch] Rate limited (429), waiting {wait_time:.1f}s (attempt {attempt + 1}/{_MAX_RETRIES})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  [AcademicSearch] Rate limited after {_MAX_RETRIES} retries: {url[:60]}...")
                    return None

            resp.raise_for_status()
            return resp

        except requests.exceptions.Timeout:
            if attempt < _MAX_RETRIES - 1:
                wait_time = _BASE_BACKOFF_SECONDS * (2 ** attempt)
                print(f"  [AcademicSearch] Timeout, retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            print(f"  [AcademicSearch] Timeout after {_MAX_RETRIES} retries: {url[:60]}...")
            return None

        except requests.exceptions.RequestException as e:
            print(f"  [AcademicSearch] Request failed: {e}")
            return None

        except Exception as e:
            print(f"  [AcademicSearch] Unexpected error: {e}")
            return None

    return None


def _extract_model_from_title(title: str) -> str:
    """
    Heuristic: extract a model name from a paper title.
    Looks for known patterns like acronyms, names with numbers, etc.
    """
    # Common patterns: "ModelName: subtitle", "ModelName - subtitle"
    colon_match = re.match(r'^([A-Z][A-Za-z0-9\-]+)[\s]*[:\-–]', title)
    if colon_match:
        return colon_match.group(1).strip()

    # Look for capitalized acronyms (2-10 chars) like BERT, GPT, ResNet
    acronym_match = re.search(r'\b([A-Z][A-Za-z0-9]{1,9}(?:Net|BERT|GPT|GAN|ViT|Former|Boost)?)\b', title)
    if acronym_match:
        candidate = acronym_match.group(1)
        # Filter out common non-model words
        skip = {"The", "For", "With", "From", "Using", "Deep", "Learning",
                "Neural", "Network", "Model", "Based", "Data", "New", "Via"}
        if candidate not in skip:
            return candidate

    return ""


# ── Semantic Scholar ──────────────────────────────────────────────

def search_semantic_scholar(query: str, limit: int = 5) -> List[Dict]:
    """
    Search Semantic Scholar (free, no API key required).
    Returns list of dicts with: title, snippet, model_name, source, url, year
    """
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={quote_plus(query)}"
        f"&limit={limit}"
        f"&fields=title,abstract,year,url,citationCount"
    )
    resp = _safe_get(url, timeout=15)
    if resp is None:
        return []

    try:
        data = resp.json()
    except Exception:
        return []

    results = []
    for paper in data.get("data", []):
        title = paper.get("title", "")
        abstract = paper.get("abstract", "") or ""
        results.append({
            "title": title,
            "snippet": abstract,  # Full abstract without truncation
            "model_name": _extract_model_from_title(title),
            "source": "semantic_scholar",
            "url": paper.get("url", ""),
            "year": paper.get("year"),
            "citations": paper.get("citationCount", 0),
        })

    return results


# ── ArXiv ─────────────────────────────────────────────────────────

def search_arxiv(query: str, limit: int = 5) -> List[Dict]:
    """
    Search ArXiv (free, XML API).
    Returns list of dicts with: title, snippet, model_name, source, url, year
    """
    url = (
        f"http://export.arxiv.org/api/query"
        f"?search_query=all:{quote_plus(query)}"
        f"&start=0&max_results={limit}"
        f"&sortBy=relevance&sortOrder=descending"
    )
    resp = _safe_get(url, timeout=15)
    if resp is None:
        return []

    results = []
    try:
        root = ET.fromstring(resp.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            link_el = entry.find("atom:id", ns)
            published_el = entry.find("atom:published", ns)

            title = title_el.text.strip().replace("\n", " ") if title_el is not None else ""
            abstract = summary_el.text.strip().replace("\n", " ") if summary_el is not None else ""  # Full abstract
            link = link_el.text.strip() if link_el is not None else ""
            year = None
            if published_el is not None and published_el.text:
                year_match = re.match(r'(\d{4})', published_el.text)
                if year_match:
                    year = int(year_match.group(1))

            results.append({
                "title": title,
                "snippet": abstract,
                "model_name": _extract_model_from_title(title),
                "source": "arxiv",
                "url": link,
                "year": year,
                "citations": 0,
            })
    except ET.ParseError as e:
        print(f"  [AcademicSearch] ArXiv XML parse error: {e}")

    return results


# ── Papers With Code ──────────────────────────────────────────────

def search_papers_with_code(query: str, limit: int = 5) -> List[Dict]:
    """
    Search Papers With Code (free JSON API).
    Returns list of dicts with: title, snippet, model_name, source, url, year
    """
    url = (
        f"https://paperswithcode.com/api/v1/papers/"
        f"?q={quote_plus(query)}"
        f"&page=1&items_per_page={limit}"
    )
    resp = _safe_get(url, timeout=15)
    if resp is None:
        return []

    try:
        data = resp.json()
    except Exception:
        return []

    papers = data.get("results", [])
    results = []
    for paper in papers:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "") or ""
        url_paper = paper.get("url_abs", "") or paper.get("paper_url", "")
        published = paper.get("published", "")
        year = None
        if published:
            year_match = re.match(r'(\d{4})', str(published))
            if year_match:
                year = int(year_match.group(1))

        results.append({
            "title": title,
            "snippet": abstract,  # Full abstract without truncation
            "model_name": _extract_model_from_title(title),
            "source": "papers_with_code",
            "url": url_paper,
            "year": year,
            "citations": 0,
        })

    return results


# ── Unified Search ────────────────────────────────────────────────

def search_all_academic_sources(
    query: str,
    limit_per_source: int = 5,
    use_cache: bool = True,
) -> List[Dict]:
    """
    Search all 3 academic sources and deduplicate by normalized title.
    Returns unified, deduplicated list sorted by citation count (desc).

    Uses persistent cache to avoid redundant API calls.
    """
    # Check cache first
    if use_cache:
        cached = get_cached_results(query)
        if cached is not None:
            print(f"  [AcademicSearch] Cache hit for: {query[:50]}...")
            return cached

    all_results: List[Dict] = []

    # Query each source (rate-limited internally)
    for search_fn in (search_semantic_scholar, search_arxiv, search_papers_with_code):
        try:
            results = search_fn(query, limit=limit_per_source)
            all_results.extend(results)
        except Exception as e:
            print(f"  [AcademicSearch] Source failed: {e}")
            continue

    # Deduplicate by normalized title
    seen_titles = set()
    unique_results = []
    for r in all_results:
        norm_title = re.sub(r'[^a-z0-9]', '', r.get("title", "").lower())
        if norm_title and norm_title not in seen_titles:
            seen_titles.add(norm_title)
            unique_results.append(r)

    # Sort by citations descending, then by year descending
    unique_results.sort(
        key=lambda x: (x.get("citations", 0), x.get("year") or 0),
        reverse=True,
    )

    # Cache the results
    if use_cache and unique_results:
        cache_results(query, unique_results)

    return unique_results
