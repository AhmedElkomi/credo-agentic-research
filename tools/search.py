import os
import re
import math
from datetime import datetime, timezone
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlparse,
    urlunparse,
)

from dotenv import load_dotenv
from tavily import TavilyClient


# ============================================================
# DOMAIN GROUPS
# ============================================================

PEER_REVIEWED_DOMAINS = {
    "aclanthology.org",
    "ieeexplore.ieee.org",
    "dl.acm.org",
    "proceedings.neurips.cc",
    "neurips.cc",
    "aaai.org",
    "jmlr.org",
    "proceedings.mlr.press",
    "pmlr.press",
    "springer.com",
    "link.springer.com",
    "sciencedirect.com",
    "mdpi.com",
    "nature.com",
    "arxiv.org",
    "ar5iv.labs.arxiv.org",
    "openreview.net",
}

ACADEMIC_INDEX_DOMAINS = {
    "semanticscholar.org",
    "paperswithcode.com",
    "alphaxiv.org",
}

OFFICIAL_DOMAINS = {
    "huggingface.co",
    "docs.anyscale.com",
    "developers.google.com",
    "cloud.google.com",
    "docs.aws.amazon.com",
    "learn.microsoft.com",
}

VENDOR_DOMAINS = {
    "ibm.com",
    "aws.amazon.com",
    "cloud.google.com",
    "microsoft.com",
    "google.com",
    "qdrant.tech",
    "meilisearch.com",
    "evidentlyai.com",
    "patronus.ai",
    "galileo.ai",
    "comet.com",
    "toloka.ai",
    "getmaxim.ai",
    "deepchecks.com",
    "atlan.com",
}

TECHNICAL_BLOG_DOMAINS = {
    "deconvoluteai.com",
    "kili-technology.com",
    "mbrenndoerfer.com",
    "karini.ai",
    "treysaddler.com",
}

MEDIUM_DOMAINS = {
    "medium.com",
}


# ============================================================
# SOURCE ROLE TERMS
# ============================================================

SURVEY_TERMS = {
    "survey",
    "systematic review",
    "literature review",
    "systematic literature review",
    "review of",
    "review on",
    "overview of",
    "state of the art",
    "mapping study",
}

PRIMARY_RESEARCH_TERMS = {
    "benchmark",
    "benchmarking",
    "dataset",
    "evaluation framework",
    "empirical study",
    "experiments",
    "experimental",
    "introducing",
    "propose",
    "proposes",
    "we present",
    "we introduce",
    "we propose",
}


# ============================================================
# URL / DOMAIN NORMALIZATION
# ============================================================

TRACKING_PARAMETERS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
    "ref",
}


def get_domain(url: str) -> str:
    """Extract a normalized hostname from a URL."""

    if not url:
        return ""

    try:
        hostname = urlparse(url).hostname

        if not hostname:
            return ""

        hostname = hostname.lower().strip()

        if hostname.startswith("www."):
            hostname = hostname[4:]

        return hostname

    except Exception:
        return ""


def canonicalize_url(url: str) -> str:
    """
    Canonicalize a URL for duplicate detection.

    Removes common tracking parameters while preserving
    meaningful query parameters such as OpenReview's id.
    """

    if not url:
        return ""

    try:
        parsed = urlparse(url)

        query_items = []

        for key, value in parse_qsl(
            parsed.query,
            keep_blank_values=True,
        ):
            if key.lower() not in TRACKING_PARAMETERS:
                query_items.append(
                    (key, value)
                )

        normalized_query = urlencode(
            sorted(query_items)
        )

        return urlunparse(
            (
                parsed.scheme.lower(),
                get_domain(url),
                parsed.path.rstrip("/"),
                "",
                normalized_query,
                "",
            )
        )

    except Exception:
        return url


# ============================================================
# TITLE / ENTITY NORMALIZATION
# ============================================================

def normalize_title(title: str) -> str:
    """Normalize titles for research-entity matching."""

    if not title:
        return ""

    title = title.strip()

    # Common search-result prefixes.
    title = re.sub(
        r"^\s*[\[\(]?(?:pdf|html)[\]\)]?\s*",
        "",
        title,
        flags=re.IGNORECASE,
    )

    # Common mirror/index suffixes.
    title = re.sub(
        r"\s*\|\s*(semantic scholar|alphaxiv|researchgate)\s*$",
        "",
        title,
        flags=re.IGNORECASE,
    )

    # arXiv identifier prefix.
    title = re.sub(
        r"^\s*\[?\d{4}\.\d{4,5}(?:v\d+)?\]?\s*",
        "",
        title,
        flags=re.IGNORECASE,
    )

    title = title.lower()

    title = re.sub(
        r"[^a-z0-9\s]",
        " ",
        title,
    )

    title = re.sub(
        r"\s+",
        " ",
        title,
    )

    return title.strip()


def title_key(title: str) -> str | None:
    """Return a title-based entity key."""

    normalized = normalize_title(title)

    if not normalized:
        return None

    return f"title:{normalized}"


# ============================================================
# RESEARCH IDENTIFIERS
# ============================================================

def extract_arxiv_id(url: str) -> str | None:
    """
    Extract an arXiv identifier.

    Supports:
    - arxiv.org/abs/2408.08067
    - arxiv.org/html/2408.08067v2
    - arxiv.org/pdf/2408.08067.pdf
    - ar5iv.labs.arxiv.org/html/2408.08067
    - alphaxiv.org/abs/2408.08067
    """

    if not url:
        return None

    match = re.search(
        r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?",
        url,
    )

    if match:
        return match.group(1)

    match = re.search(
        r"arXiv[:\s]+(\d{4}\.\d{4,5})",
        url,
        re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return None


def extract_doi(text: str) -> str | None:
    """Extract a DOI from a URL or text."""

    if not text:
        return None

    match = re.search(
        r"(?:doi\.org/|doi:\s*|doi\s+)"
        r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
        text,
        re.IGNORECASE,
    )

    if match:
        return match.group(1).rstrip(
            ").,;]"
        )

    return None


def extract_acl_id(url: str) -> str | None:
    """Extract ACL Anthology identifier."""

    if not url:
        return None

    if "aclanthology.org" not in url.lower():
        return None

    match = re.search(
        r"aclanthology\.org/([^/#?]+)",
        url,
        re.IGNORECASE,
    )

    if not match:
        return None

    identifier = match.group(1)

    identifier = re.sub(
        r"\.(pdf|html?)$",
        "",
        identifier,
        flags=re.IGNORECASE,
    )

    return identifier


def extract_openreview_id(url: str) -> str | None:
    """Extract OpenReview paper ID."""

    if not url:
        return None

    if "openreview.net" not in url.lower():
        return None

    match = re.search(
        r"[?&]id=([^&#]+)",
        url,
        re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return None


def extract_identifiers(
    url: str,
    title: str,
    content: str,
) -> dict:
    """Extract all useful identifiers from a search result."""

    combined_text = (
        f"{url}\n"
        f"{title}\n"
        f"{content[:3000]}"
    )

    return {
        "doi": extract_doi(
            combined_text
        ),
        "arxiv_id": extract_arxiv_id(
            combined_text
        ),
        "acl_id": extract_acl_id(
            url
        ),
        "openreview_id": extract_openreview_id(
            url
        ),
    }


# ============================================================
# SOURCE CLASSIFICATION
# ============================================================

def classify_source_type(
    url: str,
) -> str:
    """
    Classify the source into a broad category.
    """

    domain = get_domain(url)

    if domain in PEER_REVIEWED_DOMAINS:
        return "academic"

    if domain in ACADEMIC_INDEX_DOMAINS:
        return "academic"

    if domain in OFFICIAL_DOMAINS:
        return "official"

    if domain in MEDIUM_DOMAINS:
        return "medium"

    if domain in TECHNICAL_BLOG_DOMAINS:
        return "technical_blog"

    if domain in VENDOR_DOMAINS:
        return "vendor"

    if domain == "github.com" or domain.endswith(
        ".github.com"
    ):
        return "technical_repository"

    return "web"


def _title_contains_term(
    title: str,
    terms: set[str],
) -> bool:
    """Check whether a normalized title contains a role signal."""

    normalized = normalize_title(title)

    return any(
        term in normalized
        for term in terms
    )


def classify_source_role(
    url: str,
    source_type: str,
    title: str = "",
    content: str = "",
) -> str:
    """
    Determine the research role of a source.

    Important distinction:

    - arXiv can host either primary research or surveys.
    - AlphaXiv / Semantic Scholar / Papers With Code are treated
      as academic indexes/mirrors, not primary publication venues.
    - Official documentation is an official source.
    - Blogs/vendors are secondary sources.
    """

    domain = get_domain(url)

    # Academic indexes / mirrors are discovery sources.
    if domain in ACADEMIC_INDEX_DOMAINS:
        return "secondary_academic"

    # Official documentation.
    if source_type == "official":
        return "official"

    if source_type == "academic":

        # Detect surveys/reviews from the paper title.
        if _title_contains_term(
            title,
            SURVEY_TERMS,
        ):
            return "survey_review"

        # Strong title signals for original research.
        if _title_contains_term(
            title,
            PRIMARY_RESEARCH_TERMS,
        ):
            return "primary"

        # Look for abstract-style signals in the retrieved content.
        normalized_content = content.lower()[:2500]

        if any(
            phrase in normalized_content
            for phrase in [
                "we propose",
                "we introduce",
                "we present",
                "we evaluate",
                "we benchmark",
                "our experiments",
                "our results",
            ]
        ):
            return "primary"

        # Academic papers without enough evidence to classify them
        # as original research are treated conservatively.
        return "secondary_academic"

    if source_type in {
        "vendor",
        "technical_blog",
        "medium",
        "web",
        "technical_repository",
    }:
        return "secondary"

    return "secondary"


# ============================================================
# METADATA EXTRACTION
# ============================================================

def extract_year(
    result: dict,
) -> int | None:
    """
    Extract publication year conservatively.

    Priority:
    1. Tavily published_date
    2. Explicit year in title
    3. Published/updated date in retrieved content
    """

    published_date = result.get(
        "published_date"
    )

    if published_date:
        match = re.search(
            r"\b(20\d{2})\b",
            str(published_date),
        )

        if match:
            return int(
                match.group(1)
            )

    title = str(
        result.get(
            "title",
            "",
        )
    )

    title_match = re.search(
        r"\b(20\d{2})\b",
        title,
    )

    if title_match:
        return int(
            title_match.group(1)
        )

    content = str(
        result.get(
            "content",
            "",
        )
    )[:2000]

    patterns = [
        r"published\s*[:\-]?\s*(20\d{2})",
        r"updated\s*[:\-]?\s*(20\d{2})",
        r"posted\s*[:\-]?\s*(20\d{2})",
        r"\b(20\d{2})-\d{2}-\d{2}\b",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            content,
            re.IGNORECASE,
        )

        if match:
            return int(
                match.group(1)
            )

    return None


def extract_citation_count(
    result: dict,
) -> int | None:
    """Extract only explicitly stated citation counts."""

    fields_to_check = [
        result.get("content", ""),
        result.get("title", ""),
    ]

    patterns = [
        r"cited by\s+([\d,]+)",
        r"([\d,]+)\s+citation[s]?",
        r"citation[s]?:\s*([\d,]+)",
    ]

    for text in fields_to_check:
        if not text:
            continue

        text = str(text)

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                re.IGNORECASE,
            )

            if not match:
                continue

            try:
                return int(
                    match.group(1).replace(
                        ",",
                        "",
                    )
                )
            except ValueError:
                continue

    return None


# ============================================================
# QUALITY SCORING
# ============================================================

def venue_score(
    url: str,
    source_type: str,
) -> float:
    """Score venue/publication reliability."""

    domain = get_domain(url)

    if domain in PEER_REVIEWED_DOMAINS:
        return 1.00

    if domain in ACADEMIC_INDEX_DOMAINS:
        return 0.65

    if source_type == "official":
        return 0.85

    if source_type == "technical_repository":
        return 0.55

    if source_type == "technical_blog":
        return 0.45

    if source_type == "vendor":
        return 0.40

    if source_type == "medium":
        return 0.15

    return 0.25


def citation_score(
    citation_count: int | None,
) -> float:
    """Convert explicit citation count into a bounded score."""

    if citation_count is None:
        return 0.50

    if citation_count <= 0:
        return 0.20

    score = math.log10(
        citation_count + 1
    ) / 3.0

    return max(
        0.20,
        min(
            1.00,
            score,
        ),
    )


def recency_score(
    published_year: int | None,
) -> float:
    """Give a modest recency signal."""

    if published_year is None:
        return 0.50

    current_year = datetime.now(
        timezone.utc
    ).year

    age = max(
        0,
        current_year - published_year,
    )

    if age <= 1:
        return 1.00

    if age <= 3:
        return 0.90

    if age <= 5:
        return 0.80

    if age <= 8:
        return 0.65

    if age <= 12:
        return 0.50

    return 0.40


def primary_evidence_score(
    source_role: str,
) -> float:
    """Score directness of the source."""

    if source_role == "primary":
        return 1.00

    if source_role == "official":
        return 0.85

    if source_role == "survey_review":
        return 0.65

    if source_role == "secondary_academic":
        return 0.60

    return 0.30


def score_source_quality(
    source: dict,
) -> float:
    """
    Deterministic source quality score.

    Components:
    - source type: 45%
    - venue: 20%
    - explicit citations: 15%
    - recency: 10%
    - research role: 10%
    """

    source_type = source.get(
        "source_type",
        "web",
    )

    source_role = source.get(
        "source_role",
        "secondary",
    )

    source_type_base = {
        "academic": 1.00,
        "official": 0.85,
        "technical_repository": 0.60,
        "technical_blog": 0.45,
        "vendor": 0.40,
        "web": 0.25,
        "medium": 0.15,
    }.get(
        source_type,
        0.25,
    )

    venue = venue_score(
        source.get("url", ""),
        source_type,
    )

    citations = citation_score(
        source.get("citation_count")
    )

    recency = recency_score(
        source.get("published_year")
    )

    primary_signal = primary_evidence_score(
        source_role
    )

    score = (
        source_type_base * 0.45
        + venue * 0.20
        + citations * 0.15
        + recency * 0.10
        + primary_signal * 0.10
    )

    return round(
        score,
        4,
    )


def is_strong_source(
    source: dict,
    threshold: float = 0.70,
) -> bool:
    """
    Determine whether a source qualifies as strong evidence.

    Strong roles:
    - primary research
    - official documentation

    Surveys and academic indexes are NOT treated as primary
    evidence even when hosted on academic domains.
    """

    role = source.get(
        "source_role",
        "secondary",
    )

    quality = float(
        source.get(
            "quality_score",
            0.0,
        )
    )

    return (
        quality >= threshold
        and role in {
            "primary",
            "official",
        }
    )


# ============================================================
# ENTITY KEYS
# ============================================================

def build_entity_keys(
    source: dict,
) -> list[str]:
    """
    Build multiple identity keys for robust entity resolution.
    """

    keys = []

    source_type = source.get(
        "source_type"
    )

    title = source.get(
        "title",
        "",
    )

    identifiers = [
        source.get("doi"),
        source.get("arxiv_id"),
        source.get("acl_id"),
        source.get("openreview_id"),
    ]

    if source_type == "academic":
        key = title_key(title)

        if key:
            keys.append(key)

    for identifier in identifiers:
        if not identifier:
            continue

        identifier = str(
            identifier
        ).strip().lower()

        if identifier.startswith(
            "10."
        ):
            keys.append(
                f"doi:{identifier}"
            )

        elif re.match(
            r"^\d{4}\.\d{4,5}$",
            identifier,
        ):
            keys.append(
                f"arxiv:{identifier}"
            )

        else:
            keys.append(
                f"id:{identifier}"
            )

    canonical_url = canonicalize_url(
        source.get(
            "url",
            "",
        )
    )

    if canonical_url:
        keys.append(
            f"url:{canonical_url}"
        )

    return list(
        dict.fromkeys(keys)
    )


def extract_paper_identifier(
    url: str,
    title: str,
    content: str = "",
) -> str:
    """
    Return a stable research entity ID.

    For academic papers, normalized title is preferred so
    arXiv/PDF/mirror representations can collapse together.
    """

    source_type = classify_source_type(
        url
    )

    if source_type == "academic":
        key = title_key(title)

        if key:
            return key

    identifiers = extract_identifiers(
        url,
        title,
        content,
    )

    if identifiers.get("doi"):
        return (
            "doi:"
            + identifiers["doi"].lower()
        )

    if identifiers.get("arxiv_id"):
        return (
            "arxiv:"
            + identifiers["arxiv_id"]
        )

    if identifiers.get("acl_id"):
        return (
            "acl:"
            + identifiers["acl_id"].lower()
        )

    if identifiers.get("openreview_id"):
        return (
            "openreview:"
            + identifiers["openreview_id"].lower()
        )

    canonical_url = canonicalize_url(
        url
    )

    return (
        f"url:{canonical_url}"
    )


# ============================================================
# EVIDENCE EXTRACTION
# ============================================================

BOILERPLATE_PATTERNS = [
    "sign in",
    "log in",
    "subscribe",
    "cookie",
    "accept cookies",
    "privacy policy",
    "terms of service",
    "all rights reserved",
    "table of contents",
    "skip to",
    "menu",
    "navigation",
]


def _looks_like_boilerplate(
    text: str,
) -> bool:
    """Detect obvious navigation/page boilerplate."""

    lowered = text.lower()

    if any(
        pattern in lowered
        for pattern in BOILERPLATE_PATTERNS
    ):
        return True

    return False


def _clean_evidence_text(
    content: str,
) -> str:
    """Normalize retrieved evidence text."""

    text = re.sub(
        r"\s+",
        " ",
        str(content),
    ).strip()

    return text


def extract_evidence_span(
    content: str,
    max_length: int = 900,
) -> str:
    """
    Extract a compact retrieved evidence span.

    Important:
    This text comes from Tavily's retrieved content and is NOT
    guaranteed to be a verbatim quotation from the original page.

    The function prefers a sentence boundary and attempts to avoid
    obvious page boilerplate.
    """

    if not content:
        return ""

    cleaned = _clean_evidence_text(
        content
    )

    if not cleaned:
        return ""

    # If the beginning is obvious boilerplate, search for a better
    # sentence later in the retrieved content.
    if _looks_like_boilerplate(
        cleaned[:500]
    ):
        sentences = re.split(
            r"(?<=[.!?])\s+",
            cleaned,
        )

        useful_sentences = [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
            and not _looks_like_boilerplate(
                sentence
            )
        ]

        if useful_sentences:
            cleaned = " ".join(
                useful_sentences
            )

    # Prefer sentence boundaries rather than cutting through a word
    # or paragraph.
    if len(cleaned) <= max_length:
        return cleaned

    candidate = cleaned[:max_length]

    boundary = max(
        candidate.rfind(". "),
        candidate.rfind("? "),
        candidate.rfind("! "),
    )

    if boundary >= int(
        max_length * 0.55
    ):
        return candidate[
            : boundary + 1
        ].strip()

    return candidate.rstrip() + "..."


# ============================================================
# SEARCH
# ============================================================

def search_web(
    query: str,
    max_results: int = 5,
) -> list[dict]:
    """Search the web and return structured research evidence."""

    load_dotenv()

    api_key = os.getenv(
        "TAVILY_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY is missing. "
            "Check your .env file."
        )

    client = TavilyClient(
        api_key=api_key
    )

    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
        include_answer=False,
    )

    retrieved_at = datetime.now(
        timezone.utc
    ).isoformat()

    sources = []

    for result in response.get(
        "results",
        [],
    ):
        title = str(
            result.get(
                "title",
                "",
            )
        )

        url = str(
            result.get(
                "url",
                "",
            )
        )

        content = str(
            result.get(
                "content",
                "",
            )
        )

        source_type = classify_source_type(
            url
        )

        source_role = classify_source_role(
            url,
            source_type,
            title,
            content,
        )

        identifiers = extract_identifiers(
            url,
            title,
            content,
        )

        source = {
            "title": title,
            "url": url,
            "canonical_url": canonicalize_url(
                url
            ),
            "snippet": content,
            "evidence": [],
            "evidence_span": extract_evidence_span(
                content
            ),
            "source_type": source_type,
            "source_role": source_role,
            "paper_id": extract_paper_identifier(
                url,
                title,
                content,
            ),
            "doi": identifiers.get(
                "doi"
            ),
            "arxiv_id": identifiers.get(
                "arxiv_id"
            ),
            "acl_id": identifiers.get(
                "acl_id"
            ),
            "openreview_id": identifiers.get(
                "openreview_id"
            ),
            "published_year": extract_year(
                result
            ),
            "citation_count": extract_citation_count(
                result
            ),
            "retrieved_at": retrieved_at,
            "search_query": query,
            "retrieval_score": result.get(
                "score"
            ),
        }

        source["entity_keys"] = build_entity_keys(
            source
        )

        source["quality_score"] = score_source_quality(
            source
        )

        source["is_strong"] = is_strong_source(
            source
        )

        if source["evidence_span"]:
            source["evidence"].append(
                {
                    "quote": source[
                        "evidence_span"
                    ],
                    "url": url,
                    "title": title,
                    "verbatim": False,
                    "retrieved_at": retrieved_at,
                }
            )

        sources.append(
            source
        )

    return sources