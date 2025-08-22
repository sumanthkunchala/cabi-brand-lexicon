#!/usr/bin/env python3
"""
Extract 2–4 word brand phrases from catalog PDFs, merge into brand_lexicon.json.
- Groups by collection using first-level folder name under --input (e.g., catalogs/Fall Edit/*.pdf).
- Keeps your existing curated phrases; adds new ones (de-duped).
- Uses simple, readable luxury/fashion heuristics (no heavy NLP dependencies).

Run locally:
  python scripts/extract_brand_phrases.py --input catalogs --output brand_lexicon.json
"""

import argparse, os, re, json, glob, collections, datetime, hashlib
from typing import Dict, List, Tuple, Set
import fitz  # PyMuPDF

NGRAM_MIN, NGRAM_MAX = 2, 4
TOP_GLOBAL = 120
TOP_COLLECTION = 60

STOP = set("""the a an and or of for to with your our you we us on by from in is are it this that as at be can will just so""".split())

# Seed luxury/fashion vocabulary to bias phrase picking
LUX = set("""
refined luxe luminous silky plush tailored sculpted airy crisp elegant modern minimal elevated polished buttery velvet satin clean sleek quiet soft gentle
""".split())
FASHION = set("""
layer layers layering capsule edit wardrobe neutral neutrals palette color hue tone drape texture textural knit denim leather linen wool trench satin velvet
silhouette line lines profile contour shape fit finish comfort everyday staple essential basics classic timeless day night desk dinner weekend play
""".split())

# Allow-list of canonical brand phrases we care about (always pass filter if seen)
CANON = {
    "effortless layers","polished essentials","week-to-weekend","color story","textural mix",
    "tailored meets relaxed","versatile capsule","seasonal musts","style with ease","confidence-first",
    "modern romance","bold neutrals","capsule dressing","refined comfort","soft structure",
    "clean finish","layering edit","day-to-night","desk-to-dinner","work-to-play"
}

BAD_RE = re.compile(r"(privacy|terms|catalog|copyright|page|price|sale|%|usd|http|www|\.com)", re.I)

def clean_text(t: str) -> str:
    t = re.sub(r"-\s*\n", "", t)         # fix hyphen line breaks
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[•©®™]+", " ", t)
    return t.strip()

def load_pdfs_grouped(root: str) -> Dict[str, List[str]]:
    """Return {collection_name: [pdf_text, ...]} where collection_name is first folder under root."""
    groups: Dict[str, List[str]] = {}
    root = root.rstrip("/")

    # find all PDFs recursively
    for path in glob.glob(os.path.join(root, "**", "*.pdf"), recursive=True):
        rel = os.path.relpath(path, root)
        parts = rel.split(os.sep)
        collection = parts[0] if len(parts) > 1 else "Ungrouped"
        try:
            doc = fitz.open(path)
            pages = [page.get_text("text") or "" for page in doc]
            text = clean_text("\n".join(pages))
            if text:
                groups.setdefault(collection, []).append(text)
        except Exception:
            # skip unreadable PDFs
            continue
    return groups

def tokens_from_text(text: str) -> List[str]:
    toks = [w.lower() for w in re.findall(r"[a-zA-Z][a-zA-Z-]+", text)]
    return toks

def ok_phrase(words: List[str]) -> bool:
    if not (NGRAM_MIN <= len(words) <= NGRAM_MAX):
        return False
    if BAD_RE.search(" ".join(words)):
        return False
    # Allow canonical phrases outright
    if " ".join(words) in CANON:
        return True
    # Must include at least one fashion/luxury cue OR include connective patterns
    if any(w in LUX or w in FASHION for w in words):
        return True
    if "to" in words and ("day" in words and "night" in words):
        return True
    if "desk" in words and "dinner" in words:
        return True
    if "work" in words and "play" in words:
        return True
    if "tailored" in words and "meets" in words and "relaxed" in words:
        return True
    return False

def ngrams(tokens: List[str], n: int):
    for i in range(len(tokens)-n+1):
        yield tokens[i:i+n]

def score_text(text: str) -> List[Tuple[str, float]]:
    toks = tokens_from_text(text)
    counts = collections.Counter()
    for n in range(NGRAM_MIN, NGRAM_MAX+1):
        for gram in ngrams(toks, n):
            if ok_phrase(gram):
                phrase = re.sub(r"\s+", " ", " ".join(gram)).strip()
                counts[phrase] += 1
    total = sum(counts.values()) or 1
    # simple length-aware frequency score
    scored = {k: (v/total) * (1 + 0.25*len(k.split())) for k, v in counts.items()}
    return sorted(scored.items(), key=lambda x: x[1], reverse=True)

def unique_merge(primary: List[str], extra: List[str], limit: int = None) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for s in primary + extra:
        key = re.sub(r"\s+", " ", s.strip().lower())
        if not key: continue
        if key in seen: continue
        seen.add(key); out.append(s.strip())
        if limit and len(out) >= limit:
            break
    return out

def build_lexicon(input_dir: str, existing: dict) -> dict:
    groups = load_pdfs_grouped(input_dir)
    global_candidates: List[str] = []
    per_collection: Dict[str, List[str]] = {}

    for coll, texts in groups.items():
        local_scores: List[Tuple[str, float]] = []
        for t in texts:
            local_scores.extend(score_text(t))
        # Dedup by taking best score per phrase
        best = {}
        for phrase, sc in local_scores:
            best[phrase] = max(best.get(phrase, 0.0), sc)
        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
        phrases = [p for p, _ in ranked]
        per_collection[coll] = phrases[:TOP_COLLECTION]
        global_candidates.extend(phrases[:TOP_COLLECTION//2])

    # Start with existing curated phrases if present
    existing_global = (existing.get("global") or {}).get("brand_phrases", [])
    new_global = unique_merge(existing_global, global_candidates, limit=TOP_GLOBAL)

    # Build collections block: merge existing + new where possible
    new_collections = {}
    existing_collections = existing.get("collections") or {}
    for coll, phrases in per_collection.items():
        prev = (existing_collections.get(coll) or {}).get("brand_phrases", [])
        new_collections[coll] = {
            "brand_phrases": unique_merge(prev, phrases, limit=TOP_COLLECTION),
            # Preserve any existing flair; leave flair curation to you or another script
            "flair": (existing_collections.get(coll) or {}).get("flair", [])
        }
    # include any collections that had no new PDFs
    for coll, obj in existing_collections.items():
        if coll not in new_collections:
            new_collections[coll] = obj

    # Preserve tones section entirely (manual curation)
    tones = existing.get("tones") or {}

    # Merge default flair
    default_flair = (existing.get("global") or {}).get("flair", ["refined finish","luxe touch","sleek contour","tailored fit","polished edge","silky drape","plush hand"])

    now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    meta_src = f"PDFs in {input_dir}"
    sha = os.getenv("GITHUB_SHA", "")[:12]

    return {
        "meta": {"version": now, "source": meta_src, "commit": sha},
        "global": {"brand_phrases": new_global, "flair": default_flair},
        "collections": new_collections,
        "tones": tones
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="catalogs")
    ap.add_argument("--output", default="brand_lexicon.json")
    args = ap.parse_args()

    existing = {}
    if os.path.exists(args.output):
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    lex = build_lexicon(args.input, existing)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(lex, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
