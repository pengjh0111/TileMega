"""Freeze public-domain passages and per-model token IDs for serving runs."""
from __future__ import annotations

import argparse
from http.client import IncompleteRead
import json
import re
import urllib.request
from pathlib import Path

BOOKS = [
    (1342, "Pride and Prejudice"), (84, "Frankenstein"),
    (11, "Alice's Adventures in Wonderland"), (1661, "The Adventures of Sherlock Holmes"),
    (98, "A Tale of Two Cities"), (2701, "Moby Dick"),
    (174, "The Picture of Dorian Gray"), (345, "Dracula"),
    (76, "Adventures of Huckleberry Finn"), (1260, "Jane Eyre"),
    (2600, "War and Peace"), (43, "Strange Case of Dr Jekyll and Mr Hyde"),
    (768, "Wuthering Heights"), (120, "Treasure Island"),
    (514, "Little Women"), (996, "Don Quixote"),
]


def passages(path: Path) -> None:
    rows = []
    for book_id, title in BOOKS:
        url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        req = urllib.request.Request(url, headers={"User-Agent": "TileMega-research/1.0"})
        with urllib.request.urlopen(req, timeout=60) as stream:
            try:
                raw = stream.read(200_000)
            except IncompleteRead as exc:
                # The opening passage is fully contained in the received prefix.
                raw = exc.partial
        body = raw.decode("utf-8-sig", errors="replace")
        start = re.search(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^*]*\*\*\*", body, re.I)
        if not start:
            raise ValueError(f"missing book start marker: {book_id}")
        text = body[start.end():]
        # Use prose after the opening title/contents; only this frozen excerpt is benchmark input.
        paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
        paragraphs = [p for p in paragraphs if len(p.split()) >= 25 and not re.search(r"^(contents|chapter|illustrations|preface)\b", p, re.I)]
        words = " ".join(paragraphs[:8]).split()[:190]
        if len(words) < 120:
            raise ValueError(f"too short: {book_id}")
        rows.append({"book_id": book_id, "title": title, "source": url, "text": " ".join(words)})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def token_ids(passages_path: Path, model: Path, out: Path) -> None:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
    rows = [json.loads(line) for line in passages_path.read_text().splitlines()]
    ids = [tokenizer(row["text"], add_special_tokens=True).input_ids for row in rows]
    if len(ids) != 16 or any(len(row) < 64 for row in ids):
        raise ValueError("all 16 passages must yield at least 64 tokens")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([row[:64] for row in ids], separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--ids-out", type=Path)
    args = parser.parse_args()
    if not args.passages.exists():
        passages(args.passages)
    if args.model:
        if not args.ids_out:
            parser.error("--ids-out is required with --model")
        token_ids(args.passages, args.model, args.ids_out)


if __name__ == "__main__":
    main()
