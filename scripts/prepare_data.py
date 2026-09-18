from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-field", default="text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise ValueError("tokenizer does not contain <eos>")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document_count = 0
    token_count = 0
    with Path(args.input).open("r", encoding="utf-8") as source, destination.open("wb") as sink:
        for line_number, line in enumerate(source, 1):
            try:
                text = json.loads(line)[args.text_field]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"invalid record at line {line_number}") from error
            if not isinstance(text, str) or not text.strip():
                continue
            token_ids = tokenizer.encode(text).ids + [eos_id]
            np.asarray(token_ids, dtype=np.uint32).tofile(sink)
            document_count += 1
            token_count += len(token_ids)
    manifest = {
        "input": str(Path(args.input).resolve()),
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "output": str(destination.resolve()),
        "dtype": "uint32",
        "documents": document_count,
        "tokens": token_count,
    }
    with destination.with_suffix(destination.suffix + ".json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

