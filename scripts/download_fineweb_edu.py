from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream a bounded FineWeb-Edu experiment slice.")
    parser.add_argument("--output-dir", default="data/raw/fineweb_edu")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--name", default="sample-10BT")
    parser.add_argument("--revision", default="v1.4.0")
    parser.add_argument("--max-documents", type=int, default=5000)
    parser.add_argument("--max-characters", type=int, default=40_000_000)
    parser.add_argument("--minimum-characters", type=int, default=200)
    parser.add_argument("--validation-every", type=int, default=20)
    return parser.parse_args()


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": record.get("id"),
        "url": record.get("url"),
        "text": record["text"],
    }


def main() -> None:
    args = parse_args()
    if args.max_documents <= 0 or args.max_characters <= 0:
        raise ValueError("max-documents and max-characters must be positive")
    if args.validation_every < 2:
        raise ValueError("validation-every must be at least 2")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    dataset = load_dataset(
        args.dataset,
        name=args.name,
        split="train",
        streaming=True,
        revision=args.revision,
    )

    examined = 0
    kept = 0
    train_documents = 0
    validation_documents = 0
    total_characters = 0
    with (
        train_path.open("w", encoding="utf-8") as train_stream,
        validation_path.open("w", encoding="utf-8") as validation_stream,
    ):
        for record in dataset:
            examined += 1
            text = record.get("text")
            if not isinstance(text, str) or len(text.strip()) < args.minimum_characters:
                continue
            if kept >= args.max_documents or total_characters + len(text) > args.max_characters:
                break

            serialized = json.dumps(compact_record(record), ensure_ascii=False) + "\n"
            if kept % args.validation_every == 0:
                validation_stream.write(serialized)
                validation_documents += 1
            else:
                train_stream.write(serialized)
                train_documents += 1
            kept += 1
            total_characters += len(text)

    manifest = {
        "dataset": args.dataset,
        "name": args.name,
        "revision": args.revision,
        "license": "ODC-By",
        "examined_documents": examined,
        "kept_documents": kept,
        "train_documents": train_documents,
        "validation_documents": validation_documents,
        "characters": total_characters,
        "minimum_characters": args.minimum_characters,
        "validation_every": args.validation_every,
        "train_path": str(train_path.resolve()),
        "validation_path": str(validation_path.resolve()),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
