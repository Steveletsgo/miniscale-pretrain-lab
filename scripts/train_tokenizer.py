from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--vocab-size", type=int, default=32000)
    return parser.parse_args()


def iter_text(path: Path, field: str):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                record = json.loads(line)
                text = record[field]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(f"invalid record at line {line_number}") from error
            if isinstance(text, str) and text.strip():
                yield text


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
    )
    tokenizer.train_from_iterator(iter_text(Path(args.input), args.text_field), trainer)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(destination))
    print(f"saved tokenizer with {tokenizer.get_vocab_size()} tokens to {destination}")


if __name__ == "__main__":
    main()

