import json
from typing import Dict, List, Optional


def export_gsm8k_split(
    config_name: str = "main",
    split_name: str = "train",
    limit: Optional[int] = None,
) -> List[Dict[str, str]]:
    from datasets import load_dataset

    dataset = load_dataset("gsm8k", config_name)[split_name]
    records = []
    for index, example in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        records.append(
            {
                "id": f"gsm8k/{split_name}/{index}",
                "question": example["question"].strip(),
                "answer": example["answer"].strip(),
            }
        )
    return records


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="main", choices=["main", "socratic"])
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = export_gsm8k_split(args.config, args.split, args.limit)
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
