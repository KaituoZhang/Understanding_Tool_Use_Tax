import json
from typing import Any, Dict, List

from eval.gsm8k_semantic_distractor.gsm8k_semantic_distractor_utils import (
    NOISY_VARIANTS,
    canonicalize_variant,
    canonicalize_variants,
    force_question_last,
    split_sentences,
)


def normalize_core(record: Dict[str, Any]) -> List[str]:
    if isinstance(record.get("core_sentences"), list) and record["core_sentences"]:
        sentences = [str(sentence).strip() for sentence in record["core_sentences"] if str(sentence).strip()]
    else:
        sentences = split_sentences(record["question"])
    return force_question_last(sentences)


def augment_meta(meta: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(meta) if isinstance(meta, dict) else {}
    calc_chain = (record.get("schema") or {}).get("calc_chain") or {}
    expressions = calc_chain.get("exprs") or []
    final_answer = calc_chain.get("final")
    if "calc_chain" not in metadata and expressions:
        metadata["calc_chain"] = " ; ".join(expressions + ([f"final={final_answer}"] if final_answer is not None else []))
    return metadata


def assemble_records(records: List[Dict[str, Any]], assembly_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    target_variants = canonicalize_variants(assembly_config.get("variants")) or NOISY_VARIANTS
    include_base = bool(assembly_config.get("include_base", True))
    respect_generated_counts = bool(assembly_config.get("respect_generated_counts", True))
    drop_noise = bool(assembly_config.get("drop_noise", True))
    strip_meta_genraw = bool(assembly_config.get("strip_meta_genraw", True))
    add_calc_chain_to_meta = bool(assembly_config.get("augment_meta", True))

    assembled_rows = []
    for record in records:
        core_sentences = normalize_core(record)
        evidence_block = list(core_sentences[:-1])
        question_sentence = core_sentences[-1]
        static_fields = {
            key: value
            for key, value in record.items()
            if key not in {"variant", "question_sentences", "evidence_sentence_ids", "noise_sentence_ids", "n_before", "n_after", "position", "noise"}
        }

        def build_meta() -> Dict[str, Any]:
            meta = dict(static_fields.get("meta", {}))
            if strip_meta_genraw:
                meta.pop("gen_raw", None)
            if add_calc_chain_to_meta:
                meta = augment_meta(meta, record)
            return meta

        if include_base:
            base_row = dict(static_fields)
            base_row["variant"] = "base"
            base_row["question_sentences"] = list(core_sentences)
            base_row["evidence_sentence_ids"] = list(range(len(core_sentences) - 1))
            base_row["noise_sentence_ids"] = []
            base_row["n_before"] = 0
            base_row["n_after"] = 0
            base_row["meta"] = build_meta()
            if drop_noise:
                base_row.pop("noise", None)
            assembled_rows.append(base_row)

        grouped_noise = {variant: {"before": [], "after": []} for variant in NOISY_VARIANTS}
        for noise_record in record.get("noise", []):
            variant = canonicalize_variant(noise_record.get("type"))
            position = noise_record.get("position")
            text = noise_record.get("text")
            if variant in grouped_noise and position in {"before", "after"} and isinstance(text, str) and text.strip():
                grouped_noise[variant][position].append(text.strip())

        for variant in target_variants:
            before_sentences = grouped_noise[variant]["before"]
            after_sentences = grouped_noise[variant]["after"]
            if not before_sentences and not after_sentences:
                continue

            if not respect_generated_counts:
                before_sentences = before_sentences[:1]
                after_sentences = after_sentences[:1]

            question_sentences = list(before_sentences)
            evidence_start = len(question_sentences)
            question_sentences.extend(evidence_block)
            evidence_end = len(question_sentences)
            question_sentences.extend(after_sentences)
            question_sentences.append(question_sentence)

            row = dict(static_fields)
            row["variant"] = variant
            row["question_sentences"] = question_sentences
            row["evidence_sentence_ids"] = list(range(evidence_start, evidence_end))
            row["noise_sentence_ids"] = list(range(0, evidence_start)) + list(range(evidence_end, len(question_sentences) - 1))
            row["n_before"] = len(before_sentences)
            row["n_after"] = len(after_sentences)
            row["position"] = "both" if before_sentences and after_sentences else ("before" if before_sentences else "after")
            row["meta"] = build_meta()
            if not drop_noise:
                row["noise"] = (
                    [{"type": variant, "position": "before", "text": text} for text in before_sentences]
                    + [{"type": variant, "position": "after", "text": text} for text in after_sentences]
                )
            assembled_rows.append(row)

    return assembled_rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    rows = assemble_records(records, assembly_config={})
    with open(args.output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
