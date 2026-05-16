from eval.gsm8k_semantic_distractor.gsm8k_semantic_distractor_utils import (
    canonicalize_variant,
    canonicalize_variants,
)


def test_legacy_variants_are_canonicalized():
    assert canonicalize_variant("LX") == "TB"
    assert canonicalize_variant("HN1") == "PED"
    assert canonicalize_variant("HN2") == "HU"
    assert canonicalize_variant("LEX") == "SP"
    assert canonicalize_variants(["base", "LX", "HN1", "LEX"]) == ["base", "TB", "PED", "SP"]
