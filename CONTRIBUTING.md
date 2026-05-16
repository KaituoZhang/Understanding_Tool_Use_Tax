# Contributing

This repository is the public GSM8K slice of the Tool Tax project. Contributions that improve correctness, reproducibility, documentation, and paper-faithful behavior are welcome.

## Scope

- Keep the default public path aligned with the paper-facing GSM8K release.
- Treat `TB / PED / HU / SP` as the canonical public variant names.
- Keep legacy names such as `LX / HN1 / HN2 / LEX` only for backward compatibility with older artifacts.
- Avoid adding unrelated tasks, benchmarks, or experimental branches into the default configs.

## Development Setup

Install the base dependencies:

```bash
pip install -r requirements/basic_requirements.txt
```

Install the optional gate-training dependencies if you are working on `G-STEP` training:

```bash
pip install numpy scikit-learn
```

## Before Opening a Pull Request

- Run `bash scripts/run_tests.sh`.
- If you changed benchmark-generation code, run `bash scripts/quick_start/benchmark_generation/mock/gsm8k_semantic_distractor.sh`.
- If you changed mechanism-ablation code, run `bash scripts/quick_start/mechanism_ablation/mock/gsm8k_semantic_distractor.sh`.
- If you changed gate logic or gate training, run `bash scripts/quick_start/mechanism_ablation/mock/gsm8k_semantic_distractor_gate.sh`.
- Update `README.md`, configs, or tests when your change affects commands, file formats, or default behavior.

## Coding Guidelines

- Keep comments, docstrings, and user-facing text in English.
- Prefer small, focused pull requests over large mixed changes.
- Preserve the existing repository layout unless a structural change is necessary.
- Do not commit API keys, private data, large generated outputs, or model artifacts.
- Keep mock paths runnable offline.
- Keep paper-faithful behavior as the default path. Experimental extensions should stay clearly separated from the default configs and docs.

## Reporting Bugs

Please include the following when opening an issue:

- the exact command you ran
- the config file you used
- the relevant input path or artifact path
- the full error message or traceback
- whether the issue appeared on the mock path or the OpenAI path

## License

By contributing to this repository, you agree that your contributions may be distributed under the MIT License in [`LICENSE`](LICENSE).
