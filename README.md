# Tool Tax

Code release for the GSM8K slice of the Tool Tax paper.

This repository packages two paper-aligned components:

- `benchmark_generation`: build the GSM8K semantic-distractor benchmark from raw GSM8K-style problems.
- `mechanism_ablation`: run the function-calling ablations that decompose the CoT-vs-tool gap.

It is set up for two usage modes:

- offline smoke testing with a tiny bundled fixture and mock backends
- paper-style runs with OpenAI-based generation and evaluation

Core entrypoints:

- [`pipeline/benchmark_generation/gsm8k_semantic_distractor/`](pipeline/benchmark_generation/gsm8k_semantic_distractor/)
- [`pipeline/mechanism_ablation/gsm8k_semantic_distractor/run_mechanism_ablation.py`](pipeline/mechanism_ablation/gsm8k_semantic_distractor/run_mechanism_ablation.py)
- [`pipeline/mechanism_ablation/gsm8k_semantic_distractor/run_mechanism_ablation_gate.py`](pipeline/mechanism_ablation/gsm8k_semantic_distractor/run_mechanism_ablation_gate.py)
- [`pipeline/mechanism_ablation/gsm8k_semantic_distractor/gate/train_gate.py`](pipeline/mechanism_ablation/gsm8k_semantic_distractor/gate/train_gate.py)

## Release Scope

This public folder currently contains the `GSM8K-Sem-Distractor` path only.

Included:

- GSM8K benchmark generation
- GSM8K mechanism ablations
- paper-style `G-STEP` gate training and evaluation
- mock fixtures for end-to-end offline verification

Not included in this folder:

- HotPotQA benchmark generation and evaluation
- the full benchmark dump used in the paper
- private or large internal artifacts

## Public Variant Names

The public release uses the final paper names:

- `TB`: Thematic Background
- `PED`: Parallel Entity Distractor
- `HU`: Hedged Uncertainty
- `SP`: Semantic Paraphrase

Legacy names are still accepted when loading older files:

- `LX -> TB`
- `HN1 -> PED`
- `HN2 -> HU`
- `LEX -> SP`

All newly generated outputs in this repo use the public names.

## What Matches the Paper

- Benchmark-generation prompts follow the paper prompt design for `TB / PED / HU / SP`.
- The default OpenAI benchmark generator uses `gpt-4o-mini`.
- The mechanism runner exposes the paper conditions:
  - `fc_notool_cot`
  - `fc_notool_fcprompt`
  - `fc_noop`
  - `fc_baseline`
  - `fc_max1`
  - `fc_perfect`
  - `fc_oracle_ev`
- Gate-enabled runs expose the paper gate settings:
  - `gate_step`
  - `gate_step_critic`
- The default `G-STEP` threshold is `0.05`.
- Paper-style gate training uses:
  - labels constructed from `fc_baseline` and `fc_notool_cot`
  - 120-dimensional features
  - `StandardScaler + MLP(128, 64)`
  - `GroupKFold(n_splits=5)`

## Repository Layout

```text
.
├── configs/
│   ├── eval_config/
│   ├── management_config/
│   └── pipeline_config/
├── data/
│   └── gsm8k_semantic_distractor/
├── pipeline/
│   ├── benchmark_generation/
│   └── mechanism_ablation/
├── scripts/
│   ├── benchmark_generation/
│   ├── mechanism_ablation/
│   └── quick_start/
├── tests/
└── requirements/
```

Key bundled files:

- [`data/gsm8k_semantic_distractor/sample_raw.jsonl`](data/gsm8k_semantic_distractor/sample_raw.jsonl): tiny GSM8K-style raw fixture
- [`data/gsm8k_semantic_distractor/sample_benchmark.jsonl`](data/gsm8k_semantic_distractor/sample_benchmark.jsonl): tiny assembled benchmark fixture
- [`data/gsm8k_semantic_distractor/sample_gate_results/fc_baseline.jsonl`](data/gsm8k_semantic_distractor/sample_gate_results/fc_baseline.jsonl): sample baseline results for gate smoke tests
- [`data/gsm8k_semantic_distractor/sample_gate_results/fc_notool_cot.jsonl`](data/gsm8k_semantic_distractor/sample_gate_results/fc_notool_cot.jsonl): sample NoTool-CoT results for gate smoke tests

## Installation

All commands below assume you are running from the repository root.

Install the base dependencies:

```bash
pip install -r requirements/basic_requirements.txt
```

For paper-faithful `G-STEP` training, also install the optional gate-training dependencies:

```bash
pip install numpy scikit-learn
```

The OpenAI-based scripts additionally require:

```bash
export OPENAI_API_KEY=YOUR_KEY
```

## Quick Start

The fastest path is to run the bundled offline smoke tests first.

Generate a tiny benchmark with the mock backend:

```bash
bash scripts/quick_start/benchmark_generation/mock/gsm8k_semantic_distractor.sh
```

Run the mechanism ablation with the mock backend:

```bash
bash scripts/quick_start/mechanism_ablation/mock/gsm8k_semantic_distractor.sh
```

Train the sample `G-STEP` artifact and run the gate-enabled mock ablation:

```bash
bash scripts/quick_start/mechanism_ablation/mock/gsm8k_semantic_distractor_gate.sh
```

Run the test suite:

```bash
bash scripts/run_tests.sh
```

## Benchmark Generation Pipeline

The original monolithic workflow has been reorganized into modular stages:

1. `export_from_hf.py`: export GSM8K from Hugging Face
2. `parse_schema.py`: extract units, forbidden pairs, and calculation structure
3. `build_core.py`: split the question into core sentences and choose heuristic evidence
4. `generate_noise.py`: generate `TB / PED / HU / SP`
5. `validate.py`: check truth invariance, class constraints, and leakage
6. `assemble.py`: write final benchmark rows with public variant names

These files live under [`pipeline/benchmark_generation/gsm8k_semantic_distractor/`](pipeline/benchmark_generation/gsm8k_semantic_distractor/).

## Mechanism Conditions

The public mechanism runner exposes the seven paper conditions:

- `fc_notool_cot`: NoTool-CoT
- `fc_notool_fcprompt`: NoTool-FCStyle
- `fc_noop`: Agent-NoopTool
- `fc_baseline`: Agent-Full
- `fc_max1`: Agent-Max1Turn
- `fc_perfect`: Agent-OracleCalc
- `fc_oracle_ev`: Agent-OracleEvid

If a trained `g_step` artifact is provided, the runner also exposes the paper gate conditions:

- `gate_step`: `G-STEP` with the original continuation policy
- `gate_step_critic`: `G-STEP + CRITIC`

## OpenAI Runs

The repository ships with small configs so the commands run out of the box, but those configs default to a small bundled benchmark fixture or a small export limit. For full experiments, update the config paths before launching long runs.

Generate benchmark examples with OpenAI:

```bash
bash scripts/benchmark_generation/openai/gsm8k_semantic_distractor.sh
```

Run the seven paper mechanism conditions:

```bash
bash scripts/mechanism_ablation/openai/gsm8k_semantic_distractor.sh
```

Train a paper-style `G-STEP` artifact from a results directory that already contains both `fc_baseline.jsonl` and `fc_notool_cot.jsonl`:

```bash
python -m pipeline.mechanism_ablation.gsm8k_semantic_distractor.gate.train_gate \
  --results_dir YOUR_RESULTS_DIR \
  --out_dir output/gsm8k_semantic_distractor/gate_artifacts/openai \
  --which g_step
```

Then run the gate-enabled mechanism evaluation:

```bash
bash scripts/mechanism_ablation/openai/gsm8k_semantic_distractor_gate.sh
```

Configs to edit for larger or custom runs:

- [`configs/pipeline_config/benchmark_generation/openai/gsm8k_semantic_distractor.json`](configs/pipeline_config/benchmark_generation/openai/gsm8k_semantic_distractor.json)
- [`configs/pipeline_config/mechanism_ablation/openai/gsm8k_semantic_distractor.json`](configs/pipeline_config/mechanism_ablation/openai/gsm8k_semantic_distractor.json)
- [`configs/pipeline_config/mechanism_ablation/openai/gsm8k_semantic_distractor_gate.json`](configs/pipeline_config/mechanism_ablation/openai/gsm8k_semantic_distractor_gate.json)

## Outputs and Reproducibility

Pipeline outputs are written under `output/...`.

Each run stores:

- raw per-stage or per-condition outputs
- the config files used for the run
- copied code snapshots for reproducibility

Gate-enabled runs expect a serialized `g_step` artifact at the path provided by `g_step_model_path` in the mechanism config.

## Citation and License

- Citation metadata is provided in [`CITATION.cff`](CITATION.cff).
- The repository is released under the MIT License in [`LICENSE`](LICENSE).
- Contribution expectations are documented in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Notes

- The bundled `data/gsm8k_semantic_distractor/sample_*` files are smoke-test fixtures only.
- The default OpenAI benchmark-generation config exports only a small subset (`limit = 50`) for convenience.
- The default OpenAI mechanism configs point to the sample benchmark fixture; update `data_path` before full runs.
- If `scikit-learn` is unavailable, or the fixture is too small for a stable MLP fit, the gate trainer falls back to a lightweight linear model so the repo still runs offline end to end.
- That fallback is for smoke testing only. The paper-faithful gate backend is the `StandardScaler + MLP` path.
- Legacy `g_commit` support is still present in the codebase as a non-paper extension, but it is not part of the default paper path.
