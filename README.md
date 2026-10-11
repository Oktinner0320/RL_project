# Risk-Aware RL for Cloud–Edge Federated Training

[English](README.md) | [中文（可选）](README.zh.md)

Course project implementing reinforcement-learning scheduling for simulated
federated training on the UCI Adult dataset. The code includes native DQN/PPO,
reference adapters, experiment recovery, and a read-only results viewer.
Problem definitions, algorithm justification, tuning analysis, and conclusions
belong in the written report.

## Start here

Use **Python 3.12.14** and **uv 0.12.11**. From the repository root:

```powershell
uv sync --locked --extra comparisons
uv run --locked --extra comparisons jupyter lab
```

**Do not Run All in `experiment.ipynb` before checking the configuration.**
The checked-in `workflow.experiment` settings enable preparation, submission,
cloud refresh, collection, and verification; only resume is disabled.
For local inspection, set all six operation switches to `false` as described in
[Reproducibility](docs/REPRODUCIBILITY.md).

| Task | Entry |
|---|---|
| Inspect saved development tables | [Published results](output/fresh-seven-method-rerun-20261010-v1/report-analysis-20261010-v1/) |
| View available images and recorded playback | [visualization.ipynb](visualization.ipynb) |
| Train, check status, or collect an approved experiment | [experiment.ipynb](experiment.ipynb) |
| Run the optional N1/N3 study | [Ablations guide](ablations/README.md) |
| Prepare the course handoff | [Submission checklist](docs/SUBMISSION.md) |

## Repository map

| Path | Contents |
|---|---|
| [src/cloud_edge_rl/algorithms/](src/cloud_edge_rl/algorithms/) | Native training, attack, risk estimation, and DQN/PPO implementations |
| [src/cloud_edge_rl/](src/cloud_edge_rl/) | Environment, metrics, experiment execution, recovery, and result storage |
| [src/cloud_edge_rl/comparisons/](src/cloud_edge_rl/comparisons/) | Explicit third-party adapters |
| [config/experiment.json](config/experiment.json) | Sole editable experiment configuration |
| [pyproject.toml](pyproject.toml), [uv.lock](uv.lock) | Python requirements and locked dependencies |
| [reference_algorithms/](reference_algorithms/) | Pinned source records, downloader, and reference runtimes |
| [frontend/](frontend/README.md), [figure_assets/](figure_assets/README.md) | Interactive viewer and figure tools |
| [paper/](paper/README.md) | Manuscript entry and bibliography |

## Reproduction and limitations

- Windows uses CPU PyTorch; Linux uses the locked CUDA build. The configured
  training run uses Modal and requires an authenticated account and valid
  execution approval.
- Git includes six development CSV/JSON summaries and historical playback JSON,
  **not** all checkpoints, raw runs, test files, current snapshots, or PNGs.
  [Reproducibility](docs/REPRODUCIBILITY.md) identifies the required supplements.
- Published summaries are descriptive development results. Full local artifact
  verification and independent research confirmation are not complete.
- Predicted risk is not audited privacy risk. Reported time and energy are
  simulated quantities; unknown cloud costs remain unknown.

See [Third-party attribution](docs/THIRD_PARTY.md) for source and license records.
Git history records the actual development work; no separate history export is
required unless requested by the instructor.
