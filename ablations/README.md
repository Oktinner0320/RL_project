# Ablation Study

[Project home](../README.md) | [Reproduction](../docs/REPRODUCIBILITY.md) | [中文](README.ZH.md)

This separate entry supports the N1/N3 study. It uses the project environment
and reads `config/experiment.json` without changing the main experiment.
Study motivation and algorithm justification belong in the report.

## Prerequisites

- Install the project environment with the `comparisons` extra.
- Supply the matching `ablations/test/` supplement; tests are not in Git.
- Supply the approved `ablations/output/N-cloud-v1/` execution package, including
  its frozen source, risk inputs, and approval records.
- Authenticate Modal and check the budget before any submission.

A clean clone lacks these local records and tests; its preflight cannot be
presented as a successful execution. Preparing a new directory does not grant
additional execution or cost approval.

## Use the notebook

Open [experiment.ipynb](experiment.ipynb) in a **separate project `.venv` kernel**.
Inspect the phase before running cells: the checked-in notebook currently sets
`PHASE="continue"`, not `pilot`.

| Phase | Scope |
|---|---|
| `pilot` | First paired seed; 500 training interactions per arm |
| `continue` | First pair; add 700 per arm, reaching 1,200, then validate |
| `replicate` | Remaining paired seeds; separately approved work |

Run **Locate → Open plan → Preflight → Submit/Wait → Collect/Verify/Results**.
An existing recorded call is reused; only a verified stage record establishes
completion. Check the same workspace when reopening the notebook.
Do not invoke internal workers as an alternative to approval.

From the project root, inspect local preparation status (not cloud call status):

```powershell
uv run --locked --extra comparisons python -m ablations.runner status --name N-cloud-v1
```

Local preflight has a 90-second watchdog; phase waits are bounded at
4,800 / 9,000 / 27,000 seconds, node computation at 1,800 seconds, and download
at two hours. To stop the selected stage, call `session.cancel(PHASE)` in the
notebook. Preserve failures and unknown consumption; do not retry blindly.

## Outputs and limits

Records are stored under `ablations/output/N-cloud-v1/phases/<phase>/`, including
status, downloaded artifacts, and `closure.json`. The notebook displays available
closure records; these artifacts are not shipped in Git. Results use **predicted
risk** and do not load the official test set.
Other L/Q/E/W studies remain definitions, not implemented alternatives.
