# Risk-Aware Reinforcement Learning Framework for Cloud–Edge Federated Training

[中文](README.md) | [English](README.en.md)

Course-paper materials are in [paper/README.md](paper/README.md): a detailed Chinese outline, an English Typst skeleton, an ACM-style course adaptation, and compilation and layout-verification instructions. The final content budget is 20 pages; the current skeleton uses its actual content length.

Recent external algorithm screening and actual benchmark results are recorded in [reference_algorithms/REPORT.md](reference_algorithms/REPORT.md) (Chinese). Main-project integration results are in [BASELINE_INTEGRATION_RESULTS.md](docs/BASELINE_INTEGRATION_RESULTS.md). Six official repositories, isolated UV runtimes, and raw results reside in `reference_algorithms/`. The sole editable configuration is now `config/experiment.json`.

This project studies whether an empirical privacy-risk surrogate systematically underestimates risk when dynamic scheduling changes the training trajectory, and whether uncertainty estimates and a limited budget of gradient-inversion audits improve decision reliability. The application jointly schedules client participation and batch size, evaluating model utility, time, and energy together.

The native first implementation is available in `src/cloud_edge_rl/algorithms/`, with `core.py` retaining the public interfaces for FedSGD, gradient inversion, the risk ensemble, DQN/PPO and training recovery. Runnable third-party baselines reside separately in `comparisons/`: an upstream MLP/FedSGD and IG inversion, MLP/TabM regression, a MetaCS objective adaptation, a PyTorch PQN port, and ConformalOpt/OCP comparisons. Selecting a baseline does not automatically replace the native core. Implementation completion is not performance acceptance.

## First-Version Delivery and Acceptance

The first goal is a reproducible exposure-consistent, risk-calibrated joint scheduler, governed by [the first-version standard](docs/一版验收标准.md) (`first-acceptance-v1`, Chinese):

- Adult IID, normal resources, **three independent paired training seeds**; validation selects and freezes one primary DQN or PPO and one strong scheduling reference `b`.
- All paired means must meet: accuracy loss at most **1 percentage point**, modeled time and energy increases at most **15%** each, and at least **5% lower actually audited exposure-weighted risk**. At least **2/3** seeds must be task-feasible and have lower actual risk.
- Independent common-panel exposure-weighted MAE must be at most **1.10 times** one frozen estimator reference. A paired conditional-calibration ablation against the **uncalibrated** surrogate must reduce mean underestimation by at least **5%**, with weighted error increasing by at most **10%**.
- Retain the 30-round horizon including 12 warm-up rounds, at most 3,000 actual interactions per method/seed/partition, truth isolation, complete own-trajectory audits and independent confirmation. Spent allocations are not reset.

The five-seed, two-partition, two-resource-regime matrix and stricter 95%-of-reference goals are later research extensions, not all mandatory first-version gates. **First-version acceptance remains incomplete. Pipeline repairs and development results are not independent formal confirmation.** Register executable settings in `config/experiment.json`.

## Installation and Entry Point

Run from the project directory:

```powershell
uv sync --locked --extra comparisons
uv run --extra comparisons jupyter lab
```

Edit [config/experiment.json](config/experiment.json), then open [experiment.ipynb](experiment.ipynb). The notebook only loads, prepares, submits/resumes, queries, collects, and verifies. `.python-version` and `pyproject.toml` specify Python and UV versions; `uv.lock` determines dependencies. Local PyTorch uses CPU; the Modal Linux image uses the locked CUDA wheel.

All `workflow.experiment` operational guards default to false. Run All loads configuration and existing local status without downloading data or starting cloud work. The canonical `Experiment` lifecycle supports the `fresh` seven-method development graph, binding collection, attacks, audits, risk fitting, segmented training, and validation. Formal testing retains separate freeze and isolation requirements.

The six live controls are `prepare_enabled`, `submission_enabled`, `resume_enabled`, `refresh_enabled`, `collection_enabled`, and `verification_enabled`. Scientific settings remain frozen. `graph_admission` names the artifact binding historical allocations, source contents, and isolation approval; it is not another tuning configuration. See [pipeline documentation](docs/训练管线.md).

Check the framework:

```powershell
uv run --extra comparisons pytest
uv run --extra comparisons ruff check .
```

Test doubles check interfaces, accounting and isolation. The native core and reference adapters also have focused tests running real algorithms. Passing tests does not establish first-version performance acceptance or the research hypotheses.

## Integrated Third-Party Baselines

`CONFIG.components` defaults to `researcher`. The `reference-*`, `metacs` and `pqn` variants explicitly select separate reference components. Risk checkpoints record their implementation and architecture; evaluation preserves the saved training settings. Every policy sees the same public resources, candidate estimates and recent participation window.

PQN ports the official JAX implementation to PyTorch and samples the real Gymnasium/FedSGD environment serially. MetaCS calls the upstream objective and participation-fairness functions while enumerating the shared 12 actions; this adaptation does not execute the original LNS search.

Historical `comparisons.workflow` records and recovery APIs remain compatible; the current notebook no longer starts a competing submission/download controller. Algorithms and planning components remain reusable. Periodic sampling only measures the audited subset, not all-client or worst-client risk; final audit labels must not feed back into fitting, training, or calibration.

The 64 PQN steps and two OCP arms in the 2026-10-05 baseline pilot were historical feasibility settings. New comparisons register paired seeds and training/attack budgets in the sole JSON configuration. OCP requires enough genuinely audited uploads; data are never duplicated to satisfy its horizon. Coverage refers only to the audited sequence, and score-revelation counts do not imply fewer actual attacks.

That historical pilot's 24 risk-fitting labels comprise 12 audits from each of two runs, all during warm-up. They do not cover the full training trajectory. Held-out fixed-policy audits include later uploads. Lower risk MAE does not establish less underestimation or safer scheduling; inspect underestimation and actual reconstruction labels together. The upstream OCP loss constant is fixed at 40, so its supplied `C` argument is not an effective tuning parameter. The adapter preserves the native rules.

Pinned sources stay in `reference_algorithms/repos/` rather than being copied into the main package. Restore missing sources using the [reference directory instructions](reference_algorithms/README.md). License records and PQN port attribution are in [THIRD_PARTY.md](docs/THIRD_PARTY.md).

## Files and Responsibilities

| Entry | Purpose |
|---|---|
| `config/experiment.json` | Sole editable experiment configuration |
| `experiment.ipynb` | Submission, recovery, status, collection, and results frontend |
| `experiment.py`, `experiment_graph.py` | Canonical lifecycle, cloud dependency graph, durable state |
| `experiment_admission.py`, `experiment_transfer.py` | Historical allocation admission and whole-graph artifact return |
| `src/cloud_edge_rl/core.py`, `algorithms/` | Native public interfaces and actual FedSGD, inversion, risk and DQN/PPO implementations |
| `data.py`, `metrics.py` | Official data processing, reconstruction evaluation on original fields, and risk-error metrics |
| `environment.py`, `resources.py` | Gymnasium interaction, joint actions, resource accounting, budgets, and public state |
| `baselines.py` | Fixed, round-robin, time-greedy, and risk-greedy scheduling, plus shared warm-up |
| `comparisons/` | Explicit reference adapters, audit calibration comparisons and integrated orchestration |
| `audit.py` | Save actual uploads, perform budgeted audits, and isolate attack inputs from reference ground truth |
| `pipeline.py`, `config.py` | Stage orchestration, configuration snapshots, and source/dependency identity |
| `modal_runner.py`, `artifacts.py` | Cloud jobs, Volumes, run status, and artifact storage |
| `tests/` | Framework behavior and boundary tests |

The Python modules listed above reside in `src/cloud_edge_rl/`. See [the first-version standard](docs/一版验收标准.md), [RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md), [TASKS.md](docs/TASKS.md) and [REFERENCES.md](docs/REFERENCES.md) for gates, protocol, tasks and sources. These supporting documents are currently in Chinese. Because `docs/` is Git-ignored, this tracked README also preserves the goal summary; ignore rules are unchanged.

## Native Core Interfaces and Verification Responsibilities

| Interface | Your responsibility |
|---|---|
| `FederatedBackend.reset`, `step_round` | Initialize the PyTorch MLP; compute each selected client's mean batch gradient against the same pre-update model, aggregate, and update the global model once |
| `FederatedBackend.checkpoint`, `load_checkpoint`, `evaluate_test` | Save model, optimizer, and sampling random state; restore precisely; evaluate on the final test set |
| `Agent.act`, `checkpoint` | Select an action from the public observation and save policy state |
| `train_agent`, `load_agent` | Train DQN/PPO, restore inference, and manage optimizer/random state, replay, and training checkpoints |
| `RiskEstimator.fit`, `predict`, `checkpoint`, `load_checkpoint` | Fit risk using groups of complete runs, return mean and uncertainty, and freeze/save model versions |
| `GradientAttack.reconstruct` | Invert a single client's gradients using only `AttackInput`, returning a table of original Adult fields |

Input and output fields are defined in `types.py`. `ClientUpdate.sample_ids` belongs to the trusted evaluation side. `AttackInput` contains the pre-update model, its public construction configuration `model_config`, a single client's mean batch gradients, known labels, batch size, and public encoding information. The construction configuration must match the architecture, activations, loss, and dtype of the source upload; do not substitute the current audit-stage configuration. Do not pass reference records or true reconstruction results to the policy or attack function.

`CloudEdgeEnv` follows Gymnasium's `reset` / `step` interface. Public observations include resources, progress, sampling history, the recent `participation_window`, candidate-action utility/cost/risk, and `risk_available`. The participation window records executed client selections and is visible to every policy. The `progress` fields are ordered as validation loss, validation accuracy, elapsed-time fraction, used-round fraction, and remaining-sample-budget fraction. The last field may become negative when a complete training round exceeds the budget. Use Gymnasium's `spaces.flatten` if your algorithm needs a flat vector.

## Experiment Workflow

1. Check the local runtime, prepare the data, and inspect split and encoding metadata.
2. Implement the MLP and FedSGD, then collect actual uploads using fixed or round-robin scheduling. Explicitly pass `risk=None` in the initial mode: unavailable risk does not mean zero risk. This mode requires a zero risk-penalty weight and stores predicted mean and uncertainty as `None`.
3. Implement gradient inversion. First run the `benchmark` stage to measure attack time on actual uploads for each batch size, then select that measurement run as an `audit` input in the notebook. Apply periodic or random audits to actual uploads to obtain reconstruction labels on original fields.
4. Implement risk estimation and uncertainty. Fit/calibrate using groups of complete runs and save an explicit risk version.
5. Implement DQN/PPO. Freeze the stage's risk version while training the scheduler; all policies share candidate information and the warm-up procedure.
6. Continue auditing on new policy-induced trajectories, update the risk model, and compare fixed, periodic, random, and active audit strategies.
7. Use independent test runs and audit panels to evaluate surrogate underestimation, actual attack recovery rates, utility, and resource costs. Finding no shift, or finding that RL does not outperform a baseline, is also a valid research result.

After each new `reset`, call `baselines.run_warmup` using the notebook configuration. It executes the specified actions in sequence and stops immediately on termination or truncation. Warm-up consumes the same time, sample, and energy budgets and produces audit records. Your training loop must not execute an additional unaccounted warm-up round.

## Modal and Checkpoint Recovery

The notebook uses `Experiment` to submit a detached Modal dependency graph. Modal builds from frozen source and `uv.lock`, simulating three clients in one logical task. Hardware, deadlines, and Volume come from the JSON configuration. Compute and verification subprocesses have explicit deadlines and process-tree watchdogs.

For first-time use, authenticate following the [official Modal guide](https://modal.com/docs/guide). The cloud executes and verifies dependencies, committing each stage to Volume. A single return stage retrieves the full dependency closure for local semantic verification. Closing the notebook does not restart training; transfer failures only resume transfer.

`runs/<run_id>/` stores inputs and results. Non-content-derived IDs are generated once and frozen in the graph; reopening the same controller directory preserves them. Compute, cloud verification, transfer, and local closure are distinct states. Unknown submissions are not automatically repeated; failed or unknown consumption is retained.

Resume the same controller directory and original call. Explicit continuation requires an observed safe pause or verification-only failure; computed nodes only reverify. Failed or ambiguous computation requires reconciliation. Each continuation has an isolated transfer directory. Historical outputs are preserved and cannot silently become completed nodes under new source code.

Full recovery within the same risk version requires you to save and restore the policy, optimizer, random state, replay buffer, and wrapped environment's `snapshot()`. Restore with `load_snapshot()` without calling `reset` or repeating warm-up. When changing risk versions, retain policy weights, clear the old replay buffer, and start a new stage. Your `core.py` implementation owns these algorithm-state transitions.

The initial cloud budget is an experiment-planning target, not a hard Modal billing cap. Resource timeouts and audit budgets limit workload; check actual charges against real invocation records and the Modal bill.

## Current Status and Boundaries

See [VERIFICATION.md](docs/VERIFICATION.md) for actual runtime, notebook, test, and Modal verification records. After tuning, both `validate` and `evaluate` inherit the relevant training snapshot through `inputs.training_run_ids`. Explicit scheduling baselines use `evaluate_baselines`, so current default parameters do not overwrite a selected trial's configuration.

Code and functions include concise Chinese and English explanations. Necessary interface checks raise errors directly rather than silently fabricating inputs or switching to substitute algorithms. The historical 2026-10-05 Modal L4 baseline workflow completed 13 phases, 26 real runs and a 26-cell executed notebook; 145 tests passed at that time. The initial audit-coverage failure and subsequent artifact-I/O controller timeout remain recorded. Explicit recovery reattached 19 existing calls and submitted only seven new tasks. See the [integration results and raw CSVs](docs/BASELINE_INTEGRATION_RESULTS.md). Later native implementation and development checks are recorded in [overall acceptance](docs/NATIVE_OVERALL_ACCEPTANCE.md); first-version independent confirmation remains incomplete.

That historical run was a small feasibility check. Attacks with 200 iterations and one restart performed below the marginal prior, with zero EMR for all six scheduler audits. Four greedy/MetaCS comparisons chose identical post-warm-up actions, so their risk differences cannot establish a scheduling improvement. TabM reduced MAE but increased mean underestimation. SQT/LQT achieved full audited coverage with risk bounds near 1; OCP missed the 0.9 target. Calibration was post-hoc and did not change scheduler rewards. The first version still needs frozen-protocol independent panels, the calibration ablation, three-seed confirmation and complete actual audits.
