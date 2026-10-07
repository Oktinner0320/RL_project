# Risk-Aware Reinforcement Learning Framework for Cloud–Edge Federated Training

[中文](README.md) | [English](README.en.md)

Course-paper materials are in [paper/README.md](paper/README.md): a detailed Chinese outline, an English Typst skeleton, an ACM-style course adaptation, and compilation and layout-verification instructions. The final content budget is 20 pages; the current skeleton uses its actual content length.

Recent external algorithm screening and actual benchmark results are recorded in [reference_algorithms/REPORT.md](reference_algorithms/REPORT.md) (Chinese). Main-project integration results are in [BASELINE_INTEGRATION_RESULTS.md](docs/BASELINE_INTEGRATION_RESULTS.md). Six official repositories, isolated UV runtimes, and raw results reside in `reference_algorithms/`. All benchmark parameters remain in the notebook's single CONFIG cell.

This project studies whether an empirical privacy-risk surrogate systematically underestimates risk when dynamic scheduling changes the training trajectory, and whether uncertainty estimates and a limited budget of gradient-inversion audits improve decision reliability. The application jointly schedules client participation and batch size, evaluating model utility, time, and energy together.

Your PyTorch model, FedSGD, DQN/PPO, risk estimator and gradient inversion interfaces remain in `core.py`; unfinished interfaces still stop explicitly. With your authorization, runnable third-party baselines reside separately in `comparisons/`: FedSGD with an upstream small MLP and IG inversion, MLP/TabM risk regression, a MetaCS objective adaptation, a PyTorch PQN port, and ConformalOpt/OCP audit-sequence comparisons. Selecting a baseline does not alter or automatically replace your core. See the [integration plan and acceptance criteria](docs/BASELINE_INTEGRATION.md).

## Installation and Entry Point

Run from the project directory:

```powershell
uv sync --locked --extra comparisons
uv run --extra comparisons jupyter lab
```

Open [experiment.ipynb](experiment.ipynb). Adjust all global experiment parameters, stage switches, run inputs, and Modal resources in the notebook's single `CONFIG` cell. `.python-version` and `pyproject.toml` specify the Python and UV versions; `uv.lock` determines dependency resolution. The local environment uses CPU PyTorch for debugging and analysis, while the Modal Linux image uses the locked CUDA wheel.

By default, the notebook only checks the local runtime and prepares the Adult dataset. Research training and cloud jobs must be explicitly enabled through the stage switches in `CONFIG.workflow.jobs`. Initial data preparation requires access to the official data source. Do not introduce another tuning entry point in modules, a second configuration file, command-line hyperparameters, or other notebook cells.

Check the framework:

```powershell
uv run --extra comparisons pytest
uv run --extra comparisons ruff check .
```

Test doubles exist only in `tests/` for interfaces, accounting and isolation. The reference adapters also have focused tests that run official algorithms. Passing tests does not establish that your core is complete or that the research hypotheses hold.

## Integrated Third-Party Baselines

`CONFIG.components` defaults to `researcher`. The `reference-*`, `metacs` and `pqn` variants explicitly select separate reference components. Risk checkpoints record their implementation and architecture; evaluation preserves the saved training settings. Every policy sees the same public resources, candidate estimates and recent participation window.

PQN ports the official JAX implementation to PyTorch and samples the real Gymnasium/FedSGD environment serially. MetaCS calls the upstream objective and participation-fairness functions while enumerating the shared 12 actions; this adaptation does not execute the original LNS search.

The final notebook section uses `CONFIG.comparisons.workflow.enabled` to run the dependency-ordered comparison workflow; it is disabled by default. Actual artifacts connect collection, attack timing, audits, MLP/TabM risk fitting, PQN training, held-out validation/test audits, risk and six-scheduler comparisons, independent audits of each scheduler's trajectory, and calibration. The six schedulers receive equal attack budgets, and final audit labels are excluded from fitting, training and calibration. Current periodic sampling compares audited subsets: a fixed two-client policy may cover only one of those clients, so these results cannot establish all-client or worst-client risk. Run IDs are passed automatically, and each run preserves its configuration and source identity. Training defaults to Modal; local CPU execution supports engineering tests and explicit debugging.

The current 64 PQN steps and two OCP arms are a feasibility configuration. Formal comparisons require additional training, attack budgets and paired seeds configured in the same CONFIG cell. OCP requires enough genuinely audited uploads; data are never duplicated to satisfy its horizon conditions. Coverage refers only to the audited sequence, and score-revelation counts do not imply fewer actual attacks.

The current 24 risk-fitting labels comprise 12 audits from each of two runs, all during warm-up. They do not cover the full training trajectory. Held-out fixed-policy audits include later uploads. Lower risk MAE does not establish less underestimation or safer scheduling; inspect underestimation and actual reconstruction labels together. The upstream OCP loss constant is fixed at 40, so its supplied `C` argument is not an effective tuning parameter. The adapter preserves the native rules.

Pinned sources stay in `reference_algorithms/repos/` rather than being copied into the main package. Restore missing sources using the [reference directory instructions](reference_algorithms/README.md). License records and PQN port attribution are in [THIRD_PARTY.md](docs/THIRD_PARTY.md).

## Files and Responsibilities

| Entry | Purpose |
|---|---|
| `experiment.ipynb` | Sole entry point for configuration, submission, status queries, downloads, and result analysis |
| `src/cloud_edge_rl/core.py` | All learning and inversion cores that you implement |
| `data.py`, `metrics.py` | Official data processing, reconstruction evaluation on original fields, and risk-error metrics |
| `environment.py`, `resources.py` | Gymnasium interaction, joint actions, resource accounting, budgets, and public state |
| `baselines.py` | Fixed, round-robin, time-greedy, and risk-greedy scheduling, plus shared warm-up |
| `comparisons/` | Explicit reference adapters, audit calibration comparisons and integrated orchestration |
| `audit.py` | Save actual uploads, perform budgeted audits, and isolate attack inputs from reference ground truth |
| `pipeline.py`, `config.py` | Stage orchestration, configuration snapshots, and source/dependency identity |
| `modal_runner.py`, `artifacts.py` | Cloud jobs, Volumes, run status, and artifact storage |
| `tests/` | Framework behavior and boundary tests |

The Python modules listed above reside in `src/cloud_edge_rl/`. See [RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md) for the full research protocol, [TASKS.md](docs/TASKS.md) for implementation tasks and course-requirement mapping, and [REFERENCES.md](docs/REFERENCES.md) for third-party sources. These supporting documents are currently in Chinese.

## Core Interfaces You Implement

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

The notebook submits configuration snapshots through `modal_runner`. Modal builds an image from the same source and `uv.lock`, simulating three clients within one logical training task. This represents simulated clients rather than a network deployment across three physical edge machines. GPU type, CPU, memory, timeout, concurrency, and Volume name are read from the notebook; experiment resources are not hardcoded in Python modules.

For first-time use, authenticate locally following the [official Modal guide](https://modal.com/docs/guide). Submission, waiting, result retrieval, and artifact downloads are provided by `submit_runs`, `wait_runs` / `get_run`, and `download_run`; use the notebook for stage-specific entry points. The Volume stores the data cache, configuration, metrics, attack inputs, audit labels, and checkpoints. Remote tasks commit persistent logs on both success and failure.

`runs/<run_id>/` stores each run's inputs and results. Identical source and experiment inputs produce a stable identity, and existing runs are not silently overwritten. When `workflow.reuse_completed` is enabled, rerunning reads the completed summary; a failed or unfinished run with the same identity still raises an error. To resume, change `experiment.tag` in the sole configuration cell and specify the old checkpoint as an explicit input to the new run.

After a comparison-workflow notebook controller stops, set a new `comparisons.workflow.tag` and the prior manifest path in `comparisons.workflow.resume_manifest`, both in the sole CONFIG cell. The final section keeps the old manifest read-only, reconnects saved Modal call IDs, and carries completed phases into a new exclusive journal without resubmitting those training jobs. Algorithms, seeds, resources and budgets must match. Existing tasks retain their original provenance; future phases record their actual new source identity. Remote algorithm failures still raise errors and cannot be bypassed through recovery.

Full recovery within the same risk version requires you to save and restore the policy, optimizer, random state, replay buffer, and wrapped environment's `snapshot()`. Restore with `load_snapshot()` without calling `reset` or repeating warm-up. When changing risk versions, retain policy weights, clear the old replay buffer, and start a new stage. Your `core.py` implementation owns these algorithm-state transitions.

The initial cloud budget is an experiment-planning target, not a hard Modal billing cap. Resource timeouts and audit budgets limit workload; check actual charges against real invocation records and the Modal bill.

## Current Status and Boundaries

See [VERIFICATION.md](docs/VERIFICATION.md) for actual runtime, notebook, test, and Modal verification records. After tuning, both `validate` and `evaluate` inherit the relevant training snapshot through `inputs.training_run_ids`. Explicit scheduling baselines use `evaluate_baselines`, so current default parameters do not overwrite a selected trial's configuration.

Code and functions include concise Chinese and English explanations. Necessary interface checks raise errors directly rather than silently fabricating inputs or switching to substitute algorithms. On 2026-10-05, the Modal L4 workflow completed 13 phases, 26 real runs and a 26-cell executed notebook; 145 tests passed. The initial audit-coverage failure and subsequent artifact-I/O controller timeout remain recorded. Explicit recovery reattached 19 existing calls and submitted only seven new tasks. See the [integration results and raw CSVs](docs/BASELINE_INTEGRATION_RESULTS.md). Your core remains yours to implement.

This is a small feasibility run. Attacks with 200 iterations and one restart performed below the marginal prior, with zero EMR for all six scheduler audits. Four greedy/MetaCS comparisons chose identical post-warm-up actions, so their risk differences cannot establish a scheduling improvement. TabM reduced MAE but increased mean underestimation. SQT/LQT achieved full audited coverage with risk bounds near 1; OCP missed the 0.9 target. Calibration was a post-hoc comparison and did not change scheduler rewards. Formal research requires larger budgets, broader coverage and paired seeds.
