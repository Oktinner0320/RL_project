# Risk-Aware Reinforcement Learning Framework for Cloud–Edge Federated Training

[中文](README.md) | [English](README.en.md)

This project studies whether an empirical privacy-risk surrogate systematically underestimates risk when dynamic scheduling changes the training trajectory, and whether uncertainty estimates and a limited budget of gradient-inversion audits improve decision reliability. The application jointly schedules client participation and batch size, evaluating model utility, time, and energy together.

This is a research scaffold whose algorithms are implemented by the researcher. It provides data processing, a Gymnasium environment, resource simulation, scheduling baselines, audit records, experiment orchestration, and Modal integration. You implement the PyTorch model, FedSGD, DQN/PPO, risk estimator, and gradient inversion. These interfaces currently raise `NotImplementedError`; the framework does not supply synthetic gradients, substitute convergence curves, or fabricated research results.

## Installation and Entry Point

Run from the project directory:

```powershell
uv sync --locked
uv run jupyter lab
```

Open [experiment.ipynb](experiment.ipynb). Adjust all global experiment parameters, stage switches, run inputs, and Modal resources in the notebook's single `CONFIG` cell. `.python-version` and `pyproject.toml` specify the Python and UV versions; `uv.lock` determines dependency resolution. The local environment uses CPU PyTorch for debugging and analysis, while the Modal Linux image uses the locked CUDA wheel.

By default, the notebook only checks the local runtime and prepares the Adult dataset. Research training and cloud jobs must be explicitly enabled through the stage switches in `CONFIG.workflow.jobs`. Initial data preparation requires access to the official data source. Do not introduce another tuning entry point in modules, a second configuration file, command-line hyperparameters, or other notebook cells.

Check the framework:

```powershell
uv run pytest
uv run ruff check .
```

Test doubles exist only in `tests/` to verify interfaces, accounting, and isolation. Passing tests does not establish that learning algorithms are complete, gradient inversion is effective, or cloud training on Modal has passed algorithm-level validation.

## Files and Responsibilities

| Entry | Purpose |
|---|---|
| `experiment.ipynb` | Sole entry point for configuration, submission, status queries, downloads, and result analysis |
| `src/cloud_edge_rl/core.py` | All learning and inversion cores that you implement |
| `data.py`, `metrics.py` | Official data processing, reconstruction evaluation on original fields, and risk-error metrics |
| `environment.py`, `resources.py` | Gymnasium interaction, joint actions, resource accounting, budgets, and public state |
| `baselines.py` | Fixed, round-robin, time-greedy, and risk-greedy scheduling, plus shared warm-up |
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

`CloudEdgeEnv` follows Gymnasium's `reset` / `step` interface. Public observations include resources, progress, sampling history, candidate-action utility/cost/risk, and `risk_available`. The `progress` fields are ordered as validation loss, validation accuracy, elapsed-time fraction, used-round fraction, and remaining-sample-budget fraction. The last field may become negative when a complete training round exceeds the budget. Use Gymnasium's `spaces.flatten` if your algorithm needs a flat vector.

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

Full recovery within the same risk version requires you to save and restore the policy, optimizer, random state, replay buffer, and wrapped environment's `snapshot()`. Restore with `load_snapshot()` without calling `reset` or repeating warm-up. When changing risk versions, retain policy weights, clear the old replay buffer, and start a new stage. Your `core.py` implementation owns these algorithm-state transitions.

The initial cloud budget is an experiment-planning target, not a hard Modal billing cap. Resource timeouts and audit budgets limit workload; check actual charges against real invocation records and the Modal bill.

## Current Status and Boundaries

See [VERIFICATION.md](docs/VERIFICATION.md) for actual runtime, notebook, test, and Modal verification records. After tuning, both `validate` and `evaluate` inherit the relevant training snapshot through `inputs.training_run_ids`. Explicit scheduling baselines use `evaluate_baselines`, so current default parameters do not overwrite a selected trial's configuration.

Code and functions include concise Chinese and English explanations. Necessary interface checks raise errors directly rather than silently fabricating inputs or switching to substitute algorithms. A real Modal image build, L4 runtime check, and artifact download have completed, establishing cloud connectivity. Algorithm-level training still requires validation. Framework verification and research conclusions are recorded separately; risk calibration, RL training, independent attack evaluation, and course research conclusions remain to be completed.
