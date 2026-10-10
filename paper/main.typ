#import "course-acm.typ": course-acm

// English writing skeleton; experimental parameters remain in experiment.ipynb.
// 英文写作骨架；实验超参数仍只在 experiment.ipynb 中调整。
#let title = [Risk Prediction Reliability in Cloud-Edge Federated Learning: Uncertainty-Aware Scheduling and Budgeted Auditing]
#let authors = ("TO COMPLETE: Author Name - Student ID - Institution",)
#let abstract = [
  

  //之后等实际实验进行后再补充实验结果和结论
]
#let keywords = ("federated learning", "reinforcement learning", "gradient inversion", "uncertainty", "budgeted auditing")
#show: course-acm.with(title, authors, abstract, keywords)

= Overview
== Motivation and Context
*To write:* Explain why cloud-edge scheduling jointly affects model progress,
resource use, and exposure of individual client gradients. Introduce the risk
that a surrogate trained on one collection policy becomes unreliable along a
different training trajectory. Distinguish the deployed scheduling decision from
the trusted evaluator's later reconstruction measurement. Use the source research
summary as motivation and define the intended application scope.

== Research Questions
*RQ1:* Does dynamic scheduling systematically underestimate empirical
reconstruction risk when the surrogate was fitted on other training trajectories?

*RQ2:* Can predictive uncertainty identify underestimation, and can conservative
scoring improve the tradeoff between reliability, model quality, and cost?

*RQ3:* Under the same attack-call and computation budgets, can active auditing
and stagewise calibration improve subsequent prediction and scheduling?

*To write:* Attach a measurable outcome and a falsification condition to each
question. Answer them from independent evaluation rather than the scheduler's
own predicted rewards.

== Contributions and Scope
*To write:* Separate the implemented experimental framework, observed pilot
feasibility, and research contributions established by formal comparisons.
Restrict the main study to Adult, three simulated clients, a fixed small MLP,
FedSGD, twelve actions, and DQN. Treat PPO as an extension. State which learning
and attack cores are original implementations and which runnable third-party
baselines are adapted, with clear attribution.

= Related Work
== Federated Scheduling and Reinforcement Learning
*To write:* Introduce federated training using @fedavg, while making clear that
the present protocol uses one-step FedSGD. Relate DQN @dqn to the discrete joint
action space. Discuss PQN @pqn and MetaCS-FL @metacs as comparison or design
references, stating their assumptions and implementation differences. Compare
their scheduling objectives with the present focus on the reliability of an
empirical privacy-risk surrogate.

== Tabular Gradient Inversion
*To write:* Explain gradient matching through DLG @dlg and Inverting Gradients
@ig, then discuss mixed numerical and categorical reconstruction using TabLeak
@tableak. Describe tabular privacy profiling @tabular_gia and its official
tabular-GIA repository in the source notes,
without equating image reconstruction metrics with tabular recovery. Identify
which threat-model assumptions and evaluation procedures are reused, and which
attack implementation and hyperparameters still require validation.

== Risk Surrogates and Uncertainty
*To write:* Review learning a risk predictor from independently audited updates.
Use TabM @tabm as a tabular baseline and explain the intended uncertainty
mechanism. Distinguish accuracy, underestimation, and uncertainty ranking. An
ensemble spread is not automatically a calibrated confidence interval; assess
its empirical relationship to errors before interpreting a conservative score.

== Budgeted Auditing and Calibration
*To write:* Relate audit selection to uncertainty and optimization-oriented
calibration, using @conformalopt and @ocp as methodological references. State
where their assumptions differ from policy-dependent, sequential training data.
Do not transfer formal coverage guarantees without establishing their
conditions. Compare periodic, random, and active auditing under both attack-call
and computation budgets, with independent final evaluation.

= Problem Formulation
== Data, Clients, and FedSGD
*To write:* Document the UCI Adult dataset @adult, the official test split, and
training-only preprocessing. Specify validation selection, missing-category
encoding, numerical scaling, and IID versus label-skewed clients. Define a round
as client batch-mean gradients computed at the same pre-update weights, a
sample-weighted aggregate, and one server update. Model weights persist through
an episode. Report the fixed architecture and optimizer implemented by the author.

== Threat Model and Empirical Risk
*To write:* Define an honest-but-curious server that sees each selected client's
gradient before aggregation, the pre-update model, known batch labels, batch
size, and public feature schema. Raw records are available only to the trusted
evaluator. One audit attacks one complete client batch. Explain within-batch
matching, recovery in original feature units, and the separate exact-match
record rate. Fix numerical tolerances and categorical recovery rules before comparison.

== Observations and Joint Actions
*To write:* List currently available resources, validation progress, participation
and exposure history, remaining budgets, and candidate predicted risk and
uncertainty. Acknowledge partial observability. Exclude future resources,
candidate true gradients, reconstruction truth, and test-set metrics.
@tab:actions uses zero-based client IDs; batch sizes and action IDs align from
left to right in each row.

#figure(
  pad(x: 0.5pt, table(
    columns: (1fr, 1.4fr, 1.2fr),
    align: left,
    inset: 6pt,
    stroke: 0.5pt,
    table.header([*Action IDs*], [*Client set*], [*Batch sizes*]),
    [0, 1, 2], [{0, 1}], [8, 32, 64],
    [3, 4, 5], [{0, 2}], [8, 32, 64],
    [6, 7, 8], [{1, 2}], [8, 32, 64],
    [9, 10, 11], [{0, 1, 2}], [8, 32, 64],
  )),
  caption: [The twelve discrete scheduling actions.],
) <tab:actions>

== Rewards, Resources, and Termination
Let $Delta U_t$ denote validation-loss reduction and let $s_T$ and $s_E$ be
normalization scales fixed after the pilot. For selected clients $C(a_t)$, define
the clipped score per client and then average it:

$
  R_(t,i)^"cons" = op("clip")(hat(R)_(t,i) + kappa u_(t,i), 0, 1),
  quad R_t^"cons" = 1 / abs(C(a_t)) sum_(i in C(a_t)) R_(t,i)^"cons".
$ <eq:risk>

$
  r_t = Delta U_t - alpha T_t / s_T - beta E_t / s_E - lambda R_t^"cons".
$ <eq:reward>

*To write:* Explain @eq:risk and @eq:reward, and report worst-client risk
separately. Model a parallel synchronous round through its slowest participating
client, including download, computation, upload, and aggregation. Dense gradient
bytes depend on parameter count and precision, not proportionally on batch size.
Advance paired resource traces by simulated elapsed time. Distinguish task or
budget termination from the additional round-limit truncation.

= Methodology
== Pipeline and Risk Training
*To write:* Describe collection across clients, batch sizes, and early-to-late
training stages. Split risk-fitting, validation, and final evaluation by complete
run, so neighboring checkpoints do not leak across partitions. Freeze the risk
estimator within a stage and record its version. @fig:pipeline shows the
information boundary; only trusted auditing obtains empirical attack labels.

// Center node labels without prose justification / 流程节点居中，避免正文两端对齐拉伸。
#let pipeline-node(body) = rect(width: 100%, height: 54pt, inset: 6pt, stroke: 0.5pt, {
  set par(justify: false, first-line-indent: 0pt)
  align(center + horizon, body)
})

#figure(
  pad(x: 0.5pt, grid(
    columns: (1fr, 1fr), gutter: 12pt,
    pipeline-node[Available observation and estimated risk],
    pipeline-node[Scheduler selects one action],
    pipeline-node[FedSGD executes selected client updates],
    pipeline-node[Budgeted audit informs the next stage],
  )),
  caption: [Read left to right, then top to bottom. Auditing follows upload and cannot undo the current exposure.],
) <fig:pipeline>

== DQN and Scheduling Baselines
*To write:* Specify the author's replay buffer, target network, exploration
schedule, Bellman target, and optimizer. Compare against the selected fixed
configuration, round robin, time greedy, risk-aware greedy, and conservative
greedy. Candidate utility must come from shared warmup data and historical
estimates; greedy policies cannot execute unchosen actions to obtain true future
utility. Charge warmup to every method's budget and report baseline selection rules.

== Uncertainty-Aware Scoring
*To write:* Define the predicted mean and uncertainty in @eq:risk. Evaluate
uncertainty against absolute error and positive underestimation before assessing
scheduling effects. Select $kappa$ using validation data, then fix it for the
independent comparison. Show risk reliability together with model quality and
resource cost; clipping a score to $[0,1]$ establishes a range, not a privacy or
coverage guarantee.

== Cost-Aware Active Auditing
*To write:* Specify a reproducible acquisition rule over already executed
updates, with estimated attack cost, uncertainty, client coverage, and remaining
budget. Restrict each method by calls and computation, recording actual GPU
seconds. First compare selectors on a shared saved trajectory, then compare
closed-loop schedules separately. Reserve an independent evaluation panel that
covers updates not chosen by the active selector.

== Stagewise Calibration and Policy Continuation
*To write:* Explain the stage boundary: finish a frozen-estimator stage, audit
eligible stored updates, refit on allowed labels, and advance the estimator
version. Keep policy weights but clear replay from the old reward model before
continuing. Distinguish this deliberate transition from resuming the same stage,
which preserves state. Document which updates are evaluated independently and
never feed their labels back into training.

== Hyperparameter Selection
*To write:* Describe the sole experiment control point, `experiment.ipynb`, and
archived configuration snapshots. Separate pilot tuning, validation-based
selection, and frozen final tests. Provide search ranges, selected values, seed
pairing, attack iterations and restarts, reward weights, uncertainty coefficient,
and audit costs. Include sensitivity to attack strength and risk penalties.
Account for the initial USD 20 pilot separately from the formal experiment budget.

= Results
== Experimental Setup and Preliminary Feasibility
*Existing engineering evidence:* The recorded integration includes 26 L4 cloud
tasks across 13 phases. The current Adult split has 26,049 training, 6,512
validation, and 16,281 test records; preprocessing yields 121 encoded features.
The three IID clients contain 8,683 training records each. These records support
execution feasibility, not convergence or a five-seed performance claim.

*To write:* Cite the archived configuration and artifact IDs from the integration
report. Specify the final hardware, time budget, seeds, attacks, data skew,
statistical summaries, and common quality targets. Keep pilot observations
separate from formal result tables and state the effective denominators.

== Attack Validity
*To write:* Validate the implemented attack on controlled small batches and
compare with marginal or prior-only reconstructions. Measure feature recovery,
record exact-match rate, runtime, and sensitivity to iterations and restarts.
The current 200-iteration, one-restart pilot attack was weaker than the marginal
baseline and had zero exact-match rate. Strengthen and validate the auditor
before drawing privacy conclusions from its labels.

== Policy-Induced Underestimation
*To write:* Answer RQ1 using matched independent audits along fixed, greedy,
and learned trajectories. Report MAE, positive underestimation, worst-client
error, and early-stage risk at comparable quality and client-wise processing.
The small pilot risk comparison does not establish reliability: lower MAE and
lower underestimation need not agree. Show trajectory shift and error by stage,
with paired-seed uncertainty and the number of audited updates.

== Uncertainty and Conservative Scheduling
*To write:* Answer RQ2 with uncertainty-error associations, selective error
curves, and conservative-score sensitivity. Compare no-privacy DQN, mean-risk
DQN, conservative DQN, and conservative greedy under the same information.
Report quality-risk-cost frontiers and failure cases, including high uncertainty
that does not correspond to underestimation. Avoid treating post-hoc calibration
as an implemented closed-loop reward correction unless it actually changes later stages.

== Equal-Budget Audit Selection
*To write:* Answer the audit-selection part of RQ3 on the shared trajectory.
Compare periodic, random, and active selection under matched calls and GPU time.
Show client, batch-size, and stage coverage together with subsequent independent
prediction error. Score-query counts alone do not demonstrate attack-cost savings;
uninformative pilot scores near one cannot establish useful audit prioritization.

== Closed-Loop Evaluation and Ablation
*To write:* Test calibrated policies using the independent final panel. Report
equal simulated-time comparisons, similar client-wise processing, and cost to a
common model-quality target. Ablate uncertainty, active selection, stagewise
refitting, and policy continuation. The pilot's identical post-warmup actions
across several methods do not establish policy superiority. Attribute measured
effects to risk correction, sample selection, or long-term control using paired seeds.

= Discussion
== Interpretation and Practical Tradeoffs
*To write:* Interpret each research question using the final independent
measurements. Explain how training stage, batch size, participation, and repeated
exposure affect risk and prediction shift. Discuss when a calibrated greedy
policy may be sufficient and when learned long-term scheduling helps. Separate
simulated communication and energy from actual Modal compute costs and startup time.

== Negative Findings and Failure Cases
*To write:* Retain unsupported hypotheses and counterexamples. Examples to test
include no systematic underestimation, uncertainty with poor error ranking,
calibration with negligible scheduling benefit, and greedy performance similar
to DQN. Diagnose weak attacks, sparse risk labels, collapsed action choices, and
unstable rewards before attributing a null result to the research mechanism.

== Limitations and Validity
*To write:* Discuss the one-dataset, small-model, simulated-client, known-label,
pre-aggregation-gradient scope. Explain attack dependence, partial observability,
selection bias, temporal dependence, finite seeds, and budget overruns at round
boundaries. Do not generalize to secure aggregation, unknown-label attackers,
strict differential privacy, or real edge deployments without further evidence.
Describe the independent evaluation controls and remaining threats to validity.

== Reproducibility, Cost, and Contributions
*To write:* Identify source revision, UV lock, package versions, Notebook
snapshots, seeds, data preprocessing, checkpoints, risk versions, and per-run
artifacts. Report actual GPU seconds and monetary expenditure by stage. Attribute
third-party implementations and licenses, distinguish original cores from
adapters, and state every team member's contribution. Provide a compact rerun
procedure that does not expose authentication credentials.

= Conclusion
== Answers to the Research Questions
*To write last:* Answer RQ1, RQ2, and RQ3 individually with the corresponding
effect sizes and independent evidence. State whether the observations support
each hypothesis and identify the best practical quality-risk-cost tradeoff.
Summarize only contributions demonstrated by the completed experiments; neither
DQN superiority nor successful calibration is a required outcome.

== Next Steps
*To write:* Prioritize extensions supported by observed limitations, such as
stronger tabular inversion, additional datasets or heterogeneity, and PPO after
the DQN main study. Keep future proposals separate from completed work. The
planned final English manuscript has a 20-page content budget excluding
references; this skeleton uses only the pages its actual content needs.

#bibliography("refs/refs.bib", title: "References", style: "association-for-computing-machinery")
