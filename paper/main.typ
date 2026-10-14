#import "course-acm.typ": course-acm

// English writing skeleton; experimental parameters remain in experiment.ipynb.
// 英文写作骨架；实验超参数仍只在 experiment.ipynb 中调整。
#let title = [Risk Prediction Reliability in Cloud-Edge Federated Learning: Uncertainty-Aware Scheduling and Budgeted Auditing]
#let authors = ("TO COMPLETE: Author Name - Student ID - Institution",)
#let abstract = [
  In the cloud–edge federated learning,  client and batch selection involves trade-offs among task accuracy, resource consumption, and gradient-leakage risk. This study implements a risk-aware reinforcement learning scheduler for synchronous FedSGD on the Adult dataset under an IID setting. The scheduler links a joint action space with 12 possible actions, public observations, and an exposure-weighted risk ledger to a finite-horizon incremental reward signal. Its policy is optimized using Double DQN and PPO. In a seven-method comparison using three paired development seeds, PPO reduced mean simulated time and energy use by 19.36% and 9.43%, respectively, relative to the fixed scheduling policy, while maintaining comparable accuracy; its predicted exposure-risk proxy increased by 11.02%. In a paired multi-step-return ablation using a single seed, each arm was trained for 1,200 interactions. Compared with the 1-step return, the 3-step return reduced simulated energy use by 34.00%, alongside a 13.80% increase in the predicted risk proxy. Overall, this study brings learned scheduling strategies and return horizons into a unified analysis of the trade-off between resource efficiency and proxy-estimated risk. By combining batch-conditioned residual analysis with attack-prior comparisons, it characterizes the scope and interpretive limits of the risk proxy, providing an auditable analytical framework for evaluating cloud–edge federated learning schedulers.

]
#let keywords = ("Cloud–Edge Federated Learning", "Reinforcement Learning–Based Scheduling", "Risk Proxy", "Exposure-Weighted Risk", "Resource–Risk Trade-Off","N-Step Returns")
#show: course-acm.with(title, authors, abstract, keywords)

= Overview
== Motivation and Context
Although federated learning(FL) reduces the need to collect raw data by centre, model updates uploaded by clients may still leak information about their local training data @fedavg @fl_openproblems. Studies of gradient inversion reveal that attackers may reconstruct client training samples from observation of gradients to the server @dlg @ig. To tabular data, reconstruction evaluation also requires mapping recovered values from the encoded space back to the original fields and comparing them at the field-semantic level @tableak. Therefore, this study directly responses the question raised by privacy-profiling research on tabular FL @tabular_gia: evaluation should go beyond model accuracy, which need quantify how much information an attacker can recover from client-uploaded model updates.

Based on the above reasons, Cloud scheduling must jointly account for the privacy risk, computational cost, and learning benefit of each upload, these three elements cannot be considered in isolation. Reducing the local batch size decreases the number of records involved in each update, but may also affect the identifiability of individual records from the resulting gradients. On the another hand, increasing the participating-client set raises the number of uploads per round and may prolong synchronization delays as the server waits for slower clients. The value of a schedule action bases on its long-term effect over the training, and cannot be determined only by its proxy score in the current round. Accordingly, we first formalize the long-term objective and then describe how to estimate and optimize it.

Running a strong reconstruction attack for every candidate action is expensive to computation. and ground-truth reconstruction labels are available only after the corresponding uploads have been audited. A practical alternative is to train a risk proxy on audited uploads and use public context to estimate the risk of candidate actions. However, the scheduler actively favors actions that score well under the proxy, so proxy errors may feed back into and shape subsequent decisions. However, because the scheduler actively selects actions based on the risk proxy’s scores, the context–action distribution it visits may diverge from the proxy’s training distribution. Even if the proxy has low overall average error on existing data, the policy may concentrate on regions where it systematically underestimates risk, making training-set error statistics unrepresentative of its errors in the regions the policy actually visits @learned_reward_perils.

Motivated by these considerations, we formulate the joint selection of participating clients and local mini-batches as a sequential decision problem. At each round, the scheduler acts on public state information and receives feedback based on changes in accuracy and simulated resource consumption, together with the terminal change in the risk proxy. Rather than greedily selecting the action with the lowest current-round proxy score, the reinforcement-learning policy uses subsequent returns to assess each action’s long-term value over the remaining training horizon. Finally, proxy-prediction accuracy and policy-level risk must be evaluated separately: low prediction error on a fixed set of uploads does not necessarily imply that a policy using the predictor will achieve lower true risk.

== Research Questions
We addresses three related 3 related questions in this study: how to formalize joint client–batch scheduling with a terminal exposure-risk objective as a trainable and auditable reinforcement learning(RL) task (RQ1); what trade-offs in accuracy, resource consumption, and predicted risk learned policies and transparent rule-based baselines exhibit under matched development conditions and at a fixed training endpoint (RQ2); and what scheduling behaviors correspond to 1-step and 3-step Double DQN targets under a fixed risk-proxy version and a short interaction budget (RQ3). To address these questions, we specify the task via its interface, reward identity, and implementation. We compare seven methods across three paired seeds, and examine the effect of return length by a single N1/N3 paired comparison from a separate batch.

== Contributions and Scope
The main contributions of this study are in three part: First, we formulate and implement a risk-aware federated-scheduling task that integrates public observations, a 12-action joint space over client-set and local-batch choices, an exposure-risk ledger, and finite-horizon incremental rewards within a unified sequential decision-making framework. Second, we compare learned policies with transparent baselines under matched development conditions and a fixed training-interaction budget. Results from three paired seeds show that PPO reduces simulated time and energy relative to fixed scheduling. Third, using a single paired run, we descriptively compare recorded action choices and short-budget resource outcomes under 1-step and 3-step Double DQN return targets. The contribution lies in task formulation, mechanism comparison, and development-stage empirical evidence for this setting. It provides the analysis of the trade-off between simulated resource consumption and the predicted risk proxy.

= Related Work
== Federated Scheduling and Reinforcement Learning
FL performs collaborative optimization over data distributed across clients. Differences in local data distributions and device abilities lead to statistical and system heterogeneity respectively, which make participant selection a trade-off between model utility and execution efficiency @fedavg @oort. Oort introduces guided participant selection that jointly accounts for each client’s statistical utility for improving model accuracy and training speed, thereby improving time-to-accuracy performance @oort. These studies reveal that client participation as a resource-aware scheduling problem which balances statistical utility and system efficiency, thereby providing a methodological basis for participation decisions and resource-efficiency evaluation in cloud–edge collaboration.

RL can further utilize client-selection policies in response to training feedback. FedRLCS bases on Double DQN to select client subsets, mainly addressing convergence difficulties arising from non-IID data. It evaluates selection performance by the number of communication rounds required to reach a target accuracy @fedrlcs. In contrast, MetaCS-FL explots a metaheuristic framework to client selection, which provide a search-based comparator to learned policies @metacs. Together, these methods indecate two distinct approaches: learning selection policies from future returns and searching over candidate client combinations under an objective function. Building on the joint consideration of training utility and system costs, this study places client-set selection and local batch size in a shared action space. It also introduces a risk proxy learned from attack-derived labels, thereby linking scheduling decisions to resource consumption and predicted gradient-leakage risk.

Regarding optimization methods, DQN learns action values in discrete action spaces, Double DQN mitigates overestimation of action values, and PPO performs policy-gradient-based policy optimization @dqn @double_dqn @ppo. Gallici et al.’s PQN investigates the stability and efficiency of deep temporal-difference learning via LayerNorm-based regularization and parallelized online sampling as key design elements @pqn. PQN is a general-purpose RL algorithm, it serves as a design reference for comparing optimization methods and examining reward-propagation mechanisms.

== Tabular Gradient Inversion


== Risk Surrogates and Uncertainty


== Budgeted Auditing and Calibration


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
