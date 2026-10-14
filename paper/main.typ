#import "course-acm.typ": course-acm
#let clip = math.op("clip")
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
#figure(
  image("figures/rl-project-architecture.png", width: 100%),
  alt: "Architecture of the risk-aware federated scheduling framework. "
    + "Offline gradient-inversion audits provide reconstruction labels "
    + "for risk-ensemble fitting, residual-based bias correction, and freezing. "
    + "Public context, frozen risk predictions, and simulated resource costs "
    + "support joint client-set and batch-size selection. "
    + "One active learning algorithm or scheduling rule controls synchronous, "
    + "sample-weighted FedSGD. Round-level accounting combines predicted "
    + "exposure risk, task utility, and resource costs into incremental rewards "
    + "and feedback. Paired validation compares methods under a fixed protocol. "
    + "Resource costs are simulated, and predicted risk does not constitute "
    + "a formal privacy guarantee.",
  caption: [
    Framework for risk-aware client–batch scheduling in federated learning.
  ],
) <fig:rl-project-architecture>

== System and Synchronized Federated Updates
We consider a horizontal federated-learning system which comprise a cloud server and three logical edge clients, with the client set denoted by $cal(C) = {0, 1, 2}$. Each client retains its local records, while the server maintains a shared task model and coordinates client participation and model updates. Let $w_t$ denote the task-model parameters at the beginning of decision
round $t$, $phi$ the parameters of the frozen risk proxy, and $theta$ the parameters of the scheduling policy. The task model is updated via training. The risk proxy remains fixed within each policy comparison, whereas the scheduling policy is learned through training interactions.

For each participating client $i$, let $cal(B)_(t,i)$ denote its local batch in round $t$, and let $B_(t,i) = abs(cal(B)_(t,i))$ be the number of records in that batch. Let $ell(w\; x, y)$ denote the task loss on an individual record. Each participating client computes a batch-mean gradient at the same pre-update model parameters $w_t$:
$
  L_(t,i)(w) = 1 / B_(t,i) sum_((x, y) in cal(B)_(t,i)) ell(w\; x, y),
  quad g_(t,i) = nabla_w L_(t,i)(w_t).
$ <eq:client-batch-gradient>

The server receives gradients from the participating set $S_t$, weights them by the actual number of records in each local batch, and performs one update:
$
  w_(t+1) = w_t - eta_w sum_(i in S_t)
    (B_(t,i))/(sum_(k in S_t) B_(t,k)) g_(t,i).
$ <eq:fedsgd-update>

Where $eta_w$ is the task-model learning rate. This is synchronous FedSGD: clients upload gradients computed on their current batches @fedavg. At last, changes to the participating client set or local batch sizes affect both the composition of uploaded updates and subsequent task-model states.

== Threat Model and Empirical Risk
We introduce an honest-but-curious server threat model. The server follows the prescribed update protocol but can observe each client’s gradient before aggregation and is assumed to know the model parameters and labels need to attempt private-data reconstruction. All scheduling policies are evaluated via the same task-model architecture, training protocol, attacker knowledge, and inversion budget, so that risk comparisons reflect differences in scheduling behavior.

For each realized upload $u$, let $B_u$ denote its number of records, and let $Y_u in [0, 1]$ denote its raw-field recovery rate. $Y_u$ measures the proportion of fields correctly recovered in the reconstruction. Field matching, numerical tolerances, and categorical decision rules are specified by a common attack-evaluation protocol. The label characterizes empirical reconstructability under a fixed attack protocol; its value depends on both the information exposed by the upload and the effectiveness of attack optimization.

Let $tau$ denote a task trajectory and $cal(U)(tau)$ the set of realized post-warmup upload events in that trajectory. Define exposure-weighted risk as:
$
  R_("exp")(tau) =
    (sum_(u in cal(U)(tau)) B_u Y_u) /
    (sum_(u in cal(U)(tau)) B_u),
  quad R(pi) = bb(E)_pi[R_("exp")(tau)].
$ <risk-objective>

Weighting by record count gives uploads containing more records a greater contribution to the evaluation. $R(pi)$ denotes the expected risk over trajectories induced by policy $pi$. If a record is uploaded more than once, each upload counts as a separate exposure event. Thus, this objective measures the field-recovery proportion across exposure events, not the number of unique
affected records. If there are no post-warmup uploads, risk is undefined. If required attack labels are missing, the full-trajectory risk is indeterminate.

== Observations and Joint Actions
At round $t$, the scheduler selects a joint action $a_t = (S_t, B_t)$ using the public observation $o_t$. @tab:observations-actions summarizes the available action choices and observation components. The four client sets and three local batch sizes form a discrete joint action space of 12 actions, which allow participation scope and local aggregation size to be adjusted jointly.

#figure(
  {
    set par(justify: false, first-line-indent: 0pt)
    table(
      columns: (1fr, 1.4fr, 2.8fr),
      align: left + top,
      inset: (x: 6pt, y: 5pt),
      stroke: none,
      table.hline(stroke: 0.8pt),
      table.header([*Category*], [*Component*], [*Definition / content*]),
      table.hline(stroke: 0.5pt),
      table.cell(rowspan: 2)[Joint action],
      [Client set $S_t$], [$cal(S) = {{0, 1}, {0, 2}, {1, 2}, {0, 1, 2}}$],
      [Local batch size $B_t$], [$B_t in {8, 32, 64}$],
      table.cell(rowspan: 5)[Public observation],
      [Resource state], [Current resource conditions],
      [Validation progress], [Task-model validation feedback],
      [Participation history], [Summary of past client participation],
      [Candidate-action estimates], [Estimated resource costs and risk-proxy values for each candidate action],
      [Exposure ledger], [Cumulative exposure statistics],
      table.hline(stroke: 0.8pt),
    )
  },
  kind: table,
  caption: figure.caption(position: top)[Joint action space and public observation components.],
) <tab:observations-actions>

The action specifies the local batch size $B_t$, whereas model updates and risk statistics use the actual number of records $B_(t,i)$ in each upload. The full decision history is $h_t = (o_1, a_1, dots, a_(t-1), o_t)$, and the implemented policy acts on a fixed-dimensional summary of this history without assuming that $o_t$ is strictly Markov.

Private records, inversion ground truth, and the true gradients associated with candidate actions are excluded from the scheduler's observation. Ground-truth reconstruction labels are generated by a trusted evaluator after the corresponding upload has occurred.

== Resource Costs and the Terminal Optimization Objective
Let $A_t$ denote task-validation accuracy at the end of decision round $t$, and let $T_t$ and $E_t$ denote cumulative simulated time and energy respectively, up to that round. Resource quantities are estimated via fixed bandwidth, computational-load, and power parameters. They are used to compare the system costs of actions and are interpreted separately from measured device runtime, wall-clock training time, and cloud expenditure.

For each realized upload with public context $x_u$, the frozen risk proxy returns a score $p_u in [0, 1]$. Let $cal(U)_t(tau)$ denote the set of post-warm-up uploads up to round $t$ along trajectory $tau$. Define the cumulative risk-proxy estimate as:
$
  hat(R)_t(tau) =
    (sum_(u in cal(U)_t(tau)) B_u p_u) /
    (sum_(u in cal(U)_t(tau)) B_u).
$

The construction of the ensemble score $p_u$, which contain ensemble disagreement and calibration, is described in Methods. When cumulative exposure is nonzero, $hat(R)_t$ uses the same record-count weighting as true risk but replaces attack
labels with predicted scores, thereby providing a computable risk objective for online scheduling.

Taking the end of warm-up as the decision starting point, denote accuracy and cumulative resources by $A_0$, $T_0$, and $E_0$, and the actual decision endpoint by $H$. Define the terminal changes as $Delta A = A_H - A_0$, $Delta hat(R) = hat(R)_H - hat(R)_0$, $Delta T = T_H - T_0$, and $Delta E = E_H - E_0$. Let $cal(P)$ be the set of policies that make decisions in the specified observation and action spaces. The terminal objective optimized in this study is:
$
  max_(pi in cal(P)) J(pi),
  quad J(pi) = bb(E)_pi[
    lambda_A Delta A
    - lambda_R Delta hat(R)
    - lambda_T Delta T / s_T
    - lambda_E Delta E / s_E
  ].
$ <scheduling-objective>

Where $lambda_A$, $lambda_R$, $lambda_T$, and $lambda_E$ are fixed nonnegative weights, and $s_T, s_E > 0$ are resource-normalization scales. This weighted objective expresses preferences among accuracy gains, resource consumption, and predicted risk. Risk-proxy optimization and true-risk evaluation correspond to $hat(R)$ and $R$ respectively. The experiments report accuracy, time, energy, and predicted risk to preserve the trade-offs across these dimensions.

== Finite Horizon and Termination Conditions
The task runs for at most 30 rounds, which include 12 shared warm-up rounds and up to 18 policy-controlled decision rounds, such that $H <= 18$. The shared warm-up establishes a comparable starting point for the task model and consumes training interactions and resources, but its uploads are excluded from the post-warm-up risk set. The empty risk ledger is initialized as $hat(R)_0 = 0$. The reported terminal totals $T_H$ and $E_H$ include warm-up costs, whereas resource increments in the optimization objective are computed relative to $T_0$ and $E_0$. Under identical initial conditions, the shared warm-up cost is independent of subsequent policy choices, so not affect the ranking of policies under the objective. The task terminates when the maximum number of rounds is reached or a predefined task-level termination condition is met. The discount factor is set to $gamma = 1$, so the undiscounted sum of per-round incremental rewards is consistent with the terminal objective defined above.

= Methodology
== Attack Audit and Recovery-Rate Label Generation <attack-labels>
The risk supervision is derived from gradient-inversion audits of actual uploads. For each upload $u$, the attacker receives the pre-aggregation gradient $g_u$, task-model parameters $w_u$, the input-encoding specification, and the ground-truth batch labels $y_u$. It only optimizes the candidate input $hat(X)_u$. Following the cosine gradient-matching objective @ig, the attack
solves:
$
  min_(hat(X)_u)
    1 - cos(g_u, nabla_w cal(L)(w_u\; hat(X)_u, y_u)).
$ <attack-objective>

In the current fitting audits, candidate inputs are initialized with Gaussian random values in the encoded space, and optimized with Adam. Each of three independent restarts has 1,000 optimization steps, with the learning rate decreasing linearly from 0.1 to 0.01. The final reconstruction is the candidate with the lowest matching loss across all restarts and iterations. Only after reconstruction is complete, the trusted evaluator access the original records to calculate recovery metrics.

Evaluation is addressed in the original field space. A row-matching cost is constructed from numerical absolute distances standardized by training-data scales and the number of categorical mismatches at first. And then, the Hungarian algorithm is used to align reconstructed records with ground-truth records @tableak. For a numerical field $j$, recovery is counted as correct when the absolute error is at most $0.05 s_j$, where $s_j$ is the standard deviation of that field in the task's training data; constant fields use zero tolerance. Categorical fields must match exactly. Let $C_(r,j)$ indicate whether field $j$ of aligned record $r$ is recovered correctly, and let $d$ be the number of original fields. Then：
$
  Y_u = 1 / (B_u d)
    sum_(r = 1)^(B_u) sum_(j = 1)^d C_(r,j).
$ <recovery-label>

The row-level exact-match rate (EMR) requires every field in a record to be recovered correctly. As the units of evaluation, original fields prevents the dimensionality introduced by one-hot encoding from changing field weights in the recovery rate.

To distinguish attack information from data priors, the evaluator also generates two baselines: uniform random reconstructions over numerical ranges and categorical supports. Reconstructions sampled independently from each field's empirical marginal effect of training. Both baselines use the same matching and field-level criteria as the attack. The audits used to fit the risk proxy are restricted to actual post-warm-up uploads, with at most 12 attack calls and 1,200 seconds of attack time per collection run. Before each audit begins, the remaining budget is checked against its estimated cost. Afterward, the full measured runtime of the attack is charged. If the time budget is exceeded, the completed result is retained and further audits are stopped.

== Risk Ensemble, Empirical Disagreement, and Batch-Conditioned Calibration <risk-method>
The risk model maps the public context $x_u$ of an upload to an estimated field-recovery rate, which contain 13 input features. The model is composed of five independent MLPs. Each MLP has two 64-unit ReLU hidden layers and a sigmoid output. The fitting and validation data are split by complete collection run, with the number of validation runs set to the ceiling of 20% of all runs. Means and standard deviations for risk features are computed via fitting runs, and constant features are assigned a unit scale. Let $cal(D)_("fit")$ denote the set of uploads used for fitting. Each MFP is trained on the same recovery labels using the loss:
$
  L_("fit") =
    1 / (M abs(cal(D)_("fit")))
    sum_(u in cal(D)_("fit")) sum_(m = 1)^M
      (f_(phi,m)(x_u) - Y_u)^2.
$ <risk-fit-loss>

The optimizer is AdamW with a learning rate of $10^(-3)$, weight decay of $10^(-4)$, a mini-batch size of 16, and a maximum of 100 epochs. In each epoch, model selection is based on the ensemble-mean MSE on the internal validation groups. The weights with the lowest validation error are restored, after which the network is frozen. Thus, the loss weights uploads and MLPs equally, it not minimize exposure-weighted error directly. The ensemble mean and MLP disagreement are defined as:
$
  z_u = 1 / M sum_(m = 1)^M f_(phi,m)(x_u),
  quad
  sigma_u^2 = 1 / M sum_(m = 1)^M (f_(phi,m)(x_u) - z_u)^2.
$

In the selected weights, let $b_u$ denote the requested batch-size
condition encoded in the public risk features, and $B_u$ continues to denote the actual number of records in the upload. For each batch condition $b$, compute the mean residual $c_b$ over internal-validation uploads with that condition. If no validation uploads have condition $b$, use the global mean residual over the internal validation set:
$
  c_b =
    1 / n_b
    sum_(u in cal(D)_("val"), b_u = b) (Y_u - z_u).
$ <batch-residual>

The method applies a nonnegative residual correction followed by fixed 50% shrinkage. Define the raw prediction, the fully corrected one-sided prediction, and the calibrated point estimate as:
$
  p_("raw",u) = clip(z_u, 0, 1),
  quad
  p_(+,u) = clip(z_u + max(c_(b_u), 0), 0, 1),
$
$
  mu_u^("cal") = (1 - rho) p_("raw",u) + rho p_(+,u),
  quad rho = 1 / 2.
$ <shrinkage>

The fully corrected prediction is clipped before interpolation. Calibration adjusts only the point estimate, and $sigma_u$ is still computed from the original ensemble outputs. Because model selection and residual estimation use the same internal
validation set, this procedure is an empirical bias correction, not independent conformal calibration or full multicalibration @multicalibration. The risk score used by the environment to evaluate candidate actions and accumulate realized exposure is:
$
  p_u = clip(mu_u^("cal") + kappa sigma_u, 0, 1).
$ <risk-score>

In the executed configurations for both the seven-method comparison and the N1/N3 runs with $kappa = 0$. Therefore, the risk score uses the calibrated point estimate; ensemble disagreement of MLPs remains available as a public candidate feature and not contribute an additional penalty to the score.

== Cumulative Proxy and Terminal-Consistent Incremental Reward
For each executed decision round, let $e_t$ denote the total actual exposure from uploads, $overline(p)_t$ the exposure-weighted mean proxy score, and $N_(t-1)$ the cumulative exposure before round $t$. The cumulative proxy $hat(R)_t$ satisfies:
$
  hat(R)_t =
    (N_(t-1) hat(R)_(t-1) + e_t overline(p)_t) /
    (N_(t-1) + e_t).
$

Thus, when prior exposure is nonzero:
$
  Delta hat(R)_t =
    e_t / (N_(t-1) + e_t)
    (overline(p)_t - hat(R)_(t-1)).
$ <risk-increment>

On the first round, the empty ledger initializes the proxy to zero, so the increment equals that round's proxy. An upload with a proxy score below the historical average can reduce the cumulative proportion even it adds exposure.

To optimize @scheduling-objective, each terminal change is decomposed into per-round increments. The executed development runs used a fixed risk-proxy version and fixed reward weights. The per-round decision reward is:
$
  r_t =
    lambda_A Delta A_t
    - lambda_R Delta hat(R)_t
    - lambda_T (Delta T_t) / s_T
    - lambda_E (Delta E_t) / s_E.
$ <aligned-reward>

Summing from decision round 1 to the terminal round $H$ gives:
$
  sum_(t = 1)^H r_t =
    lambda_A (A_H - A_0)
    - lambda_R (hat(R)_H - hat(R)_0)
    - lambda_T (T_H - T_0) / s_T
    - lambda_E (E_H - E_0) / s_E.
$

The scales $s_T$ and $s_E$ normalize the reward. With a fixed risk-proxy version, fixed weights, and a discount factor of one, the max return corresponds to the terminal weighted objective over the risk proxy, accuracy, and resources. In addition, shared warm-up costs remain included in the final $T$ and $E$.

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
