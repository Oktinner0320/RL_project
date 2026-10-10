"""登记消融差异；共同实验设置只来自共享 JSON。
Register ablation differences; common scientific settings come only from the shared JSON.
"""

from copy import deepcopy

from cloud_edge_rl.fresh_benchmark import run_config

STUDIES = {
    # N 是唯一可执行假设；其余条目只登记研究问题，不意味着实现或执行获准。
    # N is the only executable hypothesis; other entries register ideas, not approval.
    'N': {'status': 'implemented', 'question': 'Does 3-step improve short-budget Double DQN?',
          'arms': {'N1': {'n_step': 1}, 'N3': {'n_step': 3}},
          'training_seeds': [96101, 96102, 96103],
          'validation_seeds': [96201, 96202, 96203],
          'seed_status': 'proposed_not_verified_unused_or_reserved',
          'pilot_steps_per_arm': 500, 'final_steps_per_arm': 1200,
          'hypothesis': 'Faster reward propagation improves fixed-budget development outcomes.',
          'falsification': 'Reject expansion on invalid numerics, counts or restore; reject the '
                           'candidate if paired fixed-endpoint outcomes fail the preregistered '
                           'utility/resource gates or show no risk-score improvement.',
          'decision': 'Pilot is an execution gate, not efficacy evidence. Three paired seeds '
                      'and an independently frozen reference are required for selection.'},
    'L': {'status': 'planned_not_implemented',
          'question': 'Fixed versus stage-frozen adaptive time/energy multipliers.',
          'requirements': ['raw replay reward components', 'no within-task lambda update']},
    'Q': {'status': 'planned_not_implemented',
          'question': 'MSE versus pinball loss at tau=0.9, initially offline without calibration.',
          'requirements': ['paired fitting seeds', 'independent run-group panel',
                           'underestimation and exposure-weighted MAE', 'no coverage guarantee']},
    'E': {'status': 'planned_not_implemented',
          'question': 'Add a separately registered total post-warmup exposure penalty.',
          'requirements': ['one frozen coefficient', 'same original mean-risk metric']},
    'W': {'status': 'planned_not_implemented',
          'question': 'Add an incremental worst-participating-client risk penalty.',
          'requirements': ['one frozen coefficient', 'participation coverage',
                           'unobserved clients are not zero-risk clients']},
}


def study_definition(study_id: str = 'N') -> dict:
    """返回已登记实验定义的副本，未知编号报错。 / Return a copy of a registered study; reject unknown IDs."""
    if study_id not in STUDIES:
        raise ValueError('Unknown ablation study')
    return deepcopy(STUDIES[study_id])


def arm_configuration(base: dict, experiment_id: str, arm: str, seed: int,
                      *, stage: str = 'train_policy', stage_steps: int = 500) -> dict:
    """生成 N1/N3 单臂配置，继承现有基准的实验设置。
    Build an N1/N3 arm configuration from the existing benchmark settings.

    不继承原管线的执行许可或科学输入规则。
    Do not inherit its execution permissions or scientific-input rules.

    N1/N3 共用奖励、风险模型和 DQN 设置，只改变回报长度 n_step。
    N1/N3 share reward, risk model, and DQN settings; only n_step changes.
    500+700 累计达到 1200，不能把阶段拆分当作新的交互额度。
    The 500+700 stages accumulate to 1200; splitting a run creates no new quota.
    """
    study = study_definition('N')
    seeds = study['training_seeds'] if stage == 'train_policy' else study['validation_seeds']
    if (arm not in study['arms'] or stage not in ('train_policy', 'validate')
            or type(seed) is not int or seed not in seeds
            or type(stage_steps) is not int or stage_steps not in (500, 700)):
        raise ValueError('Only the registered N1/N3 development stages are supported')
    cfg = run_config(base, 'dqn', stage, seed)
    cfg['experiment'] = {'tag': experiment_id, 'variant': arm}
    # 总预算固定为 1200；500/700 只限制本次调用，验证种子映射到配对训练种子。 / Fix the total budget and pair validation seeds.
    cfg['rl'].update(n_step=study['arms'][arm]['n_step'], total_steps=1200,
                     stage_interaction_limit=stage_steps, checkpoint_interval=100,
                     seed=seed if stage == 'train_policy' else
                     study['training_seeds'][study['validation_seeds'].index(seed)])
    if (cfg['rl']['gamma'] != 1 or cfg['rl'].get('double_dqn') is not True
            or cfg['rl'].get('resource_feedback') is not None
            or cfg['reward']['protocol'] != 'exposure_v1'):
        raise ValueError('N requires finite-horizon fixed-reward Double DQN')
    # 独立执行后端、云端命名和单容器设置，不启用原正式验收。 / Isolate execution and cloud names; disable native acceptance.
    cfg['execution']['backend'] = 'ablation'
    cfg['native_acceptance'] = {'enabled': False, 'version': 'first-acceptance-v1',
                                'artifact_verification': 'json_snapshot_no_hash'}
    cfg['modal'].update(app_name=f'ablation-{experiment_id}', volume_name=f'ablation-{experiment_id}',
                        max_containers=1, retries=0)
    # 原预算和后台运行选项不是本实验的授权，因此移除。 / Inherited scheduling controls do not grant permission for this study.
    cfg['modal'].pop('budget_usd', None)
    cfg['modal'].pop('detach', None)
    return cfg


def build_plan(base: dict, experiment_id: str, study_id: str = 'N') -> dict:
    """生成配对消融计划和门槛，不执行实验或授予许可。
    Build a paired ablation plan and gates, without execution or approval.

    500 步预检只验证执行链路；候选效果必须在固定终点和配对种子上比较。
    The 500-step pilot checks execution only; efficacy needs fixed-endpoint,
    paired-seed comparison. Predicted risk is not a full true-risk audit.
    """
    definition = study_definition(study_id)
    if definition['status'] != 'implemented':
        raise ValueError('This study is registered as an idea, not an executable implementation')
    configurations = {
        arm: arm_configuration(base, experiment_id, arm, definition['training_seeds'][0])
        for arm in definition['arms']
    }
    left, right = (deepcopy(configurations[arm]) for arm in ('N1', 'N3'))
    # 除臂名称和 n_step 外，两臂配置必须完全相同。 / Both arms must match except for their names and n_step.
    for item in (left, right):
        item['experiment'].pop('variant')
        item['rl'].pop('n_step')
    if left != right:
        raise ValueError('Unregistered difference between paired arms')
    # 计划仅表示准备完成；历史消耗和资金仍需核查，未知值不算零。 / Preparation is not approval; unknown spending is not zero.
    return {
        'schema': 'ablation_plan_v1', 'experiment_id': experiment_id, 'study_id': study_id,
        'status': 'prepared_not_admitted', 'definition': definition,
        'arm_configurations': configurations,
        'schedule': [
            {'stage': 'pilot', 'training_seeds': definition['training_seeds'][:1],
             'cumulative_steps_per_arm': 500, 'requires': 'new_scoped_admission'},
            {'stage': 'continue', 'training_seeds': definition['training_seeds'][:1],
             'cumulative_steps_per_arm': 1200, 'requires': 'paired_pilot_gate_and_stage_approval'},
            {'stage': 'replicate', 'training_seeds': definition['training_seeds'][1:],
             'cumulative_steps_per_arm': 1200, 'requires': 'paired_development_decision'},
        ],
        'budget': {'method_seed_partition_limit': 3000,
                   'two_arm_training_per_seed': 2400, 'shared_allocation_headroom_per_seed': 600,
                   'planned_training_physical_max': 7200, 'pilot_training_physical_max': 1000,
                    'planned_validation_physical_max': 180,
                   'actual_shared_allocation': None, 'historical_cloud_cost_usd': None,
                   'remaining_configuration_slots': None, 'approved_new_paid_usd': 0,
                   'unknown_consumption_is_zero': False, 'historical_counters_reset': False},
        'limits': {'local_preflight_seconds': 90, 'node_seconds': 1800,
                   'verification_seconds': 90, 'max_containers': 1, 'automatic_retries': 0},
        'metrics': ['final_validation_accuracy', 'simulated_time_seconds',
                    'simulated_energy_joules', 'predicted_exposure_weighted_risk',
                    'postwarm_exposure_count', 'learned_steps', 'warmup_steps', 'updates'],
        'selection_thresholds': {'accuracy_slack': .01, 'time_energy_multiplier': 1.15,
                                 'minimum_feasible_improving_seeds': 2},
        'pending_evidence': ['unused paired seeds and complete interaction ledger',
                             'configuration slot accounting and scoped execution approval',
                             'frozen developer reference and numerical decision thresholds',
                             'explicit fitted-risk source, isolation and collection allocation',
                             'funding for fair pair plus confirmation and complete audits'],
        'official_test_access': False, 'formal_acceptance': 'not_evaluated',
        'predicted_risk_is_true_risk': False, 'inherited_execution_permission': False,
    }
