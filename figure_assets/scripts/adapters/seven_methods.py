"""Fixed public seven-method CSV report; no raw graph scans or remote retrieval."""

REPORT_NAME = 'report-analysis-20261010-v1'
IDENTITY_KEYS = {'run_id', 'method', 'evaluation_seed', 'training_seed', 'checkpoint_step', 'training_run', 'raw_run_returned', 'risk_version'}


def read_seven_methods(reader, source, primitive, metric_keys):
    report_dir = source if source.name == REPORT_NAME else source / REPORT_NAME
    analysis_path = report_dir / 'analysis.json'
    analysis = reader.json(analysis_path, {})
    if not analysis:
        return None
    evaluation_path = report_dir / 'evaluation_rows.csv'
    final_path = report_dir / 'final_rows.csv'
    # 只导出显式字段，屏蔽内部文件路径。 / Export declared columns without internal file paths.
    keys = metric_keys | IDENTITY_KEYS
    rows = reader.rows(evaluation_path, keys, 'csv')
    finals = reader.rows(final_path, keys, 'csv')
    if not rows:
        return None
    experiment_id = primitive(analysis.get('experiment_id'))
    if not experiment_id:
        raise ValueError('Seven-method report lacks experiment_id')
    status = {'compute': 'reported_validation_results',
              'transport': 'partial' if analysis.get('whole_transfer_complete') is False else 'unknown',
              'verification': 'local_unverified' if analysis.get('local_semantic_verified') is False else 'unknown',
              'acceptance': primitive(analysis.get('formal_acceptance')) or 'unknown'}
    warnings = ['Descriptive validation summaries only; cloud node receipts are not local semantic verification.',
                'Checkpoint training budgets are not additional physical training interactions.',
                'Predicted exposure risk is not audited complete-policy risk.']
    by_id = {}
    final_records = {}
    for row in rows:
        run_id = row.get('run_id')
        if not isinstance(run_id, str) or not run_id:
            raise ValueError('Seven-method row lacks safe run_id')
        safe = {k: v for k, v in row.items() if k != 'source_index'}
        if run_id in by_id and by_id[run_id] != safe:
            raise ValueError('Conflicting seven-method evaluation rows: ' + run_id)
        by_id[run_id] = safe
    for row in finals:
        run_id = row.get('run_id')
        safe = {k: v for k, v in row.items() if k != 'source_index'}
        if run_id not in by_id or by_id[run_id] != safe:
            raise ValueError('Final row contradicts evaluation evidence: ' + str(run_id))
        final_records.setdefault(run_id, []).append(row['source_index'])
    runs = []
    for row in rows:
        run_id = row['run_id']
        if any(r['run_id'] == run_id for r in runs):
            continue
        checkpoint = row.get('checkpoint_step')
        summary = {k: v for k, v in row.items() if k in metric_keys}
        # 最终标记来自原报告，不由最大步数推断。 / Take final markers from the report, not the largest checkpoint.
        summary.update(checkpoint_step=checkpoint, training_interaction_budget=checkpoint,
                       is_latest=run_id in final_records, raw_run_returned=row.get('raw_run_returned'))
        metrics = dict(summary, source_index=row['source_index'])
        sources = [dict(reader.source(evaluation_path), record=row['source_index'] + 1, line=row['source_index'] + 2), reader.source(analysis_path)]
        sources.extend(dict(reader.source(final_path), record=i + 1, line=i + 2) for i in final_records.get(run_id, []))
        runs.append({'run_id': run_id, 'experiment_id': experiment_id, 'method': row.get('method'),
            'arm': None, 'phase': 'reported_validation', 'stage': 'validate',
            'train_seed': row.get('training_seed'), 'eval_seed': row.get('evaluation_seed'),
            'partition': 'validation', 'risk_measurement': 'predicted_risk', 'risk_version': row.get('risk_version'),
            'evidence_scope': 'descriptive_latest_validation_summaries_only', 'status': dict(status),
            'budget': {'new_interactions': row.get('round'), 'cumulative_interactions': row.get('round'), 'warmup_interactions': None},
            'summary': summary,
            'config_summary': {'dataset': None, 'partition': 'validation', 'num_clients': None, 'gamma': None,
                'n_step': None, 'warmup_rounds': None, 'max_rounds': None, 'action_space': [],
                'checkpoint_step': checkpoint, 'training_interaction_budget': checkpoint},
            'metrics': [metrics], 'training': [], 'decisions': [], 'uploads': [], 'sources': sources,
            'warnings': list(warnings) + ['Safe environment config and action mapping unavailable in this public report.']})
    if analysis.get('evaluation_rows') != len(rows) or analysis.get('final_rows') != len(finals):
        warnings.append('Declared report row counts differ from actual CSV records.')
    metadata_keys = ('status', 'experiment_id', 'nodes', 'returned_raw_runs', 'node_receipts_verified',
                     'whole_transfer_complete', 'local_semantic_verified', 'formal_acceptance', 'evaluation_rows', 'final_rows')
    metadata = {k: primitive(analysis[k]) for k in metadata_keys if k in analysis}
    metadata.update(report_kind='original_seven_method_analysis', source=reader.source(analysis_path),
                    actual_evaluation_rows=len(rows), actual_final_rows=len(finals), unique_latest_runs=len(final_records))
    reports = [metadata]
    diagnostics_path = report_dir / 'diagnostics.json'
    diagnostics = reader.json(diagnostics_path, {})
    if diagnostics:
        safe = {k: primitive(diagnostics[k]) for k in ('status', 'audit_aggregation', 'train_rows', 'internal_validation_rows', 'fit_seed', 'calibration_uses_epoch_selection_group') if k in diagnostics}
        safe.update(report_kind='original_descriptive_diagnostics', source=reader.source(diagnostics_path))
        reports.append(safe)
    # 检查点不代表新增物理训练交互。 / Checkpoints are not additional physical training interactions.
    experiment = {'experiment_id': experiment_id, 'name': 'Seven-method validation comparison', 'kind': 'native_experiment',
        'evidence_scope': 'descriptive_latest_validation_summaries_only',
        'sources': [reader.source(analysis_path), reader.source(evaluation_path), reader.source(final_path)],
        'status': status, 'actual_cloud_cost_usd': None,
        'counts': {'physical_train_interactions': None, 'physical_validation_interactions': None,
                   'training_seed_count': len({r['train_seed'] for r in runs if r['train_seed'] is not None})},
        'evidence_gaps': list(warnings) + ['Physical experiment interaction totals are not reconstructed from repeated checkpoint evaluation rows.',
                                        'Raw run return is partial; no missing cloud artifacts requested.', 'Actual cloud cost unavailable.']}
    return experiment, runs, reports, warnings
