"""Exact R1/R2 report layouts; never loads models or recreates official gates."""


def read_calibration(reader, source, primitive):
    # 展示原统计，不重新计算验收门槛。 / Display recorded statistics without recomputing acceptance gates.
    calibrations, reports = [], []
    files = []
    if source.name == 'risk-R1-raw-vs-batch-cpu-v1':
        files = [source / ('fit-s' + str(seed)) / 'result.json' for seed in (501, 502, 503)]
    elif source.name == 'risk-R2-positive-part-cpu-v1':
        files = [source / 'all-fits-result.json']
    for path in files:
        report = reader.json(path, {})
        if not report:
            continue
        aggregate = report.get('aggregate', report)
        scope = report.get('scope', aggregate.get('scope', 'development_only'))
        # 保留各拟合种子与面板，不混合统计。 / Keep fit-seed and panel groups separate.
        for i, row in enumerate(aggregate.get('paired_rows', [])):
            if i >= 1000:
                raise ValueError('Calibration row limit exceeded')
            trajectory = primitive(row.get('trajectory'))
            raw, cal = row.get('raw', {}), row.get('calibrated', {})
            # 无法证明模型身份时保留未知。 / Leave unproven model identity unknown.
            calibrations.append({'group_id': source.name + '-' + str(row.get('fit_seed')) + '-' + str(trajectory),
                'fit_seed': primitive(row.get('fit_seed')), 'model_id': None, 'scope': primitive(scope),
                'raw_u': primitive(raw.get('U')), 'cal_u': primitive(cal.get('U')),
                'raw_ewmae': primitive(raw.get('EWMAE')), 'cal_ewmae': primitive(cal.get('EWMAE')),
                'panel_ids': [trajectory] if trajectory else [], 'sources': [dict(reader.source(path), record=i)]})
        safe = {k: primitive(aggregate.get(k)) for k in ('scope', 'development_status', 'final_test_opened') if k in aggregate}
        safe['source'] = reader.source(path)
        safe['report_kind'] = 'original_calibration_report'
        safe['numerical_gates'] = {k: primitive(v) for k, v in aggregate.get('numerical_gates', {}).items() if k in ('U', 'EWMAE')}
        safe['thresholds'] = {k: primitive(v) for k, v in aggregate.get('thresholds', {}).items() if k in ('calibration_underestimation_multiplier', 'calibration_mae_multiplier')}
        safe['limitations'] = [primitive(v) for v in aggregate.get('limitations', [])[:100]]
        reports.append(safe)
    decision_path = source / 'decision.json'
    decision = reader.json(decision_path, {}) if files else {}
    if decision:
        safe = {k: primitive(decision[k]) for k in ('hypothesis_id', 'status', 'decision', 'scope', 'formal_acceptance_status', 'reason') if k in decision}
        safe.update(source=reader.source(decision_path), report_kind='original_decision')
        reports.append(safe)
    return calibrations, reports
