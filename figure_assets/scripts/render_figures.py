"""Read-only publication renderer for results_visualization_v1 snapshots.

No training imports, checkpoint access, experiment execution, or acceptance engine.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

# 无界面绘图，避免打开桌面窗口。 / Render headlessly without opening desktop windows.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#555555"
STYLE = Path(__file__).resolve().parents[1] / "styles" / "acm.mplstyle"


def confined(base: Path, path: str | Path) -> Path:
    """Resolve paths, including symlinks, before accepting read/write targets."""
    base = base.resolve()
    candidate = (base / path).resolve()
    if not candidate.is_relative_to(base):
        raise ValueError("Path escapes approved subtree")
    return candidate


def load_json(path: Path) -> dict:
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream, parse_constant=reject, object_pairs_hook=pairs)
    # Also rejects overflow in otherwise syntactically valid numeric literals.
    json.dumps(result, allow_nan=False)
    return result


def write_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=".render-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def number(value):
    return value if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) else None


def label(run):
    return f"{run.get('arm') or run.get('method', 'run')} · train {run.get('train_seed')} / eval {run.get('eval_seed')}"


def training_budget(run):
    declared=number(run.get("training_budget_interactions"))
    return declared if declared is not None else number(run.get("config_summary",{}).get("training_interaction_budget"))


def scope_key(run):
    # 按任务、划分、风险版本和预算分组。 / Group by task, split, risk version, and budget.
    config = run.get("config_summary", {})
    return tuple(json.dumps(v, sort_keys=True) for v in (
        run.get("experiment_id"), run.get("partition"), run.get("phase"), run.get("stage"),
        run.get("risk_measurement"), run.get("risk_version"), config.get("dataset"),
        config.get("partition"), config.get("max_rounds"),
        config.get("num_clients"), config.get("action_space"), config.get("gamma"),
        training_budget(run), config.get("checkpoint_step"),
        run.get("budget", {}).get("cumulative_interactions"),
        run.get("budget", {}).get("new_interactions"),
    ))


def unique_runs(runs):
    result = {}
    for run in runs:
        rid = run["run_id"]
        # 副本路径可不同，但结果必须一致。 / Copied paths may differ, but results must agree.
        content = {k: v for k, v in run.items() if k not in ("sources", "warnings")}
        if rid in result:
            previous = {k: v for k, v in result[rid].items() if k not in ("sources", "warnings")}
            if content != previous:
                raise ValueError(f"Contradictory duplicate run: {rid}")
        else:
            result[rid] = run
    return list(result.values())


def context(runs, rows):
    return {
        "runs": [{k: r.get(k) for k in ("run_id", "phase", "stage", "partition", "train_seed", "eval_seed", "risk_measurement", "risk_version", "budget", "training_budget_interactions", "evidence_scope")} for r in runs],
        "sample_count": len(rows), "weighting": "Original per-row observations; no seed or round uncertainty inferred",
        "missing_state": "Null observations remain missing; absent values are never imputed as zero",
    }


class Renderer:
    def __init__(self, output, snapshot, width):
        self.output, self.snapshot, self.width = output, snapshot, width
        self.figures, self.skipped = [], []
        # ACM 近似栏宽：毫米转换为英寸。 / Convert approximate ACM column widths from mm to inches.
        self.inches = (85 if width == "single" else 178) / 25.4

    def skip(self, fid, reason):
        self.skipped.append({"id": fid, "reason": reason})

    def axes(self, count=1, height=2.5):
        cols = 1 if self.width == "single" else min(count, 2)
        rows = math.ceil(count / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(self.inches, height * rows), squeeze=False, layout="constrained")
        for ax in axes.flat:
            ax.grid(True, axis="y")
            ax.set_axisbelow(True)
        for ax in list(axes.flat)[count:]:
            ax.set_visible(False)
        return fig, list(axes.flat)[:count]

    def save(self, fid, title, scope, caption, description, fig, runs, data):
        # 图片、图注和来源数据同步保存。 / Save figures, captions, and provenance together.
        evidence = context(runs, data.get("rows", []))
        splits=sorted({str(r.get('partition')) for r in runs})
        stages=sorted({str(r.get('stage')) for r in runs})
        train_seeds=sorted({r['train_seed'] for r in runs if number(r.get('train_seed')) is not None})
        eval_seeds=sorted({r['eval_seed'] for r in runs if number(r.get('eval_seed')) is not None})
        budgets=sorted({training_budget(r) for r in runs if training_budget(r) is not None})
        details=f"Split: {', '.join(splits)}; stage: {', '.join(stages)}; train seeds: {train_seeds or 'not applicable / unavailable'}; eval seeds: {eval_seeds or 'unavailable'}; declared trained interactions: {budgets or 'unavailable'}" if runs else "No empirical runs; group provenance is in source data"
        caption += f" {details}. Recorded observations: {evidence['sample_count']}. Missing values remain missing. Full source IDs, denominators, and weighting are in the figure data JSON."
        files = {}
        try:
            for extension in ("png",):
                filename = f"{fid}.{extension}"
                destination = confined(self.output, filename)
                fd, temporary = tempfile.mkstemp(prefix=".render-", suffix=f".{extension}", dir=self.output)
                os.close(fd)
                try:
                    fig.savefig(temporary, format=extension, dpi=320, facecolor="white")
                    os.replace(temporary, destination)
                finally:
                    Path(temporary).unlink(missing_ok=True)
                files[extension] = f"main/{filename}"
            # 图片目录仅含 PNG，溯源数据留在报告目录。 / Keep only PNGs in main; retain provenance in reports.
            files["data"] = f"reports/figure-data/{fid}.data.json"
            write_json(confined(self.output.parent, files["data"]), {"figure_id": fid, "snapshot_id": self.snapshot["snapshot_id"], "caption": caption, "description": description, "evidence": evidence, **data})
            self.figures.append({"id": fid, "title": title, "scope": scope, "caption": caption, "description": description, "files": files, "source_run_ids": [r["run_id"] for r in runs]})
        finally:
            plt.close(fig)

    def mechanism(self):
        # 机制示意图不是实测性能证据。 / The conceptual diagram is not measured performance evidence.
        fig, (ax,) = self.axes(height=3.0 if self.width == "double" else 4.3)
        ax.set_axis_off()
        boxes = [(0.03, .70, "State + proxy\npredicted risk"), (.38, .70, "Policy\nclient / batch action"), (.73, .70, "Federated round\nsimulated resources"), (.38, .22, "Independent true audit\nheld-out labels"), (.73, .22, "Evidence / gates\nno inferred acceptance")]
        if self.width == "single":
            boxes = [(.08, .82, boxes[0][2]), (.55, .82, boxes[1][2]), (.55, .50, boxes[2][2]), (.08, .18, boxes[3][2]), (.55, .18, boxes[4][2])]
        w = .25 if self.width == "double" else .36
        for x, y, text in boxes:
            ax.add_patch(FancyBboxPatch((x, y), w, .16, boxstyle="round,pad=0.012", facecolor="white", edgecolor=BLUE, linewidth=.8))
            ax.text(x+w/2, y+.08, text, ha="center", va="center", fontsize=7.5)
        for start, end, dashed in ((0,1,False),(1,2,False),(2,3,True),(3,4,True)):
            x,y,_ = boxes[start]
            xx,yy,_ = boxes[end]
            if y == yy:
                origin=(x+w+.013,y+.08)
                target=(xx-.013,yy+.08)
            else:
                origin=(x+w/2,y-.013)
                target=(xx+w/2,yy+.173)
            ax.annotate("", xy=target, xytext=origin, arrowprops={"arrowstyle":"->", "color":GREY, "linewidth":.8, "linestyle":"--" if dashed else "-"})
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_title("Conceptual framework\nAudit is not policy feedback" if self.width=="single" else "Conceptual framework · audit is not policy feedback")
        self.save("mechanism", "Conceptual mechanism", "conceptual", "Conceptual framework, not a measured causal or performance result. Dashed branch denotes an independent true-label audit, separate from proxy-driven actions.", "State, action, federated simulation, and independent audit evidence flow.", fig, [], {"rows": [], "nodes": [b[2] for b in boxes]})

    def learning(self, runs):
        # 各条轨迹分别绘制，缺测保留断线。 / Plot episodes separately and leave gaps for missing values.
        keys = [("validation_accuracy", "Validation accuracy"), ("elapsed_seconds", "Simulated time (s)"), ("energy_joules", "Simulated energy (J)"), ("predicted_exposure_risk", "Predicted exposure risk")]
        runs=[r for r in runs if r.get("phase")!="reported_validation" and len(r.get("metrics",[]))>1]
        if not runs:
            self.skip("learning_progress", "No validation trajectories; training and pilot metrics are not substituted.")
            return
        groups = {}
        for r in runs:
            groups.setdefault(scope_key(r), []).append(r)
        for index, group in enumerate(groups.values()):
            if not any(number(row.get(key)) is not None for r in group for row in r.get("metrics",[]) for key,_ in keys):
                self.skip(f"learning_progress_{index+1}","No actual trajectory metric values.")
                continue
            # 单栏拆为两组，保证最终字号可读。 / Split single-column panels to keep final-size text legible.
            if self.width == "single":
                for chunk_index, chunk in enumerate((keys[:2], keys[2:])):
                    self.learning_chunk(group, chunk, index, chunk_index, len(groups))
                continue
            fig, axes = self.axes(4, 2.05)
            rows = []
            for j, run in enumerate(group):
                metrics = sorted(run.get("metrics", []), key=lambda row: (row.get("episode") or 0, row.get("round") or 0))
                rows.extend({"run_id": run["run_id"], **row} for row in metrics)
                for ax, (key, ylabel) in zip(axes, keys):
                    episodes = sorted({row.get("episode") for row in metrics}, key=str)
                    for ep_index, episode in enumerate(episodes):
                        trace = [row for row in metrics if row.get("episode") == episode and number(row.get("round")) is not None]
                        ax.plot([r["round"] for r in trace], [number(r.get(key)) if number(r.get(key)) is not None else np.nan for r in trace], color=[BLUE,ORANGE,GREY][j%3], linestyle=["-","--",":"][j%3], label=label(run)+(f" / ep {episode}" if len(episodes)>1 else ""))
                        warm = [r["round"] for r in trace if r.get("is_warmup") is True]
                        if warm:
                            ax.axvspan(min(warm)-.5, max(warm)+.5, color=GREY, alpha=.05)
                    ax.set(xlabel="Validation round", ylabel=ylabel)
            for ax in axes:
                ax.legend(frameon=False, loc="best")
            fid = "learning_progress" if len(groups)==1 else f"learning_progress_{index+1}"
            self.save(fid, "Validation learning progress", "developer validation", "Compatible validation trajectories only; each seed is retained. Shading shows actual recorded warmup rounds. Risk is predicted, not audited; exposure counts are not a risk average denominator.", "Separate axes show accuracy and simulated resource and predicted-risk measurements by round.", fig, group, {"rows": rows})

    def learning_chunk(self, group, keys, index, chunk_index, group_count):
        fig, axes = self.axes(len(keys), 2.05)
        rows=[]
        for j,run in enumerate(group):
            metrics=sorted(run.get("metrics",[]),key=lambda row:(row.get("episode") or 0,row.get("round") or 0))
            rows.extend({"run_id":run["run_id"],**row} for row in metrics)
            episodes=sorted({r.get("episode") for r in metrics},key=str)
            for ax,(key,name) in zip(axes,keys):
                for episode in episodes:
                    trace=[r for r in metrics if r.get("episode")==episode and number(r.get("round")) is not None]
                    ax.plot([r["round"] for r in trace],[number(r.get(key)) if number(r.get(key)) is not None else np.nan for r in trace],color=[BLUE,ORANGE,GREY][j%3],linestyle=["-","--",":"][j%3],label=label(run)+(f" / ep {episode}" if len(episodes)>1 else ""))
                    warm=[r["round"] for r in trace if r.get("is_warmup") is True]
                    if warm:
                        ax.axvspan(min(warm)-.5,max(warm)+.5,color=GREY,alpha=.05)
                ax.set(xlabel="Validation round",ylabel=name)
                ax.legend(frameon=False)
        fid="learning_progress" if index==0 and chunk_index==0 else f"learning_progress_{index+1}_part{chunk_index+1}"
        self.save(fid,"Validation learning progress","developer validation","Compatible validation trajectories only; all seeds retained. Shading marks actual warmup rounds. Predicted risk is not audited risk; exposure counts are not a risk-average denominator.","Final-size single-column trajectory panels with missing values left as line breaks.",fig,group,{"rows":rows,"plotted_metric_keys":[key for key,_ in keys]})

    def diagnostics(self, runs):
        # 按累计步数和更新 ID 去重复历史。 / Deduplicate resumed histories by cumulative steps and update IDs.
        groups = {}
        for run in runs:
            if not run.get("training"):
                continue
            key = (run.get("experiment_id"), run.get("arm"), run.get("train_seed"), run.get("partition"), run.get("risk_version"), run.get("config_summary",{}).get("dataset"))
            groups.setdefault(key, []).append(run)
        if not groups:
            self.skip("training_diagnostics", "No actual cumulative-step training diagnostics.")
        for index, group in enumerate(groups.values()):
            unique = {}
            for run in group:
                for row in run["training"]:
                    step = number(row.get("environment_steps"))
                    if step is None:
                        continue
                    key = (step, row.get("updates"))
                    safe = {k:v for k,v in row.items() if k != "source_index"}
                    if key in unique and unique[key] != safe:
                        raise ValueError(f"Contradictory training diagnostics at {key}")
                    unique[key] = safe
            rows = sorted(unique.values(), key=lambda row: (row["environment_steps"], row.get("updates") or 0))
            if not rows:
                self.skip(f"training_diagnostics_{index+1}", "No declared cumulative environment_steps; no guessed offsets.")
                continue
            fig, axes = self.axes(2, 2.15)
            for ax, key, name in zip(axes, ("td_loss", "q_mean"), ("TD loss", "Mean learned Q")):
                ax.plot([r["environment_steps"] for r in rows], [r.get(key) if number(r.get(key)) is not None else np.nan for r in rows], color=BLUE)
                ax.set(xlabel="Cumulative training interactions", ylabel=name)
            self.save(f"training_diagnostics_{index+1}", "Training diagnostics", "developer training", "Resumed histories deduplicated by cumulative environment_steps and update identity, never concatenated or re-offset. Training diagnostics are not validation performance.", "TD loss and mean learned Q versus actual cumulative training interactions.", fig, group, {"rows": rows})

    def paired(self, runs):
        groups = {}
        for run in runs:
            if run.get("arm") in ("N1","N3"):
                groups.setdefault((scope_key(run), run.get("train_seed"), run.get("eval_seed")), []).append(run)
        pairs = [g for g in groups.values() if len(g)==2 and {r["arm"] for r in g}=={"N1","N3"}]
        if not pairs:
            self.skip("paired_ablation", "No exact compatible paired N1/N3 validation runs (same seed, split, stage, risk version, and budgets).")
            return
        for i, pair in enumerate(pairs):
            pair.sort(key=lambda r:r["arm"])
            keys = [("validation_accuracy","Final validation accuracy"),("predicted_exposure_risk","Final predicted risk")]
            fig, axes = self.axes(2, 2.3)
            rows = []
            for run in pair:
                metrics = [r for r in run.get("metrics",[]) if number(r.get("round")) is not None]
                # 每条轨迹取最后记录，不挑最好结果。 / Use each episode's last record, not its best result.
                for episode in sorted({r.get("episode") for r in metrics}, key=str):
                    terminal = max((r for r in metrics if r.get("episode")==episode), key=lambda r:r["round"])
                    rows.append({"run_id":run["run_id"],"arm":run["arm"], **terminal})
            for ax, (key,name) in zip(axes,keys):
                for row in rows:
                    if number(row.get(key)) is not None:
                        ax.scatter([row["arm"]],[row[key]], color=BLUE if row["arm"]=="N1" else ORANGE)
                ax.set(ylabel=name, xlabel="n-step arm")
                ax.set_title("Developer paired seed · no CI")
            self.save(f"paired_ablation_{i+1}", "Paired N1/N3 ablation", "developer paired validation", "One paired training seed per panel; no confidence interval, significance claim, or best-seed selection. Final recorded round in every episode is shown.", "Compatible paired n-step validation endpoints.", fig, pair, {"rows": rows})

    def actions(self, runs):
        # 不同类型的真实决策向量分别展示。 / Plot recorded action vectors separately by decision kind.
        made = False
        for run in runs:
            rows = run.get("decisions", [])
            kinds = sorted({str(row.get("kind")) for row in rows})
            for ki, kind in enumerate(kinds):
                subset = [row for row in rows if str(row.get("kind"))==kind and isinstance(row.get("values"),list) and row["values"]]
                if not subset:
                    continue
                lengths = {len(row["values"]) for row in subset}
                if len(lengths)!=1:
                    self.skip(f"actions_{len(self.figures)}", "Inconsistent action value vectors; no invented action alignment.")
                    continue
                fig, (ax,) = self.axes(height=2.7)
                matrix = np.array([[number(v) if number(v) is not None else np.nan for v in row["values"]] for row in subset], dtype=float).T
                image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="cividis")
                ax.grid(False)
                ax.set(xlabel="Recorded decision index", ylabel="Action vector index", title=f"{run.get('arm') or run.get('method')} · {kind}")
                fig.colorbar(image, ax=ax, label=f"{kind} value (original units)", shrink=.85)
                self.save(f"action_values_{len(self.figures)}", "Action values", run.get("evidence_scope", "developer"), f"Decision kind {kind} is shown separately; heuristic scores and learned Q-values are never merged. Indices are not guessed client mappings.", "Action-value heatmap; each column is one recorded decision, not an interpolated round.", fig, [run], {"rows":subset,"action_space":run.get("config_summary",{}).get("action_space",[])})
                made = True
            mapping = {a.get("action_id"):a.get("client_ids") for a in run.get("config_summary",{}).get("action_space",[]) if isinstance(a.get("client_ids"),list)}
            metrics = [r for r in run.get("metrics",[]) if r.get("action_id") in mapping]
            clients = sorted({c for ids in mapping.values() for c in ids})
            if metrics and clients:
                fig, (ax,) = self.axes(height=2.6)
                mat = np.array([[int(c in mapping[r["action_id"]]) for r in metrics] for c in clients])
                image = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0,vmax=1,interpolation="nearest")
                ax.grid(False)
                ax.set(xlabel="Recorded round index",ylabel="Client ID",yticks=range(len(clients)),yticklabels=clients)
                fig.colorbar(image,ax=ax,ticks=[0,1],label="Selected",shrink=.8)
                self.save(f"client_participation_{len(self.figures)}", "Client participation", run.get("evidence_scope","developer"), "Client participation derived only from the declared action mapping. Each column is one actual exposure opportunity; unknown actions are excluded, not interpreted as nonparticipation.", "Binary client-selection heatmap, not a risk-average denominator.", fig,[run],{"rows":metrics,"action_space":run["config_summary"]["action_space"]})
                made = True
        if not made:
            self.skip("actions", "No actual decision vectors or declared client action mappings.")

    def resources(self, runs):
        # 资源为模拟值，风险为预测值。 / Show simulated resources and predicted, not audited, risk.
        runs=[r for r in runs if r.get("phase")!="reported_validation" and len(r.get("metrics",[]))>1]
        if not runs:
            self.skip("resource_risk", "No validation simulated-resource / predicted-risk observations.")
            return
        groups = {}
        for r in runs:
            groups.setdefault(scope_key(r),[]).append(r)
        for i, group in enumerate(groups.values()):
            rows = [{"run_id":r["run_id"],**m} for r in group for m in r.get("metrics",[]) if number(m.get("predicted_exposure_risk")) is not None]
            if not rows:
                self.skip(f"resource_risk_{i+1}","Predicted risk missing; no zero imputation.")
                continue
            fig, axes = self.axes(2,2.4)
            for j,r in enumerate(group):
                for ax,key,name in zip(axes,("elapsed_seconds","energy_joules"),("Simulated time (s)","Simulated energy (J)")):
                    points=[m for m in rows if m["run_id"]==r["run_id"] and number(m.get(key)) is not None]
                    ax.scatter([p[key] for p in points],[p["predicted_exposure_risk"] for p in points],s=12,alpha=.7,color=[BLUE,ORANGE,GREY][j%3],marker=["o","s","^"][j%3],label=label(r))
                    ax.set(xlabel=name,ylabel="Predicted exposure risk")
                    ax.legend(frameon=False)
            self.save(f"resource_risk_{i+1}","Simulated resources versus predicted risk","developer validation","Per-round simulated resources versus predicted exposure risk; not audited risk, measured wall time, or monetary cloud cost. Compatible trajectories only; no fitted causal trend.","Resource/risk scatter retaining all observed rounds and seeds.",fig,group,{"rows":rows})

    def audit(self, runs):
        # 只绘制实际配对的审计真值与预测值。 / Plot only observed audit truth paired with predictions.
        actual=[a for a in self.snapshot.get("audits",[]) if number(a.get("truth")) is not None and number(a.get("prediction")) is not None and a.get("upload_id") is not None]
        if not actual:
            self.skip("proxy_reliability","No exact-ID matched actual audited labels; proxy predictions alone are insufficient.")
            self.skip("batch_residual","No actual audited truth/prediction pairs with batch sizes.")
            return
        groups={}
        for a in actual:
            groups.setdefault((a.get("model_id"),a.get("scope"),a.get("fit_seed"),a.get("panel_seed")),[]).append(a)
        has_batch=False
        for i,rows in enumerate(groups.values()):
            linked=[r for r in runs if r["run_id"] in {a["run_id"] for a in rows}]
            fig,(ax,)=self.axes(height=2.8)
            ax.scatter([a["prediction"] for a in rows],[a["truth"] for a in rows],color=BLUE,s=16)
            lo=min(min(a["truth"],a["prediction"]) for a in rows)
            hi=max(max(a["truth"],a["prediction"]) for a in rows)
            ax.plot([lo,hi],[lo,hi],color=GREY,linestyle="--",linewidth=.8)
            ax.set(xlabel="Proxy prediction",ylabel="Audited true risk")
            self.save(f"proxy_reliability_{i+1}","Proxy reliability","partial matched audit",f"Exact-ID paired labels only; partial audits are not complete-policy risk. Model/scope/fit/panel group: {list(groups)[i]}.","Actual prediction versus audited truth, with identity reference only.",fig,linked,{"rows":rows})
            batch=[a for a in rows if number(a.get("batch_size")) is not None]
            if batch:
                fig,(ax,)=self.axes(height=2.6)
                ax.scatter([a["batch_size"] for a in batch],[a["prediction"]-a["truth"] for a in batch],color=ORANGE,s=16)
                ax.axhline(0,color=GREY,linewidth=.8)
                ax.set(xlabel="Actual batch size",ylabel="Prediction − audited truth")
                self.save(f"batch_residual_{i+1}","Batch residual","partial matched audit","Residuals from actual matched audit labels only; no synthetic uncertainty band or aggregation across models/panels.","Audit residual versus recorded batch size.",fig,linked,{"rows":batch})
                has_batch=True
        if not has_batch:
            self.skip("batch_residual","Matched audit labels lack actual batch sizes.")

    def calibration(self):
        # 模型标识已知且分母为正时才计算比率。 / Require a model ID and positive denominators for ratios.
        rows=[c for c in self.snapshot.get("calibration",[]) if c.get("model_id") is not None and all(number(c.get(k)) is not None for k in ("raw_u","cal_u","raw_ewmae","cal_ewmae")) and c["raw_u"]>0 and c["raw_ewmae"]>0]
        if not rows:
            self.skip("calibration_tradeoff","No same-model scoped calibration rows with nonzero raw U/EWMAE denominators.")
            return
        groups={}
        for c in rows:
            groups.setdefault((c["model_id"],c.get("scope"),tuple(c.get("panel_ids",[]))),[]).append(c)
        for i,group in enumerate(groups.values()):
            fig,(ax,)=self.axes(height=2.7)
            ax.scatter([c["cal_u"]/c["raw_u"] for c in group],[c["cal_ewmae"]/c["raw_ewmae"] for c in group],color=BLUE,s=24)
            ax.axhline(1,color=GREY,linewidth=.7,linestyle="--")
            ax.axvline(1,color=GREY,linewidth=.7,linestyle="--")
            ax.set(xlabel="U ratio (calibrated / raw)",ylabel="EWMAE ratio (calibrated / raw)")
            self.save(f"calibration_tradeoff_{i+1}","Calibration tradeoff",str(group[0].get("scope")),"Ratios within each original row, same model, scope, and panel IDs. All fit seeds retained. Reference lines indicate unchanged metrics, not formal acceptance gates.","Calibration U and exposure-weighted error ratios.",fig,[],{"rows":group})

    def summary_suite(self, runs):
        """Native report endpoints are not fabricated round trajectories."""
        native=[r for r in runs if r.get("phase")=="reported_validation" and isinstance(r.get("summary"),dict)]
        if not native:
            self.skip("method_comparison","No native reported-validation endpoint summaries.")
            self.skip("checkpoint_learning","No actual checkpoint validation summaries.")
            return
        # 基线预算保持未知，不补成 3000。 / Keep unknown baseline budgets missing, not filled with 3000.
        groups={}
        for r in native:
            c=r.get("config_summary",{})
            task=tuple(json.dumps(v,sort_keys=True) for v in (r.get("experiment_id"),r.get("partition"),r.get("phase"),r.get("stage"),r.get("risk_version"),r.get("risk_measurement"),c.get("dataset"),c.get("partition"),c.get("max_rounds")))
            groups.setdefault(task,[]).append(r)
        keys=[("validation_accuracy","Validation accuracy"),("elapsed_seconds","Simulated time (s)"),("energy_joules","Simulated energy (J)"),("predicted_exposure_risk","Predicted exposure risk")]
        names={"dqn":"DQN","ppo":"PPO","pqn":"PQN","metacs":"MetaCS","fixed":"Fixed","time_greedy":"Time greedy","risk_greedy":"Risk greedy"}
        colors={"dqn":BLUE,"ppo":ORANGE,"pqn":"#009E73"}
        notice="Development summaries; partial transport, local unverified, not formal results. Predicted risk is not audited risk. No confidence intervals or best-seed selection. Safe dataset/config details are unavailable where null."
        for gi,group in enumerate(groups.values()):
            final=[r for r in group if r["summary"].get("is_latest") is True and (training_budget(r)==3000 or (training_budget(r) is None and r.get("train_seed") is None))]
            if len({(r.get("budget",{}).get("new_interactions"),r["summary"].get("round")) for r in final})>1:
                self.skip(f"method_comparison_{gi+1}","Final endpoints have incompatible actual evaluation interaction budgets; no equal-budget comparison implied.")
                final=[]
            methods=sorted({r["method"] for r in final},key=lambda m:list(names).index(m) if m in names else 100)
            evalseeds=sorted({r.get("eval_seed") for r in final if number(r.get("eval_seed")) is not None})
            batches=[keys] if self.width=="double" else [keys[:2],keys[2:]]
            for bi,batch in enumerate(batches):
                if not final or not any(number(r["summary"].get(k)) is not None for r in final for k,_ in batch):
                    self.skip(f"method_comparison_{gi+1}_{bi+1}","No actual 3000-trained/latest baseline endpoint values.")
                    continue
                fig,axes=self.axes(len(batch),2.25)
                for ax,(key,name) in zip(axes,batch):
                    for mi,method in enumerate(methods):
                        for r in [r for r in final if r["method"]==method]:
                            v=number(r["summary"].get(key))
                            if v is None:
                                continue
                            si=evalseeds.index(r["eval_seed"]) if r.get("eval_seed") in evalseeds else 0
                            offset=(si-(len(evalseeds)-1)/2)*.13
                            ax.scatter(mi+offset,v,s=20,color=colors.get(method,GREY),marker=["o","s","^"][si%3],edgecolors="white",linewidths=.3)
                    ax.set(ylabel=name,xticks=range(len(methods)),xticklabels=[names.get(m,m)+("*" if m not in colors else "") for m in methods])
                    ax.tick_params(axis="x",labelrotation=35)
                    ax.set_title("Development · three seed points / method")
                rows=[{"run_id":r["run_id"],"method":r["method"],"train_seed":r.get("train_seed"),"eval_seed":r.get("eval_seed"),"evaluation_budget":r.get("budget"),"training_interaction_budget":training_budget(r),**r["summary"]} for r in final]
                fid="method_comparison" if gi==0 and bi==0 else f"method_comparison_{gi+1}_part{bi+1}"
                marker_note="; ".join(f"{['circle','square','triangle'][si%3]}: eval {seed}" for si,seed in enumerate(evalseeds))
                self.save(fid,"Seven-method validation comparison","development partial-return local-unverified",notice+" Learned methods use the declared 3000-interaction checkpoint. * denotes a baseline with no learned-training budget, not 3000 training interactions. "+marker_note+"; no aggregate error bars.","Final validation endpoints for each reported method/seed; each metric has its own axis.",fig,final,{"rows":rows,"plotted_metric_keys":[k for k,_ in batch],"baseline_training_budget":"not applicable / unavailable"})
            learned=[r for r in group if r.get("method") in colors and training_budget(r) in (500,1000,1500,2000,2500,3000) and number(r.get("train_seed")) is not None and number(r.get("eval_seed")) is not None]
            for bi,batch in enumerate(batches):
                if not learned:
                    self.skip(f"checkpoint_learning_{gi+1}_{bi+1}","No declared learned training checkpoints.")
                    continue
                fig,axes=self.axes(len(batch),2.3)
                for ax,(key,name) in zip(axes,batch):
                    for method,color in colors.items():
                        matching=[r for r in learned if r["method"]==method]
                        pairs=sorted({(r["train_seed"],r["eval_seed"]) for r in matching})
                        for pi,pair in enumerate(pairs):
                            trace=sorted([r for r in matching if (r["train_seed"],r["eval_seed"])==pair],key=training_budget)
                            if len({training_budget(r) for r in trace})!=len(trace):
                                raise ValueError("Duplicate checkpoint for matched method/train/eval seed")
                            steps=[500,1000,1500,2000,2500,3000]
                            lookup={training_budget(r):number(r["summary"].get(key)) for r in trace}
                            ax.plot(steps,[lookup.get(s) if lookup.get(s) is not None else np.nan for s in steps],color=color,alpha=.45,linewidth=.8,marker=["o","s","^"][pi%3],markersize=2.4)
                        # 粗线为实际种子的描述均值，不是置信区间。 / Bold lines show descriptive seed means, not CIs.
                        means=[]
                        for step in (500,1000,1500,2000,2500,3000):
                            vals=[number(r["summary"].get(key)) for r in matching if training_budget(r)==step and number(r["summary"].get(key)) is not None]
                            means.append(float(np.mean(vals)) if vals else np.nan)
                        if matching:
                            ax.plot([500,1000,1500,2000,2500,3000],means,color=color,linewidth=1.5,label=names[method])
                    ax.set(xlabel="Actual trained interactions",ylabel=name,xticks=[500,1500,2500,3000])
                    ax.legend(frameon=False,ncol=3)
                rows=[{"run_id":r["run_id"],"method":r["method"],"train_seed":r["train_seed"],"eval_seed":r["eval_seed"],"evaluation_budget":r.get("budget"),"training_interaction_budget":training_budget(r),**r["summary"]} for r in learned]
                fid="checkpoint_learning" if gi==0 and bi==0 else f"checkpoint_learning_{gi+1}_part{bi+1}"
                lengths=sorted({r["summary"].get("round") for r in learned if number(r["summary"].get("round")) is not None})
                self.save(fid,"Checkpoint learning","development partial-return local-unverified",notice+f" Faint lines retain every matched training/evaluation seed pair; bold lines are unweighted descriptive seed means at each checkpoint. Recorded evaluation lengths: {lengths} rounds; all retained, not equal-exposure normalized. Missing checkpoints break lines. Training budgets identify checkpoints, not additional physical interactions; actual evaluation counts remain in JSON.","Validation endpoints against actual trained checkpoint budgets; variable recorded evaluation lengths are descriptive outcomes, not an equal-budget formal comparison.",fig,learned,{"rows":rows,"plotted_metric_keys":[k for k,_ in batch],"mean_weighting":"Unweighted mean of available actual seed endpoints at each method/checkpoint; not uncertainty","recorded_evaluation_lengths":lengths})

    def calibration_descriptive(self):
        rows=[c for c in self.snapshot.get("calibration",[]) if all(number(c.get(k)) is not None for k in ("raw_u","cal_u","raw_ewmae","cal_ewmae"))]
        if not rows:
            self.skip("calibration_descriptive","No actual paired raw/calibrated U and EWMAE statistics.")
            return
        groups={}
        for row in rows:
            # 按范围与来源分组，不猜测模型身份。 / Group by scope and source without inferring model identity.
            family=tuple(sorted({str(s.get("path","")).split("/fit-")[0] for s in row.get("sources",[])}))
            groups.setdefault((row.get("scope"),family),[]).append(row)
        for gi,group in enumerate(groups.values()):
            group=sorted(group,key=lambda c:(c.get("fit_seed") or 0,str(c.get("panel_ids"))))
            provenance=" ".join(str(c.get("group_id",""))+" "+" ".join(str(s.get("path","")) for s in c.get("sources",[])) for c in group)
            variant="Positive part" if "positive-part" in provenance else "Batch variant" if "raw-vs-batch" in provenance else "Reported variant"
            pairs=[("raw_u","cal_u","U (original units)"),("raw_ewmae","cal_ewmae","Exposure-weighted MAE")]
            batches=[pairs] if self.width=="double" else [[pair] for pair in pairs]
            for bi,batch in enumerate(batches):
                fig,axes=self.axes(len(batch),max(2.5,.24*len(group)+.9))
                labels=[]
                for j,c in enumerate(group):
                    panel=" / ".join(c.get("panel_ids",[]))
                    match=re.search(r"-s(\d+)(?:-|$)",panel)
                    labels.append(f"fit {c.get('fit_seed')} / panel {match.group(1) if match else j+1}")
                for ax,(raw,cal,name) in zip(axes,batch):
                    for j,c in enumerate(group):
                        ax.plot([c[raw],c[cal]],[j,j],color=GREY,linewidth=.7)
                    ax.scatter([c[raw] for c in group],range(len(group)),color=GREY,s=16,label="Raw",zorder=3)
                    ax.scatter([c[cal] for c in group],range(len(group)),color=ORANGE,s=18,label=variant,zorder=3)
                    ax.set(xlabel=name,yticks=range(len(group)),yticklabels=labels)
                    ax.invert_yaxis()
                    ax.grid(True,axis="x")
                    ax.legend(frameon=False,loc="lower center",bbox_to_anchor=(.5,1.01),ncol=2)
                scope=group[0].get("scope")
                missing=sum(c.get("model_id") is None for c in group)
                self.save(f"calibration_descriptive_{gi+1}_part{bi+1}","Paired calibration statistics",str(scope),f"Descriptive raw-versus-{variant.lower()} statistics within each original fit/panel provenance group. Scope: {scope}. All {len(group)} groups retained separately; no pooling across panels, ratios, hard gates, or formal conclusion. Model identity is missing for {missing} groups; source pairing does not prove same-model identity.","Dumbbells link only the raw and variant statistics from the same exported group. Full group/panel/source provenance remains in JSON.",fig,[],{"rows":group,"plotted_metric_keys":[[raw,cal] for raw,cal,_ in batch],"variant_label":variant,"model_identity_proven":False})


def render_snapshot(root: Path, *, snapshot: Path | None = None, width: str = "double") -> dict:
    if width not in ("single","double"):
        raise ValueError("width must be single or double")
    root=root.resolve()
    if snapshot is None:
        base=confined(root,"frontend/public/data")
        index=load_json(confined(base,"index.json"))
        if index.get("schema_version")!="results_index_v1":
            raise ValueError("Unsupported index schema")
        entries=[e for e in index["snapshots"] if e["snapshot_id"]==index["current_snapshot"]]
        if len(entries)!=1:
            raise ValueError("Current snapshot must have one index entry")
        snapshot=confined(base,entries[0]["path"])
    else:
        snapshot=confined(root,snapshot)
    data=load_json(snapshot)
    if data.get("schema_version")!="results_visualization_v1":
        raise ValueError("Unsupported results schema")
    sid=data.get("snapshot_id")
    if not isinstance(sid,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",sid) or sid in (".",".."):
        raise ValueError("Unsafe snapshot_id")
    runs=unique_runs(data.get("runs",[]))
    # 性能图只使用标注为验证的记录。 / Only declared validation records enter performance panels.
    validation=[r for r in runs if "val" in str(r.get("phase", "")).lower() or "val" in str(r.get("stage", "")).lower()]
    for run in validation:
        candidates=[r for r in runs if r not in validation and r.get("experiment_id")==run.get("experiment_id") and r.get("arm")==run.get("arm") and r.get("train_seed")==run.get("train_seed") and r.get("phase")==run.get("phase") and r.get("config_summary",{}).get("partition")==run.get("config_summary",{}).get("partition") and r.get("config_summary",{}).get("dataset")==run.get("config_summary",{}).get("dataset") and r.get("risk_version")==run.get("risk_version") and r.get("training")]
        budgets={r.get("budget",{}).get("cumulative_interactions") for r in candidates}
        # 不从含糊关联猜测预算。 / Never infer budgets from ambiguous lineage.
        run["training_budget_interactions"]=next(iter(budgets)) if len(budgets)==1 else number(run.get("config_summary",{}).get("training_interaction_budget"))
    assets=confined(root,"figure_assets")
    output=confined(assets,"main")
    output.mkdir(parents=True,exist_ok=True)
    confined(assets,"reports/figure-data").mkdir(parents=True,exist_ok=True)
    manifest_path=confined(assets,"manifest.json")
    previous=load_json(manifest_path) if manifest_path.is_file() else None
    renderer=Renderer(output,data,width)
    with plt.style.context(STYLE):
        renderer.mechanism()
        renderer.learning(validation)
        renderer.diagnostics(runs)
        renderer.paired(validation)
        renderer.actions(validation)
        renderer.resources(validation)
        renderer.summary_suite(validation)
        renderer.audit(runs)
        renderer.calibration()
        renderer.calibration_descriptive()
        renderer.skip("formal_main_results","Contract v1 supplies no authoritative full-gate schema proving frozen reference, complete audited policy risk, verification, all required seeds, and acceptance. Developer/pilot evidence is not promoted to formal results.")
    manifest={"schema_version":"figure_manifest_v1","snapshot_id":sid,"generated_at":datetime.now(timezone.utc).isoformat(),"figures":renderer.figures,"skipped":renderer.skipped}
    write_json(manifest_path,manifest)
    # 仅清理旧索引登记的过期图片，不删除未知文件。 / Remove only obsolete PNGs registered in the old index.
    if previous:
        current={f["files"]["png"] for f in renderer.figures}
        for figure in previous.get("figures", []):
            old=figure.get("files", {}).get("png")
            if isinstance(old,str) and old.startswith("main/") and old.endswith(".png") and old not in current:
                confined(assets,old).unlink(missing_ok=True)
    return {"manifest_path":str(manifest_path),"figures":renderer.figures,"skipped":renderer.skipped}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--snapshot",type=Path)
    parser.add_argument("--width",choices=("single","double"),default="double")
    args=parser.parse_args()
    print(json.dumps(render_snapshot(args.root,snapshot=args.snapshot,width=args.width),allow_nan=False,indent=2))


if __name__=="__main__":
    main()
