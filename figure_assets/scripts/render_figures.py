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
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

# 无界面绘图，避免打开桌面窗口。 / Render headlessly without opening desktop windows.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#555555"
ARM_COLORS = {"N1": BLUE, "N3": "#D55E00"}
ARM_LINES = {"N1": "-", "N3": "--"}
INK, MUTED, PALE = "#223247", "#697586", "#F0F3F6"
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
    configured=number(run.get("config_summary",{}).get("training_interaction_budget"))
    if declared is not None and configured is not None and declared != configured:
        raise ValueError(f"Contradictory declared training budgets: {run.get('run_id')}")
    return declared if declared is not None else configured


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
        "runs": [{**{k: r.get(k) for k in ("run_id", "phase", "stage", "partition", "train_seed", "eval_seed", "risk_measurement", "risk_version", "budget", "evidence_scope")},
                  "training_budget_interactions": training_budget(r),
                  "training_budget_source": "exported_budget_field" if number(r.get("training_budget_interactions")) is not None
                  else "declared_config" if training_budget(r) is not None else "unknown"} for r in runs],
        "sample_count": len(rows), "weighting": "Original per-row observations; no seed or round uncertainty inferred",
        "missing_state": "Null observations remain missing; absent values are never imputed as zero",
    }


def episode_metrics(run):
    """Keep real round identities; neither missing rounds nor episodes are joined."""
    episodes = {}
    for row in run.get("metrics", []):
        if number(row.get("round")) is None:
            continue
        identity = (row.get("episode"), row["round"])
        previous = episodes.setdefault(identity[0], {}).get(identity[1])
        if previous is not None and previous != row:
            raise ValueError(f"Contradictory round observations: {run['run_id']} {identity}")
        episodes[identity[0]][identity[1]] = row
    return {ep: [rows[k] for k in sorted(rows)] for ep, rows in episodes.items()}


def validation_pairs(runs):
    """Return exact, known-budget N1/N3 pairs, retaining each recorded episode."""
    groups = {}
    for run in runs:
        if (run.get("arm") not in ARM_COLORS
                or number(run.get("train_seed")) is None
                or number(run.get("eval_seed")) is None
                or training_budget(run) is None or training_budget(run) <= 0):
            continue
        key = (scope_key(run), run.get("method"), run["train_seed"], run["eval_seed"])
        groups.setdefault(key, []).append(run)
    pairs = []
    for group in groups.values():
        if len(group) != 2 or {r["arm"] for r in group} != set(ARM_COLORS):
            continue
        pair = sorted(group, key=lambda r: r["arm"])
        traces = [episode_metrics(r) for r in pair]
        for episode in sorted(traces[0].keys() & traces[1].keys(), key=str):
            observed = [t[episode] for t in traces]
            if observed[0][-1]["round"] == observed[1][-1]["round"]:
                pairs.append((pair, episode, observed))
    return pairs


def diagnostic_groups(runs):
    """Merge compatible arm histories, not phases, counters, or scientific settings."""
    cohorts = {}
    for run in runs:
        if not run.get("training"):
            continue
        config = run.get("config_summary", {})
        fields = (run.get("experiment_id"), run.get("method"), run.get("train_seed"),
                  run.get("partition"), run.get("risk_measurement"), run.get("risk_version"),
                  *(config.get(k) for k in ("dataset", "partition", "max_rounds", "gamma",
                                           "num_clients", "action_space", "warmup_rounds")))
        key = tuple(json.dumps(v, sort_keys=True) for v in fields)
        cohorts.setdefault(key, []).append(run)
    result = []
    for group in cohorts.values():
        arms = {}
        for run in group:
            arm = run.get("arm") or run.get("method", "run")
            records = arms.setdefault(arm, {})
            for row in run["training"]:
                step = number(row.get("environment_steps"))
                if step is None:
                    continue
                identity = (step, row.get("updates"))
                safe = {k: v for k, v in row.items() if k != "source_index"}
                if identity in records and records[identity]["observation"] != safe:
                    raise ValueError(f"Contradictory training diagnostics at {identity}")
                entry = records.setdefault(identity, {"observation": safe, "source_run_ids": []})
                if run["run_id"] not in entry["source_run_ids"]:
                    entry["source_run_ids"].append(run["run_id"])
        series = {arm: [{"arm": arm, **entry["observation"],
                         "source_run_ids": entry["source_run_ids"]}
                        for _, entry in sorted(records.items(), key=lambda item: (item[0][0], item[0][1] or 0))]
                  for arm, records in arms.items() if records}
        if series:
            result.append((group, series))
    return result


def rolling_mean(values, window=11):
    """Centered recorded-update mean; missing values break contiguous segments."""
    # 平滑仅辅助阅读，缺测位置保留，原始值不修改。 / Smooth for display only; preserve gaps and raw values.
    result = [None] * len(values)
    start = 0
    while start < len(values):
        if number(values[start]) is None:
            start += 1
            continue
        end = start
        while end < len(values) and number(values[end]) is not None:
            end += 1
        for i in range(start, end):
            result[i] = float(np.mean(values[max(start, i-window//2):min(end, i+window//2+1)]))
        start = end
    return result


def q_ranks(values):
    """Descending competition ranks; exact ties share a rank, incomplete vectors stay unknown."""
    if not isinstance(values, list) or not values or any(number(v) is None for v in values):
        return None
    return [1 + sum(other > value for other in values) for value in values]


def decision_frame(run, episode, metrics):
    """Join Q vectors to recorded actions by exact episode/round and declared IDs."""
    mapping = {}
    for action in run.get("config_summary", {}).get("action_space", []):
        aid = action.get("action_id")
        clients = action.get("client_ids")
        if (type(aid) is not int or aid < 0 or aid in mapping
                or not isinstance(clients, list) or not clients
                or any(type(c) is not int or c < 0 for c in clients)
                or len(clients) != len(set(clients))
                or number(action.get("batch_size")) is None or action["batch_size"] <= 0):
            raise ValueError("Invalid declared client/batch action mapping")
        mapping[aid] = action
    if not mapping or sorted(mapping) != list(range(len(mapping))):
        raise ValueError("Action vector indices lack a complete declared action mapping")
    decisions = {}
    for row in run.get("decisions", []):
        if row.get("episode") != episode or row.get("kind") != "q_values":
            continue
        rnd = number(row.get("round"))
        if rnd is None:
            continue
        values = row.get("values")
        if not isinstance(values, list) or len(values) != len(mapping):
            raise ValueError("Q vector width differs from the declared action space")
        if rnd in decisions and decisions[rnd]["values"] != values:
            raise ValueError("Contradictory Q vectors at the same episode/round")
        decisions[rnd] = row
    frames = []
    for row in metrics:
        decision = decisions.get(row["round"])
        values = decision["values"] if decision else None
        complete = values is not None and all(number(v) is not None for v in values)
        # 与当轮最大 Q 的距离，不是概率或真实风险。 / The within-round Q gap is neither probability nor true risk.
        gaps = [max(values)-v for v in values] if complete else [None] * len(mapping)
        action = mapping.get(row.get("action_id"))
        frames.append({"run_id": run["run_id"], "arm": run["arm"], "episode": episode,
                       "round": row["round"], "is_warmup": row.get("is_warmup"),
                       "action_id": row.get("action_id"),
                       "client_ids": action["client_ids"] if action else None,
                       "batch_size": action["batch_size"] if action else None,
                       "q_values": values, "q_gaps": gaps, "q_ranks": q_ranks(values)})
    return mapping, frames


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

    def save(self, fid, title, scope, caption, description, fig, runs, data, *, dpi=320):
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
                    fig.savefig(temporary, format=extension, dpi=dpi, facecolor="white")
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

    def compound_axes(self, count, height, ratios=None):
        cols = 1 if self.width == "single" else 2
        rows = math.ceil(count / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(self.inches, height), squeeze=False,
                                 layout="constrained", gridspec_kw={"height_ratios": ratios or [1]*rows})
        for ax in axes.flat:
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["bottom", "left"]].set_color("#CCD3DC")
            ax.tick_params(colors=MUTED, length=2.5, width=.5, pad=3)
            ax.grid(True, axis="y", color="#D8DEE6", linewidth=.45, alpha=.55)
            ax.set_axisbelow(True)
        return fig, list(axes.flat)[:count]

    def compound_title(self, fig, title, runs, footer, episode=None):
        config = runs[0].get("config_summary", {})
        dataset = config.get("dataset") or "dataset unavailable"
        partition = config.get("partition") or runs[0].get("partition") or "split unavailable"
        detail = f"{str(dataset).title()} / {str(partition).upper()}  ·  train seed {runs[0].get('train_seed')}"
        if episode is not None:
            detail += f"  ·  eval {runs[0].get('eval_seed')} / episode {episode}"
        if self.width == "single":
            title, detail, footer = (textwrap.fill(value, width) for value, width in ((title, 40), (detail, 44), (footer, 58)))
        fig.suptitle(f"{title}\n{detail}" if title else detail, fontsize=10, color=INK, fontweight="normal")
        fig.supxlabel(footer, fontsize=7, color=MUTED)

    def panel_title(self, ax, tag, title):
        ax.set_title(f"({tag})  {title}", loc="left", fontsize=9, color=INK, pad=9)

    def round_axis(self, ax, lo, hi):
        spacing = max(1, math.ceil((hi-lo)/5))
        ticks = sorted({lo, hi, *range(math.ceil(lo/spacing)*spacing, math.ceil(hi), spacing)})
        ax.set(xlim=(lo-.5, hi+.5), xticks=ticks)

    def warmup(self, ax, rows):
        warm = sorted({r["round"] for r in rows if r.get("is_warmup") is True})
        if not warm:
            return
        start = previous = warm[0]
        for rnd in [*warm[1:], None]:
            if rnd is not None and rnd == previous+1:
                previous = rnd
                continue
            ax.axvspan(start-.5, previous+.5, color=PALE, zorder=0)
            if rnd is not None:
                start = previous = rnd

    def paired(self, runs):
        pairs = validation_pairs(runs)
        if not pairs:
            self.skip("paired_ablation", "No exact known-budget paired N1/N3 validation episode with matching recorded horizon.")
            return
        for index, (pair, episode, traces) in enumerate(pairs, 1):
            ratios = [1.65, .85] if self.width == "double" else [1.35, 1.35, .8, .8]
            fig, axes = self.compound_axes(4, 4.2 if self.width == "double" else 7.2, ratios)
            self.compound_title(fig, "Paired n-step ablation: utility, resources, and proxy risk", pair,
                                "One paired seed · risk is predicted, not audited · resources are simulated", episode)
            rows, endpoints = [], []
            for run, trace in zip(pair, traces):
                rows.extend({"run_id": run["run_id"], "arm": run["arm"], **row} for row in trace)
                endpoints.append({"run_id": run["run_id"], "arm": run["arm"], **trace[-1]})
            for ai, (key, name, factor) in enumerate((("validation_accuracy", "Validation accuracy (%)", 100),
                                                     ("predicted_exposure_risk", "Exposure-weighted predicted risk", 1))):
                ax = axes[ai]
                for run, trace in zip(pair, traces):
                    arm, color = run["arm"], ARM_COLORS[run["arm"]]
                    ax.plot([r["round"] for r in trace],
                            [factor*r[key] if number(r.get(key)) is not None else np.nan for r in trace],
                            color=color, linestyle=ARM_LINES[arm], linewidth=1.45, label=arm)
                    end = trace[-1]
                    if number(end.get(key)) is not None:
                        ax.scatter(end["round"], factor*end[key], color=color, s=14, zorder=5)
                        if ai == 1:
                            ax.annotate(f"{arm}  {factor*end[key]:.3f}", (end["round"], factor*end[key]),
                                        xytext=(-3, 9), textcoords="offset points", ha="right", color=color, fontsize=7)
                self.warmup(ax, traces[0])
                ax.set(xlabel="Validation round", ylabel=name)
                self.round_axis(ax, min(t[0]["round"] for t in traces), traces[0][-1]["round"])
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                self.panel_title(ax, "ab"[ai], "Accuracy" if ai == 0 else "Proxy risk")
                if ai == 0 and all(number(t[-1].get(key)) is not None for t in traces):
                    final = " / ".join(f"{r['arm']} {100*t[-1][key]:.3f}%" for r,t in zip(pair,traces))
                    ax.text(.03, .97, f"Final: {final}", transform=ax.transAxes, color=INK, fontsize=7, va="top")
                ax.legend(frameon=False, ncol=2, fontsize=7, loc="lower right" if ai == 0 else "center left")
            deltas = {}
            for key in ("validation_accuracy", "predicted_exposure_risk", "elapsed_seconds", "energy_joules"):
                a, b = (number(r.get(key)) for r in endpoints)
                deltas[key] = {"N1": a, "N3": b,
                               "difference": b-a if a is not None and b is not None else None,
                               "relative_percent": 100*(b/a-1) if a is not None and a > 0 and b is not None else None}
            for ax, key, name, unit, tag in zip(axes[2:], ("elapsed_seconds", "energy_joules"),
                                              ("Simulated time", "Simulated energy"), ("s", "J"), "cd"):
                values = [number(r.get(key)) for r in endpoints]
                if all(v is not None for v in values):
                    ax.plot(values, [0, 0], color="#BAC4CF", linewidth=1.6, zorder=2)
                for arm, value in zip(ARM_COLORS, values):
                    if value is None:
                        continue
                    ax.scatter(value, 0, color=ARM_COLORS[arm], marker="o" if arm == "N1" else "s", s=24, zorder=3)
                    ax.annotate(f"{arm}  {value:.2f} {unit}", (value, 0), xytext=(0, 10 if arm == "N1" else -17),
                                textcoords="offset points", ha="center", color=ARM_COLORS[arm], fontsize=7.5)
                relative = deltas[key]["relative_percent"]
                if relative is not None:
                    ax.text(.98, .95, f"N3 vs N1  {relative:+.1f}%", transform=ax.transAxes,
                            ha="right", color=INK, fontsize=7.5)
                finite = [v for v in values if v is not None]
                ax.set(xlim=(0, max(finite)*1.17 if finite and max(finite) > 0 else 1), ylim=(-.8, .8),
                       xlabel=f"Final recorded {name.lower()} ({unit})", yticks=[])
                ax.spines["left"].set_visible(False)
                ax.grid(False, axis="y")
                ax.grid(True, axis="x", color="#D8DEE6", linewidth=.45, alpha=.55)
                self.panel_title(ax, tag, name)
            exposure = {r["arm"]: number(r.get("risk_exposures")) for r in endpoints}
            self.save(f"paired_ablation_{index}", "Paired n-step trade-off", "developer paired validation",
                      "All recorded rounds of one exact paired training/evaluation seed and episode. Shading marks actual warmup. "
                      "Endpoints use the final recorded round, not the best round; no CI or significance claim. "
                      f"Postwarm exposure denominators: {exposure}. Risk is predicted, never a full true audit.",
                      "Aligned accuracy and proxy-risk trajectories above compact paired time and energy endpoints.",
                      fig, pair, {"rows": rows, "endpoints": endpoints, "deltas": deltas,
                                  "exposure_denominators": exposure, "episode": episode}, dpi=600)

    def diagnostics(self, runs):
        groups = diagnostic_groups(runs)
        if not groups:
            self.skip("training_diagnostics", "No actual cumulative-step training diagnostics.")
            return
        for index, (group, series) in enumerate(groups, 1):
            fig, axes = self.compound_axes(2, 3.05 if self.width == "double" else 5.1)
            self.compound_title(fig, "Learning dynamics across n-step targets", group,
                                "Faint: recorded updates · bold: centered 11-update mean · diagnostics are not policy performance")
            rows = [row for trace in series.values() for row in trace]
            boundaries = sorted({number(r.get("budget", {}).get("cumulative_interactions"))
                                 for r in group if number(r.get("budget", {}).get("cumulative_interactions")) is not None})
            for ax, key, name, tag in zip(axes, ("td_loss", "q_mean"), ("TD loss", "Mean learned Q"), "ab"):
                finite = [r[key] for r in rows if number(r.get(key)) is not None]
                log_scale = key == "td_loss" and finite and min(finite) > 0 and max(finite)/min(finite) > 30
                if log_scale:
                    ax.set_yscale("log")
                for arm, trace in sorted(series.items()):
                    steps = [r["environment_steps"] for r in trace]
                    values = [number(r.get(key)) for r in trace]
                    color = ARM_COLORS.get(arm, GREY)
                    ax.plot(steps, [v if v is not None else np.nan for v in values], color=color, alpha=.22, linewidth=.55)
                    smooth = rolling_mean(values)
                    ax.plot(steps, [v if v is not None else np.nan for v in smooth], color=color,
                            linestyle=ARM_LINES.get(arm, "-"), linewidth=1.5, label=arm)
                first, last = min(r["environment_steps"] for r in rows), max(r["environment_steps"] for r in rows)
                for boundary in boundaries:
                    if first < boundary < last:
                        ax.axvline(boundary, color="#A7B2BF", linewidth=.7, linestyle=":")
                        ax.text(boundary, .98, f" {boundary:g}: continue", transform=ax.get_xaxis_transform(),
                                ha="left", va="top", fontsize=6.7, color=MUTED)
                ax.set(xlabel="Cumulative real training interactions", ylabel=name+(" (log scale)" if log_scale else ""))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
                ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=7.5)
                self.panel_title(ax, tag, name)
            self.save(f"training_diagnostics_{index}", "N-step learning dynamics", "developer training",
                      "Compatible histories retain every arm separately; copied prefixes deduplicated by cumulative "
                      "environment_steps and update identity. Raw observations remain visible. Centered 11-record mean "
                      "uses truncated edge windows within contiguous observed segments; gaps are never filled. "
                      "TD loss and Q scales across different n-step targets are diagnostics, not a performance ranking. "
                      "Positive TD loss spanning more than 30x uses a labeled log axis; zeros are not discarded.",
                      "Matched n-step training histories on shared interaction axes, with raw and clearly identified smoothed traces.",
                      fig, group, {"rows": rows, "continuation_boundaries": boundaries[:-1],
                                   "smoothing": {"window": 11, "unit": "recorded optimizer updates", "centered": True,
                                                 "edge_windows": "truncated", "missing_values": "preserved"}}, dpi=600)

    def decision_atlas(self, runs):
        made = False
        for index, (pair, episode, traces) in enumerate(validation_pairs(runs), 1):
            fid = f"decision_atlas_{index}"
            try:
                prepared = [decision_frame(r, episode, trace) for r, trace in zip(pair, traces)]
            except ValueError as error:
                self.skip(fid, str(error))
                continue
            if not any(row["q_ranks"] is not None for _, frames in prepared for row in frames):
                self.skip(fid, "No complete recorded Q vectors; partial values do not define known action ranks.")
                continue
            # 排名只表达相对偏好，不把差距大小解释为置信度。 / Ranks show preference, not Q-gap magnitude or confidence.
            count = 6
            ratios = [1, .45, 2.8] if self.width == "double" else [1, 1, .5, .5, 2.8, 2.8]
            fig, axes = self.compound_axes(count, 5.35 if self.width == "double" else 9.1, ratios)
            self.compound_title(fig, "", pair,
                                "Top-3 Q ranks · rings: recorded actions · gray: no Q record · ranks are not confidence", episode)
            mapping = prepared[0][0]
            clients = sorted({c for a in mapping.values() for c in a["client_ids"]})
            batches = sorted({a["batch_size"] for a in mapping.values()})
            # 分开色系，避免把批大小和排名误读成方法颜色。 / Separate measurement palettes from arm colors.
            batch_palette, rank_palette = "Purples", "Greens"
            batch_colors = {b: plt.get_cmap(batch_palette)(.3+.6*i/max(1, len(batches)-1)) for i, b in enumerate(batches)}
            maximum = max(v for _, frames in prepared for r in frames for v in r["q_gaps"] if v is not None)
            rank_colors = {rank: plt.get_cmap(rank_palette)(level) for rank, level in ((1, .88), (2, .60), (3, .35))}
            rank_sizes = {1: 22, 2: 13, 3: 7}
            all_frames, vector_rows = [], []
            for col, (run, (_, frames), trace) in enumerate(zip(pair, prepared, traces)):
                arm, color = run["arm"], ARM_COLORS[run["arm"]]
                all_frames.extend(frames)
                positions = (col, 2+col, 4+col)
                selected = [axes[p] for p in positions]
                lo, hi = min(r["round"] for r in frames), max(r["round"] for r in frames)
                rounds = np.arange(lo, hi+1)
                lookup = {r["round"]: r for r in frames}
                for row_index, ax in enumerate(selected):
                    ax.grid(False)
                    if row_index < 2:
                        self.warmup(ax, trace)
                    self.round_axis(ax, lo, hi)
                ax = selected[0]
                for yi, client in enumerate(clients):
                    ax.axhline(yi, color="#E3E8EE", linewidth=.5, zorder=1)
                    known = [r for r in frames if r["client_ids"] is not None and client in r["client_ids"]]
                    ax.scatter([r["round"] for r in known], [yi]*len(known), s=12, color=color, zorder=3)
                ax.set(yticks=range(len(clients)), yticklabels=[f"C{c}" for c in clients], ylim=(len(clients)-.5, -.5))
                ax.tick_params(labelbottom=False)
                self.panel_title(ax, "ab"[col], f"{arm} / client participation")
                ax = selected[1]
                for row in frames:
                    if row["batch_size"] in batch_colors:
                        ax.add_patch(Rectangle((row["round"]-.44, -.3), .88, .6,
                                               facecolor=batch_colors[row["batch_size"]], edgecolor="white", linewidth=.3))
                ax.set(ylim=(-.5, .5), yticks=[])
                ax.tick_params(labelbottom=False)
                ax.spines[["left", "bottom"]].set_visible(False)
                self.panel_title(ax, "cd"[col], "Batch size")
                ax = selected[2]
                missing = [rnd for rnd in rounds if rnd not in lookup or lookup[rnd]["q_ranks"] is None]
                if missing:
                    start = previous = missing[0]
                    for rnd in [*missing[1:], None]:
                        if rnd is not None and rnd == previous+1:
                            previous = rnd
                            continue
                        ax.axvspan(start-.5, previous+.5, color=PALE, zorder=0)
                        if previous-start >= 3:
                            ax.text((start+previous)/2, (len(mapping)-1)/2, "No Q\nrecord",
                                    ha="center", va="center", color=MUTED, fontsize=7)
                        if rnd is not None:
                            start = previous = rnd
                for action_id in sorted(mapping):
                    ax.axhline(action_id, color="#E7ECF1", linewidth=.4, zorder=1)
                # 并列动作使用相同名次，保留第三名边界的全部并列项。 / Keep every tie at the top-three boundary.
                for rank in (3, 2, 1):
                    points = [(r["round"], a) for r in frames if r["q_ranks"] is not None
                              for a in sorted(mapping) if r["q_ranks"][a] == rank]
                    ax.scatter([x for x, _ in points], [y for _, y in points], s=rank_sizes[rank],
                               color=rank_colors[rank], edgecolors="none", zorder=3,
                               label=f"Q rank {rank}")
                actions = [r for r in frames if r["action_id"] in mapping and r["q_ranks"] is not None]
                ax.scatter([r["round"] for r in actions], [r["action_id"] for r in actions],
                           s=40, facecolors="none", edgecolors="white", linewidths=1.8, zorder=4)
                ax.scatter([r["round"] for r in actions], [r["action_id"] for r in actions],
                           s=40, facecolors="none", edgecolors=INK, linewidths=.65, zorder=5,
                           label="Recorded action")
                labels = [f"{{{','.join(str(c) for c in mapping[a]['client_ids'])}}} / {mapping[a]['batch_size']:g}" for a in sorted(mapping)]
                ax.set(yticks=range(len(mapping)), yticklabels=labels, ylim=(len(mapping)-.5, -.5), xlabel="Actual validation round")
                ax.tick_params(axis="y", labelsize=7, length=0)
                for a in range(1, len(mapping)):
                    if mapping[a]["client_ids"] != mapping[a-1]["client_ids"]:
                        ax.axhline(a-.5, color="#C9D4DE", linewidth=.65)
                self.panel_title(ax, "ef"[col], "Top-3 action preference")
                vector_rows.extend({"run_id": run["run_id"], **r} for r in run.get("decisions", []) if r.get("episode") == episode)
            rank_handles = [Line2D([], [], linestyle="none", marker="o", markersize=math.sqrt(rank_sizes[rank]),
                                   markerfacecolor=rank_colors[rank], markeredgecolor="none", label=f"Rank {rank}")
                            for rank in (1, 2, 3)]
            rank_handles.extend([Line2D([], [], linestyle="none", marker="o", markersize=math.sqrt(40),
                                       markerfacecolor="none", markeredgecolor=INK, label="Chosen"),
                                 Patch(facecolor=PALE, edgecolor="#CCD3DC", label="No Q record")])
            fig.legend(handles=rank_handles, loc="outside lower center", frameon=False,
                       ncol=3 if self.width == "single" else 5, fontsize=7,
                       borderaxespad=4 if self.width == "single" else 1.8)
            axes[2].legend(handles=[Patch(facecolor=batch_colors[b], label=f"B{b:g}") for b in batches],
                           loc="upper center", bbox_to_anchor=(.5, -.05), frameon=False, ncol=len(batches), fontsize=6.8)
            self.save(fid, "Paired decision atlas", "developer paired validation",
                      "Client events and batch categories use only the declared action space and recorded actions. "
                      "Participation uses arm-specific blue/vermillion, batch sizes use shared purple shades, and Q ranks "
                      "use shared green shades; colors have panel-specific meanings. "
                      "Q vectors join by exact episode/round. Dot size and shade encode descending Q ranks 1–3, with the same "
                      "encoding in both arms; they do not encode Q-gap magnitude or confidence. Exact ties share competition "
                      "ranks, including all ties at rank 3. Black rings with white borders identify recorded choices even outside "
                      "the top three, never an assumed argmax. Gray bands denote absent or incomplete Q vectors; white cells without "
                      "dots denote known ranks below the top three. Unknown actions leave participation and batch missing; "
                      "warmup is shaded above. Raw Q values and gaps remain in provenance. Heuristic scores and policy probabilities "
                      "are not substituted for Q values.",
                      "Two aligned n-step columns combine client event rasters, discrete batch ribbons, and top-three action-rank dot matrices.",
                      fig, pair, {"rows": all_frames, "raw_decisions": vector_rows,
                                  "action_space": list(mapping.values()), "q_transform": "max Q within recorded round - Q(action)",
                                  "q_gap_range": [0, maximum], "episode": episode,
                                  "q_display": {"type": "top3_competition_rank_dot_matrix", "top_k": 3,
                                                "rank_rule": "1 + count(Q(other) > Q(action))",
                                                "ties": "exact equal Q values share a rank; all ranks <= 3 are displayed",
                                                "marker_sizes": rank_sizes, "palette": rank_palette,
                                                "chosen_action": "recorded action_id, including ranks > 3"},
                                  "color_encoding": {
                                      "participation": {"panels": "A/B", "by": "arm", "colors": ARM_COLORS},
                                      "batch_size": {"panels": "C/D", "palette": batch_palette,
                                                     "shared_between_arms": True,
                                                     "colors": {str(b): list(color) for b, color in batch_colors.items()}},
                                      "q_rank": {"panels": "E/F", "palette": rank_palette,
                                                 "shared_between_arms": True,
                                                 "colors": {str(rank): list(color) for rank, color in rank_colors.items()}}}}, dpi=600)
            made = True
        return made

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

    def actions(self, runs):
        # 不同类型的真实决策向量分别展示。 / Plot recorded action vectors separately by decision kind.
        made = self.decision_atlas(runs)
        for run in runs:
            if run.get("arm") in ARM_COLORS:
                continue
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
    # 只读结果已提供的预算，绝不从相似训练记录补齐。 / Use supplied budgets only, never infer from similar training runs.
    assets=confined(root,"figure_assets")
    output=confined(assets,"main")
    output.mkdir(parents=True,exist_ok=True)
    confined(assets,"reports/figure-data").mkdir(parents=True,exist_ok=True)
    manifest_path=confined(assets,"manifest.json")
    previous=load_json(manifest_path) if manifest_path.is_file() else None
    renderer=Renderer(output,data,width)
    with plt.style.context(STYLE):
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
