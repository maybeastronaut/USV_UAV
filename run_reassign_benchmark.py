#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import random
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")


@dataclass
class RunResult:
    scenario_profile: str
    extreme_event_enabled: bool
    seed: int
    reassign_algorithm: str
    wall_time_s: float
    log_path: str
    metrics: Dict[str, float | int | str | bool | None]


def _safe_scalar(raw: str) -> float | int | str | bool | None:
    txt = raw.strip()
    if txt in {"", "-"}:
        return None
    lower = txt.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if NUMERIC_RE.match(txt):
        if "." in txt:
            return float(txt)
        return int(txt)
    return txt


def _parse_kv_metrics(log_text: str, keys: Iterable[str]) -> Dict[str, float | int | str | bool | None]:
    out: Dict[str, float | int | str | bool | None] = {}
    for key in keys:
        match = re.search(rf"^{re.escape(key)}:\s*(.+)$", log_text, re.M)
        out[key] = _safe_scalar(match.group(1)) if match else None
    return out


def _find_latest_log(log_dir: Path) -> Path | None:
    logs = sorted(log_dir.glob("mission_seed*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return None
    return logs[0]


def _quantile(vals: Sequence[float], q: float) -> float:
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    arr = sorted(vals)
    pos = q * (len(arr) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    ratio = pos - lo
    return arr[lo] * (1.0 - ratio) + arr[hi] * ratio


def _mean(vals: Sequence[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _sign_test_pvalue_two_sided(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def _as_float(v: float | int | str | bool | None) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _bool_flag_name(enabled: bool) -> str:
    return "on" if enabled else "off"


def run_single(
    *,
    root: Path,
    mission_script: Path,
    python_bin: str,
    planning_algorithm: str,
    reassign_algorithm: str,
    scenario_profile: str,
    extreme_event_enabled: bool,
    seed: int,
    timeout_sec: float,
    log_dir: Path,
) -> Tuple[RunResult | None, Dict[str, str] | None]:
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_bin,
        str(mission_script),
        "--algorithm",
        planning_algorithm,
        "--reassign-algorithm",
        reassign_algorithm,
        "--seed-mode",
        "fixed",
        "--seed",
        str(seed),
        "--scenario-profile",
        scenario_profile,
        "--extreme-event" if extreme_event_enabled else "--no-extreme-event",
        "--no-visualize",
        "--no-animate",
        "--log-dir",
        str(log_dir),
    ]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, {
            "scenario_profile": scenario_profile,
            "extreme_event_enabled": str(extreme_event_enabled).lower(),
            "seed": str(seed),
            "reassign_algorithm": reassign_algorithm,
            "error_type": "timeout",
            "detail": f"timeout>{timeout_sec}s",
        }
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        return None, {
            "scenario_profile": scenario_profile,
            "extreme_event_enabled": str(extreme_event_enabled).lower(),
            "seed": str(seed),
            "reassign_algorithm": reassign_algorithm,
            "error_type": "returncode",
            "detail": f"code={proc.returncode}; stderr_tail={proc.stderr[-300:]}",
        }

    log_path = _find_latest_log(log_dir)
    if log_path is None:
        return None, {
            "scenario_profile": scenario_profile,
            "extreme_event_enabled": str(extreme_event_enabled).lower(),
            "seed": str(seed),
            "reassign_algorithm": reassign_algorithm,
            "error_type": "log_missing",
            "detail": "run success but no log found",
        }

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    wanted_keys = [
        "seed",
        "scenario_profile",
        "reassign_algorithm",
        "coverage_rate",
        "task_handling_rate",
        "turn_events",
        "turn_index",
        "load_balance_cv",
        "replans",
        "mission_completion_time_s",
        "extreme_event_enabled",
        "extreme_event_fired",
        "extreme_event_step",
        "extreme_event_time_s",
        "emergency_task_label",
        "emergency_assess_time_s",
        "emergency_first_response_time_s",
        "emergency_complete_time_s",
        "event_to_assess_s",
        "event_to_first_replan_s",
        "assess_to_first_replan_s",
        "assess_to_first_response_s",
        "assess_to_complete_s",
        "coverage_at_event",
        "coverage_at_event_plus_120s",
        "coverage_recovery_120s",
        "task_completion_at_event",
        "task_completion_at_event_plus_120s",
        "task_completion_gain_120s",
        "turn_index_at_event",
        "turn_index_at_event_plus_120s",
        "turn_index_delta_120s",
    ]
    metrics = _parse_kv_metrics(text, wanted_keys)
    if metrics.get("reassign_algorithm") != reassign_algorithm:
        return None, {
            "scenario_profile": scenario_profile,
            "extreme_event_enabled": str(extreme_event_enabled).lower(),
            "seed": str(seed),
            "reassign_algorithm": reassign_algorithm,
            "error_type": "parse_mismatch",
            "detail": "log reassign_algorithm mismatch",
        }

    return (
        RunResult(
            scenario_profile=scenario_profile,
            extreme_event_enabled=extreme_event_enabled,
            seed=seed,
            reassign_algorithm=reassign_algorithm,
            wall_time_s=wall,
            log_path=str(log_path),
            metrics=metrics,
        ),
        None,
    )


def _build_seed_table(seed_base: int, scenarios: Sequence[str], n_per_scenario: int) -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for idx, scenario in enumerate(scenarios):
        rng = random.Random(seed_base + 131 * (idx + 1))
        out[scenario] = [rng.randint(1, 2_000_000_000) for _ in range(n_per_scenario)]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Paired benchmark for task-reassignment algorithms.")
    parser.add_argument("--python-bin", default="python3", help="Python executable for mission runs.")
    parser.add_argument("--mission-script", default="demo_step3_mission.py", help="Mission entry script path.")
    parser.add_argument("--planning-algorithm", default="hierarchical_v1")
    parser.add_argument("--algorithms", default="potential_event,heuristic")
    parser.add_argument("--scenario-profiles", default="random,clustered,split-corners,high-coupling")
    parser.add_argument("--extreme-event-modes", default="on,off", help="Comma list from {on,off}.")
    parser.add_argument("--seeds-per-scenario", type=int, default=30)
    parser.add_argument("--seed-base", type=int, default=20260301)
    parser.add_argument("--timeout-sec", type=float, default=90.0)
    parser.add_argument("--min-time-improve-ratio", type=float, default=0.05)
    parser.add_argument("--coverage-noninferiority-tol", type=float, default=0.005)
    parser.add_argument("--handling-noninferiority-tol", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        default="outputs/reassign_benchmark",
        help="Output folder for csv and reports.",
    )
    args = parser.parse_args()

    root = Path.cwd()
    mission_script = (root / args.mission_script).resolve()
    if not mission_script.exists():
        raise FileNotFoundError(f"mission script not found: {mission_script}")

    algorithms = [x.strip() for x in args.algorithms.split(",") if x.strip()]
    if set(algorithms) != {"potential_event", "heuristic"}:
        raise ValueError("--algorithms must include exactly potential_event,heuristic")
    scenarios = [x.strip() for x in args.scenario_profiles.split(",") if x.strip()]
    extreme_modes_raw = [x.strip().lower() for x in args.extreme_event_modes.split(",") if x.strip()]
    extreme_modes: List[bool] = []
    for mode in extreme_modes_raw:
        if mode == "on":
            extreme_modes.append(True)
        elif mode == "off":
            extreme_modes.append(False)
        else:
            raise ValueError(f"unsupported extreme mode: {mode}")
    if not extreme_modes:
        raise ValueError("at least one extreme-event mode is required")

    out_dir = (root / args.output_dir).resolve()
    logs_root = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    seed_table = _build_seed_table(args.seed_base, scenarios, args.seeds_per_scenario)
    planned_pairs = len(scenarios) * len(extreme_modes) * args.seeds_per_scenario
    planned_runs = planned_pairs * len(algorithms)

    print(
        f"[benchmark] scenarios={scenarios}, extreme_modes={extreme_modes_raw}, "
        f"seeds_per_scenario={args.seeds_per_scenario}, planned_runs={planned_runs}"
    )

    run_results: List[RunResult] = []
    errors: List[Dict[str, str]] = []
    run_idx = 0
    for scenario in scenarios:
        for extreme_enabled in extreme_modes:
            for seed in seed_table[scenario]:
                for algo in algorithms:
                    run_idx += 1
                    run_log_dir = logs_root / scenario / f"extreme_{_bool_flag_name(extreme_enabled)}" / algo / str(seed)
                    print(
                        f"[run {run_idx:03d}/{planned_runs}] scenario={scenario} "
                        f"extreme={_bool_flag_name(extreme_enabled)} seed={seed} algo={algo}"
                    )
                    run, err = run_single(
                        root=root,
                        mission_script=mission_script,
                        python_bin=args.python_bin,
                        planning_algorithm=args.planning_algorithm,
                        reassign_algorithm=algo,
                        scenario_profile=scenario,
                        extreme_event_enabled=extreme_enabled,
                        seed=seed,
                        timeout_sec=args.timeout_sec,
                        log_dir=run_log_dir,
                    )
                    if err is not None:
                        errors.append(err)
                        print(f"  -> skip ({err['error_type']})")
                        continue
                    run_results.append(run)

    raw_csv = out_dir / "raw_runs.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "scenario_profile",
            "extreme_event_enabled",
            "seed",
            "reassign_algorithm",
            "wall_time_s",
            "log_path",
            "coverage_rate",
            "task_handling_rate",
            "turn_events",
            "turn_index",
            "load_balance_cv",
            "replans",
            "mission_completion_time_s",
            "extreme_event_fired",
            "extreme_event_time_s",
            "event_to_assess_s",
            "event_to_first_replan_s",
            "assess_to_first_replan_s",
            "assess_to_first_response_s",
            "assess_to_complete_s",
            "coverage_recovery_120s",
            "task_completion_gain_120s",
            "turn_index_delta_120s",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rr in run_results:
            row = {
                "scenario_profile": rr.scenario_profile,
                "extreme_event_enabled": str(rr.extreme_event_enabled).lower(),
                "seed": rr.seed,
                "reassign_algorithm": rr.reassign_algorithm,
                "wall_time_s": f"{rr.wall_time_s:.4f}",
                "log_path": rr.log_path,
            }
            for k in fields:
                if k in row:
                    continue
                v = rr.metrics.get(k)
                row[k] = v if v is not None else ""
            writer.writerow(row)

    errors_csv = out_dir / "errors.csv"
    with errors_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "scenario_profile",
            "extreme_event_enabled",
            "seed",
            "reassign_algorithm",
            "error_type",
            "detail",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(errors)

    by_key: Dict[Tuple[str, bool, int], Dict[str, RunResult]] = {}
    for rr in run_results:
        k = (rr.scenario_profile, rr.extreme_event_enabled, rr.seed)
        by_key.setdefault(k, {})[rr.reassign_algorithm] = rr

    paired_rows: List[Dict[str, str | float | int]] = []
    for (scenario, extreme_enabled, seed), pair_map in sorted(by_key.items()):
        if "potential_event" not in pair_map or "heuristic" not in pair_map:
            continue
        p = pair_map["potential_event"]
        h = pair_map["heuristic"]

        def val(rr: RunResult, key: str) -> float | None:
            return _as_float(rr.metrics.get(key))

        p_time = val(p, "mission_completion_time_s")
        h_time = val(h, "mission_completion_time_s")
        p_cov = val(p, "coverage_rate")
        h_cov = val(h, "coverage_rate")
        p_handle = val(p, "task_handling_rate")
        h_handle = val(h, "task_handling_rate")
        p_turn = val(p, "turn_index")
        h_turn = val(h, "turn_index")

        cov_noninferior = (
            (p_cov is not None and h_cov is not None and p_cov >= (h_cov - args.coverage_noninferiority_tol))
        )
        handle_noninferior = (
            (p_handle is not None and h_handle is not None and p_handle >= (h_handle - args.handling_noninferiority_tol))
        )
        time_superior = (
            p_time is not None
            and h_time is not None
            and p_time <= h_time * (1.0 - args.min_time_improve_ratio)
        )
        potential_pass = bool(cov_noninferior and handle_noninferior and time_superior)

        row: Dict[str, str | float | int] = {
            "scenario_profile": scenario,
            "extreme_event_enabled": str(extreme_enabled).lower(),
            "seed": seed,
            "delta_time_s_p_minus_h": (p_time - h_time) if p_time is not None and h_time is not None else "",
            "delta_coverage_p_minus_h": (p_cov - h_cov) if p_cov is not None and h_cov is not None else "",
            "delta_task_handling_p_minus_h": (
                (p_handle - h_handle) if p_handle is not None and h_handle is not None else ""
            ),
            "delta_turn_index_p_minus_h": (p_turn - h_turn) if p_turn is not None and h_turn is not None else "",
            "delta_turn_events_p_minus_h": (
                (val(p, "turn_events") - val(h, "turn_events"))
                if val(p, "turn_events") is not None and val(h, "turn_events") is not None
                else ""
            ),
            "delta_event_to_assess_s_p_minus_h": (
                (val(p, "event_to_assess_s") - val(h, "event_to_assess_s"))
                if val(p, "event_to_assess_s") is not None and val(h, "event_to_assess_s") is not None
                else ""
            ),
            "delta_assess_to_first_response_s_p_minus_h": (
                (val(p, "assess_to_first_response_s") - val(h, "assess_to_first_response_s"))
                if val(p, "assess_to_first_response_s") is not None
                and val(h, "assess_to_first_response_s") is not None
                else ""
            ),
            "delta_coverage_recovery_120s_p_minus_h": (
                (val(p, "coverage_recovery_120s") - val(h, "coverage_recovery_120s"))
                if val(p, "coverage_recovery_120s") is not None and val(h, "coverage_recovery_120s") is not None
                else ""
            ),
            "delta_task_completion_gain_120s_p_minus_h": (
                (val(p, "task_completion_gain_120s") - val(h, "task_completion_gain_120s"))
                if val(p, "task_completion_gain_120s") is not None and val(h, "task_completion_gain_120s") is not None
                else ""
            ),
            "potential_noninferior_coverage": int(cov_noninferior),
            "potential_noninferior_handling": int(handle_noninferior),
            "potential_pass_rule": int(potential_pass),
        }
        paired_rows.append(row)

    pair_csv = out_dir / "paired_diff.csv"
    with pair_csv.open("w", newline="", encoding="utf-8") as f:
        if paired_rows:
            fields = list(paired_rows[0].keys())
        else:
            fields = [
                "scenario_profile",
                "extreme_event_enabled",
                "seed",
                "delta_time_s_p_minus_h",
                "delta_coverage_p_minus_h",
                "delta_task_handling_p_minus_h",
                "delta_turn_index_p_minus_h",
                "delta_turn_events_p_minus_h",
                "potential_noninferior_coverage",
                "potential_noninferior_handling",
                "potential_pass_rule",
            ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired_rows)

    def paired_vals(key: str, *, condition: Tuple[str, bool] | None = None) -> List[float]:
        vals: List[float] = []
        for row in paired_rows:
            if condition is not None:
                if row["scenario_profile"] != condition[0]:
                    continue
                if row["extreme_event_enabled"] != str(condition[1]).lower():
                    continue
            v = row.get(key)
            if isinstance(v, (int, float)):
                vals.append(float(v))
            elif isinstance(v, str) and v not in {"", "-"} and NUMERIC_RE.match(v):
                vals.append(float(v))
        return vals

    condition_summary: List[Dict[str, str | int | float]] = []
    all_conditions = [(s, e) for s in scenarios for e in extreme_modes]
    for cond in all_conditions + [("all", True)]:
        if cond[0] == "all":
            rows_cond = paired_rows
            label = "all"
            ext_flag = "both"
            dt = paired_vals("delta_time_s_p_minus_h")
            dc = paired_vals("delta_coverage_p_minus_h")
            dti = paired_vals("delta_turn_index_p_minus_h")
        else:
            rows_cond = [
                r
                for r in paired_rows
                if r["scenario_profile"] == cond[0] and r["extreme_event_enabled"] == str(cond[1]).lower()
            ]
            label = cond[0]
            ext_flag = str(cond[1]).lower()
            dt = paired_vals("delta_time_s_p_minus_h", condition=cond)
            dc = paired_vals("delta_coverage_p_minus_h", condition=cond)
            dti = paired_vals("delta_turn_index_p_minus_h", condition=cond)
        wins_time = sum(1 for x in dt if x < 0)
        losses_time = sum(1 for x in dt if x > 0)
        ties_time = sum(1 for x in dt if x == 0)
        wins_cov = sum(1 for x in dc if x > 0)
        losses_cov = sum(1 for x in dc if x < 0)
        ties_cov = sum(1 for x in dc if x == 0)
        wins_turn = sum(1 for x in dti if x < 0)
        losses_turn = sum(1 for x in dti if x > 0)
        ties_turn = sum(1 for x in dti if x == 0)
        pass_rate = (
            sum(int(r.get("potential_pass_rule", 0)) for r in rows_cond) / len(rows_cond) if rows_cond else float("nan")
        )
        condition_summary.append(
            {
                "scenario_profile": label,
                "extreme_event_enabled": ext_flag,
                "paired_n": len(rows_cond),
                "delta_time_mean": _mean(dt),
                "delta_time_p50": _quantile(dt, 0.50),
                "delta_time_p10": _quantile(dt, 0.10),
                "delta_time_p90": _quantile(dt, 0.90),
                "delta_coverage_mean": _mean(dc),
                "delta_turn_index_mean": _mean(dti),
                "time_wins_potential": wins_time,
                "time_losses_potential": losses_time,
                "time_ties": ties_time,
                "time_sign_test_p": _sign_test_pvalue_two_sided(wins_time, losses_time),
                "coverage_wins_potential": wins_cov,
                "coverage_losses_potential": losses_cov,
                "coverage_ties": ties_cov,
                "turn_wins_potential": wins_turn,
                "turn_losses_potential": losses_turn,
                "turn_ties": ties_turn,
                "potential_pass_rule_rate": pass_rate,
            }
        )

    cond_csv = out_dir / "condition_summary.csv"
    with cond_csv.open("w", newline="", encoding="utf-8") as f:
        fields = list(condition_summary[0].keys()) if condition_summary else [
            "scenario_profile",
            "extreme_event_enabled",
            "paired_n",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(condition_summary)

    summary_lines: List[str] = []
    summary_lines.append("# Reassign Benchmark Summary")
    summary_lines.append("")
    summary_lines.append(f"- planned_runs: {planned_runs}")
    summary_lines.append(f"- successful_runs: {len(run_results)}")
    summary_lines.append(f"- skipped_runs: {len(errors)}")
    summary_lines.append(f"- paired_samples: {len(paired_rows)}")
    summary_lines.append(f"- scenarios: {', '.join(scenarios)}")
    summary_lines.append(f"- extreme_modes: {', '.join(extreme_modes_raw)}")
    summary_lines.append(f"- seeds_per_scenario: {args.seeds_per_scenario}")
    summary_lines.append(
        f"- rule: potential pass if time improves >= {args.min_time_improve_ratio*100:.1f}% "
        f"and coverage/handling are non-inferior"
    )
    summary_lines.append("")
    summary_lines.append("## Condition Summary")
    summary_lines.append("")
    summary_lines.append(
        "| scenario | extreme | paired_n | dTime(mean,p50,p10,p90) | dCoverage(mean) | dTurn(mean) | "
        "time wins/loss/tie | time p(sign) | pass_rate |"
    )
    summary_lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for row in condition_summary:
        summary_lines.append(
            "| "
            f"{row['scenario_profile']} | {row['extreme_event_enabled']} | {row['paired_n']} | "
            f"{row['delta_time_mean']:.3f}, {row['delta_time_p50']:.3f}, {row['delta_time_p10']:.3f}, {row['delta_time_p90']:.3f} | "
            f"{row['delta_coverage_mean']:.5f} | {row['delta_turn_index_mean']:.6f} | "
            f"{row['time_wins_potential']}/{row['time_losses_potential']}/{row['time_ties']} | "
            f"{row['time_sign_test_p']:.4f} | {row['potential_pass_rule_rate']:.3f} |"
        )
    summary_lines.append("")
    summary_lines.append("Interpretation note: deltas are potential_event minus heuristic.")
    summary_lines.append("Lower is better for time/turn; higher is better for coverage.")
    summary_lines.append("")
    summary_lines.append("## Outputs")
    summary_lines.append(f"- raw runs: {raw_csv}")
    summary_lines.append(f"- paired diff: {pair_csv}")
    summary_lines.append(f"- condition summary: {cond_csv}")
    summary_lines.append(f"- errors: {errors_csv}")

    summary_md = out_dir / "summary.md"
    summary_txt = out_dir / "summary.txt"
    text = "\n".join(summary_lines) + "\n"
    summary_md.write_text(text, encoding="utf-8")
    summary_txt.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
