from __future__ import annotations

import json
from math import cos, sin
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from .models import AgentParams, AgentState, AgentType, Environment, SimulationConfig


def _lerp(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t)


def _hex_color(rgb: Tuple[int, int, int]) -> str:
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def coverage_color(q: float) -> str:
    # low: pale blue, high: ocean green.
    q = max(0.0, min(1.0, q))
    low = (230, 242, 255)
    high = (17, 122, 77)
    rgb = (_lerp(low[0], high[0], q), _lerp(low[1], high[1], q), _lerp(low[2], high[2], q))
    return _hex_color(rgb)


def _rotate_translate(
    points: Sequence[Tuple[float, float]],
    x: float,
    y: float,
    psi: float,
) -> List[Tuple[float, float]]:
    c = cos(psi)
    s = sin(psi)
    out: List[Tuple[float, float]] = []
    for px, py in points:
        rx = c * px - s * py
        ry = s * px + c * py
        out.append((x + rx, y + ry))
    return out


def _usv_shape(state: AgentState, scale: float = 1.0) -> List[Tuple[float, float]]:
    base = [(8.0, 0.0), (2.5, 3.8), (-7.0, 3.5), (-9.0, 0.0), (-7.0, -3.5), (2.5, -3.8)]
    pts = [(px * scale, py * scale) for px, py in base]
    return _rotate_translate(pts, state.x, state.y, state.psi)


def _polygon_points_to_svg(
    points_world: Sequence[Tuple[float, float]],
    world_to_px,
) -> str:
    pairs = []
    for x, y in points_world:
        px, py = world_to_px(x, y)
        pairs.append(f"{px:.2f},{py:.2f}")
    return " ".join(pairs)


def _sector_polygon(
    center: Tuple[float, float],
    heading: float,
    radius: float,
    fov: float,
    n: int = 24,
) -> List[Tuple[float, float]]:
    cx, cy = center
    pts: List[Tuple[float, float]] = [(cx, cy)]
    start = heading - fov / 2.0
    if n < 2:
        n = 2
    for k in range(n + 1):
        a = start + fov * (k / n)
        pts.append((cx + radius * cos(a), cy + radius * sin(a)))
    return pts


def save_snapshot_svg(
    output_path: str,
    env: Environment,
    sim: SimulationConfig,
    trajectories: Dict[str, List[AgentState]],
    params_map: Dict[str, AgentParams],
    grid_xy: Sequence[Tuple[float, float]],
    coverage_quality: Sequence[float],
    focus_regions: Sequence[Tuple[float, float, float, str]] | None = None,
    title: str = "Step1 UAV-USV Coverage Snapshot",
) -> str:
    width = 1280
    height = 900
    margin = 65

    xmin, xmax = env.xlim
    ymin, ymax = env.ylim
    xr = max(1e-6, xmax - xmin)
    yr = max(1e-6, ymax - ymin)
    sx = (width - 2.0 * margin) / xr
    sy = (height - 2.0 * margin) / yr
    scale = min(sx, sy)

    def world_to_px(x: float, y: float) -> Tuple[float, float]:
        px = margin + (x - xmin) * scale
        py = height - margin - (y - ymin) * scale
        return px, py

    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="#f4f9ff"/>')
    lines.append(
        '<rect x="18" y="18" width="1244" height="864" rx="14" fill="none" stroke="#6f869b" stroke-width="1.2"/>'
    )

    # Coverage heatmap as grid cells.
    if sim.nx >= 2 and sim.ny >= 2:
        cell_w = (xmax - xmin) / (sim.nx - 1)
        cell_h = (ymax - ymin) / (sim.ny - 1)
        idx = 0
        for j in range(sim.ny):
            for i in range(sim.nx):
                gx, gy = grid_xy[idx]
                q = coverage_quality[idx] if idx < len(coverage_quality) else 0.0
                cx0 = gx - cell_w / 2.0
                cy0 = gy - cell_h / 2.0
                px0, py0 = world_to_px(cx0, cy0 + cell_h)
                px1, py1 = world_to_px(cx0 + cell_w, cy0)
                w = max(0.3, px1 - px0)
                h = max(0.3, py1 - py0)
                lines.append(
                    f'<rect x="{px0:.2f}" y="{py0:.2f}" width="{w:.2f}" height="{h:.2f}" '
                    f'fill="{coverage_color(q)}" fill-opacity="0.38" stroke="none"/>'
                )
                idx += 1

    # Obstacles.
    for obs in env.obstacles:
        p0 = world_to_px(obs.xmin, obs.ymax)
        p1 = world_to_px(obs.xmax, obs.ymin)
        x0, y0 = p0
        x1, y1 = p1
        w = x1 - x0
        h = y1 - y0
        lines.append(
            f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{w:.2f}" height="{h:.2f}" '
            'fill="#30374a" fill-opacity="0.55" stroke="#0f172a" stroke-width="1.5"/>'
        )

    # Domain border.
    bx0, by0 = world_to_px(xmin, ymax)
    bx1, by1 = world_to_px(xmax, ymin)
    lines.append(
        f'<rect x="{bx0:.2f}" y="{by0:.2f}" width="{bx1-bx0:.2f}" height="{by1-by0:.2f}" '
        'fill="none" stroke="#37506b" stroke-width="2"/>'
    )

    # Optional focus/problem regions.
    if focus_regions:
        for cx, cy, radius, label in focus_regions:
            px, py = world_to_px(cx, cy)
            lines.append(
                f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{max(4.0, radius * scale):.2f}" '
                'fill="#ef4444" fill-opacity="0.08" stroke="#b91c1c" stroke-width="1.8" stroke-dasharray="6,4"/>'
            )
            lines.append(
                f'<text x="{px + 6:.2f}" y="{py - 6:.2f}" font-size="12" fill="#7f1d1d">{label}</text>'
            )

    # Agent trajectories and current footprints.
    for name, states in trajectories.items():
        if not states:
            continue
        params = params_map[name]
        color = "#f97316" if params.agent_type == AgentType.UAV else "#2563eb"
        stroke_w = 2.8 if params.agent_type == AgentType.UAV else 2.2

        traj_points = []
        for st in states:
            tx, ty = world_to_px(st.x, st.y)
            traj_points.append(f"{tx:.2f},{ty:.2f}")
        lines.append(
            f'<polyline points="{" ".join(traj_points)}" fill="none" stroke="{color}" '
            f'stroke-width="{stroke_w:.2f}" stroke-linecap="round" stroke-linejoin="round" opacity="0.88"/>'
        )

        st0 = states[0]
        stf = states[-1]
        s0x, s0y = world_to_px(st0.x, st0.y)
        lines.append(
            f'<circle cx="{s0x:.2f}" cy="{s0y:.2f}" r="4.4" fill="#ffffff" stroke="{color}" stroke-width="2"/>'
        )

        if params.agent_type == AgentType.UAV:
            fp_radius = params.sensor.radius
            cxf, cyf = world_to_px(stf.x, stf.y)
            lines.append(
                f'<circle cx="{cxf:.2f}" cy="{cyf:.2f}" r="{fp_radius * scale:.2f}" fill="{color}" '
                'fill-opacity="0.08" stroke="#ea580c" stroke-width="1.5" stroke-dasharray="6,4"/>'
            )
            lines.append(
                f'<circle cx="{cxf:.2f}" cy="{cyf:.2f}" r="5.2" fill="#fb923c" stroke="#9a3412" stroke-width="2"/>'
            )
            lines.append(
                f'<line x1="{cxf - 7:.2f}" y1="{cyf:.2f}" x2="{cxf + 7:.2f}" y2="{cyf:.2f}" '
                'stroke="#9a3412" stroke-width="1.6"/>'
            )
            lines.append(
                f'<line x1="{cxf:.2f}" y1="{cyf - 7:.2f}" x2="{cxf:.2f}" y2="{cyf + 7:.2f}" '
                'stroke="#9a3412" stroke-width="1.6"/>'
            )
        else:
            sector = _sector_polygon((stf.x, stf.y), stf.psi, params.sensor.radius, params.sensor.fov, n=28)
            sector_svg = _polygon_points_to_svg(sector, world_to_px)
            lines.append(
                f'<polygon points="{sector_svg}" fill="{color}" fill-opacity="0.10" stroke="#1d4ed8" '
                'stroke-width="1.2" stroke-dasharray="5,4"/>'
            )
            shape = _usv_shape(stf, scale=1.05)
            shape_svg = _polygon_points_to_svg(shape, world_to_px)
            lines.append(
                f'<polygon points="{shape_svg}" fill="#dbeafe" stroke="#1d4ed8" stroke-width="2"/>'
            )

        p0x, p0y = world_to_px(stf.x, stf.y)
        if params.agent_type == AgentType.USV:
            # USV heading is meaningful because of non-holonomic kinematics.
            hx = stf.x + 11.0 * cos(stf.psi)
            hy = stf.y + 11.0 * sin(stf.psi)
            p1x, p1y = world_to_px(hx, hy)
            lines.append(
                f'<line x1="{p0x:.2f}" y1="{p0y:.2f}" x2="{p1x:.2f}" y2="{p1y:.2f}" '
                f'stroke="{color}" stroke-width="2.1"/>'
            )
        lines.append(
            f'<text x="{p0x + 7:.2f}" y="{p0y - 8:.2f}" font-size="13" fill="#0f172a">{name}</text>'
        )

    lines.append(f'<text x="36" y="50" font-size="24" fill="#0f172a" font-weight="700">{title}</text>')
    lines.append(
        '<text x="36" y="74" font-size="13" fill="#334155">'
        'Orange: UAV, Blue: USV, shaded map: coverage quality, dark blocks: obstacles</text>'
    )

    # Legend.
    legend_x = width - 360
    legend_y = 44
    lines.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="320" height="104" rx="8" fill="#ffffff" '
        'fill-opacity="0.88" stroke="#94a3b8"/>'
    )
    lines.append(
        f'<line x1="{legend_x + 14}" y1="{legend_y + 26}" x2="{legend_x + 62}" y2="{legend_y + 26}" '
        'stroke="#f97316" stroke-width="3"/>'
    )
    lines.append(
        f'<text x="{legend_x + 70}" y="{legend_y + 30}" font-size="13" fill="#0f172a">UAV trajectory</text>'
    )
    lines.append(
        f'<line x1="{legend_x + 14}" y1="{legend_y + 50}" x2="{legend_x + 62}" y2="{legend_y + 50}" '
        'stroke="#2563eb" stroke-width="3"/>'
    )
    lines.append(
        f'<text x="{legend_x + 70}" y="{legend_y + 54}" font-size="13" fill="#0f172a">USV trajectory</text>'
    )
    lines.append(
        f'<text x="{legend_x + 14}" y="{legend_y + 79}" font-size="12" fill="#334155">'
        'Dashed circle: UAV sensing radius</text>'
    )
    lines.append(
        f'<text x="{legend_x + 14}" y="{legend_y + 96}" font-size="12" fill="#334155">'
        'Dashed sector: USV sensing sector</text>'
    )

    lines.append("</svg>")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _palette(index: int) -> str:
    colors = [
        "#e11d48",
        "#0369a1",
        "#16a34a",
        "#ca8a04",
        "#7c3aed",
        "#0f766e",
        "#9f1239",
        "#4338ca",
    ]
    return colors[index % len(colors)]


def save_partition_svg(
    output_path: str,
    env: Environment,
    sim: SimulationConfig,
    grid_xy: Sequence[Tuple[float, float]],
    owner_by_cell: Sequence[str],
    priority_map: Sequence[float],
    states: Dict[str, AgentState],
    params_map: Dict[str, AgentParams],
    planned_paths: Dict[str, Sequence[Tuple[float, float]]],
    color_by_agent: Dict[str, str] | None = None,
    focus_regions: Sequence[Tuple[float, float, float, str]] | None = None,
    title: str = "Step2 Weighted Voronoi Partition and Paths",
) -> str:
    width = 1280
    height = 900
    margin = 65

    xmin, xmax = env.xlim
    ymin, ymax = env.ylim
    xr = max(1e-6, xmax - xmin)
    yr = max(1e-6, ymax - ymin)
    sx = (width - 2.0 * margin) / xr
    sy = (height - 2.0 * margin) / yr
    scale = min(sx, sy)

    agent_names = list(states.keys())
    computed_color_by_agent = {name: _palette(i) for i, name in enumerate(agent_names)}
    if color_by_agent:
        computed_color_by_agent.update({k: v for k, v in color_by_agent.items() if isinstance(v, str)})

    def world_to_px(x: float, y: float) -> Tuple[float, float]:
        px = margin + (x - xmin) * scale
        py = height - margin - (y - ymin) * scale
        return px, py

    lines: List[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    lines.append('<rect x="0" y="0" width="100%" height="100%" fill="#f6fbff"/>')
    lines.append(
        '<rect x="18" y="18" width="1244" height="864" rx="14" fill="none" stroke="#6f869b" stroke-width="1.2"/>'
    )

    if sim.nx >= 2 and sim.ny >= 2 and len(owner_by_cell) == len(grid_xy):
        cell_w = (xmax - xmin) / (sim.nx - 1)
        cell_h = (ymax - ymin) / (sim.ny - 1)
        for idx, (gx, gy) in enumerate(grid_xy):
            owner = owner_by_cell[idx]
            base_color = computed_color_by_agent.get(owner, "#64748b")
            pri = priority_map[idx] if idx < len(priority_map) else 0.5
            alpha = 0.18 + 0.32 * max(0.0, min(1.0, pri))

            cx0 = gx - cell_w / 2.0
            cy0 = gy - cell_h / 2.0
            px0, py0 = world_to_px(cx0, cy0 + cell_h)
            px1, py1 = world_to_px(cx0 + cell_w, cy0)
            w = max(0.3, px1 - px0)
            h = max(0.3, py1 - py0)
            lines.append(
                f'<rect x="{px0:.2f}" y="{py0:.2f}" width="{w:.2f}" height="{h:.2f}" '
                f'fill="{base_color}" fill-opacity="{alpha:.3f}" stroke="none"/>'
            )

    for obs in env.obstacles:
        p0 = world_to_px(obs.xmin, obs.ymax)
        p1 = world_to_px(obs.xmax, obs.ymin)
        x0, y0 = p0
        x1, y1 = p1
        lines.append(
            f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{x1-x0:.2f}" height="{y1-y0:.2f}" '
            'fill="#1f2937" fill-opacity="0.60" stroke="#0f172a" stroke-width="1.4"/>'
        )

    if focus_regions:
        for cx, cy, radius, label in focus_regions:
            px, py = world_to_px(cx, cy)
            lines.append(
                f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{max(4.0, radius * scale):.2f}" '
                'fill="#ef4444" fill-opacity="0.10" stroke="#b91c1c" stroke-width="1.8" stroke-dasharray="6,4"/>'
            )
            lines.append(f'<text x="{px + 6:.2f}" y="{py - 6:.2f}" font-size="12" fill="#7f1d1d">{label}</text>')

    bx0, by0 = world_to_px(xmin, ymax)
    bx1, by1 = world_to_px(xmax, ymin)
    lines.append(
        f'<rect x="{bx0:.2f}" y="{by0:.2f}" width="{bx1-bx0:.2f}" height="{by1-by0:.2f}" '
        'fill="none" stroke="#1e3a5f" stroke-width="2"/>'
    )

    for name in agent_names:
        color = computed_color_by_agent[name]
        path = list(planned_paths.get(name, []))
        if len(path) >= 2:
            pts = []
            for x, y in path:
                px, py = world_to_px(x, y)
                pts.append(f"{px:.2f},{py:.2f}")
            lines.append(
                f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.2" '
                'stroke-linejoin="round" stroke-linecap="round" opacity="0.92"/>'
            )

        st = states[name]
        params = params_map[name]
        cxf, cyf = world_to_px(st.x, st.y)
        if params.agent_type == AgentType.UAV:
            lines.append(f'<circle cx="{cxf:.2f}" cy="{cyf:.2f}" r="5.3" fill="{color}" stroke="#111827" stroke-width="1.6"/>')
            lines.append(
                f'<line x1="{cxf - 7:.2f}" y1="{cyf:.2f}" x2="{cxf + 7:.2f}" y2="{cyf:.2f}" '
                'stroke="#111827" stroke-width="1.4"/>'
            )
            lines.append(
                f'<line x1="{cxf:.2f}" y1="{cyf - 7:.2f}" x2="{cxf:.2f}" y2="{cyf + 7:.2f}" '
                'stroke="#111827" stroke-width="1.4"/>'
            )
        else:
            shape = _usv_shape(st, scale=1.05)
            shape_svg = _polygon_points_to_svg(shape, world_to_px)
            lines.append(
                f'<polygon points="{shape_svg}" fill="{color}" fill-opacity="0.25" stroke="#0f172a" stroke-width="1.6"/>'
            )
            hx = st.x + 10.0 * cos(st.psi)
            hy = st.y + 10.0 * sin(st.psi)
            p1x, p1y = world_to_px(hx, hy)
            lines.append(
                f'<line x1="{cxf:.2f}" y1="{cyf:.2f}" x2="{p1x:.2f}" y2="{p1y:.2f}" '
                'stroke="#0f172a" stroke-width="1.4"/>'
            )
        lines.append(f'<text x="{cxf + 7:.2f}" y="{cyf - 8:.2f}" font-size="12" fill="#111827">{name}</text>')

    lines.append(f'<text x="36" y="50" font-size="24" fill="#0f172a" font-weight="700">{title}</text>')
    lines.append(
        '<text x="36" y="74" font-size="13" fill="#334155">'
        'Cell color = weighted Voronoi owner; opacity increases with monitoring priority.</text>'
    )

    legend_x = width - 350
    legend_y = 44
    legend_h = 28 + 22 * len(agent_names)
    lines.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="320" height="{legend_h}" rx="8" fill="#ffffff" '
        'fill-opacity="0.90" stroke="#94a3b8"/>'
    )
    lines.append(
        f'<text x="{legend_x + 12}" y="{legend_y + 18}" font-size="12" fill="#334155">Agents and partition colors</text>'
    )
    for i, name in enumerate(agent_names):
        y = legend_y + 36 + 20 * i
        color = computed_color_by_agent[name]
        lines.append(f'<rect x="{legend_x + 12}" y="{y - 10}" width="18" height="12" fill="{color}" />')
        agent_type = params_map[name].agent_type.value.upper()
        lines.append(f'<text x="{legend_x + 38}" y="{y}" font-size="12" fill="#111827">{name} ({agent_type})</text>')

    lines.append("</svg>")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def save_mission_animation_html(
    output_path: str,
    env: Environment,
    obstacles: Sequence[Tuple[float, float, float, float]],
    focus_regions: Sequence[Tuple[float, float, float, str]],
    region_done_time: Dict[str, float] | None,
    frame_known_difficulty: Sequence[Dict[str, str]] | None,
    frame_task_assignment: Sequence[Dict[str, Sequence[str]]] | None,
    frame_coverage_rate: Sequence[float] | None,
    frame_task_completion_rate: Sequence[float] | None,
    frame_turn_rate: Sequence[float] | None,
    frame_load_table: Sequence[Dict[str, Dict[str, float]]] | None,
    frame_partition_snapshot_idx: Sequence[int] | None,
    partition_owner_snapshots: Sequence[Sequence[str]] | None,
    partition_grid_xy: Sequence[Tuple[float, float]] | None,
    partition_grid_shape: Tuple[int, int] | None,
    frame_times: Sequence[float],
    agent_types: Dict[str, str],
    agent_states: Dict[str, Sequence[Tuple[float, float, float]]],
    stage1_end_frame: int,
    current_model: Dict[str, float | bool] | None = None,
    agent_colors: Dict[str, str] | None = None,
    title: str = "UAV-USV Mission Animation",
) -> str:
    payload = {
        "xlim": [env.xlim[0], env.xlim[1]],
        "ylim": [env.ylim[0], env.ylim[1]],
        "obstacles": [[a, b, c, d] for (a, b, c, d) in obstacles],
        "regions": [[x, y, r, label] for (x, y, r, label) in focus_regions],
        "region_done": dict(region_done_time or {}),
        "frame_known": [{k: v for k, v in fr.items()} for fr in (frame_known_difficulty or [])],
        "frame_assign": [
            {k: list(v) for k, v in fr.items()}
            for fr in (frame_task_assignment or [])
        ],
        "frame_cov_rate": [float(v) for v in (frame_coverage_rate or [])],
        "frame_task_rate": [float(v) for v in (frame_task_completion_rate or [])],
        "frame_turn_rate": [float(v) for v in (frame_turn_rate or [])],
        "frame_load_table": [
            {
                name: {
                    "travel_m": float(vals.get("travel_m", 0.0)),
                    "work_s": float(vals.get("work_s", 0.0)),
                    "load_eq_m": float(vals.get("load_eq_m", 0.0)),
                }
                for name, vals in fr.items()
            }
            for fr in (frame_load_table or [])
        ],
        "frame_partition_idx": [int(v) for v in (frame_partition_snapshot_idx or [])],
        "partition_snapshots": [list(owner) for owner in (partition_owner_snapshots or [])],
        "partition_grid": [[x, y] for (x, y) in (partition_grid_xy or [])],
        "partition_nx": int(partition_grid_shape[0]) if partition_grid_shape else 0,
        "partition_ny": int(partition_grid_shape[1]) if partition_grid_shape else 0,
        "times": list(frame_times),
        "types": dict(agent_types),
        "states": {k: [[x, y, psi] for (x, y, psi) in v] for k, v in agent_states.items()},
        "current_model": dict(current_model or {}),
        "agent_colors": dict(agent_colors or {}),
        "stage1_end": int(stage1_end_frame),
        "title": title,
    }
    json_blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      font-family: "SF Pro Text", "Segoe UI", Roboto, sans-serif;
      background: linear-gradient(180deg, #eef6ff 0%, #ddefff 100%);
      color: #0f172a;
    }}
    .wrap {{ max-width: 1620px; margin: 16px auto; padding: 12px; }}
    .card {{
      background: rgba(255,255,255,0.85);
      border: 1px solid #9db4cc;
      border-radius: 12px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
      padding: 10px;
    }}
    .head {{ display: flex; justify-content: space-between; gap: 8px; margin-bottom: 8px; align-items: center; }}
    .title {{ font-size: 18px; font-weight: 700; }}
    .meta {{ font-size: 13px; color: #334155; }}
    .main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 10px;
      align-items: start;
    }}
    .view {{ min-width: 0; }}
    canvas {{ width: 100%; height: auto; border-radius: 10px; background: #f6fbff; border: 1px solid #94a3b8; }}
    .side {{
      background: rgba(255,255,255,0.92);
      border: 1px solid #94a3b8;
      border-radius: 10px;
      padding: 8px;
      max-height: 700px;
      overflow: auto;
    }}
    .panel-title {{ font-size: 13px; font-weight: 700; margin: 0 0 6px 0; color: #0f172a; }}
    .panel-sub {{ font-size: 12px; color: #475569; margin: 0 0 8px 0; }}
    .task-list {{ display: grid; gap: 6px; margin-bottom: 10px; }}
    .task-row {{
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      padding: 5px 6px;
      font-size: 12px;
      background: #f8fbff;
    }}
    .task-row.done {{
      background: #ecfdf5;
      border-color: #86efac;
    }}
    .task-empty {{ font-size: 12px; color: #64748b; padding: 4px 2px; }}
    .metric-row {{
      border: 1px solid #bfdbfe;
      border-radius: 8px;
      padding: 5px 6px;
      font-size: 12px;
      background: #eff6ff;
    }}
    .controls {{
      display: grid;
      grid-template-columns: auto auto 1fr auto auto;
      gap: 8px;
      align-items: center;
      margin-top: 10px;
    }}
    button, select {{
      border: 1px solid #94a3b8;
      background: #ffffff;
      border-radius: 8px;
      padding: 6px 10px;
      font-size: 13px;
    }}
    input[type="range"] {{ width: 100%; }}
    .foot {{ margin-top: 8px; font-size: 12px; color: #475569; }}
    @media (max-width: 1240px) {{
      .main {{
        grid-template-columns: 1fr;
      }}
      .side {{
        max-height: 360px;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="head">
        <div class="title">{title}</div>
        <div id="meta" class="meta"></div>
      </div>
      <div class="main">
        <div class="view">
          <canvas id="cv" width="1220" height="840"></canvas>
        </div>
        <div class="side">
          <p class="panel-title">运行仪表盘</p>
          <div id="dashPanel" class="task-list"></div>
          <p class="panel-title">USV 负载表</p>
          <div id="loadPanel" class="task-list"></div>
          <p class="panel-title">任务面板</p>
          <p id="taskSummary" class="panel-sub"></p>
          <p class="panel-title">已发现任务（含难度）</p>
          <div id="taskPanel" class="task-list"></div>
          <p class="panel-title">当前任务分配</p>
          <div id="assignPanel" class="task-list"></div>
        </div>
      </div>
      <div class="controls">
        <button id="playBtn">Pause</button>
        <button id="resetBtn">Reset</button>
        <input id="slider" type="range" min="0" max="0" step="1" value="0" />
        <label for="speedSel">Speed</label>
        <select id="speedSel">
          <option value="0.5">0.5x</option>
          <option value="1.0" selected>1.0x</option>
          <option value="2.0">2.0x</option>
          <option value="4.0">4.0x</option>
        </select>
      </div>
      <div class="foot">
        Blue flow arrows show ocean current field; translucent colored cells show real-time Voronoi partition; UAV marker has no trail.
      </div>
    </div>
  </div>
  <script>
    const DATA = {json_blob};
    const cv = document.getElementById("cv");
    const ctx = cv.getContext("2d");
    const slider = document.getElementById("slider");
    const playBtn = document.getElementById("playBtn");
    const resetBtn = document.getElementById("resetBtn");
    const speedSel = document.getElementById("speedSel");
    const meta = document.getElementById("meta");
    const dashPanel = document.getElementById("dashPanel");
    const loadPanel = document.getElementById("loadPanel");
    const taskSummary = document.getElementById("taskSummary");
    const taskPanel = document.getElementById("taskPanel");
    const assignPanel = document.getElementById("assignPanel");

    const names = Object.keys(DATA.states);
    const colors = {{}};
    const usvPalette = ["#2563eb", "#0f766e", "#dc2626", "#ca8a04", "#9333ea", "#be123c", "#0369a1", "#ea580c"];
    let usvIdx = 0;
    names.forEach((n) => {{
      if (DATA.agent_colors && DATA.agent_colors[n]) {{
        colors[n] = DATA.agent_colors[n];
      }} else if (DATA.types[n] === "uav") {{
        colors[n] = "#f97316";
      }} else {{
        colors[n] = usvPalette[usvIdx % usvPalette.length];
        usvIdx += 1;
      }}
    }});

    const xmin = DATA.xlim[0], xmax = DATA.xlim[1];
    const ymin = DATA.ylim[0], ymax = DATA.ylim[1];
    const W = cv.width, H = cv.height;
    const margin = 56;
    const sx = (W - 2*margin) / (xmax - xmin);
    const sy = (H - 2*margin) / (ymax - ymin);
    const s = Math.min(sx, sy);
    const toPx = (x, y) => [margin + (x - xmin) * s, H - margin - (y - ymin) * s];
    const currentModel = DATA.current_model || null;
    const partitionGrid = Array.isArray(DATA.partition_grid) ? DATA.partition_grid : [];
    const partitionSnapshots = Array.isArray(DATA.partition_snapshots) ? DATA.partition_snapshots : [];
    const framePartitionIdx = Array.isArray(DATA.frame_partition_idx) ? DATA.frame_partition_idx : [];
    const frameCovRate = Array.isArray(DATA.frame_cov_rate) ? DATA.frame_cov_rate : [];
    const frameTaskRate = Array.isArray(DATA.frame_task_rate) ? DATA.frame_task_rate : [];
    const frameTurnRate = Array.isArray(DATA.frame_turn_rate) ? DATA.frame_turn_rate : [];
    const frameLoadTable = Array.isArray(DATA.frame_load_table) ? DATA.frame_load_table : [];
    const partitionNx = Number(DATA.partition_nx || 0);
    const partitionNy = Number(DATA.partition_ny || 0);
    const canDrawPartition = (
      partitionNx >= 2
      && partitionNy >= 2
      && partitionGrid.length === partitionNx * partitionNy
      && partitionSnapshots.length > 0
    );
    const partitionCellW = canDrawPartition ? (xmax - xmin) / (partitionNx - 1) : 0;
    const partitionCellH = canDrawPartition ? (ymax - ymin) / (partitionNy - 1) : 0;
    const usvNames = names.filter((n) => DATA.types[n] !== "uav").sort();

    const N = DATA.times.length;
    slider.max = Math.max(0, N - 1);
    let frame = 0;
    let playing = true;
    let lastTs = performance.now();
    let acc = 0;
    const difficultyCN = {{ low: "下", mid: "中", high: "上", unknown: "未知" }};

    function taskLabelKey(label) {{
      const m = String(label).match(/(\\d+)/);
      return m ? Number(m[1]) : 999999;
    }}

    const taskOrder = DATA.regions
      .map((r) => String(r[3]))
      .sort((a, b) => {{
        const ka = taskLabelKey(a);
        const kb = taskLabelKey(b);
        if (ka !== kb) return ka - kb;
        return a.localeCompare(b);
      }});

    function isDone(label, tm) {{
      return Object.prototype.hasOwnProperty.call(DATA.region_done, label)
        && tm >= Number(DATA.region_done[label]);
    }}

    function escHtml(s) {{
      return String(s)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    function renderSidePanel(frameIdx, tm) {{
      const fidxKnown = Math.min(frameIdx, Math.max(0, DATA.frame_known.length - 1));
      const fidxAssign = Math.min(frameIdx, Math.max(0, DATA.frame_assign.length - 1));
      const known = DATA.frame_known[fidxKnown] || {{}};
      const assign = DATA.frame_assign[fidxAssign] || {{}};

      const discoveredLabels = Object.keys(known).sort((a, b) => {{
        const ka = taskLabelKey(a);
        const kb = taskLabelKey(b);
        if (ka !== kb) return ka - kb;
        return a.localeCompare(b);
      }});
      const doneCount = taskOrder.filter((label) => isDone(label, tm)).length;
      taskSummary.textContent = `已发现 ${{discoveredLabels.length}}/${{taskOrder.length}} | 已完成 ${{doneCount}}/${{taskOrder.length}}`;
      const metricLen = Math.max(frameCovRate.length, frameTaskRate.length, frameTurnRate.length, frameLoadTable.length);
      const fidxMetric = Math.min(frameIdx, Math.max(0, metricLen - 1));
      const covRate = Number(frameCovRate[fidxMetric]);
      const taskRate = Number(frameTaskRate[fidxMetric]);
      const turnRate = Number(frameTurnRate[fidxMetric]);
      const covTxt = Number.isFinite(covRate) ? `${{(100.0 * covRate).toFixed(2)}}%` : "-";
      const taskTxt = Number.isFinite(taskRate) ? `${{(100.0 * taskRate).toFixed(2)}}%` : "-";
      const turnTxt = Number.isFinite(turnRate) ? `${{turnRate.toFixed(5)}} rad/m` : "-";
      dashPanel.innerHTML = [
        `<div class="metric-row"><b>覆盖率</b>: ${{covTxt}}</div>`,
        `<div class="metric-row"><b>任务完成率</b>: ${{taskTxt}}</div>`,
        `<div class="metric-row"><b>总转弯率</b>: ${{turnTxt}}</div>`,
      ].join("");
      const load = frameLoadTable[fidxMetric] || {{}};
      if (usvNames.length === 0) {{
        loadPanel.innerHTML = '<div class="task-empty">暂无 USV 负载数据</div>';
      }} else {{
        loadPanel.innerHTML = usvNames.map((name) => {{
          const row = load[name] || {{}};
          const travel = Number(row.travel_m);
          const work = Number(row.work_s);
          const eq = Number(row.load_eq_m);
          const travelTxt = Number.isFinite(travel) ? `${{travel.toFixed(1)}} m` : "-";
          const workTxt = Number.isFinite(work) ? `${{work.toFixed(1)}} s` : "-";
          const eqTxt = Number.isFinite(eq) ? `${{eq.toFixed(1)}}` : "-";
          return `<div class="task-row"><b>${{escHtml(name)}}</b> | travel=${{travelTxt}} | work=${{workTxt}} | load_eq=${{eqTxt}}</div>`;
        }}).join("");
      }}

      if (discoveredLabels.length === 0) {{
        taskPanel.innerHTML = '<div class="task-empty">暂无已确认难度的任务</div>';
      }} else {{
        taskPanel.innerHTML = discoveredLabels.map((label) => {{
          const diff = String(known[label] || "unknown");
          const done = isDone(label, tm);
          const cn = difficultyCN[diff] || diff;
          return `<div class="task-row${{done ? " done" : ""}}"><b>${{escHtml(label)}}</b> | 难度: ${{escHtml(cn)}}(${{escHtml(diff)}})</div>`;
        }}).join("");
      }}

      assignPanel.innerHTML = taskOrder.map((label) => {{
        const owners = Array.isArray(assign[label]) ? assign[label] : [];
        const ownersTxt = owners.length > 0 ? owners.join(", ") : "-";
        const diff = String(known[label] || "unknown");
        const cn = difficultyCN[diff] || diff;
        const done = isDone(label, tm);
        const status = done ? "已完成" : "处理中/待处理";
        return `<div class="task-row${{done ? " done" : ""}}"><b>${{escHtml(label)}}</b> | 分配: ${{escHtml(ownersTxt)}} | 难度: ${{escHtml(cn)}}(${{escHtml(diff)}}) | 状态: ${{status}}</div>`;
      }}).join("");
    }}

    function drawUSV(x, y, psi, c) {{
      const shape = [[8,0],[2.4,3.6],[-7,3.2],[-9,0],[-7,-3.2],[2.4,-3.6]];
      ctx.beginPath();
      shape.forEach((p, i) => {{
        const rx = Math.cos(psi)*p[0] - Math.sin(psi)*p[1];
        const ry = Math.sin(psi)*p[0] + Math.cos(psi)*p[1];
        const [px, py] = toPx(x + rx, y + ry);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      }});
      ctx.closePath();
      ctx.fillStyle = c + "55";
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 1.6;
      ctx.fill();
      ctx.stroke();
    }}

    function drawUAV(x, y, c) {{
      const [px, py] = toPx(x, y);
      ctx.beginPath();
      ctx.arc(px, py, 4.8, 0, Math.PI * 2);
      ctx.fillStyle = c;
      ctx.strokeStyle = "#9a3412";
      ctx.lineWidth = 1.4;
      ctx.fill();
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(px - 7, py); ctx.lineTo(px + 7, py);
      ctx.moveTo(px, py - 7); ctx.lineTo(px, py + 7);
      ctx.strokeStyle = "#7c2d12";
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }}

    function currentVel(x, y, tm) {{
      if (!currentModel) return [0, 0];
      let u = Number(currentModel.c0x || 0) + Number(currentModel.gx || 0) * x;
      let v = Number(currentModel.c0y || 0) + Number(currentModel.gy || 0) * y;

      const tidePeriod = Math.max(1e-6, Number(currentModel.tide_period || 1));
      const omega = (2 * Math.PI) / tidePeriod;
      const tidePhase = Number(currentModel.tide_phase || 0);
      const phase = omega * tm + tidePhase;
      u += Number(currentModel.tide_amp_u || 0) * Math.sin(phase);
      v += Number(currentModel.tide_amp_v || 0) * Math.cos(0.9 * phase);

      function addGyre(prefix) {{
        const cx = Number(currentModel[prefix + "_cx"] || 0);
        const cy = Number(currentModel[prefix + "_cy"] || 0);
        const strength = Number(currentModel[prefix + "_strength"] || 0);
        const sigma = Math.max(1e-6, Number(currentModel[prefix + "_sigma"] || 1));
        const clockwise = Boolean(currentModel[prefix + "_clockwise"]);
        if (Math.abs(strength) < 1e-9) return [0, 0];
        const dx = x - cx;
        const dy = y - cy;
        const rr = (dx * dx + dy * dy) / (2 * sigma * sigma);
        const w = strength * Math.exp(-rr);
        const orient = clockwise ? -1.0 : 1.0;
        const gu = orient * (-dy / sigma) * w;
        const gv = orient * (dx / sigma) * w;
        return [gu, gv];
      }}

      const [u1, v1] = addGyre("gyre1");
      const [u2, v2] = addGyre("gyre2");
      u += u1 + u2;
      v += v1 + v2;

      const yMid = 0.5 * (ymin + ymax);
      const yHalf = Math.max(1e-6, 0.5 * (ymax - ymin));
      const yNorm = (y - yMid) / yHalf;
      u += Number(currentModel.boundary_shear || 0) * yNorm;
      return [u, v];
    }}

    function drawCurrentField(tm) {{
      if (!currentModel) return;
      const nxA = 18;
      const nyA = 12;
      const stepX = (xmax - xmin) / (nxA + 1);
      const stepY = (ymax - ymin) / (nyA + 1);
      const scale = 16.0;

      for (let j = 1; j <= nyA; j += 1) {{
        const y = ymin + j * stepY;
        for (let i = 1; i <= nxA; i += 1) {{
          const x = xmin + i * stepX;
          const [u, v] = currentVel(x, y, tm);
          const mag = Math.hypot(u, v);
          if (mag < 1e-3) continue;
          const ux = u / mag;
          const uy = v / mag;
          const len = scale * Math.min(1.45, 0.42 + 2.0 * mag);
          const x2 = x + ux * len;
          const y2 = y + uy * len;
          const [px1, py1] = toPx(x, y);
          const [px2, py2] = toPx(x2, y2);

          const alpha = Math.min(0.55, 0.16 + 0.42 * mag);
          ctx.strokeStyle = `rgba(14,116,144,${{alpha.toFixed(3)}})`;
          ctx.lineWidth = 1.05;
          ctx.beginPath();
          ctx.moveTo(px1, py1);
          ctx.lineTo(px2, py2);
          ctx.stroke();

          const ah = 3.0;
          const th = Math.atan2(py2 - py1, px2 - px1);
          ctx.beginPath();
          ctx.moveTo(px2, py2);
          ctx.lineTo(px2 - ah * Math.cos(th - 0.55), py2 - ah * Math.sin(th - 0.55));
          ctx.lineTo(px2 - ah * Math.cos(th + 0.55), py2 - ah * Math.sin(th + 0.55));
          ctx.closePath();
          ctx.fillStyle = `rgba(14,116,144,${{alpha.toFixed(3)}})`;
          ctx.fill();
        }}
      }}
    }}

    function drawPartition(frameIdx) {{
      if (!canDrawPartition) return;
      const fidx = Math.min(frameIdx, Math.max(0, framePartitionIdx.length - 1));
      const snapIdx = Number(framePartitionIdx[fidx] ?? 0);
      if (!(snapIdx >= 0 && snapIdx < partitionSnapshots.length)) return;

      const ownerByCell = partitionSnapshots[snapIdx];
      if (!Array.isArray(ownerByCell) || ownerByCell.length !== partitionGrid.length) return;

      ctx.save();
      ctx.globalAlpha = 0.24;
      for (let i = 0; i < partitionGrid.length; i += 1) {{
        const grid = partitionGrid[i];
        if (!Array.isArray(grid) || grid.length < 2) continue;
        const gx = Number(grid[0]);
        const gy = Number(grid[1]);
        if (!Number.isFinite(gx) || !Number.isFinite(gy)) continue;
        const owner = String(ownerByCell[i] || "");
        ctx.fillStyle = colors[owner] || "#64748b";

        const cx0 = gx - partitionCellW / 2.0;
        const cy0 = gy - partitionCellH / 2.0;
        const [px0, py0] = toPx(cx0, cy0 + partitionCellH);
        const [px1, py1] = toPx(cx0 + partitionCellW, cy0);
        const w = Math.max(0.3, px1 - px0);
        const h = Math.max(0.3, py1 - py0);
        ctx.fillRect(px0, py0, w, h);
      }}
      ctx.restore();
    }}

    function draw(frameIdx) {{
      ctx.clearRect(0, 0, W, H);
      const grd = ctx.createLinearGradient(0, 0, 0, H);
      grd.addColorStop(0, "#eef8ff");
      grd.addColorStop(1, "#d9eeff");
      ctx.fillStyle = grd;
      ctx.fillRect(0, 0, W, H);

      const [bx0, by0] = toPx(xmin, ymax);
      const [bx1, by1] = toPx(xmax, ymin);
      ctx.strokeStyle = "#1d4f7a";
      ctx.lineWidth = 2;
      ctx.strokeRect(bx0, by0, bx1 - bx0, by1 - by0);

      const tm = DATA.times[Math.min(frameIdx, DATA.times.length - 1)] || 0;
      drawCurrentField(tm);
      drawPartition(frameIdx);

      DATA.obstacles.forEach((o) => {{
        const [x0, y1] = toPx(o[0], o[3]);
        const [x1, y0] = toPx(o[1], o[2]);
        ctx.fillStyle = "rgba(31,41,55,0.58)";
        ctx.strokeStyle = "#111827";
        ctx.lineWidth = 1.2;
        ctx.fillRect(x0, y1, x1 - x0, y0 - y1);
        ctx.strokeRect(x0, y1, x1 - x0, y0 - y1);
      }});

      DATA.regions.forEach((r) => {{
        const [px, py] = toPx(r[0], r[1]);
        const rr = Math.max(5, r[2] * s);
        const label = String(r[3]);
        const done = isDone(label, tm);
        const fill = done ? "rgba(34,197,94,0.15)" : "rgba(239,68,68,0.10)";
        const stroke = done ? "#15803d" : "#b91c1c";
        const text = done ? "#166534" : "#7f1d1d";
        ctx.beginPath();
        ctx.arc(px, py, rr, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.strokeStyle = stroke;
        ctx.setLineDash([6,4]);
        ctx.lineWidth = 1.4;
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = text;
        ctx.font = "12px sans-serif";
        ctx.fillText(label, px + 6, py - 6);
      }});

      names.forEach((name) => {{
        const arr = DATA.states[name];
        const c = colors[name];
        const maxIdx = Math.min(frameIdx, arr.length - 1);
        if (maxIdx < 0) return;

        if (DATA.types[name] !== "uav") {{
          ctx.beginPath();
          for (let i = 0; i <= maxIdx; i += 1) {{
            const p = arr[i];
            const [px, py] = toPx(p[0], p[1]);
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          }}
          ctx.strokeStyle = c;
          ctx.lineWidth = 2.0;
          ctx.globalAlpha = 0.8;
          ctx.stroke();
          ctx.globalAlpha = 1.0;
        }}

        const p = arr[maxIdx];
        if (DATA.types[name] === "uav") drawUAV(p[0], p[1], c);
        else drawUSV(p[0], p[1], p[2], c);

        const [lx, ly] = toPx(p[0], p[1]);
        ctx.fillStyle = "#0f172a";
        ctx.font = "12px sans-serif";
        ctx.fillText(name, lx + 8, ly - 8);
      }});

      const stage = frameIdx <= DATA.stage1_end ? "Stage 1 (UAV Mapping)" : "Stage 2-3 (Joint Online)";
      meta.textContent = `t=${{tm.toFixed(1)}} s | frame ${{frameIdx+1}}/${{N}} | ${{stage}}`;
      renderSidePanel(frameIdx, tm);
      slider.value = String(frameIdx);
    }}

    playBtn.onclick = () => {{
      playing = !playing;
      playBtn.textContent = playing ? "Pause" : "Play";
    }};
    resetBtn.onclick = () => {{ frame = 0; draw(frame); }};
    slider.oninput = () => {{ frame = Number(slider.value); draw(frame); }};

    function loop(ts) {{
      const dt = (ts - lastTs) / 1000.0;
      lastTs = ts;
      const speed = Number(speedSel.value || "1");
      if (playing) {{
        acc += dt * speed;
        const stepSec = 0.06;
        while (acc >= stepSec) {{
          acc -= stepSec;
          frame += 1;
          if (frame >= N) {{
            frame = N - 1;
            playing = false;
            playBtn.textContent = "Play";
            break;
          }}
        }}
      }}
      draw(frame);
      requestAnimationFrame(loop);
    }}

    draw(0);
    requestAnimationFrame(loop);
  </script>
</body>
</html>
"""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return str(path)
