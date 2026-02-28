from __future__ import annotations

import argparse
from math import sqrt
from typing import Dict, List, Sequence, Tuple

from demo_step1 import build_demo
from maritime_coverage import (
    AgentParams,
    AgentState,
    CoverageProblem,
    WeightedVoronoiConfig,
    make_priority_map,
    plan_heterogeneous_paths,
    save_partition_svg,
    weighted_voronoi_partition,
)


def path_length(start: Tuple[float, float], waypoints: Sequence[Tuple[float, float]]) -> float:
    if not waypoints:
        return 0.0
    total = 0.0
    prev = start
    for p in waypoints:
        dx = p[0] - prev[0]
        dy = p[1] - prev[1]
        total += sqrt(dx * dx + dy * dy)
        prev = p
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step2: Weighted Voronoi coordination demo")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Export weighted Voronoi partition and planned routes to SVG",
    )
    parser.add_argument(
        "--svg",
        default="/Users/tom/Documents/Multi-Agent/outputs/step2_voronoi.svg",
        help="SVG output path",
    )
    return parser.parse_args()


def run_demo(args: argparse.Namespace) -> None:
    env, sim, agent_runners = build_demo()
    problem = CoverageProblem(env, sim)

    states: Dict[str, AgentState] = {a.name: a.state for a in agent_runners}
    params_map: Dict[str, AgentParams] = {a.name: a.params for a in agent_runners}

    hotspots = [
        (50.0, 40.0, 24.0, 0.88),
        (110.0, 80.0, 30.0, 1.00),
        (178.0, 126.0, 26.0, 0.92),
    ]
    priority_map = make_priority_map(problem.grid_xy, hotspots=hotspots, base=0.18)

    cfg = WeightedVoronoiConfig(
        lambda_energy=0.62,
        lambda_current=0.45,
        lambda_priority=1.30,
        uav_speed_factor=1.00,
        usv_speed_factor=0.90,
        uav_priority_gain=0.95,
        usv_priority_gain=1.16,
        uav_type_bias=1.15,
        usv_type_bias=0.0,
    )
    partition = weighted_voronoi_partition(
        grid_xy=problem.grid_xy,
        states=states,
        params_map=params_map,
        env=env,
        priority_map=priority_map,
        cfg=cfg,
    )
    planned_paths = plan_heterogeneous_paths(partition.cells_by_agent, states, params_map)

    total_cells = len(problem.grid_xy)
    print("=== Step 2: Weighted Voronoi Coordination Demo ===")
    print(f"Total grid cells: {total_cells}")
    print("Partition summary:")
    for name in sorted(partition.count_by_agent.keys()):
        cnt = partition.count_by_agent[name]
        ratio = cnt / total_cells if total_cells > 0 else 0.0
        mean_cost = partition.mean_cost_by_agent[name]
        st = states[name]
        length = path_length((st.x, st.y), planned_paths.get(name, []))
        print(
            f"  {name}: cells={cnt} ({ratio:.2%}), mean_cost={mean_cost:.3f}, "
            f"path_waypoints={len(planned_paths.get(name, []))}, approx_path_len={length:.1f} m"
        )

    if args.visualize:
        svg_path = save_partition_svg(
            output_path=args.svg,
            env=env,
            sim=sim,
            grid_xy=problem.grid_xy,
            owner_by_cell=partition.owner_by_cell,
            priority_map=priority_map,
            states=states,
            params_map=params_map,
            planned_paths=planned_paths,
            title="Step 2 Weighted Voronoi Partition + Heterogeneous Paths",
        )
        print(f"SVG exported: {svg_path}")


if __name__ == "__main__":
    run_demo(parse_args())
