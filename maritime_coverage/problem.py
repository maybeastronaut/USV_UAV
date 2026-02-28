from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import exp, sqrt
from typing import Dict, List, Sequence, Tuple

from .models import AgentParams, AgentState, Environment, SimulationConfig


@dataclass
class ConstraintReport:
    no_collision: bool
    no_obstacle_violation: bool
    energy_ok: bool
    coverage_ok: bool
    revisit_ok: bool
    collision_count: int
    obstacle_hits: int
    low_energy_agents: int
    uncovered_cells: int
    high_revisit_cells: int


class CoverageProblem:
    def __init__(self, env: Environment, sim: SimulationConfig):
        self.env = env
        self.sim = sim
        self.grid_xy = self._make_grid(env.xlim, env.ylim, sim.nx, sim.ny)
        num_cells = len(self.grid_xy)
        self.weights = [1.0] * num_cells

        self.coverage_intensity = [0.0] * num_cells
        self.current_gap = [0.0] * num_cells
        self.max_gap = [0.0] * num_cells

    @staticmethod
    def _make_grid(
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        nx: int,
        ny: int,
    ) -> List[Tuple[float, float]]:
        if nx < 2 or ny < 2:
            raise ValueError("nx and ny must be >= 2")
        x_step = (xlim[1] - xlim[0]) / (nx - 1)
        y_step = (ylim[1] - ylim[0]) / (ny - 1)
        grid: List[Tuple[float, float]] = []
        for j in range(ny):
            y = ylim[0] + j * y_step
            for i in range(nx):
                x = xlim[0] + i * x_step
                grid.append((x, y))
        return grid

    def update(self, fused_prob: Sequence[float], dt: float) -> None:
        for i, prob in enumerate(fused_prob):
            self.coverage_intensity[i] += prob * dt
            if prob >= self.sim.seen_prob_threshold:
                self.current_gap[i] = 0.0
            else:
                self.current_gap[i] += dt
            if self.current_gap[i] > self.max_gap[i]:
                self.max_gap[i] = self.current_gap[i]

    @property
    def coverage_quality(self) -> List[float]:
        return [1.0 - exp(-value) for value in self.coverage_intensity]

    def objective(self, control_history: Dict[str, List[Tuple[float, float]]]) -> float:
        quality = self.coverage_quality
        coverage_term = sum(w * q for w, q in zip(self.weights, quality))
        effort = 0.0
        for controls in control_history.values():
            for a, omega in controls:
                effort += (a * a + omega * omega) * self.sim.dt
        return coverage_term - self.sim.control_penalty * effort

    def evaluate_constraints(
        self,
        trajectories: Dict[str, List[AgentState]],
        final_states: Dict[str, AgentState],
        params_map: Dict[str, AgentParams],
    ) -> ConstraintReport:
        names = list(trajectories.keys())
        steps = min(len(v) for v in trajectories.values())

        collision_count = 0
        obstacle_hits = 0
        for k in range(steps):
            for i, j in combinations(names, 2):
                si = trajectories[i][k]
                sj = trajectories[j][k]
                dist = sqrt((si.x - sj.x) ** 2 + (si.y - sj.y) ** 2)
                if dist < self.sim.safe_distance:
                    collision_count += 1

            for i in names:
                s = trajectories[i][k]
                if self.env.in_obstacle(s.x, s.y):
                    obstacle_hits += 1

        low_energy_agents = 0
        for i, s in final_states.items():
            if s.energy < params_map[i].energy_min:
                low_energy_agents += 1

        quality = self.coverage_quality
        uncovered_cells = sum(1 for q in quality if q < self.sim.coverage_threshold)
        high_revisit_cells = sum(1 for g in self.max_gap if g > self.sim.revisit_max_gap)

        return ConstraintReport(
            no_collision=collision_count == 0,
            no_obstacle_violation=obstacle_hits == 0,
            energy_ok=low_energy_agents == 0,
            coverage_ok=uncovered_cells == 0,
            revisit_ok=high_revisit_cells == 0,
            collision_count=collision_count,
            obstacle_hits=obstacle_hits,
            low_energy_agents=low_energy_agents,
            uncovered_cells=uncovered_cells,
            high_revisit_cells=high_revisit_cells,
        )
