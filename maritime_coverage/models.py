from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import cos, exp, pi, sin
from typing import List, Tuple


class AgentType(str, Enum):
    UAV = "uav"
    USV = "usv"


@dataclass
class Control:
    a: float
    omega: float


@dataclass
class AgentState:
    x: float
    y: float
    psi: float
    v: float
    energy: float


@dataclass
class SensorParams:
    radius: float
    kappa: float = 0.95
    sigma_r: float = 15.0
    # UAV uses full 2*pi by default; USV can set narrow angle, e.g. pi/2.
    fov: float = 2.0 * pi
    sigma_phi: float = 0.6
    altitude: float = 40.0
    camera_fov: float = pi / 3.0


@dataclass
class AgentParams:
    agent_type: AgentType
    v_min: float
    v_max: float
    omega_max: float
    a_max: float
    energy_min: float
    base_power: float
    kv: float
    kw: float
    sensor: SensorParams


@dataclass
class ObstacleRect:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def contains(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax


@dataclass
class OceanCurrent:
    c0x: float = 0.3
    c0y: float = 0.0
    gx: float = 0.0
    gy: float = 0.0
    gyre1_cx: float = 150.0
    gyre1_cy: float = 140.0
    gyre1_strength: float = 0.0
    gyre1_sigma: float = 90.0
    gyre1_clockwise: bool = False
    gyre2_cx: float = 410.0
    gyre2_cy: float = 280.0
    gyre2_strength: float = 0.0
    gyre2_sigma: float = 95.0
    gyre2_clockwise: bool = True
    tide_amp_u: float = 0.0
    tide_amp_v: float = 0.0
    tide_period: float = 180.0
    tide_phase: float = 0.0
    boundary_shear: float = 0.0

    def velocity(self, x: float, y: float, t: float) -> Tuple[float, float]:
        u = self.c0x + self.gx * x
        v = self.c0y + self.gy * y

        omega = 2.0 * pi / max(1e-6, self.tide_period)
        phase = omega * t + self.tide_phase
        u += self.tide_amp_u * sin(phase)
        v += self.tide_amp_v * cos(0.9 * phase)

        def add_gyre(
            cx: float,
            cy: float,
            strength: float,
            sigma: float,
            clockwise: bool,
        ) -> Tuple[float, float]:
            if sigma <= 1e-6 or abs(strength) <= 1e-9:
                return (0.0, 0.0)
            dx = x - cx
            dy = y - cy
            rr = (dx * dx + dy * dy) / (2.0 * sigma * sigma)
            w = strength * exp(-rr)
            orient = -1.0 if clockwise else 1.0
            # Tangential gyre velocity with smooth decay.
            gu = orient * (-dy / sigma) * w
            gv = orient * (dx / sigma) * w
            return (gu, gv)

        gu1, gv1 = add_gyre(
            self.gyre1_cx,
            self.gyre1_cy,
            self.gyre1_strength,
            self.gyre1_sigma,
            self.gyre1_clockwise,
        )
        gu2, gv2 = add_gyre(
            self.gyre2_cx,
            self.gyre2_cy,
            self.gyre2_strength,
            self.gyre2_sigma,
            self.gyre2_clockwise,
        )
        u += gu1 + gu2
        v += gv1 + gv2

        # Mild coastal shear: stronger drift near upper/lower bands.
        y_norm = (y - 210.0) / 210.0
        u += self.boundary_shear * y_norm
        return (u, v)

    def as_dict(self) -> dict:
        return {
            "c0x": self.c0x,
            "c0y": self.c0y,
            "gx": self.gx,
            "gy": self.gy,
            "gyre1_cx": self.gyre1_cx,
            "gyre1_cy": self.gyre1_cy,
            "gyre1_strength": self.gyre1_strength,
            "gyre1_sigma": self.gyre1_sigma,
            "gyre1_clockwise": self.gyre1_clockwise,
            "gyre2_cx": self.gyre2_cx,
            "gyre2_cy": self.gyre2_cy,
            "gyre2_strength": self.gyre2_strength,
            "gyre2_sigma": self.gyre2_sigma,
            "gyre2_clockwise": self.gyre2_clockwise,
            "tide_amp_u": self.tide_amp_u,
            "tide_amp_v": self.tide_amp_v,
            "tide_period": self.tide_period,
            "tide_phase": self.tide_phase,
            "boundary_shear": self.boundary_shear,
        }


@dataclass
class WindField:
    wx: float = 0.0
    wy: float = 0.0

    def velocity(self, x: float, y: float, t: float) -> Tuple[float, float]:
        del x, y, t
        return (self.wx, self.wy)


@dataclass
class Environment:
    xlim: Tuple[float, float]
    ylim: Tuple[float, float]
    obstacles: List[ObstacleRect] = field(default_factory=list)
    current: OceanCurrent = field(default_factory=OceanCurrent)
    wind: WindField = field(default_factory=WindField)

    def in_bounds(self, x: float, y: float) -> bool:
        return self.xlim[0] <= x <= self.xlim[1] and self.ylim[0] <= y <= self.ylim[1]

    def in_obstacle(self, x: float, y: float) -> bool:
        return any(obs.contains(x, y) for obs in self.obstacles)


@dataclass
class SimulationConfig:
    dt: float
    horizon: float
    nx: int
    ny: int
    safe_distance: float
    coverage_threshold: float
    revisit_max_gap: float
    seen_prob_threshold: float = 0.25
    control_penalty: float = 0.03
