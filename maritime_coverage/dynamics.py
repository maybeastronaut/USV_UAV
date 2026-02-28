from __future__ import annotations

from math import atan2, cos, pi, sin, sqrt

from .models import AgentParams, AgentState, AgentType, Control, Environment


def clamp(value: float, vmin: float, vmax: float) -> float:
    return max(vmin, min(value, vmax))


def wrap_to_pi(angle: float) -> float:
    while angle > pi:
        angle -= 2.0 * pi
    while angle < -pi:
        angle += 2.0 * pi
    return angle


def heading_to_target(state: AgentState, tx: float, ty: float) -> float:
    return atan2(ty - state.y, tx - state.x)


def power_draw(
    body_speed: float,
    turn_rate: float,
    params: AgentParams,
    current_speed: float,
) -> float:
    base = params.base_power
    move = params.kv * (body_speed**2)
    turn = params.kw * (turn_rate**2)
    if params.agent_type == AgentType.USV:
        # Current increases hydrodynamic compensation burden.
        move += 0.4 * current_speed
    return base + move + turn


def step_agent(
    state: AgentState,
    control: Control,
    params: AgentParams,
    env: Environment,
    dt: float,
    t: float,
) -> AgentState:
    if params.agent_type == AgentType.UAV:
        # UAV is modeled as a holonomic point mass.
        ux = clamp(control.a, -params.v_max, params.v_max)
        uy = clamp(control.omega, -params.v_max, params.v_max)
        speed = sqrt(ux * ux + uy * uy)
        if speed > params.v_max and speed > 1e-9:
            scale = params.v_max / speed
            ux *= scale
            uy *= scale
            speed = params.v_max

        wx, wy = env.wind.velocity(state.x, state.y, t)
        vx, vy = ux + wx, uy + wy
        new_v = speed
        if speed > 1e-6:
            new_psi = wrap_to_pi(atan2(uy, ux))
        else:
            new_psi = state.psi
        turn_rate = 0.0
        current_speed = 0.0
    else:
        a = clamp(control.a, -params.a_max, params.a_max)
        omega = clamp(control.omega, -params.omega_max, params.omega_max)
        new_v = clamp(state.v + a * dt, params.v_min, params.v_max)
        new_psi = wrap_to_pi(state.psi + omega * dt)
        vbx = new_v * cos(new_psi)
        vby = new_v * sin(new_psi)

        cx, cy = env.current.velocity(state.x, state.y, t)
        vx, vy = vbx + cx, vby + cy
        turn_rate = omega
        current_speed = (cx * cx + cy * cy) ** 0.5

    new_x = state.x + vx * dt
    new_y = state.y + vy * dt

    # Keep agents inside simulation domain for stable coverage estimation.
    new_x = clamp(new_x, env.xlim[0], env.xlim[1])
    new_y = clamp(new_y, env.ylim[0], env.ylim[1])

    used_power = power_draw(body_speed=new_v, turn_rate=turn_rate, params=params, current_speed=current_speed)
    new_energy = max(0.0, state.energy - used_power * dt)

    return AgentState(x=new_x, y=new_y, psi=new_psi, v=new_v, energy=new_energy)
