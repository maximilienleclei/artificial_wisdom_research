from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CartPoleConstants:
    gravity: float = 9.8
    masscart: float = 1.0
    masspole: float = 0.1
    length: float = 0.5
    force_mag: float = 10.0
    tau: float = 0.02
    theta_threshold_radians: float = 12.0 * 2.0 * torch.pi / 360.0
    x_threshold: float = 2.4
    kinematics_integrator: str = "euler"

    @property
    def total_mass(self) -> float:
        return self.masspole + self.masscart

    @property
    def polemass_length(self) -> float:
        return self.masspole * self.length


def step_cartpole(
    state: torch.Tensor,
    action: torch.Tensor,
    constants: CartPoleConstants | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    c = constants or CartPoleConstants()
    x, x_dot, theta, theta_dot = state.unbind(dim=-1)
    force = torch.where(action == 1, c.force_mag, -c.force_mag).to(state.dtype)
    costheta = torch.cos(theta)
    sintheta = torch.sin(theta)

    temp = (force + c.polemass_length * theta_dot.square() * sintheta) / c.total_mass
    thetaacc = (c.gravity * sintheta - costheta * temp) / (
        c.length * (4.0 / 3.0 - c.masspole * costheta.square() / c.total_mass)
    )
    xacc = temp - c.polemass_length * thetaacc * costheta / c.total_mass

    if c.kinematics_integrator == "euler":
        x = x + c.tau * x_dot
        x_dot = x_dot + c.tau * xacc
        theta = theta + c.tau * theta_dot
        theta_dot = theta_dot + c.tau * thetaacc
    else:
        x_dot = x_dot + c.tau * xacc
        x = x + c.tau * x_dot
        theta_dot = theta_dot + c.tau * thetaacc
        theta = theta + c.tau * theta_dot

    next_state = torch.stack((x, x_dot, theta, theta_dot), dim=-1)
    terminated = (
        (x < -c.x_threshold)
        | (x > c.x_threshold)
        | (theta < -c.theta_threshold_radians)
        | (theta > c.theta_threshold_radians)
    )
    reward = torch.ones_like(x, dtype=state.dtype)
    return next_state, reward, terminated
