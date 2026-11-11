"""
Scenario terrain as MuJoCo primitives (static world geoms).

Dimensions follow the plant world (simulation/worlds/plant.sdf,
simulation/config/world_layout.yaml): floor mu 0.9, lying pipes 0.08-0.12 m,
1 cm steel plates with mu 0.15-0.2, corridor stairs with 0.178 m risers on
0.31 m treads, factory stairs 0.10 m.  Pipes are capsules (sphere-capsule
contacts are analytic in MuJoCo; the rounded ends lie outside the walking
lane), steps / stairs / plates / ramps are boxes.

Every terrain geom gets ``priority=1`` so its friction wins over the robot's
(DART uses min(mu_a, mu_b) and every robot shape has mu = 1, see mjcf.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .mjcf import TERRAIN_GROUP, rpy_to_quat, _fmt

FLOOR_MU = 0.9


@dataclass
class Prim:
    """One static primitive.  kind: plane | box | capsule.

    box:     pos = centre, size = full (x, y, z), rpy
    capsule: lying pipe, pos = axis centre, size = (radius, length), yaw = axis direction in the xy plane
    plane:   pos z = height (infinite)"""
    kind: str
    pos: Tuple[float, float, float]
    size: Tuple[float, ...] = ()
    rpy: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mu: float = FLOOR_MU
    name: str = ""
    rgba: str = "0.55 0.55 0.6 1"

    def mjcf(self, idx: int) -> str:
        name = self.name or f"terrain_{idx}"
        common = (f'name="{name}" group="{TERRAIN_GROUP}" contype="1" conaffinity="1" priority="1" '
                  f'friction="{self.mu:g} 0.005 0.0001" rgba="{self.rgba}"')
        if self.kind == "plane":
            return f'    <geom type="plane" pos="0 0 {self.pos[2]:g}" size="0 0 0.05" {common}/>\n'
        if self.kind == "box":
            q = rpy_to_quat(*self.rpy)
            return (f'    <geom type="box" pos="{_fmt(self.pos)}" quat="{_fmt(q)}" '
                    f'size="{_fmt(np.asarray(self.size) / 2.0)}" {common}/>\n')
        if self.kind == "capsule":
            r, length = self.size
            # capsule axis = local z: rotate z onto the horizontal direction yaw
            q = rpy_to_quat(0.0, math.pi / 2.0, self.rpy[2])
            return (f'    <geom type="capsule" pos="{_fmt(self.pos)}" quat="{_fmt(q)}" '
                    f'size="{r:g} {length / 2.0:g}" {common}/>\n')
        raise ValueError(self.kind)


@dataclass
class Terrain:
    name: str
    prims: List[Prim] = field(default_factory=list)
    # region (x0, y0, x1, y1) for the ground-truth height raster
    extent: Tuple[float, float, float, float] = (-3.0, -3.0, 12.0, 3.0)
    friction_regions: List[Tuple[Tuple[float, float, float, float], float]] = field(default_factory=list)

    def mjcf(self) -> str:
        return "".join(p.mjcf(i) for i, p in enumerate(self.prims))

    def mu_at(self, x: float, y: float) -> float:
        for (x0, y0, x1, y1), mu in self.friction_regions:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return mu
        return FLOOR_MU


def floor(mu: float = FLOOR_MU) -> Prim:
    return Prim("plane", (0.0, 0.0, 0.0), mu=mu, name="floor", rgba="0.42 0.44 0.46 1")


def flat() -> Terrain:
    return Terrain("flat", [floor()])


def pipe_and_step(pipe_x: float = 1.5, pipe_d: float = 0.10, step_x: float = 3.2, step_h: float = 0.10,
                  step_len: float = 1.2) -> Terrain:
    """A lying pipe across the path, then a 0.10 m step up (plateau) and down."""
    prims = [floor(),
             Prim("capsule", (pipe_x, 0.0, pipe_d / 2.0), (pipe_d / 2.0, 2.6), (0.0, 0.0, math.pi / 2.0),
                  name="pipe", rgba="0.6 0.62 0.65 1"),
             Prim("box", (step_x + step_len / 2.0, 0.0, step_h / 2.0), (step_len, 2.0, step_h),
                  name="step", rgba="0.5 0.45 0.4 1")]
    return Terrain("pipe_step", prims, extent=(-2.0, -2.5, 7.5, 2.5))


def plate(x0: float = 1.0, length: float = 4.0, mu: float = 0.15, thickness: float = 0.01) -> Terrain:
    """Slippery steel plate (F-OBS9: mu 0.15, 1 cm)."""
    prims = [floor(),
             Prim("box", (x0 + length / 2.0, 0.0, thickness / 2.0), (length, 2.4, thickness), mu=mu,
                  name="plate", rgba="0.7 0.72 0.76 1")]
    return Terrain("plate", prims, extent=(-2.0, -2.5, x0 + length + 3.0, 2.5),
                   friction_regions=[((x0, -1.2, x0 + length, 1.2), mu)])


def stairs(x0: float = 1.5, n: int = 4, rise: float = 0.17, tread: float = 0.31, landing: float = 1.6,
           width: float = 1.6) -> Terrain:
    """Straight staircase up (n steps), landing, and the same staircase down."""
    prims = [floor()]
    for k in range(n):
        h = rise * (k + 1)
        xs = x0 + k * tread
        prims.append(Prim("box", (xs + tread / 2.0, 0.0, h / 2.0), (tread, width, h), name=f"stair_up_{k}",
                          rgba="0.5 0.5 0.55 1"))
    top = rise * n
    xl = x0 + n * tread
    prims.append(Prim("box", (xl + landing / 2.0, 0.0, top / 2.0), (landing, width, top), name="landing",
                      rgba="0.48 0.48 0.52 1"))
    xd = xl + landing
    for k in range(n):
        h = top - rise * (k + 1)
        if h <= 1e-6:
            break
        prims.append(Prim("box", (xd + k * tread + tread / 2.0, 0.0, h / 2.0), (tread, width, h),
                          name=f"stair_dn_{k}", rgba="0.5 0.5 0.55 1"))
    return Terrain("stairs", prims, extent=(-2.0, -2.0, xd + n * tread + 3.0, 2.0))


def ramp(x0: float = 1.5, angle_deg: float = 16.0, length: float = 2.0, top_len: float = 1.5,
         width: float = 1.8) -> Terrain:
    """Ramp up at angle_deg, plateau, ramp down (corridor ramp is 11.8 deg)."""
    a = math.radians(angle_deg)
    h = length * math.sin(a)
    run = length * math.cos(a)
    th = 0.2
    prims = [floor()]
    # up ramp: top face through (x0, 0) .. (x0 + run, h)
    cx = x0 + run / 2.0 + math.sin(a) * th / 2.0
    cz = h / 2.0 - math.cos(a) * th / 2.0
    prims.append(Prim("box", (cx, 0.0, cz), (length, width, th), (0.0, -a, 0.0), name="ramp_up",
                      rgba="0.45 0.47 0.5 1"))
    xt = x0 + run
    prims.append(Prim("box", (xt + top_len / 2.0, 0.0, h / 2.0), (top_len, width, h), name="ramp_top",
                      rgba="0.45 0.47 0.5 1"))
    xd = xt + top_len
    cx = xd + run / 2.0 - math.sin(a) * th / 2.0
    prims.append(Prim("box", (cx, 0.0, cz), (length, width, th), (0.0, a, 0.0), name="ramp_dn",
                      rgba="0.45 0.47 0.5 1"))
    return Terrain("ramp", prims, extent=(-2.0, -2.0, xd + run + 3.0, 2.0))


TERRAINS = {"flat": flat, "pipe_step": pipe_and_step, "plate": plate, "stairs": stairs, "ramp": ramp}
