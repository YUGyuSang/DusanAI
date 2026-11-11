"""
2-D gas dispersion with a mass-consistent, gusty wind field (Module 3, ROS free).

Physical model
--------------
The near-floor layer of the turbine hall + corridor is treated as a 2-D,
depth-averaged domain.  The concentration ``c(x, y, t)`` [ppm] obeys

    dc/dt + div(u c) = div(K grad c) + S(x, y) - k c                (1)

* ``u(x, y, t)``  wind field, divergence free (div u = 0) with zero normal
  flow through walls/obstacles,
* ``K``           turbulent (eddy) diffusivity [m^2/s],
* ``S``           leak emission, a small Gaussian footprint of total rate
                  ``Q`` [ppm m^2/s] around the leak position,
* ``k``           first-order loss [1/s]: vertical entrainment of the floor
                  layer into the 7 m tall hall and roof exhaust (the 2-D model
                  has no vertical dimension; this term carries it).

Wind field (mass-consistent diagnostic model, Sherman 1978 / QUIC style)
----------------------------------------------------------------------
A *first guess* ``u*`` is built from the layout: ``wind_base`` inside the
factory and a channel flow ``(corridor_speed, 0)`` in the corridor.  Inflow
vents (west wall) prescribe the normal velocity, outflow openings (corridor
end, east-wall louvres) are pressure outlets (p = 0).  The final field is
the L2-closest divergence-free field (Helmholtz-Hodge projection)

    u = u* - grad p,   with   lap p = div u*   (Neumann at walls, p=0 at outlets)

solved on a staggered MAC grid with vectorised red-black Gauss-Seidel/SOR.
Because the projection is linear, the projected fields of the base first
guess ``W0`` and of its 90-degree rotation ``W90`` are computed once; a gust
with speed factor ``s(t)`` and direction offset ``theta(t)`` is then the exact
divergence-free field ``s (cos(theta) W0 + sin(theta) W90)``.  Smaller eddies
are added as streamfunction modes ``E_k = curl(psi_k)`` (psi_k = smoothed
seeded noise, zero on walls) with Ornstein-Uhlenbeck amplitudes.  Curl fields
are discretely divergence free and have no flux through walls, so the total
wind stays mass consistent at every instant.

Transport numerics
------------------
Finite volumes on the same MAC grid: first-order upwind advection, central
diffusion (zero flux through closed faces), forward Euler with sub-steps
limited by the per-cell positivity (CFL + diffusion) bound

    dt <= cfl / max_c( sum(outgoing face speeds)/h + n_open K / h^2 )

which makes the scheme monotone (no negative concentrations).  Clean air
enters through the vents; gas leaving through the outlets is removed and
book-kept, so the mass budget closes to round-off (unit tested).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

Rect = Tuple[float, float, float, float]


def rects_from_flat(values: Sequence[float]) -> List[Rect]:
    """[x0, y0, x1, y1, x0, y0, ...] -> list of rectangles (ROS params are flat arrays)."""
    v = [float(x) for x in values]
    if len(v) % 4:
        raise ValueError("rectangle list length must be a multiple of 4")
    return [tuple(v[i:i + 4]) for i in range(0, len(v), 4)]


def _in_rects(X: np.ndarray, Y: np.ndarray, rects: Sequence[Rect]) -> np.ndarray:
    m = np.zeros(X.shape, dtype=bool)
    for x0, y0, x1, y1 in rects:
        m |= (X >= min(x0, x1)) & (X <= max(x0, x1)) & (Y >= min(y0, y1)) & (Y <= max(y0, y1))
    return m


@dataclass
class GasConfig:
    """All tunables of the dispersion model (mirrors config/gas_dispersion.yaml)."""
    resolution: float = 0.25                 # gas grid cell size [m]
    obstacle_fraction: float = 0.15          # occupied sub-sample fraction -> solid gas cell (keeps 0.1 m walls closed)
    source_xy: Tuple[float, float] = (-3.2, 2.0)
    emission_rate: float = 170.0             # Q [ppm m^2 / s]
    source_sigma: float = 0.25               # Gaussian footprint of the leak [m]
    decay_rate: float = 0.012                # k [1/s]
    diffusivity: float = 0.03                # K [m^2/s] (upwind adds ~|u| h / 2 numerically)
    wind_base: Tuple[float, float] = (0.40, -0.10)
    corridor_speed: float = 0.20             # first-guess channel flow in the corridor [m/s]
    factory_x_max: float = 14.9              # x <= : factory first guess, x > : corridor first guess
    # openings: every wall face (fluid/solid interface) whose centre lies in a
    # rectangle [x0, y0, x1, y1] becomes an inflow vent / pressure outlet
    inflow_rects: List[Rect] = field(default_factory=lambda: [
        (-10.4, -8.9, -9.2, -1.0),           # west wall vents south of the gas tank
        (-10.4, 8.45, -9.2, 8.95),           # west wall vent north of the gas tank
        (-3.5, 8.7, 14.8, 9.3)])             # north wall vents (base wind has v < 0)
    outflow_rects: List[Rect] = field(default_factory=lambda: [
        (63.3, -1.95, 63.95, 1.95),          # corridor end
        (14.6, -9.2, 15.2, -3.4),            # east wall louvres (south of the doorway)
        (14.6, 3.4, 15.2, 9.0),              # east wall louvres (north of the doorway)
        (-10.0, -9.4, 14.85, -8.8)])         # south wall louvres
    cfl: float = 0.9
    max_substeps: int = 40
    projection_tol: float = 1e-6             # max |div u| [1/s] after projection
    projection_max_iter: int = 20000
    sor_omega: float = 0.0                   # 0 -> automatic
    gusts_enabled: bool = True
    gust_speed_amp: float = 0.15             # sinusoidal speed modulation (fraction)
    gust_speed_period: float = 47.0          # [s]
    gust_dir_amp_deg: float = 8.0            # sinusoidal direction meander [deg]
    gust_dir_period: float = 83.0            # [s]
    gust_noise_speed: float = 0.08           # OU noise std of the speed factor
    gust_noise_dir_deg: float = 5.0          # OU noise std of the direction [deg]
    gust_tau: float = 8.0                    # OU time constant [s]
    eddy_rms: float = 0.06                   # RMS speed of the eddy modes [m/s]
    eddy_scale: float = 2.5                  # spatial scale of the eddies [m]
    eddy_tau: float = 15.0                   # OU time constant of eddy amplitudes [s]
    eddy_modes: int = 6
    eddy_wall_taper: float = 0.75            # eddies fade to zero over this distance from walls [m]
    seed: int = 7


# =============================================================================
# Gusts
# =============================================================================
class GustProcess:
    """Speed factor s(t), direction offset theta(t) and eddy amplitudes a_k(t).

    s(t)     = 1 + A_s sin(2 pi t / T_s + phi_s) + n_s(t)
    theta(t) = A_d sin(2 pi t / T_d + phi_d) + n_d(t)
    n, a_k   = Ornstein-Uhlenbeck (exactly discretised filtered white noise)
    """

    def __init__(self, cfg: GasConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.t = 0.0
        self.phi_s = float(rng.uniform(0, 2 * math.pi))
        self.phi_d = float(rng.uniform(0, 2 * math.pi))
        self.n_s = float(rng.normal() * cfg.gust_noise_speed)
        self.n_d = float(rng.normal() * math.radians(cfg.gust_noise_dir_deg))
        self.a = rng.normal(size=max(0, int(cfg.eddy_modes)))

    @staticmethod
    def _ou(x, sigma, tau, dt, rng, size=None):
        f = math.exp(-dt / max(tau, 1e-6))
        return x * f + sigma * math.sqrt(max(0.0, 1.0 - f * f)) * rng.normal(size=size)

    def advance(self, dt: float) -> None:
        c = self.cfg
        self.t += dt
        self.n_s = float(self._ou(self.n_s, c.gust_noise_speed, c.gust_tau, dt, self.rng))
        self.n_d = float(self._ou(self.n_d, math.radians(c.gust_noise_dir_deg), c.gust_tau, dt, self.rng))
        if self.a.size:
            self.a = self._ou(self.a, 1.0, c.eddy_tau, dt, self.rng, size=self.a.size)

    @property
    def speed_factor(self) -> float:
        if not self.cfg.gusts_enabled:
            return 1.0
        c = self.cfg
        s = 1.0 + c.gust_speed_amp * math.sin(2 * math.pi * self.t / max(c.gust_speed_period, 1e-6) + self.phi_s)
        return float(min(2.5, max(0.2, s + self.n_s)))

    @property
    def angle(self) -> float:
        if not self.cfg.gusts_enabled:
            return 0.0
        c = self.cfg
        return float(math.radians(c.gust_dir_amp_deg) *
                     math.sin(2 * math.pi * self.t / max(c.gust_dir_period, 1e-6) + self.phi_d) + self.n_d)

    @property
    def eddy_weights(self) -> np.ndarray:
        if not self.cfg.gusts_enabled:
            return np.zeros_like(self.a)
        return self.a


# =============================================================================
# Wind field
# =============================================================================
class WindField:
    """Mass-consistent wind on a staggered (MAC) grid.

    ``u`` lives on vertical faces, shape (ny, nx+1): u[j, i] is the velocity
    between cells (j, i-1) and (j, i) at x = x0 + i h, y = y0 + (j+1/2) h.
    ``v`` lives on horizontal faces, shape (ny+1, nx): v[j, i] between cells
    (j-1, i) and (j, i) at x = x0 + (i+1/2) h, y = y0 + j h.
    """

    def __init__(self, solid: np.ndarray, resolution: float, origin: Tuple[float, float],
                 cfg: GasConfig, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.solid = np.asarray(solid, dtype=bool)
        self.fluid = ~self.solid
        self.h = float(resolution)
        self.x0, self.y0 = float(origin[0]), float(origin[1])
        ny, nx = self.solid.shape
        self.shape = (ny, nx)
        h = self.h
        self.xu = self.x0 + h * np.arange(nx + 1)
        self.yu = self.y0 + h * (np.arange(ny) + 0.5)
        self.xv = self.x0 + h * (np.arange(nx) + 0.5)
        self.yv = self.y0 + h * np.arange(ny + 1)

        fp = np.pad(self.fluid, 1, constant_values=False)
        left, right = fp[1:-1, :-1], fp[1:-1, 1:]           # (ny, nx+1)
        below, above = fp[:-1, 1:-1], fp[1:, 1:-1]          # (ny+1, nx)
        self.open_u = left & right
        self.open_v = below & above
        bnd_u = left ^ right
        bnd_v = below ^ above
        XU, YU = np.meshgrid(self.xu, self.yu)
        XV, YV = np.meshgrid(self.xv, self.yv)
        # Openings are wall faces inside the configured rectangles, filtered by
        # orientation against the base first guess: a vent must blow *into*
        # the fluid, an outlet must face downstream.  This keeps faces of
        # objects that happen to lie inside a rectangle (e.g. a machine next
        # to the wall) from becoming spurious sources / sinks.
        wx, wy = cfg.wind_base
        fg_u = np.where(XU <= cfg.factory_x_max, wx, cfg.corridor_speed)
        fg_v = np.where(XV <= cfg.factory_x_max, wy, 0.0)
        in_dir_u = (right & ~left & (fg_u > 0)) | (left & ~right & (fg_u < 0))
        in_dir_v = (above & ~below & (fg_v > 0)) | (below & ~above & (fg_v < 0))
        out_dir_u = (left & ~right & (fg_u >= 0)) | (right & ~left & (fg_u <= 0))
        out_dir_v = (below & ~above & (fg_v >= 0)) | (above & ~below & (fg_v <= 0))
        self.in_u = bnd_u & _in_rects(XU, YU, cfg.inflow_rects) & in_dir_u
        self.in_v = bnd_v & _in_rects(XV, YV, cfg.inflow_rects) & in_dir_v
        self.out_u = bnd_u & _in_rects(XU, YU, cfg.outflow_rects) & out_dir_u & ~self.in_u
        self.out_v = bnd_v & _in_rects(XV, YV, cfg.outflow_rects) & out_dir_v & ~self.in_v
        # faces whose velocity the pressure correction may change
        self.act_u = self.open_u | self.out_u
        self.act_v = self.open_v | self.out_v
        # faces that may carry flow at all
        self.flow_u = self.act_u | self.in_u
        self.flow_v = self.act_v | self.in_v
        self._XU, self._YU, self._XV, self._YV = XU, YU, XV, YV

        # ----- pressure operator: per-cell active-face masks (float) -----
        self.aE = self.act_u[:, 1:].astype(float) * self.fluid
        self.aW = self.act_u[:, :-1].astype(float) * self.fluid
        self.aN = self.act_v[1:, :].astype(float) * self.fluid
        self.aS = self.act_v[:-1, :].astype(float) * self.fluid
        n = self.aE + self.aW + self.aN + self.aS
        self.n_active = n
        self.solve_mask = (n > 0) & self.fluid
        self.inv_n = np.where(self.solve_mask, 1.0 / np.maximum(n, 1.0), 0.0)
        J, I = np.mgrid[0:ny, 0:nx]
        red = ((I + J) % 2 == 0) & self.solve_mask
        self._colors = (red.astype(float), (~red & self.solve_mask).astype(float))

        self._close_unvented_pockets()

        # ----- first guesses and projected basis fields -----
        self.last_projection_iters = 0
        self.last_projection_residual = 0.0
        u0, v0 = self.first_guess(rotate90=False)
        self.W0 = self.project(u0, v0)
        u1, v1 = self.first_guess(rotate90=True)
        self.W90 = self.project(u1, v1)
        self.eddies = self._make_eddies(rng or np.random.default_rng(cfg.seed))
        self.u, self.v = self.W0[0].copy(), self.W0[1].copy()

    # ------------------------------------------------------------ geometry
    def _close_unvented_pockets(self) -> None:
        """Fluid regions without any outlet cannot accept inflow (the Poisson
        problem would be inconsistent): close their vents."""
        try:
            import cv2
            n_lab, labels = cv2.connectedComponents(self.fluid.astype(np.uint8), connectivity=4)
        except Exception:  # pragma: no cover - OpenCV is a runtime dependency
            return
        ny, nx = self.shape
        lab_pad = np.pad(labels, 1, constant_values=0)
        # label of the fluid cell adjacent to each interface face (max of the two sides)
        lab_u = np.maximum(lab_pad[1:-1, :-1], lab_pad[1:-1, 1:])
        lab_v = np.maximum(lab_pad[:-1, 1:-1], lab_pad[1:, 1:-1])
        vented = set(np.unique(lab_u[self.out_u]).tolist()) | set(np.unique(lab_v[self.out_v]).tolist())
        for lab in set(np.unique(lab_u[self.in_u]).tolist()) | set(np.unique(lab_v[self.in_v]).tolist()):
            if lab not in vented:
                self.in_u &= lab_u != lab
                self.in_v &= lab_v != lab
        self.flow_u = self.act_u | self.in_u
        self.flow_v = self.act_v | self.in_v

    def first_guess(self, rotate90: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Face velocities of the first guess (factory wind / corridor channel)."""
        wx, wy = self.cfg.wind_base
        if rotate90:
            wx, wy = -wy, wx          # +90 deg rotation; corridor channel flow not rotated
            cu, cv = 0.0, 0.0
        else:
            cu, cv = self.cfg.corridor_speed, 0.0
        fac_u = self._XU <= self.cfg.factory_x_max
        fac_v = self._XV <= self.cfg.factory_x_max
        u = np.where(fac_u, wx, cu) * self.flow_u
        v = np.where(fac_v, wy, cv) * self.flow_v
        return u.astype(float), v.astype(float)

    # ---------------------------------------------------------- projection
    def divergence(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Cell divergence [1/s] (zero in solid cells)."""
        d = (u[:, 1:] - u[:, :-1] + v[1:, :] - v[:-1, :]) / self.h
        return d * self.fluid

    def project(self, u_star: np.ndarray, v_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Helmholtz-Hodge projection: closest divergence-free field.

        Solves  sum_active(p_nb - p_c) = h^2 div(u*)  per fluid cell (outlet
        ghosts have p = 0, walls are Neumann) with red-black SOR, then
        u = u* - grad p on active faces.
        """
        cfg = self.cfg
        h = self.h
        u = np.where(self.flow_u, u_star, 0.0)
        v = np.where(self.flow_v, v_star, 0.0)
        rhs = (h * h) * self.divergence(u, v)
        ny, nx = self.shape
        P = np.zeros((ny + 2, nx + 2))
        p = P[1:-1, 1:-1]                       # view into the padded array
        omega = cfg.sor_omega
        if omega <= 0.0:
            n_eff = 2.0 * max(ny, nx)
            omega = 2.0 / (1.0 + math.sin(math.pi / n_eff))
        aE, aW, aN, aS, inv_n = self.aE, self.aW, self.aN, self.aS, self.inv_n
        tol = cfg.projection_tol * h * h
        it = 0
        res = float("inf")
        while it < cfg.projection_max_iter:
            for col in self._colors:
                s = aE * P[1:-1, 2:] + aW * P[1:-1, :-2] + aN * P[2:, 1:-1] + aS * P[:-2, 1:-1]
                p += (omega * col) * ((s - rhs) * inv_n - p)
            it += 1
            if it % 20 == 0:
                s = aE * P[1:-1, 2:] + aW * P[1:-1, :-2] + aN * P[2:, 1:-1] + aS * P[:-2, 1:-1]
                r = (s - self.n_active * p - rhs) * self.solve_mask
                res = float(np.max(np.abs(r)))
                if res < tol:
                    break
        self.last_projection_iters = it
        self.last_projection_residual = res / (h * h)
        # u = u* - grad p (p = 0 in solid cells / ghosts)
        u = u - self.act_u * (P[1:-1, 1:] - P[1:-1, :-1]) / h
        v = v - self.act_v * (P[1:, 1:-1] - P[:-1, 1:-1]) / h
        return u, v

    # ------------------------------------------------------------- eddies
    def _make_eddies(self, rng: np.random.Generator) -> List[Tuple[np.ndarray, np.ndarray]]:
        cfg = self.cfg
        k_modes = max(0, int(cfg.eddy_modes))
        if k_modes == 0 or cfg.eddy_rms <= 0.0:
            return []
        ny, nx = self.shape
        h = self.h
        # corner grid (ny+1, nx+1): a corner touching any solid cell is a wall corner
        sp = np.pad(self.solid, 1, constant_values=True)
        wall = sp[:-1, :-1] | sp[:-1, 1:] | sp[1:, :-1] | sp[1:, 1:]
        # distance to the nearest wall corner (only needed up to the taper length)
        taper = max(cfg.eddy_wall_taper, h)
        n_it = int(math.ceil(taper / h)) + 1
        d = np.where(wall, 0.0, np.inf)
        for _ in range(n_it):
            dp = np.pad(d, 1, constant_values=np.inf)
            cand = [dp[1:-1, :-2] + h, dp[1:-1, 2:] + h, dp[:-2, 1:-1] + h, dp[2:, 1:-1] + h,
                    dp[:-2, :-2] + h * math.sqrt(2), dp[:-2, 2:] + h * math.sqrt(2),
                    dp[2:, :-2] + h * math.sqrt(2), dp[2:, 2:] + h * math.sqrt(2)]
            d = np.minimum(d, np.minimum.reduce(cand))
        taper_w = np.clip(d / taper, 0.0, 1.0)
        # smooth noise via FFT Gaussian filter (sigma = eddy_scale)
        sig = cfg.eddy_scale / h
        ky = np.fft.fftfreq(ny + 1)[:, None]
        kx = np.fft.fftfreq(nx + 1)[None, :]
        gauss = np.exp(-2.0 * (math.pi * sig) ** 2 * (kx ** 2 + ky ** 2))
        modes = []
        fl = self.fluid
        for _ in range(k_modes):
            noise = rng.normal(size=(ny + 1, nx + 1))
            psi = np.real(np.fft.ifft2(np.fft.fft2(noise) * gauss)) * taper_w
            psi[wall] = 0.0
            u = (psi[1:, :] - psi[:-1, :]) / h              # (ny, nx+1)
            v = -(psi[:, 1:] - psi[:, :-1]) / h             # (ny+1, nx)
            uc = 0.5 * (u[:, 1:] + u[:, :-1])
            vc = 0.5 * (v[1:, :] + v[:-1, :])
            rms = math.sqrt(float(np.mean((uc ** 2 + vc ** 2)[fl]))) if fl.any() else 1.0
            scale = cfg.eddy_rms / math.sqrt(k_modes) / max(rms, 1e-12)
            modes.append((u * scale, v * scale))
        return modes

    # --------------------------------------------------------------- state
    def update(self, gust: GustProcess) -> None:
        """Instantaneous field for the current gust state."""
        s, th = gust.speed_factor, gust.angle
        c0, c1 = s * math.cos(th), s * math.sin(th)
        self.u = c0 * self.W0[0] + c1 * self.W90[0]
        self.v = c0 * self.W0[1] + c1 * self.W90[1]
        for a, (eu, ev) in zip(gust.eddy_weights, self.eddies):
            self.u += a * eu
            self.v += a * ev

    def cell_centred(self) -> Tuple[np.ndarray, np.ndarray]:
        """Wind interpolated to cell centres (for plots / OccupancyGrid quivers)."""
        return 0.5 * (self.u[:, 1:] + self.u[:, :-1]), 0.5 * (self.v[1:, :] + self.v[:-1, :])

    def at(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        """Bilinear wind at world points (vectorised)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        return (_bilinear(self.u, (x - self.x0) / self.h, (y - self.y0) / self.h - 0.5),
                _bilinear(self.v, (x - self.x0) / self.h - 0.5, (y - self.y0) / self.h))


def _bilinear(field: np.ndarray, fx, fy):
    """Bilinear sample of ``field`` at fractional indices (fx along columns)."""
    ny, nx = field.shape
    fx = np.clip(fx, 0.0, nx - 1.000001)
    fy = np.clip(fy, 0.0, ny - 1.000001)
    i0 = np.floor(fx).astype(int)
    j0 = np.floor(fy).astype(int)
    tx = fx - i0
    ty = fy - j0
    i1 = np.minimum(i0 + 1, nx - 1)
    j1 = np.minimum(j0 + 1, ny - 1)
    return ((1 - tx) * (1 - ty) * field[j0, i0] + tx * (1 - ty) * field[j0, i1]
            + (1 - tx) * ty * field[j1, i0] + tx * ty * field[j1, i1])


# =============================================================================
# Dispersion
# =============================================================================
def solid_mask_from_map(grid, resolution: float, obstacle_fraction: float = 0.5,
                        samples: int = 5) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Downsample a nav OccupancyGridMap (any resolution) to a gas-grid solid
    mask.  A gas cell is solid when the occupied/unknown fraction of
    ``samples x samples`` sub-points exceeds ``obstacle_fraction``."""
    w_m = grid.width * grid.resolution
    h_m = grid.height * grid.resolution
    nx = max(1, int(round(w_m / resolution)))
    ny = max(1, int(round(h_m / resolution)))
    obst = grid.obstacle_mask(unknown_is_obstacle=True)
    sub = (np.arange(samples) + 0.5) / samples * resolution
    frac = np.zeros((ny, nx))
    xs0 = grid.origin_x + np.arange(nx) * resolution
    ys0 = grid.origin_y + np.arange(ny) * resolution
    for oy in sub:
        for ox in sub:
            X, Y = np.meshgrid(xs0 + ox, ys0 + oy)
            ix = np.floor((X - grid.origin_x) / grid.resolution).astype(int)
            iy = np.floor((Y - grid.origin_y) / grid.resolution).astype(int)
            ok = (ix >= 0) & (ix < grid.width) & (iy >= 0) & (iy < grid.height)
            o = np.ones(X.shape, dtype=bool)
            o[ok] = obst[iy[ok], ix[ok]]
            frac += o
    frac /= samples * samples
    return frac > obstacle_fraction, (grid.origin_x, grid.origin_y)


class GasDispersion2D:
    """Advection-diffusion-decay of the leak on the gas grid.

    ``c[j, i]`` is the concentration [ppm] of the cell centred at
    (x0 + (i+1/2) h, y0 + (j+1/2) h); row 0 is at y = y0 (OccupancyGrid order).
    """

    def __init__(self, solid: np.ndarray, resolution: float, origin: Tuple[float, float],
                 cfg: Optional[GasConfig] = None):
        self.cfg = cfg or GasConfig()
        self.h = float(resolution)
        self.x0, self.y0 = float(origin[0]), float(origin[1])
        self.rng = np.random.default_rng(self.cfg.seed)
        self.wind = WindField(solid, self.h, (self.x0, self.y0), self.cfg, rng=self.rng)
        self.gust = GustProcess(self.cfg, self.rng)
        self.solid = self.wind.solid
        self.fluid = self.wind.fluid
        self.fluid_f = self.fluid.astype(float)
        ny, nx = self.solid.shape
        self.shape = (ny, nx)
        self.c = np.zeros((ny, nx))
        self._cp = np.zeros((ny + 2, nx + 2))
        self.t = 0.0
        self.k_open_u = self.cfg.diffusivity / self.h * self.wind.open_u
        self.k_open_v = self.cfg.diffusivity / self.h * self.wind.open_v
        self.n_open = (self.wind.open_u[:, 1:].astype(float) + self.wind.open_u[:, :-1]
                       + self.wind.open_v[1:, :] + self.wind.open_v[:-1, :])
        self.source_weights = np.zeros((ny, nx))
        self.set_source(self.cfg.source_xy, self.cfg.emission_rate)
        # mass budget [ppm m^2]
        self.emitted = 0.0
        self.outflow = 0.0
        self.decayed = 0.0
        self.last_substeps = 0
        self.wind.update(self.gust)

    # ------------------------------------------------------------- builders
    @classmethod
    def from_nav_map(cls, grid, cfg: Optional[GasConfig] = None) -> "GasDispersion2D":
        cfg = cfg or GasConfig()
        solid, origin = solid_mask_from_map(grid, cfg.resolution, cfg.obstacle_fraction)
        return cls(solid, cfg.resolution, origin, cfg)

    # ------------------------------------------------------------- geometry
    def cell_centres(self) -> Tuple[np.ndarray, np.ndarray]:
        ny, nx = self.shape
        xs = self.x0 + (np.arange(nx) + 0.5) * self.h
        ys = self.y0 + (np.arange(ny) + 0.5) * self.h
        return np.meshgrid(xs, ys)

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        ny, nx = self.shape
        return (self.x0, self.x0 + nx * self.h, self.y0, self.y0 + ny * self.h)

    def set_source(self, xy: Tuple[float, float], rate: Optional[float] = None) -> None:
        """Gaussian emission footprint (normalised over fluid cells)."""
        if rate is not None:
            self.cfg.emission_rate = float(rate)
        self.cfg.source_xy = (float(xy[0]), float(xy[1]))
        X, Y = self.cell_centres()
        sig = max(self.cfg.source_sigma, 0.5 * self.h)
        w = np.exp(-((X - xy[0]) ** 2 + (Y - xy[1]) ** 2) / (2 * sig * sig))
        w[(X - xy[0]) ** 2 + (Y - xy[1]) ** 2 > (3.5 * sig) ** 2] = 0.0
        w *= self.fluid
        if w.sum() <= 0.0:   # leak inside an obstacle cell: use the nearest fluid cell
            d = np.where(self.fluid, (X - xy[0]) ** 2 + (Y - xy[1]) ** 2, np.inf)
            j, i = np.unravel_index(int(np.argmin(d)), d.shape)
            w[j, i] = 1.0
        self.source_weights = w / w.sum()

    @property
    def emission_field(self) -> np.ndarray:
        """Source term S [ppm/s] per cell."""
        return self.cfg.emission_rate * self.source_weights / (self.h * self.h)

    # ------------------------------------------------------------ stepping
    def stable_dt(self) -> float:
        """Largest positivity-preserving explicit time step for the current wind."""
        u, v = self.wind.u, self.wind.v
        out = (np.maximum(u[:, 1:], 0.0) + np.maximum(-u[:, :-1], 0.0)
               + np.maximum(v[1:, :], 0.0) + np.maximum(-v[:-1, :], 0.0))
        rate = (out / self.h + self.n_open * self.cfg.diffusivity / (self.h * self.h)) * self.fluid
        m = float(np.max(rate)) if rate.size else 0.0
        return self.cfg.cfl / max(m, 1e-9)

    def step(self, dt: float, advance_gusts: bool = True) -> int:
        """Advance the model by ``dt`` seconds (sub-stepped).  Returns the
        number of sub-steps.  If more than ``max_substeps`` would be needed the
        simulated interval is truncated (stability first)."""
        if dt <= 0.0:
            return 0
        if advance_gusts:
            self.gust.advance(dt)
        self.wind.update(self.gust)
        dt_lim = self.stable_dt()
        n = int(math.ceil(dt / dt_lim - 1e-9))
        if n > self.cfg.max_substeps:
            n = self.cfg.max_substeps
            dt = n * dt_lim
        n = max(n, 1)
        dts = dt / n
        S = self.emission_field
        for _ in range(n):
            self._substep(dts, S)
        self.t += dt
        self.last_substeps = n
        return n

    def _substep(self, dt: float, S: np.ndarray) -> None:
        u, v = self.wind.u, self.wind.v
        cp = self._cp
        cp[1:-1, 1:-1] = self.c
        cl, cr = cp[1:-1, :-1], cp[1:-1, 1:]
        cb, ca = cp[:-1, 1:-1], cp[1:, 1:-1]
        # upwind advective flux + diffusive flux (closed faces: u = 0, k_open = 0)
        fx = np.maximum(u, 0.0) * cl + np.minimum(u, 0.0) * cr - self.k_open_u * (cr - cl)
        fy = np.maximum(v, 0.0) * cb + np.minimum(v, 0.0) * ca - self.k_open_v * (ca - cb)
        c = self.c - (dt / self.h) * (fx[:, 1:] - fx[:, :-1] + fy[1:, :] - fy[:-1, :])
        if self.cfg.emission_rate != 0.0:
            c += dt * S
            self.emitted += dt * self.cfg.emission_rate
        h2 = self.h * self.h
        # gas pushed into solid cells left the domain through an opening
        self.outflow += float(np.sum(c[self.solid])) * h2
        c *= self.fluid_f
        if self.cfg.decay_rate > 0.0:
            f = math.exp(-self.cfg.decay_rate * dt)
            self.decayed += float(np.sum(c)) * (1.0 - f) * h2
            c *= f
        np.maximum(c, 0.0, out=c)   # guards round-off only (scheme is monotone)
        self.c = c

    def spin_up(self, duration: float, dt: float = 0.5, gusts: bool = False,
                tol: float = 0.0) -> float:
        """Run the model for ``duration`` seconds (frozen base wind unless
        ``gusts``).  Stops early when the max relative change per second drops
        below ``tol`` (> 0).  Returns the simulated time."""
        t = 0.0
        saved = self.cfg.gusts_enabled
        if not gusts:
            self.cfg.gusts_enabled = False
        try:
            while t < duration - 1e-9:
                d = min(dt, duration - t)
                prev = self.c.copy() if tol > 0.0 else None
                self.step(d, advance_gusts=gusts)
                t += d
                if prev is not None:
                    peak = max(float(self.c.max()), 1e-9)
                    if float(np.max(np.abs(self.c - prev))) / peak / d < tol:
                        break
        finally:
            self.cfg.gusts_enabled = saved
            self.wind.update(self.gust)
        return t

    # ------------------------------------------------------------- queries
    @property
    def total_mass(self) -> float:
        return float(np.sum(self.c)) * self.h * self.h

    def concentration_at(self, x, y):
        """Bilinear concentration [ppm] at world points; solid neighbours are
        excluded from the stencil (no artificial dilution near walls)."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        ny, nx = self.shape
        fx = (x - self.x0) / self.h - 0.5
        fy = (y - self.y0) / self.h - 0.5
        i0 = np.floor(fx).astype(int)
        j0 = np.floor(fy).astype(int)
        tx = fx - i0
        ty = fy - j0
        num = np.zeros(np.broadcast(x, y).shape)
        den = np.zeros_like(num)
        for dj, di, w in ((0, 0, (1 - tx) * (1 - ty)), (0, 1, tx * (1 - ty)),
                          (1, 0, (1 - tx) * ty), (1, 1, tx * ty)):
            jj = j0 + dj
            ii = i0 + di
            ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
            jc = np.clip(jj, 0, ny - 1)
            ic = np.clip(ii, 0, nx - 1)
            wf = w * ok * self.fluid_f[jc, ic]
            num += wf * self.c[jc, ic]
            den += wf
        out = np.where(den > 1e-9, num / np.maximum(den, 1e-12), 0.0)
        return float(out) if out.ndim == 0 else out

    def wind_at(self, x, y):
        u, v = self.wind.at(x, y)
        if np.ndim(u) == 0:
            return float(u), float(v)
        return u, v

    def body_points(self, x: float, y: float, yaw: float, offsets) -> np.ndarray:
        """World positions of body-frame offsets [(dx, dy), ...]."""
        off = np.asarray(offsets, dtype=float).reshape(-1, 2)
        c, s = math.cos(yaw), math.sin(yaw)
        return np.stack([x + c * off[:, 0] - s * off[:, 1], y + s * off[:, 0] + c * off[:, 1]], -1)

    def mean_wind(self) -> Tuple[float, float]:
        """Gust-modulated base wind vector (what the vents blow)."""
        s, th = self.gust.speed_factor, self.gust.angle
        wx, wy = self.cfg.wind_base
        c, sn = math.cos(th), math.sin(th)
        return s * (c * wx - sn * wy), s * (sn * wx + c * wy)

    def info(self) -> dict:
        j, i = np.unravel_index(int(np.argmax(self.c)), self.c.shape)
        X, Y = self.cell_centres()
        return {
            "t": self.t,
            "max_ppm": float(self.c[j, i]),
            "max_xy": [float(X[j, i]), float(Y[j, i])],
            "mass": self.total_mass,
            "emitted": self.emitted,
            "outflow": self.outflow,
            "decayed": self.decayed,
            "substeps": self.last_substeps,
            "gust_speed_factor": self.gust.speed_factor,
            "gust_angle_deg": math.degrees(self.gust.angle),
        }


# =============================================================================
# Sensors
# =============================================================================
class LaggedSensor:
    """First-order lag + Gaussian noise, vectorised over N channels.

        y_lag(t+dt) = y_lag + (1 - exp(-dt/tau)) (c_true - y_lag)     (exact ZOH)
        reading     = max(0, y_lag + n_abs + n_rel * y_lag)            (+ quantisation)
    """

    def __init__(self, n: int = 1, tau: float = 1.0, noise_abs: float = 0.3,
                 noise_rel: float = 0.03, quantum: float = 0.0,
                 rng: Optional[np.random.Generator] = None, floor: Optional[float] = 0.0):
        self.n = int(n)
        self.tau = float(tau)
        self.noise_abs = float(noise_abs)
        self.noise_rel = float(noise_rel)
        self.quantum = float(quantum)
        self.floor = floor
        self.rng = rng or np.random.default_rng()
        self.state: Optional[np.ndarray] = None

    def reset(self, value=None) -> None:
        self.state = None if value is None else np.full(self.n, value, dtype=float) * 1.0

    def update(self, true_values, dt: float) -> np.ndarray:
        x = np.asarray(true_values, dtype=float).reshape(self.n)
        if self.state is None:
            self.state = x.copy()
        elif self.tau > 1e-6:
            self.state += (1.0 - math.exp(-max(dt, 0.0) / self.tau)) * (x - self.state)
        else:
            self.state = x.copy()
        y = self.state + self.noise_abs * self.rng.normal(size=self.n) \
            + self.noise_rel * np.abs(self.state) * self.rng.normal(size=self.n)
        if self.quantum > 0.0:
            y = np.round(y / self.quantum) * self.quantum
        if self.floor is not None:
            y = np.maximum(y, self.floor)
        return y
