"""
Dense ADMM quadratic-program solver (OSQP algorithm, numpy only).

Solves

    minimise    1/2 x' P x + q' x
    subject to  l <= A x <= u              (l_i = -inf / u_i = +inf allowed,
                                            l_i == u_i = equality constraint)

with the operator-splitting iteration of Stellato et al., "OSQP: an operator
splitting solver for quadratic programs" (Math. Prog. Comp. 2020):

    x~(k+1) = (P + sigma I + A' diag(rho) A)^-1 (sigma x(k) - q + A'(rho z(k) - y(k)))
    z~(k+1) = A x~(k+1)
    x(k+1)  = alpha x~(k+1) + (1 - alpha) x(k)                    (over-relaxation)
    z(k+1)  = Pi_[l,u]( alpha z~(k+1) + (1 - alpha) z(k) + y(k)/rho )
    y(k+1)  = y(k) + rho (alpha z~(k+1) + (1 - alpha) z(k) - z(k+1))

Features: modified Ruiz equilibration + cost scaling, per-constraint rho
(equalities 1e3 rho, free rows rho_min), residual-balancing rho adaptation,
over-relaxation, warm start (primal and dual), residual-based termination
(absolute + relative tolerances on the unscaled problem), primal / dual
infeasibility certificates, an optional wall-clock limit (for real-time use)
and OSQP's *solution polishing*: the active set guessed from the ADMM iterate
is used to solve the reduced equality-constrained KKT system

    [ P + dI   A_act' ] [x]   [ -q    ]
    [ A_act    -dI    ] [y] = [ b_act ]

(Schur complement + iterative refinement on the unregularised system),
followed by a few primal-dual active-set corrections; the polished point is
accepted only if it is primal feasible, has correctly signed multipliers and
does not increase the residuals.  Polishing turns a cheap
moderate-accuracy ADMM solution into a high-accuracy one, which matters for
the condensed MPC whose attitude prediction is very sensitive to the forces.

The ADMM matrix P + sigma I + A' diag(rho) A is dense and is inverted once per
rho value, so every iteration costs only a few matrix-vector products (ideal for
the condensed MPC problem with a few hundred variables).  When A is sparse (the
MPC friction-pyramid matrix is block diagonal, > 99 % zeros) the equilibration
and the product A' diag(rho) A are evaluated on the nonzeros only, which gives
the same numbers at a fraction of the cost.

:func:`limit_blas_threads` caps the BLAS thread pool at run time.  For the
small dense matrices of this solver a multi-threaded OpenBLAS is slower than a
single thread, and on a loaded machine an oversubscribed pool is 10-100x
slower (environment variables only work before numpy is imported).
"""
from __future__ import annotations

import ctypes
import glob
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

INF = np.inf
RHO_MIN = 1e-6
RHO_MAX = 1e6
RHO_EQ_SCALE = 1e3

STATUS_SOLVED = "solved"
STATUS_INACCURATE = "solved_inaccurate"
STATUS_MAX_ITER = "max_iter_reached"
STATUS_TIME_LIMIT = "time_limit_reached"
STATUS_PRIMAL_INFEASIBLE = "primal_infeasible"
STATUS_DUAL_INFEASIBLE = "dual_infeasible"
STATUS_INVALID = "invalid_problem"

OK_STATUSES = (STATUS_SOLVED, STATUS_INACCURATE)


# ---------------------------------------------------------------------------
# BLAS thread-pool control (best effort, no third-party dependency)
# ---------------------------------------------------------------------------
_BLAS_SET_NAMES = ("scipy_openblas_set_num_threads64_", "scipy_openblas_set_num_threads",
                   "openblas_set_num_threads64_", "openblas_set_num_threads", "MKL_Set_Num_Threads")
_BLAS_GET_NAMES = ("scipy_openblas_get_num_threads64_", "scipy_openblas_get_num_threads",
                   "openblas_get_num_threads64_", "openblas_get_num_threads", "MKL_Get_Max_Threads")
_blas_handles: Optional[List[Tuple[object, object]]] = None


def _blas_libraries() -> List[Tuple[object, object]]:
    """(set_num_threads, get_num_threads) ctypes functions of the BLAS
    libraries numpy has loaded: the OpenBLAS bundled with the pip wheels
    (``numpy.libs`` / ``numpy/.libs`` / ``numpy/.dylibs``) and, on Linux, any
    OpenBLAS / MKL mapped into the process (Debian/Ubuntu ``python3-numpy``
    links the system ``libblas.so.3``, which is OpenBLAS when installed)."""
    global _blas_handles
    if _blas_handles is not None:
        return _blas_handles
    paths: List[str] = []
    npdir = os.path.dirname(np.__file__)
    for d in (os.path.join(os.path.dirname(npdir), "numpy.libs"), os.path.join(npdir, ".libs"),
              os.path.join(npdir, ".dylibs")):
        paths += sorted(glob.glob(os.path.join(d, "*openblas*")))
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/maps", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 6 and parts[-1].startswith("/"):
                        base = os.path.basename(parts[-1])
                        if "openblas" in base or base.startswith("libblas") or "mkl_rt" in base:
                            paths.append(parts[-1])
        except OSError:
            pass
    handles = []
    for path in dict.fromkeys(paths):
        try:
            lib = ctypes.CDLL(path)          # already loaded by numpy: returns the same module
        except OSError:
            continue
        set_fn = next((getattr(lib, n) for n in _BLAS_SET_NAMES if hasattr(lib, n)), None)
        get_fn = next((getattr(lib, n) for n in _BLAS_GET_NAMES if hasattr(lib, n)), None)
        if set_fn is None:
            continue
        set_fn.restype = None
        if get_fn is not None:
            get_fn.restype = ctypes.c_int
        handles.append((set_fn, get_fn))
    _blas_handles = handles
    return handles


def blas_threads() -> Optional[int]:
    """Current size of the BLAS thread pool, or None if it cannot be queried."""
    for _set_fn, get_fn in _blas_libraries():
        if get_fn is not None:
            try:
                return int(get_fn())
            except Exception:
                pass
    return None


def limit_blas_threads(n: int = 1) -> bool:
    """Set the BLAS thread pool to ``n`` threads at run time (best effort).
    Returns True if at least one BLAS library accepted the setting."""
    ok = False
    for set_fn, _get_fn in _blas_libraries():
        try:
            set_fn(ctypes.c_int(max(1, int(n))))
            ok = True
        except Exception:
            pass
    return ok


@dataclass
class QPResult:
    """Solution of :meth:`QPSolver.solve` (unscaled quantities)."""
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    status: str
    iterations: int
    obj: float
    primal_residual: float
    dual_residual: float
    rho: float
    solve_time: float
    rho_updates: int = 0
    polished: bool = False

    @property
    def ok(self) -> bool:
        return self.status in OK_STATUSES


class QPSolver:
    """OSQP-style dense ADMM solver.

    Parameters
    ----------
    rho, sigma, alpha : ADMM step size, primal regularisation, relaxation (1.0..1.8).
    eps_abs, eps_rel : termination tolerances.
    eps_pinf, eps_dinf : infeasibility tolerances.
    max_iter : iteration limit.
    adaptive_rho : enable residual balancing of rho.
    adaptive_rho_interval : iterations between rho updates.
    adaptive_rho_tolerance : refactor only if rho changes by this factor.
    scaling : Ruiz equilibration iterations (0 = off).
    check_termination : evaluate residuals every this many iterations.
    time_limit : optional wall-clock limit [s] (None = unlimited).
    polish : run solution polishing after ADMM.
    polish_delta : KKT regularisation of the polishing step (scaled problem).
    polish_refine_iter : iterative-refinement steps per polishing solve.
    polish_rounds : maximum active-set corrections during polishing.
    polish_retry_interval : ADMM iterations between polishing attempts when an
        attempt fails (ADMM keeps iterating until max_iter / time_limit).
    polish_max_attempts : after this many failed polishing attempts of a
        converged ADMM iterate, the (moderate-accuracy) ADMM solution is returned.
    sparse_density : A is treated as sparse (equilibration and A' diag(rho) A
        on the nonzeros only) when nnz(A) <= sparse_density * m * n (0 = never).
    """

    def __init__(self, rho: float = 0.1, sigma: float = 1e-6, alpha: float = 1.6,
                 eps_abs: float = 1e-4, eps_rel: float = 1e-4, eps_pinf: float = 1e-5,
                 eps_dinf: float = 1e-5, max_iter: int = 4000, adaptive_rho: bool = True,
                 adaptive_rho_interval: int = 25, adaptive_rho_tolerance: float = 5.0,
                 scaling: int = 10, check_termination: int = 5,
                 time_limit: Optional[float] = None, polish: bool = False,
                 polish_delta: float = 1e-10, polish_refine_iter: int = 3,
                 polish_rounds: int = 10, polish_retry_interval: int = 25,
                 polish_max_attempts: int = 3, sparse_density: float = 0.3):
        self.rho = float(rho)
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.eps_abs = float(eps_abs)
        self.eps_rel = float(eps_rel)
        self.eps_pinf = float(eps_pinf)
        self.eps_dinf = float(eps_dinf)
        self.max_iter = int(max_iter)
        self.adaptive_rho = bool(adaptive_rho)
        self.adaptive_rho_interval = max(1, int(adaptive_rho_interval))
        self.adaptive_rho_tolerance = float(adaptive_rho_tolerance)
        self.scaling = int(scaling)
        self.check_termination = max(1, int(check_termination))
        self.time_limit = time_limit
        self.polish = bool(polish)
        self.polish_delta = float(polish_delta)
        self.polish_refine_iter = int(polish_refine_iter)
        self.polish_rounds = int(polish_rounds)
        self.polish_retry_interval = max(1, int(polish_retry_interval))
        self.polish_max_attempts = max(1, int(polish_max_attempts))
        self.sparse_density = float(sparse_density)
        self._pd_inv_cache = None
        self._deadline = None

    # ------------------------------------------------------------------ scaling
    @staticmethod
    def _ruiz(P, q, A, iters: int, pattern=None):
        """Modified Ruiz equilibration of the KKT matrix [[P, A'], [A, 0]].

        Returns scaled (P, q, A) and the scalings D (n), E (m), c with
        P_s = c D P D, q_s = c D q, A_s = E A D.  With ``pattern`` = (rows,
        cols) of the nonzeros of A, the norms and the scaling of A are computed
        on the nonzeros only (identical result, O(nnz) instead of O(m n))."""
        n, m = P.shape[0], A.shape[0]
        D = np.ones(n)
        E = np.ones(m)
        c = 1.0
        Ps, qs = P.copy(), q.copy()
        if pattern is not None:
            ra, ca = pattern
            va = A[ra, ca].astype(float)
        else:
            As = A.copy()
        for _ in range(iters):
            col_p = np.abs(Ps).max(axis=0) if n else np.zeros(0)
            if m == 0:
                col_a, norm_z = np.zeros(n), np.zeros(0)
            elif pattern is not None:
                av = np.abs(va)
                col_a = np.zeros(n)
                norm_z = np.zeros(m)
                np.maximum.at(col_a, ca, av)
                np.maximum.at(norm_z, ra, av)
            else:
                aa = np.abs(As)
                col_a, norm_z = aa.max(axis=0), aa.max(axis=1)
            norm_x = np.clip(np.maximum(col_p, col_a), 1e-4, 1e4)
            norm_z = np.clip(norm_z, 1e-4, 1e4)
            dx = 1.0 / np.sqrt(norm_x)
            dz = 1.0 / np.sqrt(norm_z)
            Ps *= dx[:, None]
            Ps *= dx[None, :]
            if pattern is not None:
                va *= dz[ra] * dx[ca]
            elif m:
                As *= dz[:, None]
                As *= dx[None, :]
            qs *= dx
            D *= dx
            E *= dz
            # cost scaling
            mean_p = float(np.mean(np.abs(Ps).max(axis=0))) if n else 0.0
            nq = float(np.max(np.abs(qs))) if n else 0.0
            gamma = max(mean_p, nq)
            gamma = 1.0 / np.clip(gamma, 1e-4, 1e4) if gamma > 0 else 1.0
            Ps *= gamma
            qs *= gamma
            c *= gamma
        if pattern is not None:
            As = np.zeros(A.shape)
            As[ra, ca] = va
        return Ps, qs, As, D, E, c

    # ------------------------------------------------------------------ sparsity
    def _sparse_pattern(self, A):
        """(rows, cols) of the nonzeros of A (row-major order) if A is sparse
        enough for the sparse code paths, else None."""
        if A.size == 0 or self.sparse_density <= 0.0:
            return None
        ra, ca = np.nonzero(A)
        if ra.size > self.sparse_density * A.size:
            return None
        return ra, ca

    @staticmethod
    def _row_pairs(ra, m: int):
        """Index pairs (k1, k2) of all nonzeros that share a row, so that
        A' diag(rho) A = sum over pairs of rho_row a_k1 a_k2 e_c(k1) e_c(k2)'.
        Requires the row-major order of np.nonzero (a row's entries contiguous)."""
        counts = np.bincount(ra, minlength=m)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        cnt = counts[ra]
        k1 = np.repeat(np.arange(ra.size), cnt)
        first = np.repeat(np.cumsum(cnt) - cnt, cnt)
        k2 = starts[ra[k1]] + (np.arange(k1.size) - first)
        return k1, k2

    # ------------------------------------------------------------------ solve
    def solve(self, P, q, A, l, u, x0=None, y0=None) -> QPResult:
        """Solve the QP.  ``x0`` / ``y0`` are optional (unscaled) warm starts."""
        t_start = time.perf_counter()
        P = np.asarray(P, dtype=float)
        q = np.asarray(q, dtype=float).reshape(-1)
        n = q.size
        A = np.asarray(A, dtype=float).reshape(-1, n) if np.size(A) else np.zeros((0, n))
        m = A.shape[0]
        l = np.asarray(l, dtype=float).reshape(-1) if m else np.zeros(0)
        u = np.asarray(u, dtype=float).reshape(-1) if m else np.zeros(0)
        P = 0.5 * (P + P.T)
        if P.shape != (n, n) or l.size != m or u.size != m or np.any(l > u + 1e-12):
            return QPResult(np.zeros(n), np.zeros(m), np.zeros(m), STATUS_INVALID, 0, np.nan,
                            np.inf, np.inf, self.rho, time.perf_counter() - t_start)

        # ---------------- scaling (sparsity-aware)
        pattern = self._sparse_pattern(A) if (m and n) else None
        if self.scaling > 0 and n > 0:
            Ps, qs, As, D, E, c = self._ruiz(P, q, A, self.scaling, pattern)
        else:
            Ps, qs, As, D, E, c = P, q, A, np.ones(n), np.ones(m), 1.0
        if pattern is not None:
            # A' diag(rho) A from the products of nonzeros sharing a row
            ra, ca = pattern
            k1, k2 = self._row_pairs(ra, m)
            va_s = As[ra, ca]
            pair_val = va_s[k1] * va_s[k2]
            pair_row = ra[k1]
            pair_idx = ca[k1] * n + ca[k2]

            def atra(rv_):
                return np.bincount(pair_idx, weights=rv_[pair_row] * pair_val,
                                   minlength=n * n).reshape(n, n)
        else:
            def atra(rv_):
                return (As.T * rv_) @ As
        ls = np.where(np.isfinite(l), l * E, l)
        us = np.where(np.isfinite(u), u * E, u)
        Dinv = 1.0 / D
        Einv = 1.0 / E

        # ---------------- rho vector
        eq = np.isfinite(l) & np.isfinite(u) & (np.abs(u - l) < 1e-9)
        free = ~np.isfinite(l) & ~np.isfinite(u)
        rho = float(np.clip(self.rho, RHO_MIN, RHO_MAX))

        def rho_vec(r):
            v = np.full(m, r)
            v[eq] = RHO_EQ_SCALE * r
            v[free] = RHO_MIN
            return v

        rv = rho_vec(rho)
        sig = self.sigma
        eye = np.eye(n)

        def factor(rv_):
            K = Ps + sig * eye + atra(rv_)
            Ki = np.linalg.inv(K)
            return 0.5 * (Ki + Ki.T)

        Kinv = factor(rv)
        rho_updates = 0

        # ---------------- warm start (scaled variables)
        x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).reshape(n) * Dinv
        z = np.clip(As @ x, ls, us) if m else np.zeros(0)
        y = np.zeros(m) if y0 is None else np.asarray(y0, dtype=float).reshape(m) * Einv * c
        alpha = self.alpha
        status = STATUS_MAX_ITER
        r_prim = r_dual = np.inf
        it = 0
        q_norm_unscaled = float(np.max(np.abs(Dinv * qs))) / c if n else 0.0

        def residuals(x_, z_, y_):
            Ax = As @ x_
            Px = Ps @ x_
            Aty = As.T @ y_
            rp = float(np.max(np.abs(Einv * (Ax - z_)))) if m else 0.0
            rd = float(np.max(np.abs(Dinv * (Px + qs + Aty)))) / c if n else 0.0
            ep = self.eps_abs + self.eps_rel * (max(float(np.max(np.abs(Einv * Ax))),
                                                    float(np.max(np.abs(Einv * z_)))) if m else 0.0)
            ed = self.eps_abs + self.eps_rel * max(float(np.max(np.abs(Dinv * Px))) / c,
                                                   float(np.max(np.abs(Dinv * Aty))) / c if m else 0.0,
                                                   q_norm_unscaled)
            return rp, rd, ep, ed, Ax, Px, Aty

        polished = False
        admm_converged = False
        next_polish = 0
        polish_attempts = 0
        self._pd_inv_cache = None
        self._deadline = (t_start + self.time_limit) if self.time_limit is not None else None

        def try_polish(x_, z_, y_):
            """Polish and accept only if residuals do not get worse."""
            pol = self._polish(Ps, qs, As, ls, us, x_, z_, y_)
            if pol is None:
                return None
            xp, zp, yp = pol
            rp2, rd2, ep2, ed2, _a, _p, _t = residuals(xp, zp, yp)
            rp1, rd1, _e1, _e2, _a, _p, _t = residuals(x_, z_, y_)
            if rp2 <= max(rp1, ep2) and rd2 <= max(rd1, ed2):
                return xp, zp, yp, rp2, rd2, (rp2 <= ep2 and rd2 <= ed2)
            return None

        for it in range(1, self.max_iter + 1):
            x_prev, y_prev = x, y
            rhs = sig * x - qs + As.T @ (rv * z - y) if m else sig * x - qs
            xt = Kinv @ rhs
            x = alpha * xt + (1.0 - alpha) * x_prev
            if m:
                zt = As @ xt
                zr = alpha * zt + (1.0 - alpha) * z
                z_new = np.clip(zr + y / rv, ls, us)
                y = y + rv * (zr - z_new)
                z = z_new

            check = (it % self.check_termination == 0) or it == self.max_iter
            adapt = self.adaptive_rho and m and (it % self.adaptive_rho_interval == 0)
            if not (check or adapt):
                continue
            r_prim, r_dual, eps_p, eps_d, Ax, Px, Aty = residuals(x, z, y)
            if r_prim <= eps_p and r_dual <= eps_d:
                admm_converged = True
                if not (self.polish and m and n):
                    status = STATUS_SOLVED
                    break
                if it >= next_polish:
                    # polish; if the active-set guess is not consistent yet, keep
                    # iterating and retry a little later
                    next_polish = it + self.polish_retry_interval
                    polish_attempts += 1
                    pol = try_polish(x, z, y)
                    if pol is not None:
                        x, z, y, r_prim, r_dual, ok = pol
                        polished = True
                        status = STATUS_SOLVED if ok else STATUS_INACCURATE
                        break
                    if polish_attempts >= self.polish_max_attempts:
                        # the ADMM iterate meets the tolerances; polishing could not
                        # identify a consistent active set (heavily degenerate problem)
                        status = STATUS_SOLVED
                        break
            # infeasibility certificates
            if m and not admm_converged:
                dy = y - y_prev
                dy_u = E * dy                              # proportional to the unscaled dy
                ndy = float(np.max(np.abs(dy_u)))
                if ndy > 1e-12:
                    at_dy = float(np.max(np.abs(Dinv * (As.T @ dy))))
                    pos = np.maximum(dy_u, 0.0)
                    neg = np.minimum(dy_u, 0.0)
                    inf_u = (pos > 0) & ~np.isfinite(u)
                    inf_l = (neg < 0) & ~np.isfinite(l)
                    if not (np.any(inf_u) or np.any(inf_l)):
                        supp = float(np.sum(np.where(pos > 0, u, 0.0) * pos)
                                     + np.sum(np.where(neg < 0, l, 0.0) * neg))
                        if at_dy <= self.eps_pinf * ndy and supp < -self.eps_pinf * ndy:
                            status = STATUS_PRIMAL_INFEASIBLE
                            break
            dx = x - x_prev
            dx_u = D * dx
            ndx = float(np.max(np.abs(dx_u))) if n else 0.0
            if ndx > 1e-12 and not admm_converged:
                pdx = float(np.max(np.abs(Dinv * (Ps @ dx)))) / c
                qdx = float(np.dot(qs, dx)) / c
                if pdx <= self.eps_dinf * ndx and qdx < -self.eps_dinf * ndx:
                    ok = True
                    if m:
                        adx = Einv * (As @ dx)
                        tol = self.eps_dinf * ndx
                        hi_ok = np.where(np.isfinite(u), adx <= tol, True)
                        lo_ok = np.where(np.isfinite(l), adx >= -tol, True)
                        ok = bool(np.all(hi_ok & lo_ok))
                    if ok:
                        status = STATUS_DUAL_INFEASIBLE
                        break
            if self.time_limit is not None and time.perf_counter() - t_start > self.time_limit:
                status = STATUS_TIME_LIMIT
                break
            if adapt:
                # residual balancing in the scaled space (OSQP eq. 22)
                rp_s = float(np.max(np.abs(Ax - z)))
                rd_s = float(np.max(np.abs(Px + qs + Aty)))
                den_p = max(float(np.max(np.abs(Ax))), float(np.max(np.abs(z))), 1e-12)
                den_d = max(float(np.max(np.abs(Px))), float(np.max(np.abs(Aty))),
                            float(np.max(np.abs(qs))), 1e-12)
                ratio = (rp_s / den_p) / max(rd_s / den_d, 1e-12)
                rho_new = float(np.clip(rho * np.sqrt(ratio), RHO_MIN, RHO_MAX))
                tol = self.adaptive_rho_tolerance
                if rho_new > tol * rho or rho_new < rho / tol:
                    rho = rho_new
                    rv = rho_vec(rho)
                    Kinv = factor(rv)
                    rho_updates += 1

        if status in (STATUS_MAX_ITER, STATUS_TIME_LIMIT):
            # one last polishing attempt (allowed to exceed the ADMM time limit slightly)
            self._deadline = None if self._deadline is None else time.perf_counter() + 0.5 * self.time_limit
            if self.polish and m and n and not polished:
                pol = try_polish(x, z, y)
                if pol is not None:
                    x, z, y, r_prim, r_dual, ok = pol
                    polished = True
                    status = STATUS_SOLVED if ok else STATUS_INACCURATE
            if not polished:
                r_prim, r_dual, eps_p, eps_d, _Ax, _Px, _Aty = residuals(x, z, y)
                if admm_converged or (r_prim <= 10.0 * eps_p and r_dual <= 10.0 * eps_d):
                    status = STATUS_INACCURATE

        # ---------------- unscale
        x_u = D * x
        z_u = Einv * z
        y_u = E * y / c
        obj = 0.5 * float(x_u @ P @ x_u) + float(q @ x_u)
        return QPResult(x=x_u, y=y_u, z=z_u, status=status, iterations=it, obj=obj,
                        primal_residual=r_prim, dual_residual=r_dual, rho=rho,
                        solve_time=time.perf_counter() - t_start, rho_updates=rho_updates,
                        polished=polished)

    # ------------------------------------------------------------------ polish
    @staticmethod
    def _independent_rows(As, candidates, tol: float = 1e-9):
        """Greedy (order-preserving) selection of linearly independent rows, so
        that the reduced KKT system stays regular (e.g. both opposite faces of
        a friction pyramid can never be active together).  Uses a QR
        factorisation of the candidate rows: a negligible |R_jj| means row j
        lies in the span of the rows before it; repeated on the kept subset
        until stable (normally one or two LAPACK calls)."""
        cand = list(candidates)
        if not cand:
            return []
        for _ in range(4):
            M = As[cand].T                                    # (n, k)
            norms = np.linalg.norm(M, axis=0)
            # with more candidates than variables only the first n columns can be pivots
            R = np.linalg.qr(M, mode="r")
            k = min(R.shape[0], M.shape[1])
            diag = np.abs(np.diag(R[:k, :k])) if k else np.zeros(0)
            ok = np.zeros(M.shape[1], dtype=bool)
            ok[:k] = diag > tol * np.maximum(norms[:k], 1e-300)
            if ok.all():
                return cand
            cand = [c for c, o in zip(cand, ok) if o]
            if not cand:
                return []
        return cand

    # The polishing systems are solved WITHOUT the usual regularisation P + dI
    # whenever P is numerically positive definite.  For an ill-conditioned P
    # (the condensed MPC Hessian has cond ~ 5e9: internal "squeeze" forces
    # between stance feet are only weakly penalised) the smallest scaled
    # eigenvalue can be below d, so the regularised solution is biased in those
    # directions and the iterative refinement removes the bias only at the rate
    # d / (lambda_min + d) per step (measured: several newtons of internal force
    # error).  The regularised path remains as a fallback for singular P (LPs).
    def _solve_p(self, Ps, rhs) -> np.ndarray:
        """x = P^-1 rhs (unregularised if P is PD, else (P + dI) + refinement)."""
        scale = max(1.0, float(np.max(np.abs(rhs)))) if rhs.size else 1.0
        try:
            x = np.linalg.solve(Ps, rhs)
            if np.all(np.isfinite(x)) and float(np.max(np.abs(Ps @ x - rhs))) <= 1e-7 * scale:
                return x
        except np.linalg.LinAlgError:
            pass
        Pd = Ps + self.polish_delta * np.eye(Ps.shape[0])
        x = np.linalg.solve(Pd, rhs)
        for _ in range(self.polish_refine_iter):
            r = rhs - Ps @ x
            if float(np.max(np.abs(r))) <= 1e-12 * scale:
                break
            x = x + np.linalg.solve(Pd, r)
        return x

    def _inv_p(self, Ps) -> np.ndarray:
        """Symmetric P^-1 (unregularised if P is PD, else (P + dI)^-1)."""
        n = Ps.shape[0]
        probe = np.linspace(1.0, 2.0, n)
        try:
            Pi = np.linalg.inv(Ps)
            if np.all(np.isfinite(Pi)) and \
                    float(np.max(np.abs(Ps @ (Pi @ probe) - probe))) <= 1e-5 * float(np.max(probe)):
                return 0.5 * (Pi + Pi.T)
        except np.linalg.LinAlgError:
            pass
        Pi = np.linalg.inv(Ps + self.polish_delta * np.eye(n))
        return 0.5 * (Pi + Pi.T)

    def _polish(self, Ps, qs, As, ls, us, x, z, y):
        """OSQP-style solution polishing on the scaled problem, extended with
        primal-dual active-set corrections: constraints violated by the polished
        point are added (most violated first, keeping the active rows linearly
        independent), wrong-signed multipliers are dropped.  Returns (x, z, y)
        or None if no consistent active set was found."""
        m = As.shape[0]
        low = (z - ls) < -y                 # lower bound active (y < 0)
        upp = ((us - z) < y) & ~low         # upper bound active (y > 0)
        # initial guess: strongest multipliers first, independent rows only
        cand = np.nonzero(low | upp)[0]
        cand = cand[np.argsort(-np.abs(y[cand]))]
        sel = np.zeros(m, dtype=bool)
        sel[self._independent_rows(As, cand)] = True
        low &= sel
        upp &= sel
        d = self.polish_delta
        # P^-1 is formed lazily: with an empty active set (the common case for
        # the MPC: forces strictly inside the friction pyramid) one LU solve
        # (2/3 n^3 flops) replaces the explicit inverse (2 n^3 flops)
        Pd_inv = self._pd_inv_cache
        for _round in range(max(1, self.polish_rounds)):
            if self._deadline is not None and time.perf_counter() > self._deadline:
                return None
            act = low | upp
            idx = np.nonzero(act)[0]
            na = idx.size
            try:
                if na == 0 and Pd_inv is None:
                    xp = self._solve_p(Ps, -qs)
                    yact = np.zeros(0)
                elif na == 0:
                    xp = -Pd_inv @ qs
                    for _ in range(self.polish_refine_iter):
                        xp = xp + Pd_inv @ (-qs - Ps @ xp)
                    yact = np.zeros(0)
                else:
                    if Pd_inv is None:
                        Pd_inv = self._inv_p(Ps)
                        self._pd_inv_cache = Pd_inv
                    Aa = As[idx]
                    b = np.where(low[idx], ls[idx], us[idx])
                    if not np.all(np.isfinite(b)):
                        return None
                    PA = Pd_inv @ Aa.T
                    S = Aa @ PA + d * np.eye(na)
                    S_inv = np.linalg.inv(S)

                    def kkt_solve(r1, r2):
                        # [P(+dI) A'; A -dI] [x; y] = [r1; r2] via the Schur complement
                        t = Pd_inv @ r1
                        yy = S_inv @ (Aa @ t - r2)
                        return t - PA @ yy, yy

                    xp, yact = kkt_solve(-qs, b)
                    for _ in range(self.polish_refine_iter):
                        # iterative refinement on the unregularised KKT system
                        r1 = -qs - (Ps @ xp + Aa.T @ yact)
                        r2 = b - Aa @ xp
                        dx, dy = kkt_solve(r1, r2)
                        xp = xp + dx
                        yact = yact + dy
            except np.linalg.LinAlgError:
                return None
            if not (np.all(np.isfinite(xp)) and np.all(np.isfinite(yact))):
                return None
            yp = np.zeros(m)
            yp[idx] = yact
            Ax = As @ xp
            tol_p = 1e-7 * max(1.0, float(np.max(np.abs(Ax))))
            tol_y = 1e-7 * max(1.0, float(np.max(np.abs(yact)))) if na else 1e-7
            viol = np.where(~act, np.maximum(ls - Ax, Ax - us), -np.inf)
            wrong_lo = low & (yp > tol_y)
            wrong_hi = upp & (yp < -tol_y)
            new = np.nonzero(viol > tol_p)[0]
            if new.size == 0 and not (wrong_lo.any() or wrong_hi.any()):
                return xp, np.clip(Ax, ls, us), yp
            # update: drop wrong signs, add violations (largest first, independent rows only)
            low &= ~wrong_lo
            upp &= ~wrong_hi
            keep = np.nonzero(low | upp)[0]
            new = new[np.argsort(-viol[new])]
            chosen = self._independent_rows(As, list(keep) + list(new))
            add = np.array([i for i in chosen if i in set(new.tolist())], dtype=int)
            if add.size:
                lo_add = add[Ax[add] < ls[add]]
                hi_add = add[Ax[add] >= ls[add]]
                low[lo_add] = True
                upp[hi_add] = True
            if add.size == 0 and not (wrong_lo.any() or wrong_hi.any()):
                return None          # violations only by dependent rows: give up
        return None


def solve_qp(P, q, A, l, u, x0=None, y0=None, **settings) -> QPResult:
    """Convenience wrapper: ``QPSolver(**settings).solve(P, q, A, l, u, x0, y0)``."""
    return QPSolver(**settings).solve(P, q, A, l, u, x0=x0, y0=y0)
