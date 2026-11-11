"""Dense ADMM QP solver: known solutions, scipy cross-check, infeasibility, warm start,
sparse code path, polishing accuracy on ill-conditioned problems, BLAS thread control."""
import numpy as np
import pytest

from module1_locomotion import qp as Q
from module1_locomotion.qp import (STATUS_DUAL_INFEASIBLE, STATUS_PRIMAL_INFEASIBLE, STATUS_SOLVED,
                                   QPSolver, solve_qp)

TOL = dict(eps_abs=1e-7, eps_rel=1e-7, max_iter=20000)


def test_unconstrained_quadratic():
    P = np.array([[4.0, 1.0], [1.0, 2.0]])
    q = np.array([1.0, 1.0])
    r = solve_qp(P, q, np.zeros((0, 2)), np.zeros(0), np.zeros(0), **TOL)
    assert r.status == STATUS_SOLVED
    assert np.allclose(r.x, np.linalg.solve(P, -q), atol=1e-6)


def test_box_constrained_known_solution():
    # min (x-3)^2 + (y+2)^2  s.t. 0 <= x <= 1, -1 <= y <= 1  ->  (1, -1)
    P = 2.0 * np.eye(2)
    q = np.array([-6.0, 4.0])
    r = solve_qp(P, q, np.eye(2), [0.0, -1.0], [1.0, 1.0], **TOL)
    assert r.ok and np.allclose(r.x, [1.0, -1.0], atol=1e-5)
    # multipliers: upper bound active on x (y > 0), lower bound on y (y < 0)
    assert r.y[0] > 0 and r.y[1] < 0


def test_equality_constraint():
    # min x1^2 + x2^2  s.t. x1 + x2 = 1  -> (0.5, 0.5), multiplier -1
    r = solve_qp(2 * np.eye(2), np.zeros(2), [[1.0, 1.0]], [1.0], [1.0], **TOL)
    assert r.ok and np.allclose(r.x, [0.5, 0.5], atol=1e-6)
    assert r.y[0] == pytest.approx(-1.0, abs=1e-4)


def test_osqp_documentation_example():
    # OSQP "Setup and solve" demo problem, solution (0.3, 0.7)
    P = np.array([[4.0, 1.0], [1.0, 2.0]])
    q = np.array([1.0, 1.0])
    A = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    l = np.array([1.0, 0.0, 0.0])
    u = np.array([1.0, 0.7, 0.7])
    r = solve_qp(P, q, A, l, u, **TOL)
    assert r.ok and np.allclose(r.x, [0.3, 0.7], atol=1e-5)
    assert r.obj == pytest.approx(1.88, abs=1e-4)
    rp = solve_qp(P, q, A, l, u, eps_abs=1e-3, eps_rel=1e-3, polish=True)
    assert rp.polished and np.allclose(rp.x, [0.3, 0.7], atol=1e-9)


@pytest.mark.parametrize("seed", range(6))
def test_random_qps_against_reference(seed):
    rng = np.random.default_rng(seed)
    n, m = 12, 20
    M = rng.normal(size=(n, n))
    P = M @ M.T + 0.1 * np.eye(n)
    q = rng.normal(size=n) * 5
    A = rng.normal(size=(m, n))
    x_feas = rng.normal(size=n)
    Ax = A @ x_feas
    l = Ax - rng.uniform(0.1, 1.0, m)
    u = Ax + rng.uniform(0.1, 1.0, m)
    l[:3] = -np.inf                                     # one-sided rows
    r = solve_qp(P, q, A, l, u, **TOL)
    rp = solve_qp(P, q, A, l, u, eps_abs=1e-4, eps_rel=1e-4, polish=True)
    assert r.ok and rp.ok
    try:
        from scipy.optimize import LinearConstraint, minimize
        lin = LinearConstraint(A, l, u)
        ref = minimize(lambda x: 0.5 * x @ P @ x + q @ x, x_feas, jac=lambda x: P @ x + q,
                       hess=lambda x: P, constraints=[lin], method="trust-constr",
                       options={"gtol": 1e-12, "xtol": 1e-14, "maxiter": 5000})
        f_ref = ref.fun
    except ImportError:          # pragma: no cover - scipy is only a test aid
        f_ref = r.obj
    assert r.obj == pytest.approx(f_ref, rel=1e-5, abs=1e-6)
    assert rp.obj == pytest.approx(f_ref, rel=1e-5, abs=1e-6)
    for res, tol in ((r, 1e-5), (rp, 1e-3 if not rp.polished else 1e-7)):
        Ax = A @ res.x
        assert np.all(Ax >= l - tol) and np.all(Ax <= u + tol)


def test_primal_infeasible():
    # x >= 1 and x <= 0
    r = solve_qp(np.eye(1), np.zeros(1), [[1.0], [1.0]], [1.0, -np.inf], [np.inf, 0.0])
    assert r.status == STATUS_PRIMAL_INFEASIBLE


def test_dual_infeasible():
    # min -x  s.t. x >= 0   (unbounded)
    r = solve_qp(np.zeros((1, 1)), np.array([-1.0]), [[1.0]], [0.0], [np.inf])
    assert r.status == STATUS_DUAL_INFEASIBLE


def test_warm_start_reduces_iterations():
    rng = np.random.default_rng(7)
    n, m = 30, 45
    M = rng.normal(size=(n, n))
    P = M @ M.T + np.eye(n)
    q = rng.normal(size=n)
    A = rng.normal(size=(m, n))
    l = -rng.uniform(0.5, 1.5, m)
    u = rng.uniform(0.5, 1.5, m)
    s = QPSolver(eps_abs=1e-6, eps_rel=1e-6, max_iter=20000)
    cold = s.solve(P, q, A, l, u)
    warm = s.solve(P, q + 1e-3 * rng.normal(size=n), A, l, u, x0=cold.x, y0=cold.y)
    assert cold.ok and warm.ok
    assert warm.iterations < cold.iterations


def test_invalid_and_time_limit():
    r = solve_qp(np.eye(2), np.zeros(2), np.eye(2), [1.0, 1.0], [0.0, 0.0])
    assert r.status == "invalid_problem"
    rng = np.random.default_rng(8)
    n = 60
    M = rng.normal(size=(n, n))
    P = M @ M.T + 1e-6 * np.eye(n)
    r = solve_qp(P, rng.normal(size=n), rng.normal(size=(90, n)), -np.ones(90), np.ones(90),
                 eps_abs=1e-12, eps_rel=1e-12, max_iter=10 ** 7, time_limit=0.05, adaptive_rho=False)
    assert r.status in ("time_limit_reached", "solved_inaccurate", "solved")
    assert r.solve_time < 1.0


def _friction_qp(rng, n_legs=12, mu=0.5):
    """MPC-like QP: forces (fx, fy, fz) per leg, friction pyramid + normal force
    bounds (block-diagonal A, > 90 % zeros), random SPD cost."""
    n = 3 * n_legs
    blk = np.array([[1.0, 0.0, -mu], [-1.0, 0.0, -mu], [0.0, 1.0, -mu], [0.0, -1.0, -mu], [0.0, 0.0, 1.0]])
    A = np.kron(np.eye(n_legs), blk)
    lo = np.tile([-np.inf, -np.inf, -np.inf, -np.inf, 1.0], n_legs)
    hi = np.tile([0.0, 0.0, 0.0, 0.0, 120.0], n_legs)
    M = rng.normal(size=(n, n))
    P = M @ M.T / n + 1e-3 * np.eye(n)
    q = rng.normal(size=n) * 20.0
    return P, q, A, lo, hi


@pytest.mark.parametrize("seed", range(3))
def test_sparse_and_dense_code_paths_agree(seed):
    """The sparsity-aware equilibration / A' diag(rho) A give the same solution."""
    P, q, A, lo, hi = _friction_qp(np.random.default_rng(seed))
    sp = QPSolver(eps_abs=1e-6, eps_rel=1e-6, max_iter=20000, polish=True, sparse_density=0.3)
    de = QPSolver(eps_abs=1e-6, eps_rel=1e-6, max_iter=20000, polish=True, sparse_density=0.0)
    assert sp._sparse_pattern(A) is not None and de._sparse_pattern(A) is None
    rs, rd = sp.solve(P, q, A, lo, hi), de.solve(P, q, A, lo, hi)
    assert rs.ok and rd.ok
    assert np.allclose(rs.x, rd.x, atol=1e-6 * max(1.0, np.max(np.abs(rd.x))))
    # the product of nonzeros sharing a row reproduces A' diag(rho) A exactly
    ra, ca = sp._sparse_pattern(A)
    k1, k2 = QPSolver._row_pairs(ra, A.shape[0])
    rho = np.random.default_rng(seed).uniform(0.1, 2.0, A.shape[0])
    n = A.shape[1]
    M = np.bincount(ca[k1] * n + ca[k2], weights=rho[ra[k1]] * A[ra, ca][k1] * A[ra, ca][k2],
                    minlength=n * n).reshape(n, n)
    assert np.allclose(M, (A.T * rho) @ A, atol=1e-12)


def test_polish_is_exact_for_ill_conditioned_hessian():
    """cond(P) ~ 1e11 with a very weakly penalised direction (like the internal
    'squeeze' forces of the condensed MPC): the polished solution must equal
    P^-1 (-q), i.e. polishing must not be biased by a regularisation larger than
    the smallest eigenvalue of the scaled Hessian."""
    rng = np.random.default_rng(11)
    n = 40
    Qm, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = np.logspace(-6, 5, n)
    P = (Qm * lam) @ Qm.T
    P = 0.5 * (P + P.T)
    q = rng.normal(size=n)
    x_ref = np.linalg.solve(P, -q)
    box = 10.0 * np.max(np.abs(x_ref)) + 1.0                 # inactive bounds
    r = solve_qp(P, q, np.eye(n), -box * np.ones(n), box * np.ones(n),
                 eps_abs=1e-3, eps_rel=1e-3, max_iter=4000, polish=True)
    assert r.ok and r.polished
    assert np.linalg.norm(r.x - x_ref) <= 1e-6 * np.linalg.norm(x_ref)


def test_polish_falls_back_to_regularisation_for_lp():
    """Singular P (linear program): polishing uses (P + dI) + refinement.
    max x1 + x2  s.t. x1 + 2 x2 <= 4, 3 x1 + x2 <= 6, x >= 0  ->  (1.6, 1.2)."""
    A = np.array([[1.0, 2.0], [3.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    r = solve_qp(np.zeros((2, 2)), np.array([-1.0, -1.0]), A, [-np.inf, -np.inf, 0.0, 0.0],
                 [4.0, 6.0, np.inf, np.inf], eps_abs=1e-5, eps_rel=1e-5, max_iter=20000, polish=True)
    assert r.ok
    assert np.allclose(r.x, [1.6, 1.2], atol=1e-4)
    assert r.obj == pytest.approx(-2.8, abs=1e-4)


def test_polish_attempts_are_capped():
    """A converged ADMM iterate is returned after polish_max_attempts failures
    instead of iterating to max_iter."""
    P, q, A, lo, hi = _friction_qp(np.random.default_rng(5))
    s = QPSolver(eps_abs=1e-4, eps_rel=1e-4, max_iter=5000, polish=True, polish_max_attempts=2)
    s._polish = lambda *a, **k: None                          # force every polishing attempt to fail
    r = s.solve(P, q, A, lo, hi)
    assert r.status == STATUS_SOLVED and not r.polished
    assert r.iterations < 5000


def test_blas_thread_helpers():
    """Best-effort BLAS thread control never raises; when supported it works."""
    before = Q.blas_threads()
    assert before is None or before >= 1
    ok = Q.limit_blas_threads(1)
    assert isinstance(ok, bool)
    try:
        if ok and before is not None:
            assert Q.blas_threads() == 1
    finally:
        if ok and before is not None:
            Q.limit_blas_threads(before)
