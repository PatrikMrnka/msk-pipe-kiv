# SPDX-License-Identifier: Apache-2.0
"""Surface registration for atlas transfer: similarity ICP and smooth non-rigid warps.

Pure NumPy/SciPy on point samples of the surfaces (no VTK):

* :func:`similarity_icp` - symmetric trimmed ICP with an optional uniform scale
  (Umeyama; point-to-point or point-to-plane), robust to bones that cover a different
  extent (cropped images, sacrum).
* :func:`fit_coherent_icp` - *coherent ICP*: non-rigid ICP whose displacement field
  ``v(x) = sum_j g(x, c_j) w_j`` (Gaussian kernel of width ``beta``) carries the
  motion-coherence regulariser of CPD, fitted to hard, symmetric, trimmed closest-point
  correspondences projected on the target normals (point-to-plane).
* :func:`fit_cpd` - non-rigid Coherent Point Drift (Myronenko & Song 2010): the same
  kind of field, fitted by EM with soft (GMM) correspondences and an outlier weight.

All return a :class:`GaussianWarp` in the units of the input (mm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial import cKDTree

from mskpipe.io.mesh_io import TriMesh


def sample_surface(
    mesh: TriMesh, n: int, rng: np.random.Generator, *, normals: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """``n`` points uniformly distributed over the surface (area-weighted).

    With ``normals=True`` also the unit normal of the triangle of each point.
    """
    tri = mesh.vertices[mesh.faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    if not area.sum() > 0:
        raise ValueError("mesh has no area")
    idx = rng.choice(len(tri), size=n, p=area / area.sum())
    a, b = rng.random(n), rng.random(n)
    flip = a + b > 1.0
    a[flip], b[flip] = 1.0 - a[flip], 1.0 - b[flip]
    t = tri[idx]
    pts = t[:, 0] + a[:, None] * (t[:, 1] - t[:, 0]) + b[:, None] * (t[:, 2] - t[:, 0])
    if not normals:
        return pts
    nrm = cross[idx] / np.maximum(np.linalg.norm(cross[idx], axis=1), 1e-12)[:, None]
    return pts, nrm


def apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.asarray(points, float) @ matrix[:3, :3].T + matrix[:3, 3]


def umeyama(src: np.ndarray, dst: np.ndarray, scale: bool = True) -> np.ndarray:
    """4x4 similarity (rotation, uniform scale, translation) minimising ||s R src + t - dst||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(xd.T @ xs / len(src))
    d = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1.0
    r = u @ np.diag(d) @ vt
    s = float((sig * d).sum() / (xs**2).sum(axis=1).mean()) if scale else 1.0
    m = np.eye(4)
    m[:3, :3] = s * r
    m[:3, 3] = mu_d - s * r @ mu_s
    return m


@dataclass(frozen=True)
class RigidResult:
    matrix: np.ndarray  # 4x4, source -> target
    scale: float
    rms: float  # trimmed symmetric RMS of the last iteration (point-to-point)
    iterations: int


def similarity_icp(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    init: np.ndarray | None = None,
    scale: bool = True,
    iterations: int = 100,
    trim: float = 0.9,
    tol: float = 1e-6,
    src_normals: np.ndarray | None = None,
    dst_normals: np.ndarray | None = None,
) -> RigidResult:
    """Symmetric trimmed ICP: correspondences src->dst and dst->src, worst ``1 - trim`` dropped.

    Without normals each iteration is the closed-form point-to-point similarity
    (Umeyama). With both ``src_normals`` and ``dst_normals`` it is point-to-plane
    (linearised similarity, Low 2004): only the normal component of each pair is
    minimised, so the surfaces may slide along each other. This converges in a few
    iterations even for elongated bones and does not depend on the sampling density
    of the target (point-to-point needs a target sampled much more densely than the
    source, else the sampling noise biases the result). Point-to-plane needs a start
    that is already roughly aligned - run point-to-point first.
    """
    m = np.eye(4) if init is None else np.asarray(init, float)
    plane = src_normals is not None and dst_normals is not None
    tree_dst = cKDTree(dst)
    prev, rms, it = np.inf, np.inf, 0
    for step in range(1, iterations + 1):
        moved = apply(m, src)
        d1, i1 = tree_dst.query(moved)
        d2, i2 = cKDTree(moved).query(dst)
        s1 = d1 <= np.quantile(d1, trim)
        s2 = d2 <= np.quantile(d2, trim)
        rms = float(np.sqrt((np.sum(d1[s1] ** 2) + np.sum(d2[s2] ** 2)) / (s1.sum() + s2.sum())))
        it = step
        if plane:
            a = np.vstack([moved[s1], moved[i2[s2]]])
            b = np.vstack([dst[i1[s1]], dst[s2]])
            rot = m[:3, :3] / np.cbrt(abs(np.linalg.det(m[:3, :3])))
            n = np.vstack([dst_normals[i1[s1]], src_normals[i2[s2]] @ rot.T])
            delta = _point_to_plane_step(a, b, n, scale)
            m = delta @ m
            change = np.abs(delta - np.eye(4)).max()
            if change < tol:
                break
        else:
            a = np.vstack([src[s1], src[i2[s2]]])
            b = np.vstack([dst[i1[s1]], dst[s2]])
            m = umeyama(a, b, scale)
            if abs(prev - rms) < tol:
                break
        prev = rms
    s = float(np.cbrt(abs(np.linalg.det(m[:3, :3]))))
    return RigidResult(matrix=m, scale=s, rms=rms, iterations=it)


def _point_to_plane_step(a: np.ndarray, b: np.ndarray, n: np.ndarray, scale: bool) -> np.ndarray:
    """Incremental similarity (4x4) minimising sum (n . (T a - b))^2, linearised about the
    centroid of ``a``: T x = c + (1 + sigma) R(omega) (x - c) + tau."""
    c = a.mean(0)
    x = a - c
    cols = [np.cross(x, n), n]
    if scale:
        cols.insert(0, np.sum(x * n, axis=1)[:, None])
    jac = np.hstack(cols)
    res = -np.sum((a - b) * n, axis=1)
    sol, *_ = np.linalg.lstsq(jac, res, rcond=None)
    sigma = sol[0] if scale else 0.0
    omega, tau = sol[-6:-3], sol[-3:]
    r = _rodrigues(omega)
    t = np.eye(4)
    t[:3, :3] = (1.0 + sigma) * r
    t[:3, 3] = c + tau - t[:3, :3] @ c
    return t


def _rodrigues(omega: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(omega))
    if theta < 1e-15:
        return np.eye(3)
    k = omega / theta
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * kx @ kx


@dataclass(frozen=True)
class GaussianWarp:
    """Smooth displacement field; ``warp(points)`` returns the moved points."""

    centers: np.ndarray  # (N, 3)
    weights: np.ndarray  # (N, 3)
    beta: float

    def displacement(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, float)
        out = np.zeros_like(points)
        for start in range(0, len(points), 2048):  # bounded memory
            p = points[start : start + 2048]
            g = np.exp(-_sqdist(p, self.centers) / (2.0 * self.beta**2))
            out[start : start + 2048] = g @ self.weights
        return out

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, float) + self.displacement(points)


def fit_coherent_icp(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    dst_normals: np.ndarray | None = None,
    beta: float = 40.0,
    lam: float = 2.0,
    iterations: int = 30,
    trim: float = 0.9,
) -> tuple[GaussianWarp, float]:
    """Coherent ICP: warp ``src`` (already rigidly aligned) onto ``dst``; returns the warp
    and the RMS of the last correspondences.

    Each iteration takes symmetric closest-point targets (src->dst and dst->src, worst
    ``1 - trim`` dropped), averages them per source point and solves
    ``(G + lam I) W = D`` - the CPD motion-coherence update with hard correspondences.
    With ``dst_normals`` only the normal part of each correspondence is used
    (point-to-plane), so sampling noise does not slide the surface sideways.
    Larger ``beta`` / ``lam`` = smoother (more rigid) warp.
    """
    src = np.asarray(src, float)
    g = np.exp(-_sqdist(src, src) / (2.0 * beta**2))
    factor = cho_factor(g + lam * np.eye(len(src)))
    w = np.zeros_like(src)
    tree_dst = cKDTree(dst)
    rms = np.inf
    for _ in range(iterations):
        disp = g @ w
        moved = src + disp
        d1, i1 = tree_dst.query(moved)
        d2, i2 = cKDTree(moved).query(dst)
        s1 = d1 <= np.quantile(d1, trim)
        s2 = d2 <= np.quantile(d2, trim)
        step1 = dst[i1[s1]] - moved[s1]
        step2 = dst[s2] - moved[i2[s2]]
        if dst_normals is not None:
            n1, n2 = dst_normals[i1[s1]], dst_normals[s2]
            step1 = np.sum(step1 * n1, axis=1)[:, None] * n1
            step2 = np.sum(step2 * n2, axis=1)[:, None] * n2
        acc = np.zeros_like(src)
        cnt = np.zeros(len(src))
        acc[s1] += disp[s1] + step1
        cnt[s1] += 1
        np.add.at(acc, i2[s2], disp[i2[s2]] + step2)
        np.add.at(cnt, i2[s2], 1)
        target = np.where(cnt[:, None] > 0, acc / np.maximum(cnt, 1)[:, None], disp)
        w = cho_solve(factor, target)
        rms = float(np.sqrt(np.mean(d1[s1] ** 2)))
    return GaussianWarp(centers=src, weights=w, beta=beta), rms


@dataclass(frozen=True)
class CpdResult:
    warp: GaussianWarp
    iterations: int
    sigma2: float  # final GMM variance [units^2]
    beta: float  # kernel width [units]


def fit_cpd(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    beta: float = 2.0,
    lam: float = 2.0,
    w: float = 0.1,
    iterations: int = 150,
    tol: float = 1e-5,
    sigma_init: float | None = None,
) -> CpdResult:
    """Non-rigid CPD of ``src`` (M points, already rigidly aligned) onto ``dst`` (N points).

    As in Myronenko & Song (2010), both sets are normalised (here by the mean and RMS
    radius of ``dst``, so the rigid alignment is kept); ``beta`` and ``lam`` are in these
    normalised units (paper / probreg defaults 2, 2). ``w`` is the weight of the uniform
    outlier component (0 = none, as probreg's default). ``sigma_init`` (units of the
    input) replaces the CPD start ``sigma^2 = mean of all squared pair distances``: that
    start pulls every point towards the centroid first, which on pre-aligned bones lets
    the points drift along the surface. Memory ~ M x N doubles.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    mu = dst.mean(0)
    s = float(np.sqrt(np.mean(np.sum((dst - mu) ** 2, axis=1))))
    x = (dst - mu) / s
    y = (src - mu) / s
    m, d = y.shape
    n = len(x)
    g = np.exp(-_sqdist(y, y) / (2.0 * beta**2))
    wts = np.zeros_like(y)
    t = y
    if sigma_init is None:
        sigma2 = float(_sqdist(y, x).sum() / (d * m * n))
    else:
        sigma2 = (sigma_init / s) ** 2
    xx = np.sum(x**2, axis=1)
    it = 0
    while it < iterations:
        it += 1
        p = np.exp(-_sqdist(t, x) / (2.0 * sigma2))
        c = (2.0 * np.pi * sigma2) ** (d / 2.0) * w / (1.0 - w) * m / n
        den = p.sum(0) + c
        p /= np.where(den > 0, den, np.finfo(float).tiny)
        p1, pt1 = p.sum(1), p.sum(0)
        np_ = float(p1.sum())
        px = p @ x
        # symmetric form of (d(P1) G + lam sigma2 I) W = PX - d(P1) Y
        inv = 1.0 / np.maximum(p1, 1e-12)
        factor = cho_factor(g + lam * sigma2 * np.diag(inv))
        wts = cho_solve(factor, inv[:, None] * px - y)
        t = y + g @ wts
        new = float(
            (xx @ pt1 - 2.0 * np.sum(px * t) + np.sum(p1 * np.sum(t**2, axis=1))) / (np_ * d)
        )
        new = max(new, 1e-10)
        done = abs(sigma2 - new) < tol
        sigma2 = new
        if done:
            break
    warp = GaussianWarp(centers=src, weights=wts * s, beta=beta * s)
    return CpdResult(warp=warp, iterations=it, sigma2=sigma2 * s**2, beta=beta * s)


def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum((a**2).sum(1)[:, None] + (b**2).sum(1)[None, :] - 2.0 * a @ b.T, 0.0)


__all__ = [
    "CpdResult",
    "GaussianWarp",
    "RigidResult",
    "apply",
    "fit_coherent_icp",
    "fit_cpd",
    "sample_surface",
    "similarity_icp",
    "umeyama",
]
