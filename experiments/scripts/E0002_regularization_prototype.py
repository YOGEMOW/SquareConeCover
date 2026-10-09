#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E0002 网格规整化原型 v2（Stage A / A2 位置规整）

纯 numpy（无 scipy），验证 docs/08 的能量与两条求解路线。v2 的关键修正：
  1. 面积约束改为"全局面积项"为主（逐面面积项默认关闭），避免与形状规整化互相拉扯；
  2. 平滑项改用真正的拉普拉斯形式 sum |v - mean(N(v))|^2（平面网格上为零），
     替换 v1 的"膜能量" Sum |v_i - v_j|^2（后者会推动收缩、与形状约束冲突）；
  3. 优化前加少量拉普拉斯预平滑，改善初值。

运行：
  python experiments/scripts/E0002_regularization_prototype.py quick   # 只跑 T1 诊断
  python experiments/scripts/E0002_regularization_prototype.py        # 全量
输出：experiments/results/E0002/{results.json, results.csv, report.md, *.obj}
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(REPO, "experiments", "results", "E0002")
RNG = np.random.default_rng(20261009)


class Mesh:
    def __init__(self, V, quads=None, tris=None, bases=None, anchors=None, name=""):
        self.V0 = np.asarray(V, float)
        self.V = self.V0.copy()
        self.quads = np.zeros((0, 4), int) if quads is None else np.asarray(quads, int)
        self.tris = np.zeros((0, 3), int) if tris is None else np.asarray(tris, int)
        self.bases = bases
        self.anchors = np.zeros(len(self.V0), bool) if anchors is None else np.asarray(anchors, bool)
        self.name = name
        self._h = None

    def scale(self):
        if self._h is None:
            E = mesh_edges(self)
            self._h = float(np.mean(np.linalg.norm(self.V0[E[:, 0]] - self.V0[E[:, 1]], axis=1)))
        return self._h


def mesh_edges(mesh):
    E = set()
    for f in mesh.quads:
        for k in range(4):
            E.add(tuple(sorted((int(f[k]), int(f[(k + 1) % 4])))))
    for f in mesh.tris:
        for k in range(3):
            E.add(tuple(sorted((int(f[k]), int(f[(k + 1) % 3])))))
    return np.array(sorted(E), int)


def adjacency(nv, E):
    nbr = [[] for _ in range(nv)]
    for (i, j) in E:
        nbr[int(i)].append(int(j))
        nbr[int(j)].append(int(i))
    return nbr


def face_areas(V, faces):
    if len(faces) == 0:
        return np.zeros(0)
    A = V[faces]
    acc = np.zeros((len(faces), 3))
    for k in range(1, faces.shape[1] - 1):
        acc += np.cross(A[:, k] - A[:, 0], A[:, k + 1] - A[:, 0])
    return 0.5 * np.linalg.norm(acc, axis=1)


def quad_area_grads(V, quads):
    """返回每个四边面 dA/dv，形如 (m,4,3)（面是凸四边形时有效）。"""
    P = V[quads]
    d1 = P[:, 2] - P[:, 0]
    d2 = P[:, 3] - P[:, 1]
    g = np.cross(d1, d2)
    gn = np.linalg.norm(g, axis=1)
    gh = g / (gn[:, None] + 1e-30)
    gd1 = 0.5 * np.cross(d2, gh)
    gd2 = 0.5 * np.cross(gh, d1)
    out = np.zeros((len(quads), 4, 3))
    out[:, 0] = -gd1
    out[:, 2] = gd1
    out[:, 1] = -gd2
    out[:, 3] = gd2
    return out


def tri_area_grads(V, tris):
    P = V[tris]
    e1 = P[:, 1] - P[:, 0]
    e2 = P[:, 2] - P[:, 0]
    g = np.cross(e1, e2)
    gn = np.linalg.norm(g, axis=1)
    gh = g / (gn[:, None] + 1e-30)
    g1 = 0.5 * np.cross(e2, gh)
    g2 = 0.5 * np.cross(gh, e1)
    out = np.zeros((len(tris), 3, 3))
    out[:, 1] = g1
    out[:, 2] = g2
    out[:, 0] = -(g1 + g2)
    return out


# ---------------------------------------------------------------- 网格构造
def grid_quads(nx, ny):
    idx = lambda i, j: j * (nx + 1) + i
    return np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)]
                     for j in range(ny) for i in range(nx)], int)


def boundary_mask(nx, ny):
    m = np.zeros((ny + 1, nx + 1), bool)
    m[0, :] = m[-1, :] = True
    m[:, 0] = m[:, -1] = True
    return m.reshape(-1)


def make_flat_quad(nx=12, ny=12, noise=0.03):
    xs = np.linspace(0, 1, nx + 1)
    ys = np.linspace(0, 1, ny + 1)
    V = np.array([[x, y, 0.0] for y in ys for x in xs])
    m = boundary_mask(nx, ny)
    jit = noise * (RNG.random(V.shape) - 0.5)
    jit[:, 2] *= 3.0
    V[~m] += jit[~m]
    return Mesh(V, quads=grid_quads(nx, ny), anchors=m, name="T1-flat-quad")


def make_cylinder_quad(nx=12, ny=10, R=1.0, ang=1.2, noise=0.01, anchor="rail0"):
    xs = np.linspace(-ang / 2, ang / 2, nx + 1)
    ys = np.linspace(0, 1.2, ny + 1)
    V = np.array([[R * math.sin(a), y, R * math.cos(a)] for y in ys for a in xs])
    jit = noise * (RNG.random(V.shape) - 0.5)
    m = boundary_mask(nx, ny)
    V[~m] += jit[~m]
    rails = np.zeros((ny + 1, nx + 1), bool)
    if anchor == "rail0":
        rails[:, 0] = True
        name = "T2-cylinder-quad"
    else:
        rails[:, 0] = rails[:, -1] = True
        name = "T2b-cylinder-quad-both-rails"
    return Mesh(V, quads=grid_quads(nx, ny), anchors=rails.reshape(-1), name=name)


def make_sphere_cap_quad(nx=10, ny=10, R=1.0, psi0=-0.5, psi1=0.5, th0=0.7, th1=1.2, noise=0.005):
    psis = np.linspace(psi0, psi1, nx + 1)
    ths = np.linspace(th0, th1, ny + 1)
    V = np.array([[R * math.sin(t) * math.cos(p), R * math.cos(t), R * math.sin(t) * math.sin(p)]
                  for t in ths for p in psis])
    jit = noise * (RNG.random(V.shape) - 0.5)
    m = boundary_mask(nx, ny)
    V[~m] += jit[~m]
    return Mesh(V, quads=grid_quads(nx, ny), anchors=m, name="T3-spherecap-quad")


def make_flat_tri(nx=12, ny=12, noise=0.03, rule="diagonal"):
    mesh = make_flat_quad(nx, ny, noise=noise)
    q = mesh.quads
    tris = np.vstack([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    if rule in ("shortest", "longest"):
        bases = np.zeros((len(tris), 2), int)
        for t, f in enumerate(tris):
            P = mesh.V[f]
            lens = [np.linalg.norm(P[1] - P[2]), np.linalg.norm(P[2] - P[0]), np.linalg.norm(P[0] - P[1])]
            k = int(np.argmin(lens) if rule == "shortest" else np.argmax(lens))
            bases[t] = [(k + 1) % 3, (k + 2) % 3]
        name = "T4b-flat-tri-shortest-base" if rule == "shortest" else "T4-flat-tri-longest-base"
    else:
        bases = np.zeros((len(tris), 2), int)
        bases[:len(q)] = [0, 2]
        bases[len(q):] = [0, 1]
        name = "T4-flat-tri"
    return Mesh(mesh.V, tris=tris, bases=bases, anchors=mesh.anchors, name=name)


# ---------------------------------------------------------------- 度量
def quad_metrics(V, quads):
    if len(quads) == 0:
        return {}
    P = V[quads]
    dev = []
    for k in range(4):
        a = P[:, k] - P[:, (k + 1) % 4]
        b = P[:, (k + 2) % 4] - P[:, (k + 1) % 4]
        ang = np.arccos(np.clip(np.einsum("ij,ij->i", a, b) /
                                (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30), -1, 1))
        dev.append(np.abs(np.degrees(ang) - 90.0))
    dev = np.concatenate(dev)
    e01 = np.linalg.norm(P[:, 1] - P[:, 0], axis=1)
    e12 = np.linalg.norm(P[:, 2] - P[:, 1], axis=1)
    e23 = np.linalg.norm(P[:, 3] - P[:, 2], axis=1)
    e30 = np.linalg.norm(P[:, 0] - P[:, 3], axis=1)
    esc = 0.5 * (e01 + e12 + e23 + e30) + 1e-30
    opp = np.concatenate([np.abs(e01 - e23) / esc, np.abs(e12 - e30) / esc])
    par = P[:, 0] + P[:, 2] - P[:, 1] - P[:, 3]
    par_rel = np.linalg.norm(par, axis=1) / (2 * esc)
    u = P[:, 1] - P[:, 0]
    w = P[:, 3] - P[:, 0]
    z = P[:, 2] - P[:, 0]
    vol = np.abs(np.einsum("ij,ij->i", np.cross(u, w), z))
    plan_rel = vol / (2 * esc) ** 3
    return {
        "angle_p95_deg": float(np.percentile(dev, 95)),
        "angle_max_deg": float(dev.max()),
        "opposite_rel_p95": float(np.percentile(opp, 95)),
        "parallel_rel_p95": float(np.percentile(par_rel, 95)),
        "planarity_rel_p95": float(np.percentile(plan_rel, 95)),
    }


def tri_metrics(V, tris, bases):
    if len(tris) == 0:
        return {}
    leg = []
    mn = []
    for t, f in enumerate(tris):
        ia, ib = int(f[bases[t][0]]), int(f[bases[t][1]])
        ic = int([x for x in f if x not in (ia, ib)][0])
        A, B, C = V[ia], V[ib], V[ic]
        base = np.linalg.norm(B - A) + 1e-30
        leg.append(abs(np.linalg.norm(C - A) - np.linalg.norm(C - B)) / base)
        for (p, q, r) in ((A, B, C), (B, C, A), (C, A, B)):
            u = q - p
            w = r - p
            mn.append(math.degrees(math.acos(max(-1.0, min(1.0, float(u @ w) / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-30))))))
    return {"leg_rel_p95": float(np.percentile(leg, 95)),
            "leg_rel_max": float(np.max(leg)),
            "min_angle_deg": float(np.min(mn))}


def flip_count(V, V0, faces):
    if len(faces) == 0:
        return 0
    def nrm(X):
        A = X[faces]
        acc = np.zeros((len(faces), 3))
        for k in range(1, faces.shape[1] - 1):
            acc += np.cross(A[:, k] - A[:, 0], A[:, k + 1] - A[:, 0])
        return acc
    return int(np.sum(np.einsum("ij,ij->i", nrm(V0), nrm(V)) < 0))


def vertex_defects(V, quads, tris):
    acc = {}
    def add(f):
        P = V[f]
        for k in range(len(f)):
            p = P[(k - 1) % len(f)] - P[k]
            q = P[(k + 1) % len(f)] - P[k]
            acc[int(f[k])] = acc.get(int(f[k]), 0.0) + math.acos(max(-1.0, min(1.0, float(p @ q) / (np.linalg.norm(p) * np.linalg.norm(q) + 1e-30))))
    for f in quads:
        add(f)
    for f in tris:
        add(f)
    d = np.array([2 * math.pi - acc[i] for i in sorted(acc)])
    return {"mean_defect_deg": float(np.degrees(d.mean())), "abs_mean_defect_deg": float(np.degrees(np.abs(d).mean()))}


# ---------------------------------------------------------------- 残差（路线 A）与线性最小二乘（路线 B）
def residuals(mesh, V, w, with_jac=True):
    nv = len(V)
    n = 3 * nv
    blocks, rows, cols, vals = [], [], [], []
    r0 = 0

    def push(row_vals, entries):
        nonlocal r0
        if with_jac:
            for (r, c, v) in entries:
                vtx = c // 3
                if mesh.fmap[vtx] < 0:
                    continue
                rows.append(r0 + r); cols.append(3 * int(mesh.fmap[vtx]) + (c % 3)); vals.append(v)
        blocks.append(np.asarray(row_vals, float))
        r0 += len(row_vals)

    h = mesh.scale()
    if len(mesh.quads):
        P = V[mesh.quads]
        wq = math.sqrt(w["par"]) / h
        for i, f in enumerate(mesh.quads):
            rv = wq * (P[i, 0] + P[i, 2] - P[i, 1] - P[i, 3])
            e = []
            for k in range(3):
                e += [(k, 3 * int(f[0]) + k, wq), (k, 3 * int(f[1]) + k, -wq),
                      (k, 3 * int(f[2]) + k, wq), (k, 3 * int(f[3]) + k, -wq)]
            push(rv, e)
        wd = math.sqrt(w["diag"]) / (h * h)
        for i, f in enumerate(mesh.quads):
            d1 = P[i, 0] - P[i, 2]
            d2 = P[i, 1] - P[i, 3]
            rv = np.array([wd * (float(d1 @ d1) - float(d2 @ d2))])
            e = []
            for k in range(3):
                e += [(0, 3 * int(f[0]) + k, 2 * wd * d1[k]), (0, 3 * int(f[2]) + k, -2 * wd * d1[k]),
                      (0, 3 * int(f[1]) + k, -2 * wd * d2[k]), (0, 3 * int(f[3]) + k, 2 * wd * d2[k])]
            push(rv, e)
        wp = math.sqrt(w["plan"]) / (h ** 3)
        for i, f in enumerate(mesh.quads):
            u = P[i, 1] - P[i, 0]
            wv = P[i, 3] - P[i, 0]
            z = P[i, 2] - P[i, 0]
            rv = np.array([wp * float(np.cross(u, wv) @ z)])
            gu = np.cross(wv, z) * wp
            gw = np.cross(z, u) * wp
            gz = np.cross(u, wv) * wp
            e = []
            for k in range(3):
                e += [(0, 3 * int(f[1]) + k, gu[k]), (0, 3 * int(f[3]) + k, gw[k]),
                      (0, 3 * int(f[2]) + k, gz[k]), (0, 3 * int(f[0]) + k, -(gu[k] + gw[k] + gz[k]))]
            push(rv, e)
    if len(mesh.tris):
        h2 = h * h
        for i, f in enumerate(mesh.tris):
            ia, ib = int(f[mesh.bases[i][0]]), int(f[mesh.bases[i][1]])
            ic = int([x for x in f if x not in (ia, ib)][0])
            a, b, c = V[ia], V[ib], V[ic]
            ca = c - a
            cb = c - b
            wi = math.sqrt(w["iso"]) / (h2 * h2)
            rv = np.array([wi * (float(ca @ ca) - float(cb @ cb))])
            e = []
            for k in range(3):
                e += [(0, 3 * ic + k, 2.0 * wi * (ca[k] - cb[k])),
                      (0, 3 * ib + k, 2.0 * wi * cb[k]),
                      (0, 3 * ia + k, -2.0 * wi * ca[k])]
            push(rv, e)
    # 全局面积项
    if w.get("area", 0.0) > 0 and (len(mesh.quads) or len(mesh.tris)):
        A0 = mesh.A0_total + 1e-30
        Aq = face_areas(V, mesh.quads)
        At = face_areas(V, mesh.tris)
        wa = math.sqrt(w["area"]) / A0
        rv = np.array([wa * (Aq.sum() + At.sum() - mesh.A0_total)])
        gq = quad_area_grads(V, mesh.quads) if len(mesh.quads) else np.zeros((0, 4, 3))
        gt = tri_area_grads(V, mesh.tris) if len(mesh.tris) else np.zeros((0, 3, 3))
        e = []
        for i, f in enumerate(mesh.quads):
            for k in range(4):
                for c in range(3):
                    e.append((0, 3 * int(f[k]) + c, wa * gq[i, k, c]))
        for i, f in enumerate(mesh.tris):
            for k in range(3):
                for c in range(3):
                    e.append((0, 3 * int(f[k]) + c, wa * gt[i, k, c]))
        push(rv, e)
    # 拉普拉斯平滑（仅自由顶点）
    if w.get("smooth", 0.0) > 0:
        ws = math.sqrt(w["smooth"]) / h
        for v in range(nv):
            if mesh.anchors[v]:
                continue
            nb = mesh.nbr[v]
            if not nb:
                continue
            d = len(nb)
            mean = V[nb].mean(axis=0)
            rv = ws * (V[v] - mean)
            e = []
            for k in range(3):
                e.append((k, 3 * v + k, ws))
                for j in nb:
                    e.append((k, 3 * int(j) + k, -ws / d))
            push(rv, e)
    # 位置项（仅自由顶点，防整体漂移）
    if w.get("pos", 0.0) > 0:
        wpv = math.sqrt(w["pos"]) / h
        for v in range(nv):
            if mesh.anchors[v]:
                continue
            rv = wpv * (V[v] - mesh.V0[v])
            push(rv, [(k, 3 * v + k, wpv) for k in range(3)])
    r = np.concatenate(blocks) if blocks else np.zeros(0)
    J = (np.array(rows, int), np.array(cols, int), np.array(vals, float)) if rows else (np.zeros(0, int), np.zeros(0, int), np.zeros(0))
    return r, J


def spmv(rows, cols, vals, x, n):
    return np.bincount(rows, weights=vals * x[cols], minlength=n)


def cg(matvec, b, tol=1e-12, maxit=800):
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs = float(r @ r)
    bnorm = max(float(b @ b), 1e-30)
    for _ in range(maxit):
        Ap = matvec(p)
        den = float(p @ Ap)
        if den <= 1e-30:
            break
        a = rs / den
        x += a * p
        r -= a * Ap
        rs2 = float(r @ r)
        if rs2 / bnorm < tol:
            break
        p = r + (rs2 / rs) * p
        rs = rs2
    return x



def pcg(matvec, b, diag, tol=1e-12, maxit=3000):
    """Jacobi 预条件共轭梯度（对角预条件）。"""
    x = np.zeros_like(b)
    r = b.copy()
    z = r / diag
    p = z.copy()
    rz = float(r @ z)
    bnorm = max(float(b @ b), 1e-30)
    for _ in range(maxit):
        Ap = matvec(p)
        den = float(p @ Ap)
        if den <= 1e-30:
            break
        a = rz / den
        x += a * p
        r -= a * Ap
        if float(r @ r) / bnorm < tol:
            break
        z = r / diag
        rz2 = float(r @ z)
        if rz <= 1e-300:
            break
        p = z + (rz2 / rz) * p
        rz = rz2
    return x
    V = V.copy()
    for _ in range(passes):
        Vn = V.copy()
        for v in range(len(V)):
            if mesh.anchors[v] or not mesh.nbr[v]:
                continue
            Vn[v] = (1 - omega) * V[v] + omega * V[mesh.nbr[v]].mean(axis=0)
        V = Vn
    return V


def presmooth(mesh, V, passes=3, omega=0.3):
    """少量拉普拉斯预平滑（仅移动自由顶点），用于改善优化初值。"""
    V = V.copy()
    for _ in range(passes):
        Vn = V.copy()
        for v in range(len(V)):
            if mesh.anchors[v] or not mesh.nbr[v]:
                continue
            Vn[v] = (1 - omega) * V[v] + omega * V[mesh.nbr[v]].mean(axis=0)
        V = Vn
    return V

def prep(mesh):
    mesh.E = mesh_edges(mesh)
    mesh.nbr = adjacency(len(mesh.V0), mesh.E)
    mesh.A0_quads = face_areas(mesh.V0, mesh.quads)
    mesh.A0_tris = face_areas(mesh.V0, mesh.tris)
    mesh.A0_total = float(mesh.A0_quads.sum() + mesh.A0_tris.sum())
    mesh.free = ~mesh.anchors
    mesh.fmap = np.full(len(mesh.V0), -1, int)
    mesh.fmap[mesh.free] = np.arange(int(mesh.free.sum()))
    mesh.nfree = int(mesh.free.sum())
    return mesh


def solve_local_global(mesh, w, iters=300, omega=0.7):
    V = mesh.V.copy()
    nv = len(V)
    free = mesh.free
    fmap = mesh.fmap
    trace = []
    _r0, _ = residuals(mesh, V, w, with_jac=False)
    trace.append(0.5 * float(_r0 @ _r0))
    for it in range(iters):
        Tq = quad_targets(V, mesh)
        Tt, order = tri_targets(V, mesh)
        rowlist = []
        for i, f in enumerate(mesh.quads):
            for k in range(4):
                for c in range(3):
                    rowlist.append(([(int(f[k]), c, 1.0)], float(Tq[i, k, c])))
        for i in range(len(mesh.tris)):
            for k in range(3):
                for c in range(3):
                    rowlist.append(([(int(order[i, k]), c, 1.0)], float(Tt[i, k, c])))
        if w.get("smooth", 0.0) > 0:
            ws = math.sqrt(w["smooth"]) / mesh.scale()
            for v in range(nv):
                if mesh.anchors[v] or not mesh.nbr[v]:
                    continue
                d = len(mesh.nbr[v])
                for c in range(3):
                    coefs = [(v, c, ws)] + [(int(j), c, -ws / d) for j in mesh.nbr[v]]
                    rowlist.append((coefs, 0.0))
        if w.get("pos", 0.0) > 0:
            wpv = math.sqrt(w["pos"]) / mesh.scale()
            for v in range(nv):
                if mesh.anchors[v]:
                    continue
                for c in range(3):
                    rowlist.append(([(v, c, wpv)], wpv * float(mesh.V0[v, c])))
        rows, cols, vals, yv = [], [], [], []
        r0 = 0
        for coefs, val in rowlist:
            ent = [(v, c, co) for (v, c, co) in coefs if free[v]]
            if not ent:
                continue
            const = 0.0
            for (v, c, co) in coefs:
                if not free[v]:
                    const += co * float(V[v, c])
            for (v, c, co) in ent:
                rows.append(r0); cols.append(3 * int(fmap[v]) + c); vals.append(co)
            yv.append(val - const)
            r0 += 1
        rows = np.array(rows, int); cols = np.array(cols, int); vals = np.array(vals, float); yv = np.array(yv, float)
        n = 3 * mesh.nfree
        b = spmv(cols, rows, vals, yv, n)
        lam = 1e-9
        mv = lambda x: spmv(cols, rows, vals, spmv(rows, cols, vals, x, len(yv)), n) + lam * x
        diag = np.bincount(cols, weights=vals * vals, minlength=n) + lam
        x = pcg(mv, b, np.maximum(diag, 1e-30), tol=1e-12, maxit=3000)
        xs = x.reshape(-1, 3)
        Vn = V.copy()
        Vn[free] = V[free] + omega * (xs - V[free])
        dmove = float(np.abs(Vn - V).max())
        V = Vn
        mesh.V = V
        rt, _ = residuals(mesh, V, w, with_jac=False)
        trace.append(0.5 * float(rt @ rt))
        if dmove < 1e-13:
            break
    mesh.V = V
    return {"iterations": it + 1, "energy_start": trace[0], "energy_end": trace[-1], "trace": trace}

def solve_gauss_newton(mesh, w, iters=100, lam=1e-4):
    V = mesh.V.copy()
    r, (rows, cols, vals) = residuals(mesh, V, w)
    energy = 0.5 * float(r @ r)
    n = 3 * mesh.nfree
    trace = [energy]
    if energy < 1e-18:
        mesh.V = V
        return {"iterations": 0, "energy_start": energy, "energy_end": energy, "trace": trace}
    for it in range(iters):
        Jt = spmv(cols, rows, vals, -r, n)
        if float(Jt @ Jt) < 1e-24:
            break
        Hf = lambda x: spmv(cols, rows, vals, spmv(rows, cols, vals, x, len(r)), n) + lam * x
        diag = np.bincount(cols, weights=vals * vals, minlength=n) + lam
        dx = pcg(Hf, Jt, np.maximum(diag, 1e-30), tol=1e-13, maxit=3000)
        step, improved = 1.0, False
        for _ in range(20):
            Vt = V.copy()
            Vt[mesh.free] = V[mesh.free] + (step * dx).reshape(-1, 3)
            rt, _ = residuals(mesh, Vt, w, with_jac=False)
            et = 0.5 * float(rt @ rt)
            if et < energy * (1 - 1e-14):
                V, energy = Vt, et
                lam = max(lam * 0.5, 1e-12)
                improved = True
                break
            step *= 0.5
        if not improved:
            lam *= 10.0
        r, (rows, cols, vals) = residuals(mesh, V, w)
        trace.append(energy)
        if len(trace) > 6 and abs(trace[-2] - trace[-1]) < 1e-12 * max(1.0, trace[-1]):
            break
    mesh.V = V
    return {"iterations": it + 1, "energy_start": trace[0], "energy_end": trace[-1], "trace": trace}

def quad_targets(V, mesh):
    if len(mesh.quads) == 0:
        return np.zeros((0, 4, 3))
    P = V[mesh.quads]
    cen = P.mean(axis=1)
    Q = P - cen[:, None, :]
    n = np.zeros((len(P), 3))
    for k in range(4):
        n += np.cross(P[:, k], P[:, (k + 1) % 4])
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-30)
    Qp = Q - np.einsum("mij,mj->mi", Q, n)[:, :, None] * n[:, None, :]
    ea = 0.5 * ((Qp[:, 1] - Qp[:, 0]) + (Qp[:, 2] - Qp[:, 3]))
    eb = 0.5 * ((Qp[:, 2] - Qp[:, 1]) + (Qp[:, 3] - Qp[:, 0]))
    u = ea / (np.linalg.norm(ea, axis=1, keepdims=True) + 1e-30)
    eb = eb - np.einsum("mi,mi->m", eb, u)[:, None] * u
    wv = eb / (np.linalg.norm(eb, axis=1, keepdims=True) + 1e-30)
    x = np.einsum("mij,mj->mi", Q, u)
    y = np.einsum("mij,mj->mi", Q, wv)
    a2 = np.abs(x).mean(axis=1)
    b2 = np.abs(y).mean(axis=1)
    A0 = mesh.A0_quads
    s = np.sqrt(np.where(4 * a2 * b2 > 1e-30, A0 / np.maximum(4 * a2 * b2, 1e-30), 1.0))
    a2 = a2 * s
    b2 = b2 * s
    sx = np.where(x >= 0, 1.0, -1.0)
    sy = np.where(y >= 0, 1.0, -1.0)
    return cen[:, None, :] + ((a2[:, None] * sx)[:, :, None] * u[:, None, :]) + ((b2[:, None] * sy)[:, :, None] * wv[:, None, :])


def tri_targets(V, mesh):
    if len(mesh.tris) == 0:
        return np.zeros((0, 3, 3)), np.zeros((0, 3), int)
    tris, bases = mesh.tris, mesh.bases
    t = len(tris)
    ridx = np.arange(t)
    ia = tris[ridx, bases[:, 0]]
    ib = tris[ridx, bases[:, 1]]
    keep = np.array([[k for k in range(3) if k != bases[i][0] and k != bases[i][1]][0] for i in ridx])
    ic = tris[ridx, keep]
    a, b, c = V[ia], V[ib], V[ic]
    m = 0.5 * (a + b)
    d = b - a
    dd = np.einsum("ij,ij->i", d, d) + 1e-30
    tpar = np.einsum("ij,ij->i", c - m, d) / dd
    c1 = c - tpar[:, None] * d
    L = np.sqrt(dd)
    A0 = mesh.A0_tris
    h0 = 2 * A0 / (L + 1e-30)
    perp = (c1 - m) - (np.einsum("ij,ij->i", c1 - m, d) / dd)[:, None] * d
    hp = np.linalg.norm(perp, axis=1)
    dirp = perp / (hp[:, None] + 1e-30)
    along = (np.einsum("ij,ij->i", c1 - m, d) / dd)[:, None] * d
    ct = m + along + dirp * h0[:, None]
    return np.stack([a, b, ct], axis=1), np.stack([ia, ib, ic], axis=1)


def evaluate(mesh, Vb, Va):
    out = {}
    if len(mesh.quads):
        out["quad_before"] = quad_metrics(Vb, mesh.quads)
        out["quad_after"] = quad_metrics(Va, mesh.quads)
    if len(mesh.tris):
        out["tri_before"] = tri_metrics(Vb, mesh.tris, mesh.bases)
        out["tri_after"] = tri_metrics(Va, mesh.tris, mesh.bases)
    Aq0, Aq1 = mesh.A0_quads, face_areas(Va, mesh.quads)
    At0, At1 = mesh.A0_tris, face_areas(Va, mesh.tris)
    per = []
    if len(Aq0):
        per.append(np.abs(Aq1 - Aq0) / (Aq0 + 1e-30))
    if len(At0):
        per.append(np.abs(At1 - At0) / (At0 + 1e-30))
    per = np.concatenate(per) if per else np.zeros(0)
    out["area"] = {"max_rel_drift": float(per.max()) if per.size else 0.0,
                   "p95_rel_drift": float(np.percentile(per, 95)) if per.size else 0.0,
                   "global_rel_drift": float((Aq1.sum() + At1.sum() - mesh.A0_total) / (mesh.A0_total + 1e-30))}
    out["flips"] = flip_count(Va, Vb, mesh.quads) + flip_count(Va, Vb, mesh.tris)
    out["anchor_move"] = float(np.abs(Va[mesh.anchors] - Vb[mesh.anchors]).max()) if mesh.anchors.any() else 0.0
    out["max_disp"] = float(np.linalg.norm(Va - Vb, axis=1).max())
    out["defect"] = vertex_defects(Va, mesh.quads, mesh.tris)
    return out


def write_obj(path, mesh, V):
    with open(path, "w", encoding="utf-8") as f:
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for q in mesh.quads:
            f.write("f " + " ".join(str(int(i) + 1) for i in q) + "\n")
        for t in mesh.tris:
            f.write("f " + " ".join(str(int(i) + 1) for i in t) + "\n")


W_DEFAULT = {"par": 1.0, "diag": 1.0, "plan": 1.0, "iso": 1.0,
             "area": 100.0, "smooth": 0.3, "pos": 1e-6}


def run_test(mesh, routes=("B", "A"), w=None, iters=(300, 60), presmooth_passes=3):
    w = dict(w or W_DEFAULT)
    prep(mesh)
    res = {"name": mesh.name, "n_vertices": int(len(mesh.V0)), "n_quads": int(len(mesh.quads)),
           "n_tris": int(len(mesh.tris)), "n_anchors": int(mesh.anchors.sum()), "weights": w}
    Vb = mesh.V0.copy()
    res["before"] = evaluate(mesh, Vb, Vb)
    t0 = time.time()
    if presmooth_passes:
        mesh.V = presmooth(mesh, mesh.V, passes=presmooth_passes)
    infos = {}
    for r in routes:
        if r == "B":
            infos["B"] = solve_local_global(mesh, w, iters=iters[0])
        else:
            infos["A"] = solve_gauss_newton(mesh, w, iters=iters[1])
    res["solver"] = infos
    res["runtime_s"] = time.time() - t0
    res["after"] = evaluate(mesh, Vb, mesh.V.copy())
    if len(mesh.V0) <= 400 and mesh.name == "T1-flat-quad":
        m2 = Mesh(mesh.V0, quads=mesh.quads, tris=mesh.tris, bases=mesh.bases, anchors=mesh.anchors, name=mesh.name)
        prep(m2)
        if presmooth_passes:
            m2.V = presmooth(m2, m2.V, passes=presmooth_passes)
        for r in routes:
            if r == "B":
                solve_local_global(m2, w, iters=iters[0])
            else:
                solve_gauss_newton(m2, w, iters=iters[1])
        res["determinism_max_diff"] = float(np.abs(m2.V - mesh.V).max())
    write_obj(os.path.join(OUTDIR, mesh.name + "-before.obj"), mesh, Vb)
    write_obj(os.path.join(OUTDIR, mesh.name + "-after.obj"), mesh, mesh.V)
    return res


def main():
    quick = len(sys.argv) > 1 and sys.argv[1] == "quick"
    os.makedirs(OUTDIR, exist_ok=True)
    results = []
    print("=" * 84)
    if quick:
        m = prep(make_flat_quad())
        r = run_test(m)
        print(json.dumps({k: v for k, v in r["after"].items() if k != "defect"}, ensure_ascii=False, indent=1))
        print("solver:", {k: {"iters": v["iterations"], "E0": v["energy_start"], "E1": v["energy_end"]}
                          for k, v in r["solver"].items()})
        return
    results.append(run_test(make_flat_quad()))
    for tag, routes in (("T1-routeB-only", ("B",)), ("T1-routeA-only", ("A",))):
        mm = make_flat_quad()
        rr = run_test(mm, routes=routes)
        rr["name"] = tag
        results.append(rr)
    results.append(run_test(make_cylinder_quad()))
    results.append(run_test(make_cylinder_quad(anchor="both")))
    results.append(run_test(make_sphere_cap_quad()))
    results.append(run_test(make_flat_tri(rule="diagonal")))
    results.append(run_test(make_flat_tri(rule="shortest")))
    for r in results:
        qa = r["after"].get("quad_after", {})
        ta = r["after"].get("tri_after", {})
        print(f"[{r['name']}] V={r['n_vertices']} Q={r['n_quads']} T={r['n_tris']} "
              f"angle_p95={qa.get('angle_p95_deg', float('nan')):.4f} "
              f"leg_p95={ta.get('leg_rel_p95', float('nan')):.4f} "
              f"flips={r['after']['flips']} area_p95={r['after']['area']['p95_rel_drift']:.2e} "
              f"t={r['runtime_s']:.2f}s")
    with open(os.path.join(OUTDIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    rows = []
    for r in results:
        row = {"test": r["name"], "vertices": r["n_vertices"], "quads": r["n_quads"], "tris": r["n_tris"],
               "runtime_s": round(r["runtime_s"], 3), "flips": r["after"]["flips"],
               "area_p95": r["after"]["area"]["p95_rel_drift"], "area_global": r["after"]["area"]["global_rel_drift"],
               "max_disp": r["after"]["max_disp"], "anchor_move": r["after"]["anchor_move"]}
        qa, ta = r["after"].get("quad_after"), r["after"].get("tri_after")
        if qa:
            row.update({"angle_p95_deg": qa["angle_p95_deg"], "angle_max_deg": qa["angle_max_deg"],
                        "opposite_p95": qa["opposite_rel_p95"], "par_p95": qa["parallel_rel_p95"],
                        "planarity_p95": qa["planarity_rel_p95"]})
        if ta:
            row.update({"leg_p95": ta["leg_rel_p95"], "leg_max": ta["leg_rel_max"], "min_angle_deg": ta["min_angle_deg"]})
        rows.append(row)
    keys = sorted({k for row in rows for k in row})
    with open(os.path.join(OUTDIR, "results.csv"), "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=keys)
        wtr.writeheader()
        for row in rows:
            wtr.writerow(row)
    print("results ->", OUTDIR)
    print("=" * 84)


if __name__ == "__main__":
    main()