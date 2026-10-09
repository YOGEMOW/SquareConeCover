#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E0006 雅可比缺陷二分定位"""
import sys, numpy as np
sys.path.insert(0, r"E:\Git\repositoris\SquareConeCover\experiments\scripts")
import E0002_regularization_prototype as core

W0 = dict(core.W_DEFAULT)

def blocks(mesh, W):
    nq, nt, nv = len(mesh.quads), len(mesh.tris), len(mesh.V0)
    cur = 0
    out = []
    for name, n in (("par", 3*nq), ("diag", nq), ("plan", nq), ("iso", nt)):
        out.append((name, cur, cur + n)); cur += n
    if W.get("area", 0) > 0 and (nq + nt):
        out.append(("area", cur, cur + 1)); cur += 1
    if W.get("crease", 0) > 0:
        T, pf, pg = None, None, None
        out.append(("crease", cur, cur + 999999))  # 仅占位
    lv = [v for v in range(nv) if mesh.free[v] and mesh.nbr[v]]
    if W.get("smooth", 0) > 0:
        out.append(("lap", cur, cur + 3*len(lv))); cur += 3*len(lv)
    fv = np.where(mesh.free)[0]
    if W.get("pos", 0) > 0:
        out.append(("pos", cur, cur + 3*len(fv))); cur += 3*len(fv)
    return out, cur

def check(W, tag):
    m = core.make_flat_tri(nx=3, ny=3, noise=0.05, rule="longest")
    core.prep(m)
    r0, (rows, cols, vals) = core.residuals(m, m.V, W)
    J = np.zeros((len(r0), 3 * m.nfree))
    for i in range(len(vals)):
        v = cols[i] // 3
        if m.fmap[v] >= 0:
            J[rows[i], 3 * m.fmap[v] + cols[i] % 3] += vals[i]
    eps = 1e-6
    maxerr = 0.0; worst = None
    for k in range(0, 3 * m.nfree, 3):
        Vp = m.V.copy().reshape(-1); Vp[k] += eps
        Vm = m.V.copy().reshape(-1); Vm[k] -= eps
        fd = (core.residuals(m, Vp.reshape(-1, 3), W, with_jac=False)[0]
              - core.residuals(m, Vm.reshape(-1, 3), W, with_jac=False)[0]) / (2 * eps)
        e = np.abs(J[:, k] - fd)
        if e.max() > maxerr:
            maxerr = float(e.max()); worst = (k, int(np.argmax(e)))
    bl, _ = blocks(m, W)
    def block_of(row):
        for name, a, b in bl:
            if a <= row < b:
                return name
        return "?"
    print(f"[{tag}] rows={len(r0)} nnz={len(vals)} max_err={maxerr:.3e} worst(col,row)=({worst[0]},{worst[1]}) block={block_of(worst[1])}")

base = dict(W0)
check(dict(base, iso=1.0, area=0.0, smooth=0.0, pos=0.0, crease=0.0), "iso only")
check(dict(base, iso=1.0, area=1.0, smooth=0.0, pos=0.0, crease=0.0), "iso+area")
check(dict(base, iso=1.0, area=0.0, smooth=0.3, pos=0.0, crease=0.0), "iso+lap")
check(dict(base, iso=1.0, area=0.0, smooth=0.0, pos=1e-6, crease=0.0), "iso+pos")
check(dict(base, iso=1.0, area=1.0, smooth=0.3, pos=1e-6, crease=0.0), "all(no crease)")