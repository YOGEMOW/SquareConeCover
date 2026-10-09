#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E0005 三角块雅可比校验 + 翻转防护 + 过程渲染图"""

from __future__ import annotations
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import E0002_regularization_prototype as core
import E0004_vectorized_assembly as v4
import E0004_render_figures as r4

OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "experiments", "results", "E0005")
os.makedirs(OUT, exist_ok=True)
W0 = dict(core.W_DEFAULT)


def tri_faces(mesh):
    T = [list(map(int, f)) for f in mesh.tris]
    for q in mesh.quads:
        T.append([int(q[0]), int(q[1]), int(q[2])])
        T.append([int(q[0]), int(q[2]), int(q[3])])
    return np.asarray(T, int) if T else np.zeros((0, 3), int)


def normals(V, T):
    A = V[T]
    return np.cross(A[:, 1] - A[:, 0], A[:, 2] - A[:, 0])


def flip_count(V, V0, T):
    if len(T) == 0:
        return 0
    return int(np.sum(np.einsum("ij,ij->i", normals(V, T), normals(V0, T)) < 0))


def coplanarity(V):
    Q = V - V.mean(0)
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    return float(np.abs(Q @ vt[2]).max())


def jacobian_check():
    m = core.make_flat_tri(nx=3, ny=3, noise=0.05, rule="longest")
    core.prep(m)
    W = dict(W0); W["crease"] = 0.0
    r1, (rows1, cols1, vals1) = core.residuals(m, m.V, W)
    r2, (rows2, cols2, vals2), st = v4.residuals_vec(m, m.V, W)
    jac_scalar = np.zeros((len(r1), 3 * m.nfree)); jac_vec = np.zeros_like(jac_scalar)
    fmap = m.fmap
    for i in range(len(vals1)):
        if fmap[cols1[i] // 3] >= 0:
            jac_scalar[rows1[i], 3 * fmap[cols1[i] // 3] + cols1[i] % 3] += vals1[i]
    for i in range(len(vals2)):
        jac_vec[rows2[i], cols2[i]] += vals2[i]
    eps = 1e-6
    errs = []
    for k in range(0, 3 * m.nfree, 7):
        Vp = m.V.copy().reshape(-1); Vp[k] += eps
        Vm = m.V.copy().reshape(-1); Vm[k] -= eps
        fd = (core.residuals(m, Vp.reshape(-1, 3), W, with_jac=False)[0]
              - core.residuals(m, Vm.reshape(-1, 3), W, with_jac=False)[0]) / (2 * eps)
        errs.append(max(np.abs(jac_scalar[:, k] - fd).max(), np.abs(jac_vec[:, k] - fd).max()))
    # 折痕块单独校验（向量化实现）
    Wc = dict(W0); Wc["crease"] = 2.0
    rc, (rowsc, colsc, valsc), _ = v4.residuals_vec(m, m.V, Wc, None)
    jc = np.zeros((len(rc), 3 * m.nfree))
    for i in range(len(valsc)):
        jc[rowsc[i], colsc[i]] += valsc[i]
    errsc = []
    for k in range(0, 3 * m.nfree, 5):
        Vp = m.V.copy().reshape(-1); Vp[k] += eps
        Vm = m.V.copy().reshape(-1); Vm[k] -= eps
        fd = (v4.residuals_vec(m, Vp.reshape(-1, 3), Wc, None)[0]
              - v4.residuals_vec(m, Vm.reshape(-1, 3), Wc, None)[0]) / (2 * eps)
        errsc.append(np.abs(jc[:, k] - fd).max())
    return {"residual_diff": float(np.abs(r1 - r2).max()), "jac_scalar_vs_fd_max": float(np.max(errs)),
            "jac_crease_vs_fd_max": float(np.max(errsc)),
            "n_tri": int(len(m.tris)), "n_res": int(len(r1))}


def solve_guarded(mesh, w, iters=40, pcg_maxit=200, pcg_tol=1e-12, ckpt=(), on_ckpt=None):
    T = tri_faces(mesh)
    V0 = mesh.V0.copy()
    st = v4.prep_struct(mesh, w)
    r, (rows, cols, vals), _ = v4.residuals_vec(mesh, mesh.V, w, st)
    energy = 0.5 * float(r @ r)
    V = mesh.V.copy()
    lam = 1e-4
    hist = []
    flips = flip_count(V, V0, T)
    hist.append({"iter": 0, "energy": energy, "flips": flips,
                 "leg_p95": core.tri_metrics(V, mesh.tris, mesh.bases)["leg_rel_p95"] if len(mesh.tris) else 0.0,
                 "copl": coplanarity(V)})
    if on_ckpt:
        on_ckpt(0, V, hist[-1])
    t0 = time.time()
    for it in range(1, iters + 1):
        Jt = core.spmv(cols, rows, vals, -r, len(r) and 3 * mesh.nfree)
        diag = np.bincount(cols, weights=vals * vals, minlength=3 * mesh.nfree) + lam
        Hf = lambda x: core.spmv(cols, rows, vals, core.spmv(rows, cols, vals, x, len(r)), 3 * mesh.nfree) + lam * x
        dx = core.pcg(Hf, Jt, np.maximum(diag, 1e-30), tol=pcg_tol, maxit=pcg_maxit)
        step, improved = 1.0, False
        for _ in range(20):
            Vt = V.copy()
            Vt[mesh.free] = V[mesh.free] + (step * dx).reshape(-1, 3)
            if flip_count(Vt, V0, T) <= flips:          # 翻转防护：不允许新增翻转面
                rt, _, _ = v4.residuals_vec(mesh, Vt, w, st)
                et = 0.5 * float(rt @ rt)
                if et < energy * (1 - 1e-14):
                    V, energy = Vt, et
                    lam = max(lam * 0.5, 1e-12)
                    improved = True
                    break
            step *= 0.5
        if not improved:
            lam *= 10.0
        r, (rows, cols, vals), _ = v4.residuals_vec(mesh, V, w, st)
        flips = flip_count(V, V0, T)
        rec = {"iter": it, "energy": energy, "flips": flips,
               "leg_p95": core.tri_metrics(V, mesh.tris, mesh.bases)["leg_rel_p95"] if len(mesh.tris) else 0.0,
               "copl": coplanarity(V)}
        hist.append(rec)
        if it in ckpt and on_ckpt:
            on_ckpt(it, V, rec)
        if len(hist) > 6 and abs(hist[-2]["energy"] - hist[-1]["energy"]) < 1e-13 * max(1.0, energy):
            break
    mesh.V = V
    return {"iters": it, "energy": energy, "flips": flips, "wall_s": time.time() - t0, "history": hist}


def main():
    chk = jacobian_check()
    print("雅可比校验:", chk)

    mesh = core.make_flat_tri(nx=40, ny=40, noise=0.01, rule="longest")
    core.prep(mesh)
    mesh.V = core.presmooth(mesh, mesh.V, passes=5)
    W = dict(W0); W.update(iso=1.0, area=1.0, smooth=0.3, pos=1e-6, crease=0.5)
    panels = []

    def on_ckpt(it, V, rec):
        img = r4.render_mesh(V, mesh.tris, size=(430, 360), view=(0.85, -1.15, 0.95),
                             title=f"iter {it}: leg_p95={rec['leg_p95']*100:.2f}% copl={rec['copl']:.4f} flips={rec['flips']}")
        panels.append(img)

    info = solve_guarded(mesh, W, iters=40, pcg_maxit=200, pcg_tol=1e-12,
                         ckpt=(2, 5, 10, 20, 40), on_ckpt=on_ckpt)
    tm = core.tri_metrics(mesh.V, mesh.tris, mesh.bases)
    print(f"结果: iters={info['iters']} leg_p95={tm['leg_rel_p95']:.4f} min_angle={tm['min_angle_deg']:.2f} "
          f"copl={coplanarity(mesh.V):.5f} flips={info['flips']} wall={info['wall_s']:.1f}s")
    if panels:
        w = sum(p.width for p in panels) + 8 * (len(panels) - 1)
        h = max(p.height for p in panels)
        from PIL import Image, ImageDraw
        canvas = Image.new("RGB", (w, h + 20), (255, 255, 255))
        dr = ImageDraw.Draw(canvas)
        x = 0
        for p, rec in zip(panels, [h for h in info["history"] if h["iter"] in (0, 2, 5, 10, 20, 40)][:len(panels)]):
            canvas.paste(p, (x, 0))
            x += p.width + 8
        canvas.save(os.path.join(OUT, "fig-tri-process.png"))
    # 曲线图
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (860, 420), (255, 255, 255)); dr = ImageDraw.Draw(img)
    h = info["history"]
    it = [r["iter"] for r in h]
    def px(i): return 70 + (i - min(it)) / max(1, max(it) - min(it)) * 740
    def py(v, vmax): return 360 - v / max(vmax, 1e-9) * 300
    vmax = max(max(r["leg_p95"] for r in h), 1e-6)
    prev = None
    for r in h:
        p = (px(r["iter"]), py(r["leg_p95"], vmax))
        if prev: dr.line([prev, p], fill=(200, 60, 60), width=3)
        prev = p
    prev = None
    cmax = max(r["copl"] for r in h)
    for r in h:
        p = (px(r["iter"]), py(r["copl"], cmax))
        if prev: dr.line([prev, p], fill=(40, 90, 170), width=3)
        prev = p
    prev = None
    fmax = max(1, max(r["flips"] for r in h))
    for r in h:
        p = (px(r["iter"]), py(r["flips"], fmax))
        if prev: dr.line([prev, p], fill=(30, 140, 70), width=2)
        prev = p
    dr.text((70, 10), "E0005 flip-guarded solve: red=leg_p95  blue=coplanarity  green=flips (x=iteration)", fill=(20, 20, 20))
    img.save(os.path.join(OUT, "fig-tri-metrics.png"))
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"jacobian_check": chk, "solve": {k: v for k, v in info.items() if k != "history"},
                   "history": info["history"]}, f, ensure_ascii=False, indent=2)
    print("renders ->", OUT)


if __name__ == "__main__":
    main()