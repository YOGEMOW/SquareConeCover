#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E0004 向量化装配 + 规模复测（Stage A/A2）

内容：
  1. 向量化装配：把雅可比/线性系统的 rows/cols/vals 全部用 numpy 批量构造（不再用 Python 三级循环）；
  2. 正确性对照：向量化残差/雅可比 vs E0002 的标量实现（小网格，逐项比对）；
  3. 规模复测：20/40/60/100 平面四边网格的每次迭代耗时；
  4. 5 万顶点测试：约 4.9 万顶点/4.8 万四边面，验证 Q18 目标（<=10 s/轮）。

输出：experiments/results/E0004/{results.json, results.csv, report.md}
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import E0002_regularization_prototype as core

REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "experiments", "results", "E0004")
W = dict(core.W_DEFAULT)


def crease_struct(mesh):
    """内部边的面-面配对（把四边面拆成两个三角形），用于折痕/二面角正则。"""
    tris = [list(f) for f in mesh.tris]
    for q in mesh.quads:
        tris.append([int(q[0]), int(q[1]), int(q[2])])
        tris.append([int(q[0]), int(q[2]), int(q[3])])
    T = np.asarray(tris, int)
    if len(T) == 0:
        return T, np.zeros(0, int), np.zeros(0, int)
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=0)
    fid = np.tile(np.arange(len(T)), 3)
    key = np.sort(E, axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    k = key[order]
    same = np.all(k[1:] == k[:-1], axis=1)
    idx = np.where(same)[0]
    # 只取恰好出现两次的边
    _, counts = np.unique(k, axis=0, return_counts=True)
    OK = counts == 2
    pairs = []
    cnt = {}
    for i in range(len(k)):
        kk = (int(k[i, 0]), int(k[i, 1]))
        cnt.setdefault(kk, []).append(order[i])
    for kk, ids in cnt.items():
        if len(ids) == 2:
            pairs.append(ids)
    P = np.asarray(pairs, int) if pairs else np.zeros((0, 2), int)
    if len(P) == 0:
        return T, np.zeros(0, int), np.zeros(0, int)
    return T, fid[P[:, 0]], fid[P[:, 1]]


def skew(u):
    z = np.zeros((len(u), 3, 3))
    z[:, 0, 1] = -u[:, 2]; z[:, 0, 2] = u[:, 1]
    z[:, 1, 0] = u[:, 2];  z[:, 1, 2] = -u[:, 0]
    z[:, 2, 0] = -u[:, 1]; z[:, 2, 1] = u[:, 0]
    return z

def prep_struct(mesh, w):
    """预计算与坐标无关的索引结构。"""
    st = {}
    q, t = mesh.quads, mesh.tris
    mq, mt, nv = len(q), len(t), len(mesh.V0)
    h = mesh.scale()
    st.update(wq=math.sqrt(w["par"]) / h, wd=math.sqrt(w["diag"]) / (h * h),
              wp=math.sqrt(w["plan"]) / (h ** 3))
    st["wa"] = math.sqrt(w["area"]) / (mesh.A0_total + 1e-30) if w.get("area", 0) > 0 else 0.0
    st["ws"] = math.sqrt(w.get("smooth", 0.0)) / h
    st["wpos"] = math.sqrt(w.get("pos", 0.0)) / h

    def cols_of(faces):
        return (3 * faces[:, :, None] + np.arange(3)[None, None, :]).reshape(len(faces), -1)

    n_par = 3 * mq
    st["n_par"], st["n_diag"], st["n_plan"], st["n_iso"] = n_par, mq, mq, mt
    st["q_cols"] = cols_of(q) if mq else np.zeros((0, 12), int)
    st["t_cols"] = cols_of(t) if mt else np.zeros((0, 9), int)
    st["rows_par"] = (np.arange(mq)[:, None] * 3 + np.tile(np.arange(3), 4)[None, :]).ravel()
    st["rows_diag"] = n_par + np.arange(mq)
    st["rows_plan"] = n_par + mq + np.arange(mq)
    st["rows_iso"] = n_par + 2 * mq + np.arange(mt)
    st["sign_par"] = np.repeat(np.array([1.0, -1.0, 1.0, -1.0]), 3)
    # 三角形底边
    if mt:
        ridx = np.arange(mt)
        bases = mesh.bases
        st["ia"] = t[ridx, bases[:, 0]]
        st["ib"] = t[ridx, bases[:, 1]]
        keep = np.array([[k for k in range(3) if k != bases[i][0] and k != bases[i][1]][0] for i in ridx])
        st["ic"] = t[ridx, keep]
        st["wi"] = math.sqrt(w["iso"]) / (h * h)
        st["t_cols"] = (3 * np.stack([st["ia"], st["ib"], st["ic"]], axis=1)[:, :, None]
                        + np.arange(3)[None, None, :]).reshape(mt, 9)
    # 全局面积（1 行）
    st["row_area"] = n_par + 2 * mq + mt
    T, pf, pg = crease_struct(mesh)
    st["crease_T"], st["crease_pf"], st["crease_pg"] = T, pf, pg
    st["n_crease"] = len(pf)
    st["wc"] = math.sqrt(max(0.0, w.get("crease", 0.0)))
    if st["wc"] <= 0:
        st["n_crease"] = 0
    st["row_crease"] = n_par + 2 * mq + mt + (1 if (w.get("area", 0) > 0 and (mq + mt)) else 0)
    st["q_cols_nat"] = st["q_cols"]
    st["t_cols_nat"] = cols_of(t) if mt else np.zeros((0, 9), int)
    st["cols_area"] = np.concatenate([st["q_cols_nat"].ravel(), st["t_cols_nat"].ravel()]) if (mq + mt) else np.zeros(0, int)
    st["n_area"] = 1 if (w.get("area", 0) > 0 and (mq + mt)) else 0
    # 拉普拉斯（只含自由且有点邻接的顶点）
    lv = np.array([v for v in range(nv) if mesh.free[v] and mesh.nbr[v]], int)
    st["lap_v"] = lv
    st["lap_deg"] = np.array([len(mesh.nbr[v]) for v in lv], float) if len(lv) else np.zeros(0)
    st["lap_nbr"] = np.concatenate([np.array(mesh.nbr[int(v)], int) for v in lv]) if len(lv) else np.zeros(0, int)
    st["rows_crease"] = st["row_crease"] + np.arange(st["n_crease"])
    st["row_lap"] = st["row_crease"] + st["n_crease"]
    st["n_lap"] = 3 * len(lv) if st["ws"] > 0 else 0
    # 位置项（自由顶点）
    fv = np.where(mesh.free)[0]
    st["pos_v"] = fv
    st["row_pos"] = st["row_lap"] + st["n_lap"]
    st["n_pos"] = 3 * len(fv) if st["wpos"] > 0 else 0
    st["n_rows"] = st["row_pos"] + st["n_pos"]
    return st


def residuals_vec(mesh, V, w, st=None):
    if st is None:
        st = prep_struct(mesh, w)
    q, t = mesh.quads, mesh.tris
    mq, mt = len(q), len(t)
    blocks_r, blocks_rows, blocks_cols, blocks_vals = [], [], [], []

    def add(rvals, rows, cols, vals):
        blocks_r.append(np.asarray(rvals, float).ravel())
        if len(rows):
            blocks_rows.append(np.asarray(rows).ravel())
            blocks_cols.append(np.asarray(cols).ravel())
            blocks_vals.append(np.asarray(vals, float).ravel())

    if mq:
        P = V[q]
        rpar = (st["wq"] * (P[:, 0] + P[:, 2] - P[:, 1] - P[:, 3]))
        add(rpar, st["rows_par"], st["q_cols"], (st["wq"] * st["sign_par"])[None, :].repeat(mq, axis=0))
        d1 = P[:, 0] - P[:, 2]
        d2 = P[:, 1] - P[:, 3]
        add(st["wd"] * ((d1 * d1).sum(1) - (d2 * d2).sum(1)), np.repeat(st["rows_diag"], 12), st["q_cols"].ravel(),
            np.stack([2 * st["wd"] * d1, -2 * st["wd"] * d2, -2 * st["wd"] * d1, 2 * st["wd"] * d2], axis=1).reshape(mq, 12))
        u = P[:, 1] - P[:, 0]
        wv = P[:, 3] - P[:, 0]
        z = P[:, 2] - P[:, 0]
        add(st["wp"] * np.einsum("ij,ij->i", np.cross(u, wv), z), np.repeat(st["rows_plan"], 12), st["q_cols"].ravel(),
            np.stack([-(st["wp"] * (np.cross(wv, z) + np.cross(z, u) + np.cross(u, wv))),
                      st["wp"] * np.cross(wv, z), st["wp"] * np.cross(u, wv), st["wp"] * np.cross(z, u)], axis=1).reshape(mq, 12))
    if mt:
        a, b, c = V[st["ia"]], V[st["ib"]], V[st["ic"]]
        d = b - a
        add(st["wi"] * (np.einsum("ij,ij->i", c, d) - 0.5 * (np.einsum("ij,ij->i", b, b) - np.einsum("ij,ij->i", a, a))),
            np.repeat(st["rows_iso"], 9), st["t_cols"].ravel(),
            np.concatenate([st["wi"] * (a - c), st["wi"] * (c - b), st["wi"] * d], axis=1).ravel())
    if st["n_area"]:
        Aq = core.face_areas(V, q) if mq else np.zeros(0)
        At = core.face_areas(V, t) if mt else np.zeros(0)
        add(st["wa"] * ((Aq.sum() + At.sum()) - mesh.A0_total), np.full(st["cols_area"].size, st["row_area"]), st["cols_area"].ravel(),
            st["wa"] * np.concatenate([g.ravel() for g in
                                       ([core.quad_area_grads(V, q)] if mq else []) +
                                       ([core.tri_area_grads(V, t)] if mt else [])]))
    if st["n_crease"] and st["wc"] > 0:
        T, pf, pg = st["crease_T"], st["crease_pf"], st["crease_pg"]
        a = V[T[:, 1]] - V[T[:, 0]]
        b = V[T[:, 2]] - V[T[:, 0]]
        c = np.cross(a, b)
        L = np.linalg.norm(c, axis=1) + 1e-30
        nrm = c / L[:, None]
        ident = np.eye(3)[None, :, :].repeat(len(T), axis=0)
        M = ident - nrm[:, :, None] * nrm[:, None, :]
        dnp1 = -np.einsum("nij,njk->nik", M, skew(b)) / L[:, None, None]
        dnp2 = np.einsum("nij,njk->nik", M, skew(a)) / L[:, None, None]
        dnp0 = -(dnp1 + dnp2)
        nf, ng = nrm[pf], nrm[pg]
        cosang = np.einsum("ij,ij->i", nf, ng)
        add(st["wc"] * (1.0 - cosang), st["rows_crease"], np.zeros(st["n_crease"], int),
            np.zeros(st["n_crease"], float))
        r_ent, c_ent, v_ent = [], [], []
        for k, (fi, gi) in enumerate(zip(pf, pg)):
            for j in range(3):
                vtx = int(T[fi, j])
                grad = -st["wc"] * (np.stack([dnp0[fi], dnp1[fi], dnp2[fi]])[j] @ ng[gi])
                for comp in range(3):
                    r_ent.append(st["rows_crease"][k]); c_ent.append(3 * vtx + comp); v_ent.append(float(grad[comp]))
            for j in range(3):
                vtx = int(T[gi, j])
                grad = -st["wc"] * (np.stack([dnp0[gi], dnp1[gi], dnp2[gi]])[j] @ nf[k])
                for comp in range(3):
                    r_ent.append(st["rows_crease"][k]); c_ent.append(3 * vtx + comp); v_ent.append(float(grad[comp]))
        blocks_rows.append(np.array(r_ent)); blocks_cols.append(np.array(c_ent)); blocks_vals.append(np.array(v_ent))
    if st["n_lap"] and st["ws"] > 0:
        lv, deg = st["lap_v"], st["lap_deg"]
        nv_l = len(lv)
        i_of = np.repeat(np.arange(nv_l), 3)
        c_of = np.tile(np.arange(3), nv_l)
        rows_c = st["row_lap"] + 3 * i_of + c_of
        cols_c = 3 * lv[i_of] + c_of
        vals_c = np.full(len(rows_c), st["ws"])
        rep_i = np.repeat(np.arange(nv_l), deg.astype(int))
        own = np.repeat(rep_i, 3)
        c_n = np.tile(np.arange(3), len(rep_i))
        rows_n = st["row_lap"] + 3 * own + c_n
        cols_n = 3 * np.repeat(st["lap_nbr"], 3) + c_n
        vals_n = -st["ws"] / deg[own]
        rows = np.concatenate([rows_c, rows_n])
        cols = np.concatenate([cols_c, cols_n])
        vals = np.concatenate([vals_c, vals_n])
        # 残差：ws * (v - mean(neighbors))
        means = np.zeros((nv_l, 3))
        np.add.at(means, rep_i, V[st["lap_nbr"]])
        means /= deg[:, None]
        rlap = st["ws"] * (V[lv] - means)
        add(rlap, rows, cols, vals)
    if st["n_pos"] and st["wpos"] > 0:
        fv = st["pos_v"]
        rows = st["row_pos"] + np.arange(3 * len(fv))
        cols = 3 * np.repeat(fv, 3) + np.tile(np.arange(3), len(fv))
        vals = np.full(len(rows), st["wpos"])
        add(st["wpos"] * (V[fv] - mesh.V0[fv]), rows, cols, vals)

    r = np.concatenate(blocks_r) if blocks_r else np.zeros(0)
    if blocks_rows:
        rows = np.concatenate(blocks_rows)
        cols = np.concatenate(blocks_cols)
        vals = np.concatenate(blocks_vals)
        fmap = mesh.fmap[cols // 3]
        keep = fmap >= 0
        rows, cols, vals = rows[keep], 3 * fmap[keep] + (cols[keep] % 3), vals[keep]
    else:
        rows, cols, vals = np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return r, (rows, cols, vals), st


def solve_gn_vec(mesh, w, iters=3, pcg_maxit=80, pcg_tol=1e-12, lam=1e-4):
    r, (rows, cols, vals), st = residuals_vec(mesh, V=mesh.V, w=w)
    energy = 0.5 * float(r @ r)
    n = 3 * mesh.nfree
    trace = [energy]
    info = {"iterations": 0, "energy_start": energy, "energy_end": energy, "trace": trace,
            "pcg_iters": [], "assemble_s": [], "solve_s": []}
    if energy < 1e-18:
        return info
    V = mesh.V.copy()
    for it in range(iters):
        t0 = time.time()
        Jt = core.spmv(cols, rows, vals, -r, n)
        diag = np.bincount(cols, weights=vals * vals, minlength=n) + lam
        t1 = time.time()
        Hf = lambda x: core.spmv(cols, rows, vals, core.spmv(rows, cols, vals, x, len(r)), n) + lam * x
        dx = core.pcg(Hf, Jt, np.maximum(diag, 1e-30), tol=pcg_tol, maxit=pcg_maxit)
        t2 = time.time()
        step, improved = 1.0, False
        for _ in range(20):
            Vt = V.copy()
            Vt[mesh.free] = V[mesh.free] + (step * dx).reshape(-1, 3)
            rt, _, _ = residuals_vec(mesh, Vt, w, st)
            et = 0.5 * float(rt @ rt)
            if et < energy * (1 - 1e-14):
                V, energy = Vt, et
                lam = max(lam * 0.5, 1e-12)
                improved = True
                break
            step *= 0.5
        if not improved:
            lam *= 10.0
        t3 = time.time()
        r, (rows, cols, vals), _ = residuals_vec(mesh, V, w, st)
        info["pcg_iters"].append(pcg_maxit)
        info["assemble_s"].append(t0 - t0)  # 占位，实际装配在 residuals_vec 内
        info["solve_s"].append(t2 - t1)
        trace.append(energy)
        info["iterations"] = it + 1
        if len(trace) > 4 and abs(trace[-2] - trace[-1]) < 1e-13 * max(1.0, trace[-1]):
            break
    mesh.V = V
    info["energy_end"] = energy
    return info


def time_iteration(mesh, w, iters=3, pcg_maxit=80, pcg_tol=1e-8):
    """测量每次迭代的总耗时（装配 + 求解 + 线搜索）。"""
    core.prep(mesh)
    t0 = time.time()
    info = solve_gn_vec(mesh, w, iters=iters, pcg_maxit=pcg_maxit, pcg_tol=pcg_tol)
    dt = time.time() - t0
    n_it = max(1, info["iterations"])
    return {"name": mesh.name, "vertices": int(len(mesh.V0)), "quads": int(len(mesh.quads)),
            "tris": int(len(mesh.tris)), "iterations": int(info["iterations"]),
            "total_s": dt, "s_per_iter": dt / n_it, "energy_end": info["energy_end"]}


def main():
    os.makedirs(OUT, exist_ok=True)
    # 正确性对照
    small = core.prep(core.make_flat_quad(nx=8, ny=8, noise=0.05))
    r_scalar, (rs, cs, vs) = core.residuals(small, small.V, W)
    r_vec, (rv, cv, vv), st = residuals_vec(small, small.V, W)
    diff_r = float(np.abs(r_scalar - r_vec).max()) if len(r_scalar) == len(r_vec) else float("nan")
    # 稀疏结构应完全相同（同一构造顺序）
    same_struct = (len(rs) == len(rv)) and bool(np.all(rs == rv)) and bool(np.all(cs == cv))
    diff_j = float(np.abs(vs - vv).max()) if same_struct and len(vs) else float("nan")
    checks = {"residual_max_abs_diff": diff_r, "jacobian_same_structure": same_struct,
              "jacobian_max_abs_diff": diff_j, "n_residual": int(len(r_scalar)), "nnz": int(len(vs))}
    print("正确性对照:", checks)

    perf = []
    for n in (20, 40, 60, 100):
        m = core.make_flat_quad(nx=n, ny=n, noise=0.03)
        perf.append(time_iteration(m, W, iters=3, pcg_maxit=80))
        p = perf[-1]
        print(f"[n={n}] V={p['vertices']} Q={p['quads']} iters={p['iterations']} "
              f"{p['s_per_iter']:.2f} s/iter  E={p['energy_end']:.3e}")

    big = core.make_flat_quad(nx=221, ny=221, noise=0.03)
    t0 = time.time()
    big_res = time_iteration(big, W, iters=2, pcg_maxit=60, pcg_tol=1e-8)
    print(f"[50k] V={big_res['vertices']} Q={big_res['quads']} iters={big_res['iterations']} "
          f"{big_res['s_per_iter']:.2f} s/iter  E={big_res['energy_end']:.3e}")
    big_res["wall_s"] = time.time() - t0

    payload = {"experiment": "E0004", "date": "2026-10-09", "weights": W,
               "correctness_checks": checks, "performance": perf, "large_test": big_res}
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    rows = []
    for p in perf + [big_res]:
        rows.append({"name": p["name"], "vertices": p["vertices"], "quads": p["quads"], "tris": p["tris"],
                     "iterations": p["iterations"], "total_s": p["total_s"], "s_per_iter": p["s_per_iter"],
                     "energy_end": p["energy_end"]})
    with open(os.path.join(OUT, "results.csv"), "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        for r in rows:
            wtr.writerow(r)
    L = ["# E0004 运行报告（向量化装配 + 规模复测）", "",
         "## 正确性对照（向量化 vs 标量实现，8×8 网格）", "",
         f"- 残差最大绝对差：{checks['residual_max_abs_diff']:.3e}",
         f"- 雅可比结构一致：{checks['jacobian_same_structure']}；数值最大绝对差：{checks['jacobian_max_abs_diff']:.3e}",
         f"- 残差行数 {checks['n_residual']}，非零元 {checks['nnz']}", "",
         "## 规模-性能", "",
         "| 网格 n | 顶点 | 四边面 | 迭代 | 每次迭代 s | 结束能量 |", "| --- | --- | --- | --- | --- | --- |"]
    for p in perf:
        L.append("| {} | {} | {} | {} | {:.3f} | {:.3e} |".format(p["name"], p["vertices"], p["quads"], p["iterations"], p["s_per_iter"], p["energy_end"]))
    L.append("")
    L.append(f"### 5 万顶点测试（n=221）")
    L.append("")
    L.append(f"- 顶点 {big_res['vertices']}，四边面 {big_res['quads']}，迭代 {big_res['iterations']} 轮，"
             f"每次迭代 {big_res['s_per_iter']:.2f} s（总 {big_res['total_s']:.2f} s）")
    L.append(f"- Q18 目标：<=10 s/轮 → {'达标' if big_res['s_per_iter'] <= 10 else '未达标'}")
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("results ->", OUT)


if __name__ == "__main__":
    main()