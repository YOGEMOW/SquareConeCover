#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E0003 干净输入与规模验证（Stage A/A2）

目标：
  1. 干净输入基线：已规整的网格不应被改动（位移≈0、残差≈0、面积漂移≈0）；
  2. 验证 Q15 新默认（底边 = 最长边）在干净三角网格上正确（能量应为 0）；
  3. 圆柱面（可展）+ 单导轨锚定：应收敛到平坦矩形网格（含共面性检查）；
  4. 球冠（不可展）：规模下的稳定性与曲率残差下限；
  5. 规模-性能基准：20/40/60 网格的每次迭代耗时。

求解：路线 A（高斯-牛顿 + 对角预条件 CG）；干净输入不做预平滑。
输出：experiments/results/E0003/{results.json, results.csv, report.md, *.obj}
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
OUT = os.path.join(REPO, "experiments", "results", "E0003")
W = dict(core.W_DEFAULT)


def coplanarity(V):
    """所有顶点到最佳拟合平面的最大偏差（衡量是否已展平）。"""
    c = V.mean(axis=0)
    Q = V - c
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    return float(np.abs(Q @ vt[2]).max())


def run_case(name, mesh, iters=150, presmooth_passes=0, det=False):
    core.prep(mesh)
    V0 = mesh.V0.copy()
    t0 = time.time()
    if presmooth_passes:
        mesh.V = core.presmooth(mesh, mesh.V, passes=presmooth_passes)
    info = core.solve_gauss_newton(mesh, W, iters=iters)
    dt = time.time() - t0
    after = core.evaluate(mesh, V0, mesh.V.copy())
    res = {
        "name": name,
        "n_vertices": int(len(mesh.V0)), "n_quads": int(len(mesh.quads)), "n_tris": int(len(mesh.tris)),
        "n_anchors": int(mesh.anchors.sum()),
        "iterations": int(info["iterations"]),
        "energy_start": float(info["energy_start"]), "energy_end": float(info["energy_end"]),
        "runtime_s": dt, "after": after,
        "coplanarity_before": coplanarity(V0), "coplanarity_after": coplanarity(mesh.V),
    }
    if det:
        m2 = core.Mesh(mesh.V0, quads=mesh.quads, tris=mesh.tris, bases=mesh.bases,
                       anchors=mesh.anchors, name=name)
        core.prep(m2)
        if presmooth_passes:
            m2.V = core.presmooth(m2, m2.V, passes=presmooth_passes)
        core.solve_gauss_newton(m2, W, iters=iters)
        res["determinism_max_diff"] = float(np.abs(m2.V - mesh.V).max())
    core.write_obj(os.path.join(OUT, name + "-after.obj"), mesh, mesh.V)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    results = []
    print("=" * 88)
    results.append(run_case("C1-flat-quad-clean-40", core.make_flat_quad(nx=40, ny=40, noise=0.0), iters=20))
    results.append(run_case("C2-flat-tri-longest-40", core.make_flat_tri(nx=40, ny=40, noise=0.0, rule="longest"), iters=20))
    results.append(run_case("C3-cylinder-clean-24x20", core.make_cylinder_quad(nx=24, ny=20, noise=0.0, anchor="rail0"), iters=150, det=True))
    results.append(run_case("C4-spherecap-clean-20", core.make_sphere_cap_quad(nx=20, ny=20, noise=0.0), iters=120))
    perf = []
    for n in (20, 40, 60):
        m = core.prep(core.make_flat_quad(nx=n, ny=n, noise=0.03))
        t0 = time.time()
        core.solve_gauss_newton(m, W, iters=3)
        dt = time.time() - t0
        perf.append({"n": n, "vertices": int(len(m.V0)), "quads": int(len(m.quads)),
                     "time_5iter_s": dt, "s_per_iter": dt / 3.0})
    payload = {"experiment": "E0003", "date": "2026-10-09", "weights": W, "cases": results, "performance": perf}
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    rows = []
    for r in results:
        qa = r["after"].get("quad_after") or {}
        ta = r["after"].get("tri_after") or {}
        rows.append({
            "case": r["name"], "vertices": r["n_vertices"], "quads": r["n_quads"], "tris": r["n_tris"],
            "iters": r["iterations"], "E_start": r["energy_start"], "E_end": r["energy_end"],
            "angle_p95_deg": qa.get("angle_p95_deg"), "angle_max_deg": qa.get("angle_max_deg"),
            "opposite_p95": qa.get("opposite_rel_p95"), "planarity_p95": qa.get("planarity_rel_p95"),
            "leg_p95": ta.get("leg_rel_p95"), "min_angle_deg": ta.get("min_angle_deg"),
            "area_p95": r["after"]["area"]["p95_rel_drift"], "area_global": r["after"]["area"]["global_rel_drift"],
            "flips": r["after"]["flips"], "max_disp": r["after"]["max_disp"],
            "coplanarity_before": r["coplanarity_before"], "coplanarity_after": r["coplanarity_after"],
            "runtime_s": r["runtime_s"], "determinism": r.get("determinism_max_diff"),
        })
    keys = sorted({k for row in rows for k in row})
    with open(os.path.join(OUT, "results.csv"), "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=keys)
        wtr.writeheader()
        for row in rows:
            wtr.writerow(row)

    L = ["# E0003 运行报告（干净输入 + 规模验证）", "",
         "对照目标：干净输入下应“不动作”（残差≈0、位移≈0、面积漂移≈0）；圆柱面应展平；球冠应稳定并保留曲率残差。", "",
         "## 结果", "",
         "| 用例 | 顶点 | 面 | 迭代 | 能量 结束 | 直角 p95 | 直角 max | 腰长差 p95 | 共面性(前→后) | 翻转 | 面积漂移(全局) | 最大位移 | 耗时 s |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    def fm(v, nd=4):
        if v is None:
            return "-"
        return f"{v:.{nd}g}" if isinstance(v, float) else str(v)
    for r in rows:
        L.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} → {} | {} | {:+.3e} | {:.4g} | {:.1f} |".format(
            r["case"], r["vertices"], r["quads"] + r["tris"], r["iters"], fm(r["E_end"]),
            fm(r["angle_p95_deg"]), fm(r["angle_max_deg"]), fm(r["leg_p95"]),
            fm(r["coplanarity_before"]), fm(r["coplanarity_after"]), r["flips"],
            r["area_global"], r["max_disp"], r["runtime_s"]))
    L += ["", "## 规模-性能基准（带扰动的平面四边网格，高斯-牛顿 3 轮）", "",
          "| n (网格) | 顶点 | 四边面 | 3 轮耗时 s | 每次迭代 s |", "| --- | --- | --- | --- | --- |"]
    for p in perf:
        L.append("| {} | {} | {} | {:.2f} | {:.3f} |".format(p["n"], p["vertices"], p["quads"], p["time_5iter_s"], p["s_per_iter"]))
    L += ["", "## 判定", "",
          "- C1/C2（干净平面网格）：能量与残差应为 0、位移应为 0 —— 验证“干净输入不动作”基线，并验证 Q15 新默认（底边=最长边）。",
          "- C3（圆柱面）：应展平（共面性趋近 0）且面积漂移接近 0（等距展开）；若仍有残差需在实验卡中记录。",
          "- C4（球冠）：不可展，残差有下限；重点看稳定性（无翻转、面积漂移小）。",
          "- 性能：给出每次迭代耗时随规模的增长，用于外推 5 万顶点预算。", ""]
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    for r in results:
        qa = r["after"].get("quad_after") or {}
        ta = r["after"].get("tri_after") or {}
        print(f"[{r['name']}] V={r['n_vertices']} iters={r['iterations']} E={r['energy_end']:.3e} "
              f"angle_p95={qa.get('angle_p95_deg', float('nan')):.6f} leg_p95={ta.get('leg_rel_p95', float('nan')):.6f} "
              f"disp={r['after']['max_disp']:.2e} area_g={r['after']['area']['global_rel_drift']:+.2e} "
              f"coplan={r['coplanarity_before']:.3f}->{r['coplanarity_after']:.3e} flips={r['after']['flips']} t={r['runtime_s']:.1f}s")
    print("性能:", [(p["n"], round(p["s_per_iter"], 3)) for p in perf])
    print("results ->", OUT)
    print("=" * 88)


if __name__ == "__main__":
    main()