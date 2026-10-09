#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E0001 锥体公式数值复核（标准库实现，无第三方依赖）。

验证 docs/02-primitive-geometry.md 中的核心公式：
  Theta(A,B,H) = Int_0^{2pi} sqrt(B^2 H^2 cos^2 t + A^2 B^2 + A^2 H^2 sin^2 t)
                 / (H^2 + A^2 cos^2 t + B^2 sin^2 t) dt
  delta = 2 pi - Theta
  侧面积 = 1/2 Int sqrt(...) dt

运行：python experiments/scripts/E0001_cone_formula_check.py
输出：实验结论 + experiments/results/E0001/{results.csv,results.json,report.md}
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TAU = 2.0 * math.pi
DEG = 180.0 / math.pi
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTDIR = os.path.join(ROOT, "experiments", "results", "E0001")
N_DEFAULT = 1 << 18


def integrand_num(A, B, H, t):
    c = math.cos(t)
    s = math.sin(t)
    return math.sqrt(B * B * H * H * c * c + A * A * B * B + A * A * H * H * s * s)


def integrand_den(A, B, H, t):
    c = math.cos(t)
    s = math.sin(t)
    return H * H + A * A * c * c + B * B * s * s


def theta_num(A, B, H, N=N_DEFAULT):
    """周期梯形法（光滑周期函数上指数收敛）。"""
    dt = TAU / N
    acc = 0.0
    for k in range(N):
        t = (k + 0.5) * dt
        acc += integrand_num(A, B, H, t) / integrand_den(A, B, H, t)
    return acc * dt


def lateral_num(A, B, H, N=N_DEFAULT):
    dt = TAU / N
    acc = 0.0
    for k in range(N):
        t = (k + 0.5) * dt
        acc += 0.5 * integrand_num(A, B, H, t)
    return acc * dt


def theta_circular_analytic(A, H):
    return TAU * A / math.sqrt(H * H + A * A)


def lateral_circular_analytic(A, H):
    return math.pi * A * math.sqrt(H * H + A * A)


def H_for_defect(A, delta):
    """圆底反解：由角亏 delta 求高 H（修正公式）。"""
    return A * math.sqrt((TAU / (TAU - delta)) ** 2 - 1.0)


def H_for_defect_doc_bug(A, delta):
    """doc 旧式（错误）版本，仅用于复现缺陷 F1。"""
    return A * math.sqrt((TAU * A / (TAU - delta)) ** 2 - 1.0)


def L_of(t, A, B, H):
    return math.sqrt(H * H + A * A * math.cos(t) ** 2 + B * B * math.sin(t) ** 2)


def dphi_expr(t, A, B, H):
    """文档采用的绕顶角度测度 dphi/dt = |p' x (p-apex)| / |p-apex|^2。"""
    return integrand_num(A, B, H, t) / integrand_den(A, B, H, t)


def speed_p(t, A, B, H):
    """|p'(t)|，p(t) = (A cos t, 0, B sin t)。"""
    return math.sqrt((A * math.sin(t)) ** 2 + (B * math.cos(t)) ** 2)


def phi(t, A, B, H, M=200000):
    dt = t / M
    acc = 0.0
    for k in range(M):
        acc += dphi_expr((k + 0.5) * dt, A, B, H)
    return acc * dt


def developed_rim_polyline_length(A, B, H, N):
    total = 0.0
    px = L_of(0.0, A, B, H)
    prev = (px, 0.0)
    acc_phi = 0.0
    dt = TAU / N
    for k in range(1, N + 1):
        t = k * dt
        acc_phi += dphi_expr(t - dt / 2, A, B, H) * dt
        rr = L_of(t, A, B, H)
        cur = (rr * math.cos(acc_phi), rr * math.sin(acc_phi))
        total += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        prev = cur
    return total


def rim_polyline_length_3d(A, B, H, N):
    total = 0.0
    prev = (A, 0.0, 0.0)
    dt = TAU / N
    for k in range(1, N + 1):
        t = k * dt
        cur = (A * math.cos(t), 0.0, B * math.sin(t))
        total += math.dist(cur, prev)
        prev = cur
    return total


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    checks = []
    rows = []

    def record(check_id, desc, value, ref, rel_err, passed, note=""):
        rows.append({
            "check": check_id, "desc": desc, "value": value, "reference": ref,
            "rel_err": rel_err, "passed": bool(passed), "note": note,
        })

    # ---------- 命题 1 / 5：圆底解析对照 ----------
    worst1 = 0.0
    worst5 = 0.0
    for A in (0.5, 1.0, 0.3):
        for H in (0.01, 0.5, 1.0, 3.0):
            th = theta_num(A, A, H)
            th_a = theta_circular_analytic(A, H)
            e1 = abs(th - th_a) / th_a
            worst1 = max(worst1, e1)
            area = lateral_num(A, A, H)
            area_a = lateral_circular_analytic(A, H)
            e5 = abs(area - area_a) / area_a
            worst5 = max(worst5, e5)
            record("P1", f"Theta circular A={A} H={H}", th, th_a, e1, e1 < 1e-10)
            record("P5", f"lateral circular A={A} H={H}", area, area_a, e5, e5 < 1e-10)
    checks.append(("P1 圆底解析对照 Theta = 2piA/sqrt(H^2+A^2)", worst1 < 1e-10, worst1))
    checks.append(("P5 圆底侧面积 = pi*A*sqrt(H^2+A^2)", worst5 < 1e-10, worst5))

    # ---------- 命题 3：H -> 0 与 H -> infinity ----------
    A = 0.5
    delta_small = []
    for H in (1e-2, 1e-3, 1e-4):
        th = theta_num(A, A, H)
        delta_small.append(abs(th - TAU))
    ratios = [delta_small[i] / delta_small[i + 1] for i in range(len(delta_small) - 1)]
    p3a = all(r > 50.0 for r in ratios) and delta_small[0] > delta_small[1] > delta_small[2]
    for i, H in enumerate((1e-2, 1e-3, 1e-4)):
        record("P3a", f"H->0 |Theta-2pi| H={H}", delta_small[i], 0.0, None,
               delta_small[i] < 1e-3, note="期望按 H^2 收敛")
    record("P3a-ratio", "相邻 H 步长比值 (~100)", ratios[0], 100.0, abs(ratios[0] - 100) / 100, p3a)

    big = []
    for H in (1e3, 1e4, 1e5):
        th = theta_num(A, A, H)
        big.append(abs(th * H - TAU * A) / (TAU * A))
    p3b = all(e < 1e-6 for e in big)
    for i, H in enumerate((1e3, 1e4, 1e5)):
        record("P3b", f"H->inf rel err of Theta*H vs 2piA, H={H}", big[i], 0.0, big[i], big[i] < 1e-6)
    checks.append(("P3a H->0 时 Theta->2pi（H^2 收敛）", p3a, max(delta_small)))
    checks.append(("P3b H->inf 时 Theta*H->2piA", p3b, max(big)))

    # ---------- 命题 4：B -> 0（H 固定）Theta -> 4 atan(A/H) ----------
    A, H = 0.5, 1.0
    limit4 = 4.0 * math.atan(A / H)
    errs = []
    for B in (1e-2, 1e-3, 1e-4, 1e-5):
        th = theta_num(A, B, H, N=1 << 19)
        errs.append(th - limit4)
    p4 = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1)) and errs[-1] < 1e-6
    for i, B in enumerate((1e-2, 1e-3, 1e-4, 1e-5)):
        record("P4", f"B->0 signed err, B={B}", errs[i], 0.0, None, errs[i] < 1e-5,
               note="理论量级 O(B^2 log(1/B))")
    checks.append(("P4 B->0 时 Theta->4*atan(A/H)", p4, errs[-1]))

    # ---------- 命题 6：展开等距（度量一致性） ----------
    worst6 = 0.0
    rng = random.Random(20261009)
    configs = [(0.5, 0.5, 1.0), (0.5, 0.25, 0.8), (0.7, 0.2, 0.3), (0.3, 0.3, 2.0)]
    for (A2, B2, H2) in configs:
        for _ in range(6):
            t = rng.uniform(0.15, TAU - 0.15)
            L = L_of(t, A2, B2, H2)
            h = 1e-6
            Lp = (L_of(t + h, A2, B2, H2) - L_of(t - h, A2, B2, H2)) / (2 * h)
            phip = dphi_expr(t, A2, B2, H2)
            lhs = speed_p(t, A2, B2, H2) ** 2
            rhs = Lp * Lp + (L * phip) ** 2
            worst6 = max(worst6, abs(lhs - rhs) / lhs)
    p6a = worst6 < 1e-6
    record("P6a", "度量一致性 |p'|^2 = L'^2 + (L phi')^2", worst6, 0.0, worst6, p6a)
    checks.append(("P6a 展开映射保度量（数值微分）", p6a, worst6))

    p6b = {}
    for N in (2000, 20000):
        dev = developed_rim_polyline_length(A, B, H, N)
        ref = rim_polyline_length_3d(A, B, H, N)
        rel = abs(dev - ref) / ref
        p6b[N] = rel
        record("P6b", f"展开底口折线 vs 三维底口折线 N={N}", dev, ref, rel, rel < (3e-3 if N == 2000 else 4e-4))
    checks.append(("P6b 展开底口折线长度收敛一致", p6b[20000] < p6b[2000] and p6b[20000] < 4e-4, p6b[20000]))

    # ---------- 参考数值表核对 ----------
    ref_rows = [
        ("default cone", 0.5, 0.5, 1.0, 160.99, 199.01),
        ("sphere n=4", 0.5, 0.5, H_for_defect(0.5, math.pi), 180.00, 180.00),
        ("sphere n=6", 0.5, 0.5, H_for_defect(0.5, 2.0 * math.pi / 3.0), 240.00, 120.00),
        ("sphere n=8", 0.5, 0.5, H_for_defect(0.5, math.pi / 2.0), 270.00, 90.00),
        ("sphere n=12", 0.5, 0.5, H_for_defect(0.5, math.pi / 3.0), 300.00, 60.00),
        ("flat disk", 0.5, 0.5, 0.01, 359.93, 0.07),
    ]
    worst_tab = 0.0
    for (name, A3, B3, H3, th_exp, de_exp) in ref_rows:
        th = theta_num(A3, B3, H3)
        de = TAU - th
        worst_tab = max(worst_tab, abs(th * DEG - th_exp), abs(de * DEG - de_exp))
        record("TAB", f"{name} Theta", th * DEG, th_exp, None, abs(th * DEG - th_exp) < 0.01)
        record("TAB", f"{name} delta", de * DEG, de_exp, None, abs(de * DEG - de_exp) < 0.01)
    checks.append(("参考数值表 (0.01 度容差)", worst_tab < 0.01, worst_tab))

    # ---------- 楔形形态精确值（修正文档旧值） ----------
    A, B, H = 0.5, 0.005, 1.0
    th_w = theta_num(A, B, H, N=1 << 19)
    de_w = TAU - th_w
    limit_w = 4.0 * math.atan(A / H)
    record("F2", "wedge Theta (B=0.005)", th_w * DEG, limit_w * DEG, None, True,
           note="文档旧表写的 106.26 度是 B->0 极限，实际值偏高")

    # ---------- 反解公式（修正版 vs 文档旧式） ----------
    worst_rev = 0.0
    for d_deg in (180.0, 120.0, 90.0, 60.0, 30.0):
        d = d_deg / DEG
        H_fix = H_for_defect(0.5, d)
        th = theta_num(0.5, 0.5, H_fix)
        worst_rev = max(worst_rev, abs(th - (TAU - d)))
        record("F1", f"reverse formula (fixed) delta={d_deg} deg", th * DEG, (TAU - d) * DEG,
               abs(th - (TAU - d)) / (TAU - d), abs(th - (TAU - d)) < 1e-9)
    checks.append(("F1 反解公式（修正版）", worst_rev < 1e-9, worst_rev))

    A = 0.5
    H_bug = H_for_defect_doc_bug(A, math.pi)
    th_bug = theta_num(A, A, H_bug) if H_bug > 0 else float("nan")
    bug_repro = (H_bug < 1e-9) or (abs(th_bug - math.pi) > 1.0)
    record("F1b", "doc 旧反解式 delta=180 -> H", H_bug, 0.8660254, None, bug_repro,
           note="旧式给出 H=0，Theta=2pi，应为 0.8660254")

    # ---------- 补充：随机参数下 0 < Theta < 2pi ----------
    rng = random.Random(7)
    viol = 0
    n_rand = 300
    for _ in range(n_rand):
        A4 = 10 ** rng.uniform(-2, 1.5)
        B4 = 10 ** rng.uniform(-2, 1.5)
        H4 = 10 ** rng.uniform(-2, 1.5)
        th = theta_num(A4, B4, H4, N=1 << 15)
        if not (0.0 - 1e-9 < th < TAU + 1e-9):
            viol += 1
    record("P7", f"随机 {n_rand} 组参数越界次数", viol, 0, None, viol == 0)
    checks.append(("P7 0 < Theta < 2pi（随机抽样）", viol == 0, viol))

    # ---------- 输出 ----------
    os.makedirs(OUTDIR, exist_ok=True)
    csv_path = os.path.join(OUTDIR, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["check", "desc", "value", "reference", "rel_err", "passed", "note"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = {
        "experiment": "E0001",
        "date": "2026-10-09",
        "python": "3.10.6",
        "n_default": N_DEFAULT,
        "checks": [{"id": c[0], "passed": c[1], "metric": c[2]} for c in checks],
        "all_passed": all(c[1] for c in checks),
        "wedge_theta_deg_B0.005": th_w * DEG,
        "wedge_theta_limit_deg": limit_w * DEG,
        "findings": [
            "F1: docs/02 圆底反解式写错（多了因子 A），实验给出修正式",
            "F2: docs/02 三角薄楔行应区分 B=0.005 实际值与该极限值",
        ],
    }
    with open(os.path.join(OUTDIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    lines = ["# E0001 运行报告", "",
             f"- 实验：E0001 锥体公式数值复核",
             f"- 日期：2026-10-09",
             f"- 结论：{'全部通过' if summary['all_passed'] else '存在未通过项'}", "",
             "## 检查结果", "", "| 检查 | 通过 | 指标 |", "| --- | --- | --- |"]
    for c in checks:
        v = c[2]
        vtxt = f"{v:.3e}" if isinstance(v, float) else str(v)
        lines.append(f"| {c[0]} | {'通过' if c[1] else '未通过'} | {vtxt} |")
    lines += ["", "## 发现", ""] + [f"- {x}" for x in summary["findings"]]
    lines += ["", f"- 楔形（B=0.005）Theta = {th_w * DEG:.4f} 度；B->0 极限 = {limit_w * DEG:.4f} 度", ""]
    with open(os.path.join(OUTDIR, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("=" * 72)
    for c in checks:
        v = c[2]
        vtxt = f"{v:.3e}" if isinstance(v, float) else str(v)
        print(f"[{'PASS' if c[1] else 'FAIL'}] {c[0]}  (metric={vtxt})")
    print("-" * 72)
    print(f"wedge Theta (B=0.005) = {th_w * DEG:.4f} deg; limit = {limit_w * DEG:.4f} deg")
    print(f"results -> {OUTDIR}")
    print("=" * 72)
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())