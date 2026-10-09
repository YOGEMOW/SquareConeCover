#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E0004 渲染图与性能图生成（PIL 实现，无第三方 3D 依赖）。"""

from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import E0002_regularization_prototype as core
import E0004_vectorized_assembly as v4

REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "experiments", "results", "E0004", "renders")
os.makedirs(OUT, exist_ok=True)
W = dict(core.W_DEFAULT)


def project(V, view=(0.85, -1.15, 0.95), up=(0.0, 1.0, 0.0)):
    d = np.array(view, float)
    d /= np.linalg.norm(d)
    up = np.array(up, float)
    right = np.cross(up, d)
    right /= np.linalg.norm(right)
    top = np.cross(d, right)
    return V @ right, V @ top, V @ d


def render_mesh(V, faces, size=(620, 520), view=(0.85, -1.15, 0.95), fit=None, title="", bg=(248, 246, 242)):
    img = Image.new("RGB", size, bg)
    dr = ImageDraw.Draw(img)
    if len(faces) == 0:
        return img
    u, v, w = project(V, view)
    if fit is None:
        fit = (u.min(), u.max(), v.min(), v.max())
    u0, u1, v0, v1 = fit
    pad = 34
    Wp, Hp = size[0] - 2 * pad, size[1] - 2 * pad - 22
    s = min(Wp / max(u1 - u0, 1e-9), Hp / max(v1 - v0, 1e-9))
    cx = pad + Wp / 2 - s * (u0 + u1) / 2
    cy = size[1] - pad - 22 - Hp / 2 + s * (v0 + v1) / 2
    px = cx + s * u
    py = cy - s * v
    P3 = V[faces]
    n = np.cross(P3[:, 1] - P3[:, 0], P3[:, 2] - P3[:, 0])
    nn = np.linalg.norm(n, axis=1)
    ok = nn > 1e-12
    n = np.where(ok[:, None], n / (nn[:, None] + 1e-30), 0.0)
    L = np.array([0.35, -0.62, 0.70])
    L /= np.linalg.norm(L)
    inten = 0.30 + 0.70 * np.clip(np.abs(n @ L), 0, 1)
    depth = w[faces].mean(axis=1)
    order = np.argsort(-depth)
    areas = np.abs(u[faces[:, 0]] * (v[faces[:, 1]] - v[faces[:, 2]]) +
                   u[faces[:, 1]] * (v[faces[:, 2]] - v[faces[:, 0]]) +
                   u[faces[:, 2]] * (v[faces[:, 0]] - v[faces[:, 1]])) * s * s / 2
    for i in order:
        f = faces[i]
        poly = [(float(px[j]), float(py[j])) for j in f]
        c = int(255 * inten[i])
        col = (int(c * 0.80), int(c * 0.86), int(c * 0.95))
        edge = (70, 78, 92) if areas[i] > 6.0 else None
        dr.polygon(poly, fill=col, outline=edge)
    if title:
        dr.rectangle([0, 0, size[0], 22], fill=(236, 233, 226))
        dr.text((8, 6), title, fill=(40, 40, 40))
    return img


def compose(images, labels, path, gap=10):
    w = sum(im.width for im in images) + gap * (len(images) - 1)
    h = max(im.height for im in images) + 24
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    dr = ImageDraw.Draw(canvas)
    x = 0
    for im, lab in zip(images, labels):
        canvas.paste(im, (x, 0))
        dr.text((x + 8, im.height + 6), lab, fill=(30, 30, 30))
        x += im.width + gap
    canvas.save(path)


def solve_quick(mesh, iters=60, pcg=300, presmooth=5, tol=1e-13):
    core.prep(mesh)
    if presmooth:
        mesh.V = core.presmooth(mesh, mesh.V, passes=presmooth)
    v4.solve_gn_vec(mesh, W, iters=iters, pcg_maxit=pcg, pcg_tol=tol)
    return mesh


def main():
    # 1) 平面四边（带扰动）
    m = core.make_flat_quad(nx=40, ny=40, noise=0.01)
    Vb = m.V0.copy()
    before = render_mesh(Vb, m.quads, title="flat quad 40x40 before (perturbed)")
    solve_quick(m, iters=60)
    after = render_mesh(m.V, m.quads, title="flat quad 40x40 after (regularized)")
    compose([before, after], ["before", "after"], os.path.join(OUT, "fig1-flat-quad.png"))

    # 2) 平面三角（底=最长边）
    m = core.make_flat_tri(nx=40, ny=40, noise=0.01, rule="longest")
    before = render_mesh(m.V0, m.tris, title="triangle grid 40x40 before (base=longest)")
    solve_quick(m, iters=60)
    after = render_mesh(m.V, m.tris, title="triangle grid 40x40 after")
    compose([before, after], ["before", "after"], os.path.join(OUT, "fig2-triangle.png"))

    # 3) 圆柱面（可展，单导轨锚定）
    m = core.make_cylinder_quad(nx=24, ny=20, noise=0.0, anchor="rail0")
    before = render_mesh(m.V0, m.quads, title="cylinder patch before (developable)")
    solve_quick(m, iters=150, presmooth=0)
    after = render_mesh(m.V, m.quads, title="cylinder patch after (unrolled flat)")
    compose([before, after], ["before (curved)", "after (unrolled)"], os.path.join(OUT, "fig3-cylinder.png"))

    # 4) 球冠（不可展）
    m = core.make_sphere_cap_quad(nx=20, ny=20, noise=0.0)
    before = render_mesh(m.V0, m.quads, title="sphere cap before (non-developable)")
    solve_quick(m, iters=120, presmooth=0)
    after = render_mesh(m.V, m.quads, title="sphere cap after (curvature limited)")
    compose([before, after], ["before", "after"], os.path.join(OUT, "fig4-spherecap.png"))

    # 5) 大网格（100x100）
    m = core.make_flat_quad(nx=100, ny=100, noise=0.004)
    solve_quick(m, iters=40, pcg=60)
    img = render_mesh(m.V, m.quads, size=(760, 640), title="flat quad 100x100 (10201 vertices) after")
    img.save(os.path.join(OUT, "fig5-large-100x100.png"))

    # 6) 性能图（对数坐标）
    data = json.load(open(os.path.join(REPO, "experiments", "results", "E0004", "results.json"), encoding="utf-8"))
    pts = [(p["vertices"], p["s_per_iter"]) for p in data["performance"]]
    lt = data["large_test"]
    pts.append((lt["vertices"], lt["s_per_iter"]))
    size = (900, 560)
    img = Image.new("RGB", size, (255, 255, 255))
    dr = ImageDraw.Draw(img)
    ox, oy, ex, ey = 90, 470, 60, 380
    def sx(v):
        return ox + (math.log10(v) - math.log10(300)) / (math.log10(80000) - math.log10(300)) * (size[0] - ox - ex)
    def sy(t):
        return oy - (math.log10(max(t, 0.005)) - math.log10(0.005)) / (math.log10(20) - math.log10(0.005)) * (oy - ey)
    dr.line([(ox, ey - 4), (ox, oy), (size[0] - ex + 10, oy)], fill=(60, 60, 60), width=2)
    for v in (400, 1000, 4000, 10000, 40000):
        dr.line([(sx(v), oy), (sx(v), oy + 6)], fill=(60, 60, 60), width=1)
        dr.text((sx(v) - 14, oy + 10), f"{v//1000}k" if v >= 1000 else str(v), fill=(60, 60, 60))
    for t in (0.01, 0.1, 1, 10):
        dr.line([(ox - 6, sy(t)), (ox, sy(t))], fill=(60, 60, 60), width=1)
        dr.text((ox - 78, sy(t) - 6), f"{t:g} s", fill=(60, 60, 60))
    dr.line([(ox, sy(10)), (size[0] - ex + 10, sy(10))], fill=(200, 60, 60), width=2)
    dr.text((ox + 8, sy(10) - 18), "target <= 10 s / iteration (Q18)", fill=(200, 60, 60))
    prev = None
    for (v, t) in sorted(pts):
        x, y = sx(v), sy(t)
        if prev:
            dr.line([prev, (x, y)], fill=(40, 90, 170), width=3)
        prev = (x, y)
    for (v, t) in sorted(pts):
        x, y = sx(v), sy(t)
        dr.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(40, 90, 170))
        dr.text((x + 8, y - 16), f"{v} vtx, {t:.2f}s", fill=(30, 30, 30))
    dr.text((ox, 20), "E0004 vectorized assembly: seconds per GN iteration vs mesh size", fill=(20, 20, 20))
    img.save(os.path.join(OUT, "fig6-performance.png"))
    print("renders ->", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)


if __name__ == "__main__":
    main()