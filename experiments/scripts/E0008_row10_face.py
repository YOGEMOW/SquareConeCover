import sys, numpy as np
sys.path.insert(0, r"E:\Git\repositoris\SquareConeCover\experiments\scripts")
import E0002_regularization_prototype as core
W = dict(core.W_DEFAULT); W.update(iso=1.0, area=0.0, smooth=0.0, pos=0.0, crease=0.0)
m = core.make_flat_tri(nx=3, ny=3, noise=0.05, rule="longest")
core.prep(m)
r, (rows, cols, vals) = core.residuals(m, m.V, W)
n = 3 * m.nfree
# 解析行（按 fmap 重映射）
arow = {}
for i in range(len(vals)):
    if rows[i] == 10:
        v = cols[i] // 3
        if m.fmap[v] >= 0:
            arow[3 * m.fmap[v] + cols[i] % 3] = arow.get(3 * m.fmap[v] + cols[i] % 3, 0.0) + vals[i]
        else:
            arow.setdefault("anchored_vertex_%d" % v, 0)
eps = 1e-6
fdrow = {}
for k in range(n):
    Vp = m.V.copy().reshape(-1); Vp[k] += eps
    Vm = m.V.copy().reshape(-1); Vm[k] -= eps
    d = (core.residuals(m, Vp.reshape(-1, 3), W, with_jac=False)[0][10]
         - core.residuals(m, Vm.reshape(-1, 3), W, with_jac=False)[0][10]) / (2 * eps)
    if abs(d) > 1e-9:
        fdrow[k] = round(float(d), 4)
print("row 10 -> triangle slots:", list(map(int, m.tris[10])), " bases:", list(map(int, m.bases[10])))
ia = int(m.tris[10][m.bases[10][0]]); ib = int(m.tris[10][m.bases[10][1]])
ic = int([x for x in m.tris[10] if x not in (ia, ib)][0])
print("代码解读：(ia,ib,ic) =", (ia, ib, ic), " -> 顶点", {ia, ib, ic}, " 其中自由顶点列:", [c for c in range(n) if c // 3 in (ia, ib, ic)])
print("槽位解读：slots 0/1/2 =", list(map(int, m.tris[10])), " 自由顶点列:", [c for c in range(n) if m.tris[10][c // 3] in np.where(m.free)[0]])
print("解析行 entries:", {k: round(v, 4) for k, v in arow.items() if isinstance(k, int)})
print("有限差分非零列:", fdrow, " -> 对应网格顶点:", sorted({k // 3 for k in fdrow}))
print("自由顶点列表(网格编号):", list(map(int, np.where(m.free)[0])), " fmap:", {int(v): int(m.fmap[v]) for v in np.where(m.free)[0]})