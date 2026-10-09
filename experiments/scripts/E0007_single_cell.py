import sys, numpy as np
sys.path.insert(0, r"E:\Git\repositoris\SquareConeCover\experiments\scripts")
import E0002_regularization_prototype as core
W = dict(core.W_DEFAULT); W.update(iso=1.0, area=0.0, smooth=0.0, pos=0.0, crease=0.0)
m = core.make_flat_tri(nx=3, ny=3, noise=0.05, rule="longest")
core.prep(m)
r, (rows, cols, vals) = core.residuals(m, m.V, W)
n = 3 * m.nfree
J = {}
for i in range(len(vals)):
    v = cols[i] // 3
    if m.fmap[v] >= 0:
        key = (rows[i], 3 * m.fmap[v] + cols[i] % 3)
        J[key] = J.get(key, 0.0) + vals[i]
eps = 1e-6
FD = np.zeros((len(r), n))
for k in range(n):
    Vp = m.V.copy().reshape(-1); Vp[k] += eps
    Vm = m.V.copy().reshape(-1); Vm[k] -= eps
    FD[:, k] = (core.residuals(m, Vp.reshape(-1, 3), W, with_jac=False)[0]
                - core.residuals(m, Vm.reshape(-1, 3), W, with_jac=False)[0]) / (2 * eps)
AN = np.zeros((len(r), n))
for (rr, cc), vv in J.items():
    AN[rr, cc] = vv
err = np.abs(AN - FD)
rr, kk = np.unravel_index(np.argmax(err), err.shape)
print(f"worst cell: row={rr} col={kk} err={err[rr,kk]:.4f}  (col-> vertex {kk//3} axis {kk%3})")
print("analytic row:", {c: round(AN[rr, c], 4) for c in range(n) if abs(AN[rr, c]) > 1e-12})
print("fd       row:", {c: round(FD[rr, c], 4) for c in range(n) if abs(FD[rr, c]) > 1e-9})
print("analytic col:", {q: round(AN[q, kk], 4) for q in range(len(r)) if abs(AN[q, kk]) > 1e-12})
print("fd       col:", {q: round(FD[q, kk], 4) for q in range(len(r)) if abs(FD[q, kk]) > 1e-9})
print("triangles (first 6):", [list(map(int, m.tris[i])) for i in range(min(6, len(m.tris)))])
print("bases     (first 6):", [list(map(int, m.bases[i])) for i in range(min(6, len(m.bases)))])
import inspect
src = inspect.getsource(core.residuals)
i0 = src.find('等腰' if False else 'wi = math.sqrt(w["iso"])')
print(src[i0-380:i0+520])