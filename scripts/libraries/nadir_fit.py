"""
nadir_fit.py - nadir estimate from limb rays. Plain Python (no numpy), so the SAME
file runs on the laptop test and on the OpenMV.

Idea: every limb ray r (unit vector) makes the same angle alpha with the nadir n,
so r.n = cos(alpha). Solve that for n.
  1. RANSAC: try many small-subset solutions, keep the one most rays agree with
  2. refine with least squares on the agreeing rays, drop rays that still disagree
Returns (n, inlier_mask, rms_error_deg) or None.
"""
import math

def _det3(m):
    return (m[0]*(m[4]*m[8]-m[5]*m[7]) - m[1]*(m[3]*m[8]-m[5]*m[6])
            + m[2]*(m[3]*m[7]-m[4]*m[6]))

def _solve3(m, b):
    d = _det3(m)
    if abs(d) < 1e-12:
        return None
    out = []
    for c in range(3):
        mc = list(m)
        mc[c] = b[0]; mc[3 + c] = b[1]; mc[6 + c] = b[2]
        out.append(_det3(mc) / d)
    return out

def _unit(v):
    s = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    if s < 1e-12:
        return None
    return (v[0]/s, v[1]/s, v[2]/s)

def _lsq(rays, idx, cosa):
    m00 = m01 = m02 = m11 = m12 = m22 = 0.0
    sx = sy = sz = 0.0
    for i in idx:
        x, y, z = rays[i]
        m00 += x*x; m01 += x*y; m02 += x*z
        m11 += y*y; m12 += y*z; m22 += z*z
        sx += x; sy += y; sz += z
    sol = _solve3([m00, m01, m02, m01, m11, m12, m02, m12, m22],
                  [cosa*sx, cosa*sy, cosa*sz])
    return None if sol is None else _unit(sol)

def _err_deg(r, n, alpha_deg):
    d = r[0]*n[0] + r[1]*n[1] + r[2]*n[2]
    if d > 1.0: d = 1.0
    if d < -1.0: d = -1.0
    return math.degrees(math.acos(d)) - alpha_deg

def fit_nadir(rays, alpha, tol_deg=1.5, trials=40, min_inliers=20):
    N = len(rays)
    if N < max(min_inliers, 6):
        return None
    cosa = math.cos(alpha)
    alpha_deg = math.degrees(alpha)
    tol_cos = math.sin(alpha) * math.radians(tol_deg)

    xs = [r[0] for r in rays]
    ys = [r[1] for r in rays]
    axis = 0 if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else 1
    order = sorted(range(N), key=lambda i: rays[i][axis])
    t = N // 3
    groups = (order[:t], order[t:2*t], order[2*t:])

    s = 12345
    per = 3
    best_n, best_cnt = None, 0
    for _ in range(trials):
        sub = []
        for g in groups:
            for _k in range(per):
                s = (75 * s + 74) % 65537
                sub.append(g[s % len(g)])
        n = _lsq(rays, sub, cosa)
        if n is None:
            continue
        cnt = 0
        for r in rays:
            if abs(r[0]*n[0] + r[1]*n[1] + r[2]*n[2] - cosa) < tol_cos:
                cnt += 1
        if cnt > best_cnt:
            best_cnt, best_n = cnt, n
            if cnt > 0.9 * N:
                break
    if best_n is None or best_cnt < min_inliers:
        return None

    n = best_n
    mask = [abs(r[0]*n[0] + r[1]*n[1] + r[2]*n[2] - cosa) < tol_cos for r in rays]
    for _ in range(4):
        idx = [i for i in range(N) if mask[i]]
        if len(idx) < min_inliers:
            break
        n2 = _lsq(rays, idx, cosa)
        if n2 is None:
            break
        n = n2
        errs = [_err_deg(r, n, alpha_deg) for r in rays]
        ae = sorted(abs(errs[i]) for i in idx)
        lim = max(3.0 * 1.4826 * ae[len(ae) // 2], 0.2)
        if lim > 3.0 * tol_deg:
            lim = 3.0 * tol_deg
        new = [abs(e) < lim for e in errs]
        if new == mask:
            break
        mask = new

    idx = [i for i in range(N) if mask[i]]
    if len(idx) < min_inliers:
        return None
    sq = 0.0
    for i in idx:
        e = _err_deg(rays[i], n, alpha_deg)
        sq += e * e
    return n, mask, math.sqrt(sq / len(idx))
