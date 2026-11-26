"""
horizon_core.py - the ONE shared implementation. Plain Python, no numpy/cv2.
Runs unchanged on the laptop and on the OpenMV H7 Plus (MicroPython).

Pipeline:
  gray buffer (W*H bytes, row-major) -> extract_limb() -> rays_from()
  -> nadir_fit.fit_nadir() -> gates -> result

Limb extraction scans IN from the frame border through dark "space" pixels and
keeps only the FIRST transition to bright. Interior coastlines, clouds and city
lights are never visited, which is what broke the old "any edge near dark" method.
"""
import math
from nadir_fit import fit_nadir

R_E = 6371.0
D_ATM = 20.0


def cone_alpha(alt_km):
    return math.asin((R_E + D_ATM) / (R_E + alt_km))


def fx_from_hfov(width_px, hfov_deg):
    return (width_px / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)


# --------------------------------------------------------------------------- #
# Limb extraction
# --------------------------------------------------------------------------- #
def _scan(buf, start, stride, n, thr, min_run, confirm):
    """Index k of first bright pixel after >= min_run dark pixels, else -1.
    A bright blob shorter than `confirm` pixels (a star) is skipped."""
    k = 0
    while k < n and buf[start + k * stride] < thr:
        k += 1
    if k < min_run or k >= n:
        return -1
    while k < n:
        ok = True
        for j in range(confirm):
            kk = k + j
            if kk >= n or buf[start + kk * stride] < thr:
                ok = False
                break
        if ok:
            return k
        k += 1
        while k < n and buf[start + k * stride] < thr:
            k += 1
    return -1


def extract_limb(buf, W, H, thr=30, step=4, min_run=6, confirm=3, max_pts=280):
    pts = []
    for x in range(0, W, step):
        k = _scan(buf, x, W, H, thr, min_run, confirm)            # from top
        if k >= 0:
            pts.append((float(x), k - 0.5))
        k = _scan(buf, (H - 1) * W + x, -W, H, thr, min_run, confirm)  # from bottom
        if k >= 0:
            pts.append((float(x), H - 1 - k + 0.5))
    for y in range(0, H, step):
        k = _scan(buf, y * W, 1, W, thr, min_run, confirm)        # from left
        if k >= 0:
            pts.append((k - 0.5, float(y)))
        k = _scan(buf, y * W + W - 1, -1, W, thr, min_run, confirm)    # from right
        if k >= 0:
            pts.append((W - 1 - k + 0.5, float(y)))
    if len(pts) > max_pts:
        s = len(pts) // max_pts
        pts = pts[::s]
    return pts


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def rays_from(pts, f, cx, cy, k1=0.0, k2=0.0):
    out = []
    for (u, v) in pts:
        xn = (u - cx) / f
        yn = (v - cy) / f
        if k1 != 0.0 or k2 != 0.0:
            x0, y0 = xn, yn
            for _ in range(4):
                r2 = xn * xn + yn * yn
                d = 1.0 + k1 * r2 + k2 * r2 * r2
                xn = x0 / d
                yn = y0 / d
        m = math.sqrt(xn * xn + yn * yn + 1.0)
        out.append((xn / m, yn / m, 1.0 / m))
    return out


def arc_span_deg(rays, mask, n):
    """Angular extent of the inlier arc, measured around the nadir axis."""
    a = (n[1] * 0 - n[2] * 1, n[2] * 0 - n[0] * 0, n[0] * 1 - n[1] * 0)  # n x (0,1,0)
    la = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    if la < 1e-6:
        a = (1.0, 0.0, 0.0)
    else:
        a = (a[0] / la, a[1] / la, a[2] / la)
    b = (n[1] * a[2] - n[2] * a[1], n[2] * a[0] - n[0] * a[2], n[0] * a[1] - n[1] * a[0])
    ang = []
    for r, k in zip(rays, mask):
        if not k:
            continue
        pa = r[0] * a[0] + r[1] * a[1] + r[2] * a[2]
        pb = r[0] * b[0] + r[1] * b[1] + r[2] * b[2]
        ang.append(math.degrees(math.atan2(pb, pa)))
    if len(ang) < 2:
        return 0.0
    ang.sort()
    gap = 360.0 - (ang[-1] - ang[0])
    for i in range(1, len(ang)):
        if ang[i] - ang[i - 1] > gap:
            gap = ang[i] - ang[i - 1]
    return 360.0 - gap


# --------------------------------------------------------------------------- #
# Full estimate with gates. Returns a dict (same keys on both platforms).
# --------------------------------------------------------------------------- #
def estimate(buf, W, H, f, alpha, thr=30, cx=None, cy=None, k1=0.0, k2=0.0,
             rms_max=None, min_span_deg=8.0, min_inlier_frac=0.5, min_inliers=20,
             confirm=3, min_run=6, max_span_deg=200.0):
    if cx is None:
        cx = W / 2.0
    if cy is None:
        cy = H / 2.0
    if rms_max is None:
        # one pixel is degrees(1/f) wide; quantisation alone gives ~0.3-1 px rms
        rms_max = 1.5 * math.degrees(1.0 / f) + 0.1
    res = {"status": "NO_PTS", "n": None, "roll": 0.0, "pitch": 0.0,
           "rms": 0.0, "inliers": 0, "npts": 0, "span": 0.0, "mask": None, "pts": []}
    pts = extract_limb(buf, W, H, thr=thr, confirm=confirm, min_run=min_run)
    res["pts"] = pts
    res["npts"] = len(pts)
    if len(pts) < min_inliers:
        return res
    rays = rays_from(pts, f, cx, cy, k1, k2)
    fit = fit_nadir(rays, alpha, min_inliers=min_inliers)
    if fit is None:
        res["status"] = "NO_FIT"
        return res
    n, mask, rms = fit
    n_in = sum(1 for m in mask if m)
    span = arc_span_deg(rays, mask, n)
    res.update({"n": n, "rms": rms, "inliers": n_in, "span": span, "mask": mask,
                "roll": math.degrees(math.atan2(n[0], n[2])),
                "pitch": math.degrees(math.atan2(n[1], n[2]))})
    if rms > rms_max:
        res["status"] = "BAD_RMS"
    elif n_in < min_inlier_frac * len(pts):
        res["status"] = "LOW_FRAC"
    elif span < min_span_deg:
        res["status"] = "SHORT_ARC"
    elif span > max_span_deg:
        # a wide lens cannot see most of a 140+ deg limb circle: points all around
        # the frame mean the fit locked onto something that is not the limb
        res["status"] = "BAD_ARC"
    else:
        res["status"] = "OK"
    return res


def limb_curve(n, alpha, f, W, H, steps=120):
    """Fitted limb as a list of (x, y) pixel points (None = off-screen break).
    Shared by the laptop overlay and the OpenMV overlay so both draw the same curve."""
    a = (-n[2], 0.0, n[0])                      # n x (0,1,0)
    la = math.sqrt(a[0] * a[0] + a[2] * a[2])
    a = (1.0, 0.0, 0.0) if la < 1e-6 else (a[0] / la, 0.0, a[2] / la)
    b = (n[1] * a[2] - n[2] * a[1], n[2] * a[0] - n[0] * a[2], n[0] * a[1] - n[1] * a[0])
    ca, sa = math.cos(alpha), math.sin(alpha)
    out = []
    for i in range(steps + 1):
        ph = 2.0 * math.pi * i / steps
        c, s = math.cos(ph), math.sin(ph)
        p0 = ca * n[0] + sa * (c * a[0] + s * b[0])
        p1 = ca * n[1] + sa * (c * a[1] + s * b[1])
        p2 = ca * n[2] + sa * (c * a[2] + s * b[2])
        if p2 > 0.02:
            x = f * p0 / p2 + W / 2.0
            y = f * p1 / p2 + H / 2.0
            out.append((x, y) if abs(x) < 2000 and abs(y) < 2000 else None)
        else:
            out.append(None)
    return out


def nad_line(r):
    if r["status"] != "OK":
        return "NAD,nan,nan,nan,nan,nan,nan,%d,%s" % (r["npts"], r["status"])
    n = r["n"]
    return "NAD,%.6f,%.6f,%.6f,%+.3f,%+.3f,%.3f,%d,OK" % (
        n[0], n[1], n[2], r["roll"], r["pitch"], r["rms"], r["inliers"])


def horz_line(counter, r):
    if r["status"] != "OK":
        return "HORZ,C%d,NO_FIX,pts%d\r\n" % (counter, r["npts"])
    n = r["n"]
    return "HORZ,C%d,ex%.6f,ey%.6f,ez%.6f,roll%.2f,pitch%.2f,res%.2f,pts%d\r\n" % (
        counter, n[0], n[1], n[2], r["roll"], r["pitch"], r["rms"], r["inliers"])
