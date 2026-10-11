"""Pinhole camera over a planar scene, homographies and polygon utilities.

World frame (mm): X right along the tray, Y toward the tray front, Z pointing
DOWN into the table. The tray top is Z = 0; things above it have negative Z.
All image coordinates are edge-based (pixel (i, j) spans [i, i+1) x [j, j+1)).
"""
import numpy as np


def rot_axis(axis, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Camera:
    def __init__(self, width, height, hfov_deg, tilt_deg, tilt_azimuth_deg, roll_deg, target, distance):
        self.W, self.H = int(width), int(height)
        self.f = 0.5 * self.W / np.tan(np.deg2rad(hfov_deg) / 2)
        self.hfov, self.tilt, self.tilt_az, self.roll = hfov_deg, tilt_deg, tilt_azimuth_deg, roll_deg
        self.target = np.asarray(target, float)
        self.distance = float(distance)
        self._build()

    def _build(self):
        # Start looking straight down (+Z), image x = world X, image y = world Y.
        R0 = np.eye(3)
        # tilt: rotate the viewing direction away from vertical, around an axis in the XY plane
        tilt = rot_axis("z", self.tilt_az) @ rot_axis("x", self.tilt) @ rot_axis("z", -self.tilt_az)
        Rc2w = tilt @ R0 @ rot_axis("z", self.roll)   # columns: camera axes in world
        self.R = Rc2w.T                               # world -> camera
        view_dir = Rc2w[:, 2]
        self.C = np.array([self.target[0], self.target[1], 0.0]) - self.distance * view_dir
        self.K = np.array([[self.f, 0, self.W / 2], [0, self.f, self.H / 2], [0, 0, 1]])

    def set_distance(self, d):
        self.distance = float(d)
        self._build()

    def plane_h(self, z=0.0):
        """Homography from plane (X, Y, Z=z) in mm to image edge coordinates."""
        t = -self.R @ self.C
        M = np.column_stack([self.R[:, 0], self.R[:, 1], self.R[:, 2] * z + t])
        # not normalised: the homogeneous w stays equal to camera depth (> 0 in front)
        return self.K @ M

    def params(self):
        return dict(cam_hfov_deg=self.hfov, cam_tilt_deg=self.tilt, cam_tilt_az_deg=self.tilt_az,
                    cam_roll_deg=self.roll, cam_distance_mm=self.distance)


def apply_h(H, pts):
    pts = np.asarray(pts, float)
    p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def projects_into(H, pts, W, H_img, margin=0.0):
    """True if all points are in front of the camera and their bbox overlaps the image."""
    pts = np.asarray(pts, float)
    p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    if np.any(p[:, 2] <= 1e-9):
        return False
    q = p[:, :2] / p[:, 2:3]
    return bool(q[:, 0].max() > -margin and q[:, 0].min() < W + margin and
                q[:, 1].max() > -margin and q[:, 1].min() < H_img + margin)


def local_scale(H, pt):
    """sqrt(|det J|) and max singular value of the homography Jacobian at a plane point."""
    eps = 1e-3
    p0 = apply_h(H, [pt])[0]
    px = apply_h(H, [(pt[0] + eps, pt[1])])[0]
    py = apply_h(H, [(pt[0], pt[1] + eps)])[0]
    J = np.column_stack([(px - p0) / eps, (py - p0) / eps])
    sv = np.linalg.svd(J, compute_uv=False)
    return float(np.sqrt(abs(np.linalg.det(J)))), float(sv[0])


def translate(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], float)


def scale(sx, sy=None):
    sy = sx if sy is None else sy
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], float)


def rotate(deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


def to_cv(H):
    """Edge-based homography -> OpenCV pixel-centre convention."""
    return translate(-0.5, -0.5) @ H @ translate(0.5, 0.5)


def polygon_area(poly):
    p = np.asarray(poly, float)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clip_polygon_rect(poly, x0, y0, x1, y1):
    """Sutherland-Hodgman clip of a polygon to an axis-aligned rectangle."""
    def clip(points, inside, intersect):
        out = []
        if not points:
            return out
        prev = points[-1]
        for cur in points:
            if inside(cur):
                if not inside(prev):
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(intersect(prev, cur))
            prev = cur
        return out

    def ix(a, b, x):
        t = (x - a[0]) / (b[0] - a[0])
        return (x, a[1] + t * (b[1] - a[1]))

    def iy(a, b, y):
        t = (y - a[1]) / (b[1] - a[1])
        return (a[0] + t * (b[0] - a[0]), y)

    pts = [tuple(p) for p in np.asarray(poly, float)]
    pts = clip(pts, lambda p: p[0] >= x0, lambda a, b: ix(a, b, x0))
    pts = clip(pts, lambda p: p[0] <= x1, lambda a, b: ix(a, b, x1))
    pts = clip(pts, lambda p: p[1] >= y0, lambda a, b: iy(a, b, y0))
    pts = clip(pts, lambda p: p[1] <= y1, lambda a, b: iy(a, b, y1))
    return np.array(pts, float).reshape(-1, 2)


def convex_gap(a, b):
    """Separating-axis gap between two convex polygons (mm or px).

    > 0: separated by at least this much along some edge normal; <= 0: they
    touch or overlap (value = -penetration along the best axis).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    best = -np.inf
    for poly in (a, b):
        edges = np.roll(poly, -1, axis=0) - poly
        for e in edges:
            n = np.array([-e[1], e[0]])
            norm = np.linalg.norm(n)
            if norm < 1e-12:
                continue
            n /= norm
            pa, pb = a @ n, b @ n
            best = max(best, pb.min() - pa.max(), pa.min() - pb.max())
    return float(best)


def quad_ppm(corners, modules):
    """Pixels per module measured from projected symbol corners."""
    c = np.asarray(corners, float)
    sides = np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)
    return dict(ppm=float(np.sqrt(polygon_area(c)) / modules),
                ppm_min_side=float(sides.min() / modules),
                ppm_max_side=float(sides.max() / modules))
