import numpy as np

# ---------- basics ----------

def _compose_vector(p1, p2):
    """Vector from p1 to p2."""
    return np.asarray(p2) - np.asarray(p1)

def _unit(v, eps=1e-12):
    """Unit vector with safe handling of near-zero norm."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v if n < eps else v / n

def _compose_unit_vector(p1, p2):
    """Vector from p1 to p2."""
    return _unit(np.asarray(p2) - np.asarray(p1))

def _compose_plane_normal(p1, p2, p3):
    """Normal of plane defined by three points."""
    v1 = _compose_unit_vector(p1, p2)
    v2 = _compose_unit_vector(p1, p3)
    return np.cross(v1, v2)


# ---------- Unified Cost Scaling Constants ----------
# We align angular errors so that 15 degrees of error produces a cost of ~0.01 (1 cm).

REF_POS_ERR = 0.01
REF_ROT_ERR_RAD = np.deg2rad(20.0)

# 1. Linear angular error scaling (e.g. _theta_cost)
# Cost = |error_rad| * THETA_COST_SCALE
# 0.01 = REF_ROT_ERR_RAD * THETA_COST_SCALE
THETA_COST_SCALE = REF_POS_ERR / REF_ROT_ERR_RAD

# 2. Parallel cost scaling (linear angle error)
# Cost = |error_rad| * PARALLEL_COST_SCALE
# 0.01 = REF_ROT_ERR_RAD * PARALLEL_COST_SCALE
PARALLEL_COST_SCALE = REF_POS_ERR / REF_ROT_ERR_RAD

# 3. Orthogonal cost scaling (linear angle error)
# Cost = |error_rad| * ORTHOGONAL_COST_SCALE
# 0.01 = REF_ROT_ERR_RAD * ORTHOGONAL_COST_SCALE
ORTHOGONAL_COST_SCALE = REF_POS_ERR / REF_ROT_ERR_RAD


# ---------- Vector–vector angular relationships ----------

def _pos_cost(p1, p2):
    return np.linalg.norm(p1 - p2)

def _high_precision_pos_cost(p1, p2):
    return np.linalg.norm(p1 - p2) * 10

def _parallel_cost(v1, v2):
    """
    Enforce v1 ∥ v2.
    Cost scaled so that 15° misalignment ~= 0.01.
    """
    u1, u2 = _unit(v1), _unit(v2)
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    angle = np.arccos(abs(dot))
    return angle * PARALLEL_COST_SCALE

def _samedir_cost(v1, v2):
    """
    Enforce v1 to has the same direction with v2.
    Cost scaled so that 15° misalignment ~= 0.01.
    """
    u1, u2 = _unit(v1), _unit(v2)
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    angle = np.arccos(dot)
    return angle * PARALLEL_COST_SCALE

def _samedir_xy_cost(v1, v2):
    u1 = v1.copy()
    u2 = v2.copy()
    u1[2] = 0
    u2[2] = 0
    u1, u2 = _unit(u1), _unit(u2)
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    angle = np.arccos(dot)
    return angle * PARALLEL_COST_SCALE

def _orthogonal_cost(v1, v2):
    """
    Enforce v1 ⟂ v2.
    Cost scaled so that 15° deviation from orthogonality ~= 0.01.
    """
    u1, u2 = _unit(v1), _unit(v2)
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    # deviation from 90 deg = |arccos(dot) - pi/2| = arcsin(|dot|)
    return np.arcsin(abs(dot)) * ORTHOGONAL_COST_SCALE

def _theta_cost(v1, v2, theta):
    """
    Enforce angle(v1, v2) == theta (radians).
    Cost is absolute angle error (radians) scaled so 15° error ~= 0.01.
    """
    u1, u2 = _unit(v1), _unit(v2)
    dot = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    angle = np.arccos(dot)
    return abs(angle - float(theta)) * THETA_COST_SCALE

def _xy_project(v):
    v1 = v.copy()
    v1[2] = 0
    v1 = _unit(v1)
    return v1

def _gripper_left_to_right(keypoints):
    gripper_left = keypoints[-2]
    gripper_right = keypoints[-1]
    v_left_to_right = _unit(gripper_right - gripper_left)
    return v_left_to_right

# ---------- 2D triangle membership (for plane projections) ----------

def _barycentric_violation_cost_2d(p, a, b, c, eps=1e-12):
    """
    Return 0 if p is inside triangle (a,b,c) in 2D; otherwise a soft penalty.
    Penalty is the sum of ReLU violations of barycentric coordinates.
    """
    p, a, b, c = map(lambda x: np.asarray(x, dtype=float), (p, a, b, c))
    v0 = b - a
    v1 = c - a
    v2 = p - a

    # Solve for barycentric (u, v) s.t. p = a + u*v0 + v*v1
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01

    if abs(denom) < eps:
        # Degenerate triangle: treat as distance to line/point via simple fallback
        # Here, return small penalty equal to min distance to edges
        def seg_dist(pa, pb, pp):
            v = pb - pa
            t = 0.0 if np.allclose(v, 0) else np.clip(np.dot(pp - pa, v) / (np.dot(v, v) + eps), 0.0, 1.0)
            proj = pa + t * v
            return np.linalg.norm(pp - proj)
        return min(seg_dist(a, b, p), seg_dist(b, c, p), seg_dist(c, a, p))

    u = (d11 * d20 - d01 * d21) / denom
    v = (d00 * d21 - d01 * d20) / denom
    w = 1.0 - u - v

    # Penalize leaving [0,1] for each barycentric coord
    relu = lambda x: max(0.0, x)
    return relu(-u) + relu(u - 1.0) + relu(-v) + relu(v - 1.0) + relu(-w) + relu(w - 1.0)

def _point_in_plane_xy_cost(p, plane_p1, plane_p2, plane_p3):
    """
    Enforce that p's XY projection lies inside the XY triangle formed by plane points.
    Cost >= 0 (0 if inside).
    """
    P  = np.array([p[0],         p[1]],         dtype=float)
    A  = np.array([plane_p1[0],  plane_p1[1]],  dtype=float)
    B  = np.array([plane_p2[0],  plane_p2[1]],  dtype=float)
    C  = np.array([plane_p3[0],  plane_p3[1]],  dtype=float)
    return _barycentric_violation_cost_2d(P, A, B, C)

def _point_in_plane_yz_cost(p, plane_p1, plane_p2, plane_p3):
    """
    Enforce that p's YZ projection lies inside the YZ triangle formed by plane points.
    Cost >= 0 (0 if inside).
    """
    P  = np.array([p[1],         p[2]],         dtype=float)
    A  = np.array([plane_p1[1],  plane_p1[2]],  dtype=float)
    B  = np.array([plane_p2[1],  plane_p2[2]],  dtype=float)
    C  = np.array([plane_p3[1],  plane_p3[2]],  dtype=float)
    return _barycentric_violation_cost_2d(P, A, B, C)

def _point_in_plane_xz_cost(p, plane_p1, plane_p2, plane_p3):
    """
    Enforce that p's XZ projection lies inside the XZ triangle formed by plane points.
    Cost >= 0 (0 if inside).
    """
    P  = np.array([p[0],         p[2]],         dtype=float)
    A  = np.array([plane_p1[0],  plane_p1[2]],  dtype=float)
    B  = np.array([plane_p2[0],  plane_p2[2]],  dtype=float)
    C  = np.array([plane_p3[0],  plane_p3[2]],  dtype=float)
    return _barycentric_violation_cost_2d(P, A, B, C)

# ---------- Axis-aligned bounding box (AABB) ----------

def _point_in_aabb_cost(p, c1, c2, c3, c4):
    """
    Enforce that point p lies inside the axis-aligned bounding box defined
    by four corner points (min/max taken componentwise across provided corners).
    Cost is the L2 norm of per-axis violations (0 if inside).
    """
    corners = np.vstack([c1, c2, c3, c4]).astype(float)
    p = np.asarray(p, dtype=float)
    lo = np.min(corners, axis=0)
    hi = np.max(corners, axis=0)

    # Per-axis outside distances (violations)
    below = np.maximum(0.0, lo - p)
    above = np.maximum(0.0, p - hi)
    violation = below + above
    return float(np.linalg.norm(violation))

def _angle_30deg_from_neg_z_cost(v):
    """
    Enforce angle(v, [0,0,-1]) == 30°.
    Returns absolute angle error in radians (0 when exactly 30°).
    """
    target = np.array([0.0, 0.0, -1.0], dtype=float)
    return _theta_cost(v, target, np.deg2rad(30.0))


def _angle_60deg_from_neg_z_cost(v):
    """
    Enforce angle(v, [0,0,-1]) == 60°.
    Returns absolute angle error in radians (0 when exactly 60°).
    """
    target = np.array([0.0, 0.0, -1.0], dtype=float)
    return _theta_cost(v, target, np.deg2rad(60.0))


def _angle_75deg_from_neg_z_cost(v):
    """
    Enforce angle(v, [0,0,-1]) == 60°.
    Returns absolute angle error in radians (0 when exactly 75°).
    """
    target = np.array([0.0, 0.0, -1.0], dtype=float)
    return _theta_cost(v, target, np.deg2rad(75.0))
