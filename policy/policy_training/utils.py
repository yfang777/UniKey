import numpy as np
import torch
from torch.nn import functional as F

def rot6d_to_rotmat(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)

def rotmat_to_rot6d(matrix):
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

def _calculate_action_state(dataloader, horizon):
    '''
    Calculates the min and max of relative actions (geometrically relative to obs_0)
    across the entire dataset.
    
    Args:
        dataloader: torch.utils.data.DataLoader
        horizon: int
        
    Returns:
        action_stats_max: np.array [horizon, 10]
        action_stats_min: np.array [horizon, 10]
    '''
    
    all_relative_actions = []
    
    for batch in dataloader:
        # batch['action'] is updated to be relative to batch['observation.state'][:, 0]
        rel_batch = abs_to_rel_actions(batch)
        
        # Extract the relative actions
        actions = rel_batch["action"]
        actions = actions.cpu().numpy()
            
        all_relative_actions.append(actions)
    
    # Stack all batches: [N_total, T, 10]
    total_relative_actions = np.concatenate(all_relative_actions, axis=0)
    
    # Calculate global stats across the dataset
    action_stats_max = np.max(total_relative_actions, axis=0)   # [horizon, 10]
    action_stats_min = np.min(total_relative_actions, axis=0)   # [horizon, 10]
    
    

    return action_stats_max, action_stats_min

def update_relative_action_stats(stats, horizon, dataloader):
    '''
    Updates the stats dictionary with the calculated relative action max/min.
    Args:
        stats: dict
        horizon: int
        dataloader: torch.utils.data.DataLoader
    '''

    action_max, action_min = _calculate_action_state(dataloader, horizon)

    stats["action"]["max"] = action_max
    stats["action"]["min"] = action_min

    assert action_max.shape == (horizon, 10)
    assert action_min.shape == (horizon, 10)

def abs_to_rel_actions(batch: dict) -> dict:
    """
    Convert absolute actions to relative actions w.r.t. obs_0 for each batch item.
    """
    
    state  = batch['observation.state']
    action = batch['action']
    assert len(state.size()) == 3 and state.size()[2] == 10
    assert len(action.size()) == 3 and action.size()[2] == 10

    B, T, _ = action.shape

    # 1. Extract obs_0 pose (Position and Rotation)
    obs0 = state[:, 0, :]                       # [B, 10]
    p_0 = obs0[:, :3].unsqueeze(1)              # [B, 1, 3]
    R_0 = rot6d_to_rotmat(obs0[:, 3:9])   # [B, 3, 3]
    R_0 = R_0.unsqueeze(1)                      # [B, 1, 3, 3]

    # 2. Extract action pose (Position and Rotation)
    p_t = action[:, :, :3]                      # [B, T, 3]
    
    # Reshape to (Batch * Time, 6) for conversion, then back to (Batch, Time, 3, 3)
    rot_6d_flat = action[:, :, 3:9].reshape(-1, 6)
    R_t = rot6d_to_rotmat(rot_6d_flat).reshape(B, T, 3, 3)
    
    gripper_t = action[:, :, 9].unsqueeze(-1)   # [B, T, 1]

    # 3. Calculate Relative Rotation: R_rel = R_0^T @ R_t
    R_0_inv = R_0.transpose(-1, -2)             # Orthogonal matrix inverse is transpose
    R_rel_mat = torch.matmul(R_0_inv, R_t)      # [B, T, 3, 3]

    # 4. Calculate Relative Position: p_rel = R_0^T @ (p_t - p_0)
    delta_p = p_t - p_0                         # [B, T, 3]
    p_rel = torch.matmul(R_0_inv, delta_p.unsqueeze(-1)).squeeze(-1) # [B, T, 3]

    # 5. Convert Relative Rotation back to 6D
    R_rel_6d = rotmat_to_rot6d(R_rel_mat.reshape(-1, 3, 3)).reshape(B, T, 6)

    # 6. Concatenate: [pos, rot6d, gripper]
    rel_action = torch.cat([p_rel, R_rel_6d, gripper_t], dim=-1)

    new_batch = dict(batch)
    new_batch['action'] = rel_action
    return new_batch


def rel_to_abs_actions(batch: dict) -> dict:
    """
    Batch dictionary wrapper for rel_to_abs_action.

    Args:
        batch:
            state: (B, K, 10) -> [pos(3), rot6d(6), gripper(1)]
            action: (B, T, 10) -> [rel_pos(3), rel_rot6d(6), gripper(1)]

    Returns:
        batch:
            state: (B, K, 10) -> [pos(3), rot6d(6), gripper(1)]
            action: (B, T, 10) -> [rel_pos(3), rel_rot6d(6), gripper(1)]
    """
    state = batch['observation.state']
    action_rel = batch['action']
    
    
    assert len(state.size()) == 3 and state.size()[2] == 10
    assert len(action_rel.size()) == 3 and action_rel.size()[2] == 10
    B, T, _ = action_rel.size()

    p_0 = state[:, 0:1, :3].expand(B, T, 3)
    R_0 = rot6d_to_rotmat(state[:, 0, 3:9]).unsqueeze(1).expand(B, T, 3, 3)
    p_rel = action_rel[:, :, :3]                # [B, T, 3]
    R_rel = rot6d_to_rotmat(action_rel[:, :, 3:9].reshape(-1, 6)).reshape(B, T, 3, 3)
    gripper = action_rel[:, :, 9].unsqueeze(-1)

    R_t_mat = torch.matmul(R_0.reshape(-1, 3, 3) , R_rel.reshape(-1, 3, 3))
    R_t_6d = rotmat_to_rot6d(R_t_mat).reshape(B, T, 6)

    p_t = p_0 + torch.matmul(R_0.reshape(-1, 3, 3), p_rel.reshape(-1, 3, 1)).reshape(B, T, 3)


    # Concatenate
    abs_action = torch.cat([p_t, R_t_6d, gripper], dim=-1)

    new_batch = dict(batch)
    new_batch['action'] = abs_action
    return new_batch

def add_noise_to_state(batch: dict):
    """
    Adds Gaussian noise to observation state
    """
    state = batch['observation.state']

    assert len(state.size()) == 3 and state.size()[2] == 10, \
        f"Expected state shape (Batch, Time, 10), got {state.size()}"

    noisy_state = state.clone()
    
    device = state.device
    dtype = state.dtype
    batch_size, time_steps, _ = state.shape

    pos_std = 0.001
    pos_noise = torch.randn(batch_size, time_steps, 3, device=device, dtype=dtype) * pos_std
    noisy_state[:, :, :3] += pos_noise

    # 2. Add noise to Angles (indices 3:9) - Rotation 6D
    angle_std_deg = 1.0
    angle_std_rad = np.deg2rad(angle_std_deg)

    # A. Generate random axis-angle perturbations
    # Random 3D axis vectors
    rand_axes = torch.randn(batch_size, time_steps, 3, device=device, dtype=dtype)
    rand_axes = rand_axes / (torch.norm(rand_axes, dim=-1, keepdim=True) + 1e-8)
    
    # Random angles (magnitudes) from Gaussian
    rand_angles = torch.randn(batch_size, time_steps, 1, device=device, dtype=dtype) * angle_std_rad
    
    # B. Convert Axis-Angle to Rotation Matrix (Rodrigues' formula)
    # K is the skew-symmetric matrix of the axis
    kx, ky, kz = rand_axes[..., 0], rand_axes[..., 1], rand_axes[..., 2]
    zeros = torch.zeros_like(kx)
    
    # Stack to form K matrix: (B, T, 3, 3)
    K = torch.stack([
        zeros, -kz, ky,
        kz, zeros, -kx,
        -ky, kx, zeros
    ], dim=-1).view(batch_size, time_steps, 3, 3)
    
    I = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3).expand(batch_size, time_steps, 3, 3)
    theta = rand_angles.unsqueeze(-1) # Broadcastable shape
    
    # R_noise = I + sin(theta)K + (1-cos(theta))K^2
    R_noise = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)

    # C. Apply rotation to the 6D representation
    # The 6D representation consists of two 3D vectors (col1, col2) flattened.
    # Reshape to (Batch, Time, 2, 3) then permute to (Batch, Time, 3, 2) for matrix mult
    rot6d_vecs = state[:, :, 3:9].view(batch_size, time_steps, 2, 3).permute(0, 1, 3, 2)
    
    # Rotate the vectors: R_noise @ [v1, v2]
    rotated_vecs = torch.matmul(R_noise, rot6d_vecs)
    
    # Flatten back to 6D: (B, T, 3, 2) -> (B, T, 2, 3) -> (B, T, 6)
    noisy_state[:, :, 3:9] = rotated_vecs.permute(0, 1, 3, 2).reshape(batch_size, time_steps, 6)

    # Update batch
    new_batch = dict(batch)
    new_batch['observation.state'] = noisy_state

    return new_batch

def _rot6d_to_mat_np(d6: np.ndarray) -> np.ndarray:
    """Convert a (6,) 6D rotation repr to a (3,3) rotation matrix (numpy, Gram-Schmidt).
    Matches the torch ``rot6d_to_rotmat`` convention: b1, b2, b3 are **rows**."""
    a1, a2 = d6[:3], d6[3:]
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)            # (3, 3)  rows = b1, b2, b3


def inject_noise(
    kin_helper,
    batch: dict,
    alpha: float = 0.1,
    pos_cap: float = 0.02,
    rot_cap: float = 0.1,
    rot_scale: float = 0.1,
) -> dict:
    """
    Jacobian-based TCP noise injection for a training batch.

    For every (sample, timestep) in the action chunk the procedure is:

    1.  Recover the current joint configuration *q* from ``observation.state``
        via IK (using kin_helper).
    2.  Compute the 6×13 end-effector Jacobian **J** at *q*.
    3.  Compute the intended TCP displacement  Δx  from the action delta
        (position part only used for magnitude).
    4.  Generate a 6-D TCP noise vector ε_tcp:
          - Position:  σ_pos = α · ‖Δx_pos‖,  capped at ±pos_cap (2 cm).
          - Rotation:  σ_rot = rot_scale · σ_pos,  capped at ±rot_cap (≈5.7°).
    5.  Project to joint space and back:  ε_tcp_proj = J · (J⁺ · ε_tcp).
        This ensures the noise is physically achievable by the arm.
    6.  Add the projected TCP noise to the action.

    Args:
        kin_helper:  ``KinHelper`` instance (provides IK + Jacobian).
        batch:       Training batch dict with keys
                       ``observation.state``  (B, K, 10)  [pos3, rot6d6, gripper1]
                       ``action``             (B, T, 10)  [pos3, rot6d6, gripper1]
        alpha:       Fraction of TCP displacement → σ_pos  (default 0.1 = 10 %).
        pos_cap:     Maximum positional noise (m).  Default ±0.02 = ±2 cm.
        rot_cap:     Maximum rotational noise (rad). Default ±0.1 ≈ ±5.7°.
        rot_scale:   σ_rot = rot_scale * σ_pos.     Default 0.1.

    Returns:
        New batch dict with noise added to ``action``.
    """
    state  = batch["observation.state"]              # (B, K, 10)
    action = batch["action"]                         # (B, T, 10)

    B, T, _ = action.shape

    # Work in numpy for the Jacobian / IK calls; convert back at the end.
    action_np = action.detach().cpu().numpy().copy()  # (B, T, 10)
    state_np  = state.detach().cpu().numpy()           # (B, K, 10)

    reset_qpos = np.array([0., -0.78539816, 0., 0.52359878, 0., 1.30899694, 0.,
                           0., 0., 0., 0., 0., 0.])  # 13-D default

    for b in range(B):
        # --- recover joint config from obs_0 via IK ---
        obs0 = state_np[b, 0]                       # (10,)
        pos  = obs0[:3]                              # (3,)
        r6d  = obs0[3:9]                             # (6,)

        R = _rot6d_to_mat_np(r6d)                    # (3, 3)
        tf = np.eye(4)
        tf[:3, :3] = R
        tf[:3, 3]  = pos

        qpos, ik_ok, _err = kin_helper.compute_ik_from_mat(reset_qpos.copy(), tf)
        if not ik_ok:
            # Fallback: use reset config (noise will still be generated,
            # just with a less accurate Jacobian).
            qpos = reset_qpos.copy()

        # --- Jacobian at this config ---
        kin_helper.robot_model.compute_full_jacobian(qpos)
        J = np.array(
            kin_helper.robot_model.get_link_jacobian(
                kin_helper.sapien_eef_idx, local=False
            )
        )  # (6, 13)

        for t in range(T):
            act = action_np[b, t]                    # (10,)

            # Intended TCP delta (position part) relative to obs_0
            dx_pos = act[:3] - pos                   # (3,)
            intended_dist = np.linalg.norm(dx_pos)

            # σ computation
            sigma_pos = min(alpha * intended_dist, pos_cap)
            sigma_rot = min(rot_scale * sigma_pos, rot_cap)

            # Generate noise in TCP space  (6-D: [pos3, angvel3])
            eps_pos = np.random.normal(0, max(sigma_pos, 1e-12), size=3)
            eps_pos = np.clip(eps_pos, -pos_cap, pos_cap)

            eps_rot = np.random.normal(0, max(sigma_rot, 1e-12), size=3)
            eps_rot = np.clip(eps_rot, -rot_cap, rot_cap)

            eps_tcp = np.concatenate([eps_pos, eps_rot])  # (6,)

            # Project through Jacobian pseudo-inverse → joint → TCP
            #   ε_q   = J⁺ · ε_tcp      (least-squares, 13-D)
            #   ε_proj = J · ε_q         (6-D, physically realisable)
            eps_q, _, _, _ = np.linalg.lstsq(J, eps_tcp, rcond=None)
            eps_proj = J @ eps_q                     # (6,)

            # Apply position noise
            action_np[b, t, :3] += eps_proj[:3]

            # Apply rotation noise (small-angle: axis-angle → rotation matrix)
            dtheta = eps_proj[3:]                    # angular velocity noise
            angle = np.linalg.norm(dtheta)
            if angle > 1e-8:
                k = dtheta / angle
                K = np.array([[   0., -k[2],  k[1]],
                              [ k[2],    0., -k[0]],
                              [-k[1],  k[0],    0.]])
                R_noise = (np.eye(3)
                           + np.sin(angle) * K
                           + (1 - np.cos(angle)) * (K @ K))
                R_act = _rot6d_to_mat_np(action_np[b, t, 3:9])
                R_new = R_noise @ R_act
                # Write back as row-major 6D (first two rows of R)
                action_np[b, t, 3:9] = R_new[:2, :].reshape(6)

            # Gripper (index 9) is untouched.

    noisy_action = torch.from_numpy(action_np).to(
        dtype=action.dtype, device=action.device
    )

    new_batch = dict(batch)
    new_batch["action"] = noisy_action
    return new_batch



if __name__ == "__main__":
    
    print("--- Test 1: Rotation Normalization Robustness ---")
    p2_rot6d = torch.Tensor([[1, 0, 0, 0.1, 0.9, 0]]) # Batch size 1
    
    rot_mat = rot6d_to_rotmat(p2_rot6d)
    det = torch.det(rot_mat)
    print(det)
    print("rot_mat:", rot_mat)
    # Check orthogonality: R @ R^T should be Identity
    identity_check = torch.matmul(rot_mat, rot_mat.transpose(1, 2))
    is_identity = torch.allclose(identity_check, torch.eye(3).unsqueeze(0), atol=1e-6)
    
    p2_rot6d_mapped_back = rotmat_to_rot6d(rot_mat)
    print("p2_rot6d_mapped_back:", p2_rot6d_mapped_back)
    assert is_identity
    print("Pass.\n")

    print("--- Test 2: Absolute -> Relative -> Absolute Round Trip ---")
    B, T = 2, 5
    torch.manual_seed(42)

    base_pos = torch.tensor([0.5, 0.5, 0.5]).expand(B, 3)
    base_rot6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).expand(B, 6)
    base_grip = torch.tensor([0.0]).expand(B, 1)

    obs_clean = torch.cat([base_pos, base_rot6d, base_grip], dim=-1).unsqueeze(1) # [B, 1, 10]
    
    # Add Noise to Obs
    obs_state = obs_clean + 0.01 * torch.randn(B, 1, 10)
    
    steps = torch.arange(1, T + 1, dtype=torch.float32).reshape(1, T, 1).expand(B, T, 1)
    delta_pos = torch.zeros(B, T, 3)
    delta_pos[:, :, 0] = steps[:, :, 0] * 0.1
    action_pos_clean = base_pos.unsqueeze(1) + delta_pos
    action_rot6d_clean = base_rot6d.unsqueeze(1).expand(B, T, 6)
    action_grip_clean = base_grip.unsqueeze(1).expand(B, T, 1)

    actions_clean = torch.cat([action_pos_clean, action_rot6d_clean, action_grip_clean], dim=-1)

    abs_actions_original = actions_clean + 0.01 * torch.randn(B, T, 10)


    batch = {
        'observation.state': obs_clean,
        'action': abs_actions_original
    }
    
    batch_rel = abs_to_rel_actions(batch)
    rel_actions = batch_rel['action']

    batch_recovered = rel_to_abs_actions(batch_rel)
    abs_actions_recovered = batch_recovered['action']
    
    pos_diff = torch.abs(abs_actions_original[..., :3] - abs_actions_recovered[..., :3]).max()
    grip_diff = torch.abs(abs_actions_original[..., 9] - abs_actions_recovered[..., 9]).max()
    
    rot6d_orig = abs_actions_original[..., 3:9].reshape(-1, 6)
    rot6d_recov = abs_actions_recovered[..., 3:9].reshape(-1, 6)
    
    mat_orig = rot6d_to_rotmat(rot6d_orig)
    mat_recov = rot6d_to_rotmat(rot6d_recov)
    
    rot_diff = torch.abs(mat_orig - mat_recov).max()
    
    print("pos_diff:", pos_diff)
    print("grip_diff:", grip_diff)
    print("rot_diff:", rot_diff)

    assert pos_diff < 1e-5, "Position reconstruction failed"
    assert grip_diff < 1e-5, "Gripper reconstruction failed"
    assert rot_diff < 1e-5,  "Rotation reconstruction failed"
    print("Pass.\n")
