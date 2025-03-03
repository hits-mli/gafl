# Copyright (c) 2024 HITS gGmbH

# replace the quaternion calculation of openfold by a version that does not use torch.eigh and is thus differentiable wrt the rotation matrices.

import torch

def rot_to_quat(rot):
    if rot.shape[-2:] != (3, 3):
        raise ValueError("Input rotation matrix must be of shape [..., 3, 3]")
    
    orig_shape = rot.shape

    rot = rot.view(-1, 3, 3)

    # Compute the trace of the rotation matrices
    r_trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]

    # Preallocate the quaternion tensor with zeros
    q = torch.zeros(rot.shape[:-2] + (4,), dtype=rot.dtype, device=rot.device)

    # Determine conditions for each case
    case1 = r_trace > 0
    case2 = (rot[:, 0, 0] > rot[:, 1, 1]) & (rot[:, 0, 0] > rot[:, 2, 2]) & ~case1
    case3 = (rot[:, 1, 1] > rot[:, 2, 2]) & ~case1 & ~case2
    case4 = ~(case1 | case2 | case3)  # Remaining cases

    # Compute quaternions for case 1
    s1 = torch.sqrt(r_trace[case1] + 1.0) * 2  # S=4*qw
    qw1 = 0.25 * s1
    qx1 = (rot[case1, 2, 1] - rot[case1, 1, 2]) / s1
    qy1 = (rot[case1, 0, 2] - rot[case1, 2, 0]) / s1
    qz1 = (rot[case1, 1, 0] - rot[case1, 0, 1]) / s1
    q[case1] = torch.stack([qw1, qx1, qy1, qz1], dim=-1)

    # Compute quaternions for case 2
    s2 = torch.sqrt(1.0 + rot[case2, 0, 0] - rot[case2, 1, 1] - rot[case2, 2, 2]) * 2
    qw2 = (rot[case2, 2, 1] - rot[case2, 1, 2]) / s2
    qx2 = 0.25 * s2
    qy2 = (rot[case2, 0, 1] + rot[case2, 1, 0]) / s2
    qz2 = (rot[case2, 0, 2] + rot[case2, 2, 0]) / s2
    q[case2] = torch.stack([qw2, qx2, qy2, qz2], dim=-1)

    # Compute quaternions for case 3
    s3 = torch.sqrt(1.0 + rot[case3, 1, 1] - rot[case3, 0, 0] - rot[case3, 2, 2]) * 2
    qw3 = (rot[case3, 0, 2] - rot[case3, 2, 0]) / s3
    qx3 = (rot[case3, 0, 1] + rot[case3, 1, 0]) / s3
    qy3 = 0.25 * s3
    qz3 = (rot[case3, 1, 2] + rot[case3, 2, 1]) / s3
    q[case3] = torch.stack([qw3, qx3, qy3, qz3], dim=-1)

    # Compute quaternions for case 4
    s4 = torch.sqrt(1.0 + rot[case4, 2, 2] - rot[case4, 0, 0] - rot[case4, 1, 1]) * 2
    qw4 = (rot[case4, 1, 0] - rot[case4, 0, 1]) / s4
    qx4 = (rot[case4, 0, 2] + rot[case4, 2, 0]) / s4
    qy4 = (rot[case4, 1, 2] + rot[case4, 2, 1]) / s4
    qz4 = 0.25 * s4
    q[case4] = torch.stack([qw4, qx4, qy4, qz4], dim=-1)

    q = q.view(orig_shape[:-2] + (4,))

    return q