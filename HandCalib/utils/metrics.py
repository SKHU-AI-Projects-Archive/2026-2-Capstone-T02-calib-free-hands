"""AnyCalib와 공유할 기본 intrinsic camera error를 계산한다."""

import torch


def camera_errors(pred_intrinsics, gt_intrinsics, width, height):
    """[fx, fy, cx, cy] 두 intrinsic 벡터의 상대 오차를 반환한다."""
    pred = torch.as_tensor(pred_intrinsics, dtype=torch.float32)
    gt = torch.as_tensor(gt_intrinsics, dtype=torch.float32, device=pred.device)
    rel_fx_error = torch.abs(pred[..., 0] - gt[..., 0]) / torch.abs(gt[..., 0])
    rel_fy_error = torch.abs(pred[..., 1] - gt[..., 1]) / torch.abs(gt[..., 1])
    max_rel_f_error = torch.maximum(rel_fx_error, rel_fy_error)
    max_rel_c_error = 2 * torch.maximum(
        torch.abs(pred[..., 2] - gt[..., 2]) / width,
        torch.abs(pred[..., 3] - gt[..., 3]) / height,
    )
    return {
        "rel_fx_error": rel_fx_error,
        "rel_fy_error": rel_fy_error,
        "max_rel_f_error": max_rel_f_error,
        "max_rel_c_error": max_rel_c_error,
    }


def focal_error(pred_intrinsics, gt_intrinsics):
    """두 focal length의 최대 상대 오차를 반환한다."""
    return camera_errors(pred_intrinsics, gt_intrinsics, 1, 1)["max_rel_f_error"]
