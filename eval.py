from conformal import get_coverage

def evaluate_conformal(
    preds: torch.Tensor,
    targets: torch.Tensor,
    calibration_scores: torch.Tensor,
    return_intervals: bool = False
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Evaluates conformal prediction performance by calculating coverage percentages.

    Args:
        preds (torch.Tensor): Model predictions
            Shape: (n_samples, pred_len, num_channels)
        targets (torch.Tensor): Ground truth values
            Shape: (n_samples, pred_len, num_channels)
        calibration_scores (torch.Tensor): Critical calibration scores
            Shape: (pred_len,)
        return_intervals (bool, optional): Whether to return prediction interval metrics (mean and std width).

    Returns:
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
            - If return_intervals=False:
                torch.Tensor: Coverage percentages
                Shape: (2,) containing [independent_coverage%, joint_coverage%]
            - If return_intervals=True:
                Tuple containing:
                    - coverage_percentages (torch.Tensor): Shape (2,)
                    - intervals (torch.Tensor): Shape (n_samples, 2, pred_len, num_channels)
    """
    ic, jc, intervals = get_coverage(preds, targets, calibration_scores) # (independent_coverages, joint_coverages, intervals)
    ic_percent = ic.float().mean()*100
    jc_percent = jc.float().mean()*100

    # Compute interval widths
    interval_widths = intervals[:, 1] - intervals[:, 0]  # shape: [n_samples, pred_len, 1]
    mean_width = interval_widths.mean()
    std_width = interval_widths.std()

    if return_intervals:
        return torch.stack([ic_percent, jc_percent]), torch.stack([mean_width, std_width])
    else:
        return torch.stack([ic_percent, jc_percent])
