# Adapted from: https://github.com/kamilest/conformal-rnn/blob/master/models/cfrnn.py

import torch
import torch.nn as nn
from torch.nn.functional import l1_loss
from typing import Tuple

def coverage(
    intervals: torch.Tensor,
    target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Determines whether intervals cover the target prediction
    considering each target horizon either separately or jointly.

    Args:
        intervals (torch.Tensor): Prediction intervals containing lower and upper bounds
            Shape: (n_samples, 2, horizon, num_channels)
            Axis 1 contains [lower_bound, upper_bound]
        target (torch.Tensor): Ground truth forecast values
            Shape: (n_samples, horizon, num_channels)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - horizon_coverages (torch.Tensor): Coverage for each horizon separately
              Shape: (n_samples, horizon, num_channels)
              Boolean tensor indicating if target is within bounds
            - joint_coverages (torch.Tensor): Coverage across all horizons
              Shape: (n_samples, num_channels)
              Boolean tensor indicating if target is within bounds for all horizons
    """

    lower, upper = intervals[:, 0], intervals[:, 1]
    # [n_samples, horizon, num_channels]
    horizon_coverages = torch.logical_and(target >= lower, target <= upper)
    # [n_samples, horizon, num_channels], [n_samples, num_channels]
    return horizon_coverages, torch.all(horizon_coverages, dim=1)

def get_critical_scores(
    calibration_scores: torch.Tensor,
    q: float
) -> torch.Tensor:
    """
    Computes critical calibration scores from scores in the calibration set.

    Args:
        calibration_scores (torch.Tensor): Calibration scores for each example in the
            calibration set. Shape: (num_channels, n_samples, pred_len)
        q (float): Target quantile for which to return the calibration score,
            value between 0 and 1

    Returns:
        torch.Tensor: Critical calibration scores for each target horizon
            Shape: (pred_len, num_channels)
            Contains quantile values computed for each position and feature
    """

    return torch.tensor(
        [
            [
                torch.quantile(position_calibration_scores, q=q)
                for position_calibration_scores in feature_calibration_scores
            ]
            for feature_calibration_scores in calibration_scores
        ]
    ).T

def nonconformity(
    output: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Measures the nonconformity between output and target time series.

    Args:
        output (torch.Tensor): Model forecast for the target
            Shape: (batch_size, pred_len, num_channels)
        target (torch.Tensor): The target time series
            Shape: (batch_size, pred_len, num_channels)

    Returns:
        torch.Tensor: Average MAE loss for every step in the sequence
            Shape: (pred_len, num_channels)
            Computed using element-wise L1 loss without reduction
    """
    # Average MAE loss for every step in the sequence.
    return l1_loss(output, target, reduction="none")

def get_all_critical_scores(
    preds: torch.Tensor,
    targets: torch.Tensor,
    alpha: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the nonconformity scores for the calibration dataset.

    Args:
        preds (torch.Tensor): All model predictions over the entire calibration set
            Shape: (n_samples, pred_len, num_channels)
        targets (torch.Tensor): All target values over the entire calibration set
            Shape: (n_samples, pred_len, num_channels)
        alpha (float): The significance level

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - critical_calibration_scores (torch.Tensor): Uncorrected critical scores
              Shape: (pred_len,)
            - corrected_critical_calibration_scores (torch.Tensor): Corrected critical scores
              Shape: (pred_len,)
    """
    n_samples, pred_len, _ = preds.size() # (n_samples, pred_len, num_channels)
    calibration_scores = nonconformity(preds, targets).T # (num_channels, pred_len, n_samples)


    # Uncorrected critical calibration scores
    q = min((n_samples + 1.0) * (1 - alpha) / n_samples, 1)
    corrected_q = min((n_samples + 1.0) * (1 - alpha / pred_len) / n_samples, 1)
    critical_calibration_scores = get_critical_scores(calibration_scores=calibration_scores, q=q) # (pred_len, num_channels)

    # Bonferroni-corrected critical calibration scores.
    corrected_critical_calibration_scores = get_critical_scores(
        calibration_scores=calibration_scores,
        q=corrected_q,
    ) # (pred_len, num_channels)

    return critical_calibration_scores, corrected_critical_calibration_scores


# Renamed from (def predict())
def get_intervals(
    pred: torch.Tensor,
    scores: torch.Tensor
) -> torch.Tensor:
    """
    Forecasts the time series with conformal uncertainty intervals.

    Args:
        pred (torch.Tensor): The model forecast for the time series
            Shape: (batch_size, pred_len, num_channels)
        scores (torch.Tensor): The critical calibration scores (corrected or noncorrected)
            for the model

    Returns:
        torch.Tensor: Tensor containing lower and upper forecast bounds
            Shape: (batch_size, 2, pred_len, num_channels) where axis 1 contains
            [lower_bound, upper_bound]
    """

    # [batch_size, pred_len, num_channels]

    with torch.no_grad():
        lower = pred - scores
        upper = pred + scores

    # [batch_size, 2, pred_len, num_channels]
    return torch.stack((lower, upper), dim=1)

def get_coverage(
    preds: torch.Tensor,
    targets: torch.Tensor,
    scores: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluates coverage of predictions against target values using calibration scores.

    Args:
        preds (torch.Tensor): Model predictions
        targets (torch.Tensor): Ground truth target values
        scores (torch.Tensor): Critical calibration scores (corrected or noncorrected)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - independent_coverages (torch.Tensor): Coverage metrics computed independently
            - joint_coverages (torch.Tensor): Coverage metrics computed jointly
            - intervals (torch.Tensor): Forecast uncertainty intervals with shape [n_samples, 2, pred_len, num_channels]
    """

    intervals = get_intervals(preds, scores) # [n_samples, 2, pred_len, num_channels] containing lower and upper bounds
    independent_coverages, joint_coverages = coverage(intervals, targets) # (n_samples, (1 | pred_len), num_channels) booleans

    return independent_coverages, joint_coverages, intervals
