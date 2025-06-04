# /home/ubuntu/dashboard_v2/darkpool_analytics_utils.py
# -*- coding: utf-8 -*-
"""
Utility functions for Darkpool analysis calculations.
"""
import pandas as pd
import numpy as np
from typing import Union, Optional, Dict, Tuple # Added Optional, Dict, Tuple for new functions

def calculate_zscore(series: pd.Series) -> pd.Series:
    """
    Calculates the Z-score for each element in a pandas Series.
    Z-score = (value - mean) / std_dev.

    Args:
        series (pd.Series): A pandas Series of numerical data.

    Returns:
        pd.Series: A pandas Series containing the Z-scores.
                   Returns a Series of NaNs if the input series is empty or all NaNs.
                   Returns a Series of 0s if standard deviation is 0 (and mean exists).
    """
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series.")

    if series.empty:
        return pd.Series([], dtype=float)

    # Ensure numeric type for calculations, coercing errors will turn non-numeric to NaN
    numeric_series = pd.to_numeric(series, errors='coerce')

    if numeric_series.isnull().all():
        return pd.Series([np.nan] * len(numeric_series), index=numeric_series.index, dtype=float)

    mean = numeric_series.mean()
    std_dev = numeric_series.std()

    if pd.isna(mean) or pd.isna(std_dev): # Should be rare if not all NaNs, but good check
        return pd.Series([np.nan] * len(numeric_series), index=numeric_series.index, dtype=float)

    if std_dev == 0:
        # If std is 0, z-score is 0 for values equal to mean, NaN otherwise.
        # However, if all values are the same, they are all equal to the mean.
        # If there are NaNs that were not numeric, they should remain NaN.
        z_scores = np.where(numeric_series == mean, 0.0, np.nan)
        # Preserve original NaNs if any existed in the numeric_series
        z_scores = pd.Series(z_scores, index=numeric_series.index, dtype=float)
        z_scores[numeric_series.isnull()] = np.nan
        return z_scores

    z_scores = (numeric_series - mean) / std_dev
    return z_scores

def get_quantile_threshold(series: pd.Series, quantile_val: float) -> Union[float, None]:
    """
    Calculates the value at a specific quantile for a pandas Series.

    Args:
        series (pd.Series): A pandas Series of numerical data.
        quantile_val (float): The quantile to calculate (e.g., 0.9 for 90th percentile).
                              Must be between 0 and 1, inclusive.

    Returns:
        Union[float, None]: The value at the specified quantile.
                            Returns None if the series is empty, all NaNs, or quantile_val is invalid.
    """
    if not isinstance(series, pd.Series):
        # Or raise TypeError, depending on desired strictness
        return None

    if not (0 <= quantile_val <= 1):
        return None

    if series.empty:
        return None

    # Ensure numeric type, coercing errors will turn non-numeric to NaN, then drop them for quantile calculation
    numeric_series = pd.to_numeric(series, errors='coerce')
    cleaned_series = numeric_series.dropna()

    if cleaned_series.empty: # If all values were NaN or became NaN after coercion
        return None

    try:
        threshold = cleaned_series.quantile(quantile_val)
        return float(threshold) if pd.notna(threshold) else None
    except Exception: # Broad exception for any pandas internal error
        return None

# --- Appended functions start here ---

def scale_iqr(series: pd.Series, clip_range: Optional[Tuple[float, float]] = None) -> pd.Series:
    """
    Scales a pandas Series using Interquartile Range (IQR).
    Scaled Value = (value - Q1) / (Q3 - Q1).
    More robust to outliers than min-max scaling.

    Args:
        series (pd.Series): A pandas Series of numerical data.
        clip_range (Optional[tuple[float, float]]): If provided, clips the output
                                                     to (min_val, max_val). E.g., (0, 1).

    Returns:
        pd.Series: Scaled series. Returns NaNs or original values if IQR is 0 or undefined.
    """
    if not isinstance(series, pd.Series): # Added type check for consistency
        raise TypeError("Input must be a pandas Series.")

    if series.empty: # Return empty series matching input type if possible
        return pd.Series([], dtype=float if series.dtype in [float, int, np.number] else object)


    numeric_series = pd.to_numeric(series, errors='coerce')
    if numeric_series.isnull().all(): # All coerced to NaN or were already all NaN
        return pd.Series([np.nan] * len(series), index=series.index, dtype=float)

    q1 = numeric_series.quantile(0.25)
    q3 = numeric_series.quantile(0.75)

    if pd.isna(q1) or pd.isna(q3):
        # This can happen if there are too few non-NaN values after coercion
        # (e.g., less than 2 for Q1/Q3 to be distinct, or less than 4 for pandas default interpolation)
        # In such cases, scaling is ill-defined; return series of NaNs matching original NaNs
        return pd.Series(np.where(numeric_series.notna(), np.nan, np.nan), index=series.index, dtype=float)


    iqr = q3 - q1

    if iqr == 0:
        # If IQR is 0, all non-NaN values in the core distribution are the same (Q1=Q3=median).
        # Values equal to Q1 are scaled to 0.5 (center of typical 0-1 scaled range from this method).
        # Values different from Q1 (outliers if IQR is 0) become inf or -inf, then NaN.
        # Original NaNs remain NaN.
        scaled_series = pd.Series(np.where(numeric_series == q1, 0.5, np.nan), index=series.index, dtype=float)
        # Ensure original NaNs from numeric_series are preserved (they would be NaN in scaled_series already)
        scaled_series[numeric_series.isnull()] = np.nan
    else:
        scaled_series = (numeric_series - q1) / iqr

    if clip_range:
        if clip_range[0] is not None and clip_range[1] is not None and clip_range[0] > clip_range[1]:
            # print("Warning (scale_iqr): clip_range min > max. Not clipping.") # Or raise error
            pass # Don't clip if range is invalid
        else:
            scaled_series = scaled_series.clip(lower=clip_range[0], upper=clip_range[1])

    return scaled_series

def scale_tanh(series: pd.Series, scale_factor: Optional[float] = None) -> pd.Series:
    """
    Normalizes data into a [-1, 1] range using np.tanh.
    scaled_value = np.tanh(value / scale_factor).
    If scale_factor is None, it's estimated using the mean of absolute values.

    Args:
        series (pd.Series): A pandas Series of numerical data.
        scale_factor (Optional[float]): The factor to divide values by before tanh.
                                        If None, estimated from data. Must be > 0 if provided.

    Returns:
        pd.Series: Scaled series in the range [-1, 1].
    """
    if not isinstance(series, pd.Series): # Added type check
        raise TypeError("Input must be a pandas Series.")

    if series.empty:
        return pd.Series([], dtype=float)

    numeric_series = pd.to_numeric(series, errors='coerce')
    if numeric_series.isnull().all():
        return pd.Series([np.nan] * len(series), index=series.index, dtype=float)

    if scale_factor is None:
        abs_mean = numeric_series.abs().mean()
        effective_scale_factor = abs_mean if pd.notna(abs_mean) and abs_mean > 1e-9 else 1.0
    elif scale_factor <= 0:
        # print("Error (scale_tanh): scale_factor must be positive.") # Or raise error
        return pd.Series([np.nan] * len(series), index=series.index, dtype=float)
    else:
        effective_scale_factor = scale_factor

    scaled_series = np.tanh(numeric_series / effective_scale_factor)
    return scaled_series

def determine_simplified_regime(metric_value: Optional[float], thresholds: Dict[str, float]) -> str:
    """
    Categorizes a market metric (like VIX or HV) into 'Low', 'Medium', 'High' regimes.

    Args:
        metric_value (Optional[float]): The value of the metric (e.g., VIX).
        thresholds (Dict[str, float]): Dictionary with keys like
                                       'low_medium_boundary' and 'medium_high_boundary'.
                                       Example: {'low_medium_boundary': 15, 'medium_high_boundary': 25}

    Returns:
        str: Regime string ("Low", "Medium", "High", or "Undefined" if metric_value is None or thresholds are invalid).
    """
    if metric_value is None or pd.isna(metric_value):
        return "Undefined"

    low_medium_boundary = thresholds.get('low_medium_boundary')
    medium_high_boundary = thresholds.get('medium_high_boundary')

    if not (isinstance(low_medium_boundary, (int, float)) and isinstance(medium_high_boundary, (int, float))):
        # print("Error (determine_simplified_regime): Thresholds dict must contain valid numeric "
        # "'low_medium_boundary' and 'medium_high_boundary'.") # Or raise
        return "Undefined"

    if low_medium_boundary >= medium_high_boundary:
        # print("Error (determine_simplified_regime): low_medium_boundary must be less than medium_high_boundary.")
        return "Undefined" # Indicate error in thresholds logic

    if metric_value < low_medium_boundary:
        return "Low"
    elif metric_value < medium_high_boundary:
        return "Medium"
    else:
        return "High"

# --- Update the __main__ block to include tests for new functions ---
if __name__ == '__main__':
    # (Existing tests for calculate_zscore, get_quantile_threshold)
    print("--- Testing calculate_zscore (existing) ---")
    s1 = pd.Series([1, 2, 3, 4, 5])
    print(f"s1: {s1.tolist()}, z-scores: {calculate_zscore(s1).round(4).tolist()}")
    s_mixed_type = pd.Series([1, 'a', 3, 4, 'b'])
    print(f"s_mixed_type: {s_mixed_type.tolist()}, z-scores: {calculate_zscore(s_mixed_type).round(4).tolist()}")


    print("\n--- Testing get_quantile_threshold (existing) ---")
    s_quant = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"s_quant: {s_quant.tolist()}, 0.9 quantile: {get_quantile_threshold(s_quant, 0.9)}")


    print("\n--- Testing scale_iqr ---")
    s_iqr1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(f"s_iqr1: {s_iqr1.tolist()}\nscaled: {scale_iqr(s_iqr1).round(3).tolist()}")
    print(f"s_iqr1 clipped [0,1]: {scale_iqr(s_iqr1, clip_range=(0,1)).round(3).tolist()}")

    s_iqr_outlier = pd.Series([1, 2, 3, 4, 5, 100])
    print(f"s_iqr_outlier: {s_iqr_outlier.tolist()}\nscaled: {scale_iqr(s_iqr_outlier).round(3).tolist()}")

    s_iqr_zero = pd.Series([5, 5, 5, 5, 5])
    print(f"s_iqr_zero: {s_iqr_zero.tolist()}, scaled: {scale_iqr(s_iqr_zero).tolist()}")

    s_iqr_empty = pd.Series([], dtype=float)
    print(f"s_iqr_empty: {s_iqr_empty.tolist()}, scaled: {scale_iqr(s_iqr_empty).tolist()}")

    s_iqr_nan = pd.Series([1, np.nan, 3, 4, np.nan, 6])
    print(f"s_iqr_nan: {s_iqr_nan.tolist()}\nscaled: {scale_iqr(s_iqr_nan).round(3).tolist()}")

    s_iqr_one_val = pd.Series([7.0])
    print(f"s_iqr_one_val: {s_iqr_one_val.tolist()}, scaled: {scale_iqr(s_iqr_one_val).tolist()}") # Expect [nan]

    s_iqr_two_val = pd.Series([7.0, 8.0]) # Q1=7, Q3=8, IQR=1 => (7-7)/1=0, (8-7)/1=1
    print(f"s_iqr_two_val: {s_iqr_two_val.tolist()}, scaled: {scale_iqr(s_iqr_two_val).tolist()}")

    s_iqr_three_val = pd.Series([7.0, 8.0, 9.0]) # Q1=7.5, Q3=8.5, IQR=1 => (7-7.5)/1 = -0.5, (8-7.5)/1 = 0.5, (9-7.5)/1 = 1.5
    print(f"s_iqr_three_val: {s_iqr_three_val.tolist()}, scaled: {scale_iqr(s_iqr_three_val).tolist()}")


    print("\n--- Testing scale_tanh ---")
    s_tanh1 = pd.Series([-10, -5, 0, 5, 10])
    print(f"s_tanh1: {s_tanh1.tolist()}\nscaled (auto factor): {scale_tanh(s_tanh1).round(3).tolist()}")
    print(f"s_tanh1 scaled (factor=5): {scale_tanh(s_tanh1, scale_factor=5).round(3).tolist()}")

    s_tanh_positive = pd.Series([1, 2, 3, 4, 5])
    print(f"s_tanh_positive: {s_tanh_positive.tolist()}\nscaled (auto factor): {scale_tanh(s_tanh_positive).round(3).tolist()}")

    s_tanh_empty = pd.Series([], dtype=float)
    print(f"s_tanh_empty: {s_tanh_empty.tolist()}, scaled: {scale_tanh(s_tanh_empty).tolist()}")

    s_tanh_zero_factor = pd.Series([1,2,3])
    print(f"s_tanh_zero_factor (factor=0): {scale_tanh(s_tanh_zero_factor, scale_factor=0).tolist()}") # Expect NaNs

    s_tanh_all_zeros = pd.Series([0,0,0,0])
    print(f"s_tanh_all_zeros (auto factor): {scale_tanh(s_tanh_all_zeros).tolist()}") # Expect all zeros, auto factor becomes 1


    print("\n--- Testing determine_simplified_regime ---")
    regime_thresh = {'low_medium_boundary': 15, 'medium_high_boundary': 25}
    print(f"VIX=10, thresholds={regime_thresh}: {determine_simplified_regime(10, regime_thresh)}")
    print(f"VIX=20, thresholds={regime_thresh}: {determine_simplified_regime(20, regime_thresh)}")
    print(f"VIX=30, thresholds={regime_thresh}: {determine_simplified_regime(30, regime_thresh)}")
    print(f"VIX=15 (at boundary), thresholds={regime_thresh}: {determine_simplified_regime(15, regime_thresh)}")
    print(f"VIX=None, thresholds={regime_thresh}: {determine_simplified_regime(None, regime_thresh)}")
    print(f"VIX=20, thresholds={{'low_medium_boundary': 15}}: {determine_simplified_regime(20, {'low_medium_boundary': 15})}")
    print(f"VIX=20, thresholds={{'low_medium_boundary': 20, 'medium_high_boundary': 15}}: {determine_simplified_regime(20, {'low_medium_boundary': 20, 'medium_high_boundary': 15})}") # Invalid thresholds
```
