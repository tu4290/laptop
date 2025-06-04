# /home/ubuntu/dashboard_v2/darkpool_analyzer.py
# -*- coding: utf-8 -*-
"""
Implements the seven Darkpool analysis methodologies.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Attempt to import from the new utils location
try:
    from .darkpool_analytics_utils import calculate_zscore, get_quantile_threshold
    _utils_imported_successfully = True
except ImportError:
    _utils_imported_successfully = False
    # Fallback for direct execution or if utils is not found in package structure
    # This assumes darkpool_analytics_utils.py is in the same directory when running directly
    try:
        from darkpool_analytics_utils import calculate_zscore, get_quantile_threshold
        print("Warning: darkpool_analyzer.py using fallback import for darkpool_analytics_utils.")
        _utils_imported_successfully = True
    except ImportError:
        print("CRITICAL ERROR: darkpool_analytics_utils.py not found. Cannot proceed.")
        # Define dummy functions if import fails to prevent further load-time errors,
        # but operations will fail.
        def calculate_zscore(series): return pd.Series([np.nan] * len(series), index=series.index if isinstance(series, pd.Series) else None)
        def get_quantile_threshold(series, quantile_val): return np.nan


def analyze_high_gamma_imbalance(df: pd.DataFrame, gxoi_col: str = 'gxoi', quantile: float = 0.9) -> pd.DataFrame:
    """
    Identifies strikes with unusually high gamma concentration (gxoi).
    Methodology 1.
    """
    empty_res = pd.DataFrame(columns=['strike', 'method_score', 'methodology'])
    if gxoi_col not in df.columns:
        # print(f"Warning (HGI): Column '{gxoi_col}' not found.")
        return empty_res

    metric_series = pd.to_numeric(df[gxoi_col], errors='coerce')
    if metric_series.isnull().all():
        # print(f"Warning (HGI): Column '{gxoi_col}' contains all NaNs after numeric conversion.")
        return empty_res

    metric_zscore = calculate_zscore(metric_series)

    threshold = get_quantile_threshold(metric_zscore.dropna(), quantile)
    if threshold is None or pd.isna(threshold):
        # print(f"Warning (HGI): Could not determine threshold for quantile {quantile}.")
        return empty_res

    significant_mask = metric_zscore > threshold
    if not significant_mask.any():
        return empty_res

    result_df = df.loc[significant_mask].copy() # Make a copy to avoid SettingWithCopyWarning
    result_df['method_score'] = metric_zscore[significant_mask]
    result_df['methodology'] = 'High Gamma Imbalance'
    return result_df[['strike', 'method_score', 'methodology']].drop_duplicates(subset=['strike']).reset_index(drop=True)


def analyze_delta_gamma_divergence(df: pd.DataFrame, dxoi_col: str = 'dxoi', gxoi_col: str = 'gxoi', quantile: float = 0.9) -> pd.DataFrame:
    """
    Identifies strikes where delta imbalance (dxoi) and gamma concentration (gxoi) diverge significantly.
    Methodology 2.
    """
    empty_res = pd.DataFrame(columns=['strike', 'method_score', 'methodology'])
    if dxoi_col not in df.columns or gxoi_col not in df.columns:
        # print(f"Warning (DGD): Columns '{dxoi_col}' or '{gxoi_col}' not found.")
        return empty_res

    delta_series = pd.to_numeric(df[dxoi_col], errors='coerce')
    gamma_series = pd.to_numeric(df[gxoi_col], errors='coerce')

    if delta_series.isnull().all() or gamma_series.isnull().all():
        # print(f"Warning (DGD): One or both columns ('{dxoi_col}', '{gxoi_col}') are all NaNs after conversion.")
        return empty_res

    delta_zscore = calculate_zscore(delta_series)
    gamma_zscore = calculate_zscore(gamma_series)

    valid_z_mask = delta_zscore.notna() & gamma_zscore.notna()
    if not valid_z_mask.any():
        return empty_res

    divergence_score = (gamma_zscore - delta_zscore).abs() # Calculate on full series, then filter by valid_z_mask for thresholding

    threshold = get_quantile_threshold(divergence_score[valid_z_mask].dropna(), quantile)
    if threshold is None or pd.isna(threshold):
        # print(f"Warning (DGD): Could not determine threshold for quantile {quantile}.")
        return empty_res

    significant_mask = (divergence_score > threshold) & valid_z_mask
    if not significant_mask.any():
        return empty_res

    result_df = df.loc[significant_mask].copy()
    result_df['method_score'] = divergence_score[significant_mask]
    result_df['methodology'] = 'Delta-Gamma Divergence'
    return result_df[['strike', 'method_score', 'methodology']].drop_duplicates(subset=['strike']).reset_index(drop=True)


def analyze_flow_anomaly(df: pd.DataFrame, volmbs_15m_col: str = 'volmbs_15m', volmbs_60m_col: str = 'volmbs_60m', quantile: float = 0.9) -> pd.DataFrame:
    """
    Identifies strikes with unusual flow patterns across different timeframes.
    Methodology 3.
    """
    empty_res = pd.DataFrame(columns=['strike', 'method_score', 'methodology'])
    if volmbs_15m_col not in df.columns or volmbs_60m_col not in df.columns:
        # print(f"Warning (FA): Columns '{volmbs_15m_col}' or '{volmbs_60m_col}' not found.")
        return empty_res

    volmbs_15m_series = pd.to_numeric(df[volmbs_15m_col], errors='coerce')
    volmbs_60m_series = pd.to_numeric(df[volmbs_60m_col], errors='coerce')

    if volmbs_15m_series.isnull().all() or volmbs_60m_series.isnull().all():
        # print(f"Warning (FA): One or both columns ('{volmbs_15m_col}', '{volmbs_60m_col}') are all NaNs after conversion.")
        return empty_res

    volmbs_15m_zscore = calculate_zscore(volmbs_15m_series)
    volmbs_60m_zscore = calculate_zscore(volmbs_60m_series)

    valid_z_mask = volmbs_15m_zscore.notna() & volmbs_60m_zscore.notna()
    if not valid_z_mask.any():
        return empty_res

    flow_anomaly_score = volmbs_15m_zscore.abs() + (volmbs_15m_zscore - volmbs_60m_zscore).abs()

    threshold = get_quantile_threshold(flow_anomaly_score[valid_z_mask].dropna(), quantile)
    if threshold is None or pd.isna(threshold):
        # print(f"Warning (FA): Could not determine threshold for quantile {quantile}.")
        return empty_res

    significant_mask = (flow_anomaly_score > threshold) & valid_z_mask
    if not significant_mask.any():
        return empty_res

    result_df = df.loc[significant_mask].copy()
    result_df['method_score'] = flow_anomaly_score[significant_mask]
    result_df['methodology'] = 'Flow Anomaly'
    return result_df[['strike', 'method_score', 'methodology']].drop_duplicates(subset=['strike']).reset_index(drop=True)


def analyze_volatility_sensitivity(df: pd.DataFrame, vannaxoi_col: str = 'vannaxoi', vommaxoi_col: str = 'vommaxoi', quantile: float = 0.9) -> pd.DataFrame:
    """
    Identifies strikes with high vanna (vannaxoi) and vomma (vommaxoi) exposure.
    Methodology 4.
    """
    empty_res = pd.DataFrame(columns=['strike', 'method_score', 'methodology'])
    if vannaxoi_col not in df.columns or vommaxoi_col not in df.columns:
        # print(f"Warning (VS): Columns '{vannaxoi_col}' or '{vommaxoi_col}' not found.")
        return empty_res

    vanna_series = pd.to_numeric(df[vannaxoi_col], errors='coerce')
    vomma_series = pd.to_numeric(df[vommaxoi_col], errors='coerce')

    if vanna_series.isnull().all() or vomma_series.isnull().all():
        # print(f"Warning (VS): One or both columns ('{vannaxoi_col}', '{vommaxoi_col}') are all NaNs after conversion.")
        return empty_res

    vanna_zscore = calculate_zscore(vanna_series)
    vomma_zscore = calculate_zscore(vomma_series)

    valid_z_mask = vanna_zscore.notna() & vomma_zscore.notna()
    if not valid_z_mask.any():
        return empty_res

    vol_sensitivity_score = vanna_zscore.abs() + vomma_zscore.abs()

    threshold = get_quantile_threshold(vol_sensitivity_score[valid_z_mask].dropna(), quantile)
    if threshold is None or pd.isna(threshold):
        # print(f"Warning (VS): Could not determine threshold for quantile {quantile}.")
        return empty_res

    significant_mask = (vol_sensitivity_score > threshold) & valid_z_mask
    if not significant_mask.any():
        return empty_res

    result_df = df.loc[significant_mask].copy()
    result_df['method_score'] = vol_sensitivity_score[significant_mask]
    result_df['methodology'] = 'Volatility Sensitivity'
    return result_df[['strike', 'method_score', 'methodology']].drop_duplicates(subset=['strike']).reset_index(drop=True)


def analyze_charm_adjusted_gamma(df: pd.DataFrame, gxoi_col: str = 'gxoi', charmxoi_col: str = 'charmxoi', quantile: float = 0.9) -> pd.DataFrame:
    """
    Identifies strikes with high gamma that are also sensitive to time decay (charm).
    Methodology 5.
    """
    empty_res = pd.DataFrame(columns=['strike', 'method_score', 'methodology'])
    if gxoi_col not in df.columns or charmxoi_col not in df.columns:
        # print(f"Warning (CAG): Columns '{gxoi_col}' or '{charmxoi_col}' not found.")
        return empty_res

    gamma_series = pd.to_numeric(df[gxoi_col], errors='coerce')
    charm_series = pd.to_numeric(df[charmxoi_col], errors='coerce')

    if gamma_series.isnull().all() or charm_series.isnull().all():
        # print(f"Warning (CAG): One or both columns ('{gxoi_col}', '{charmxoi_col}') are all NaNs after conversion.")
        return empty_res

    gamma_zscore = calculate_zscore(gamma_series)
    charm_zscore = calculate_zscore(charm_series)

    valid_z_mask = gamma_zscore.notna() & charm_zscore.notna()
    if not valid_z_mask.any():
        return empty_res

    charm_adj_gamma_score = gamma_zscore * (1 + charm_zscore.abs())

    threshold = get_quantile_threshold(charm_adj_gamma_score[valid_z_mask].dropna(), quantile)
    if threshold is None or pd.isna(threshold):
        # print(f"Warning (CAG): Could not determine threshold for quantile {quantile}.")
        return empty_res

    significant_mask = (charm_adj_gamma_score > threshold) & valid_z_mask
    if not significant_mask.any():
        return empty_res

    result_df = df.loc[significant_mask].copy()
    result_df['method_score'] = charm_adj_gamma_score[significant_mask]
    result_df['methodology'] = 'Charm-Adjusted Gamma'
    return result_df[['strike', 'method_score', 'methodology']].drop_duplicates(subset=['strike']).reset_index(drop=True)


def analyze_active_hedging_detection(df: pd.DataFrame, gxoi_col: str = 'gxoi', gxvolm_col: str = 'gxvolm', quantile: float = 0.9) -> pd.DataFrame:
    """
    Identifies strikes with high gamma (gxoi) and high gamma-weighted volume (gxvolm).
    Methodology 6.
    """
    empty_res = pd.DataFrame(columns=['strike', 'method_score', 'methodology'])
    if gxoi_col not in df.columns or gxvolm_col not in df.columns:
        # print(f"Warning (AHD): Columns '{gxoi_col}' or '{gxvolm_col}' not found.")
        return empty_res

    gamma_series = pd.to_numeric(df[gxoi_col], errors='coerce')
    gxvolm_series = pd.to_numeric(df[gxvolm_col], errors='coerce')

    if gamma_series.isnull().all() or gxvolm_series.isnull().all():
        # print(f"Warning (AHD): One or both columns ('{gxoi_col}', '{gxvolm_col}') are all NaNs after conversion.")
        return empty_res

    gamma_zscore = calculate_zscore(gamma_series)
    gxvolm_zscore = calculate_zscore(gxvolm_series)

    valid_z_mask = gamma_zscore.notna() & gxvolm_zscore.notna()
    if not valid_z_mask.any():
        return empty_res

    active_hedging_score = gamma_zscore * gxvolm_zscore

    threshold = get_quantile_threshold(active_hedging_score[valid_z_mask].dropna(), quantile)
    if threshold is None or pd.isna(threshold):
        # print(f"Warning (AHD): Could not determine threshold for quantile {quantile}.")
        return empty_res

    significant_mask = (active_hedging_score > threshold) & valid_z_mask
    if not significant_mask.any():
        return empty_res

    result_df = df.loc[significant_mask].copy()
    result_df['method_score'] = active_hedging_score[significant_mask]
    result_df['methodology'] = 'Active Hedging Detection'
    return result_df[['strike', 'method_score', 'methodology']].drop_duplicates(subset=['strike']).reset_index(drop=True)


def analyze_value_volume_divergence(df: pd.DataFrame, value_bs_col: str = 'value_bs', volmbs_15m_col: str = 'volmbs_15m', quantile: float = 0.9) -> pd.DataFrame:
    """
    Identifies strikes where value flow (value_bs) and volume flow (volmbs_15m) diverge significantly.
    Methodology 7.
    """
    empty_res = pd.DataFrame(columns=['strike', 'method_score', 'methodology'])
    if value_bs_col not in df.columns or volmbs_15m_col not in df.columns:
        # print(f"Warning (VVD): Columns '{value_bs_col}' or '{volmbs_15m_col}' not found.")
        return empty_res

    value_bs_series = pd.to_numeric(df[value_bs_col], errors='coerce')
    volmbs_15m_series = pd.to_numeric(df[volmbs_15m_col], errors='coerce')

    if value_bs_series.isnull().all() or volmbs_15m_series.isnull().all():
        # print(f"Warning (VVD): One or both columns ('{value_bs_col}', '{volmbs_15m_col}') are all NaNs after conversion.")
        return empty_res

    value_bs_zscore = calculate_zscore(value_bs_series)
    volmbs_15m_zscore = calculate_zscore(volmbs_15m_series)

    valid_z_mask = value_bs_zscore.notna() & volmbs_15m_zscore.notna()
    if not valid_z_mask.any():
        return empty_res

    val_vol_divergence_score = (value_bs_zscore - volmbs_15m_zscore).abs()

    threshold = get_quantile_threshold(val_vol_divergence_score[valid_z_mask].dropna(), quantile)
    if threshold is None or pd.isna(threshold):
        # print(f"Warning (VVD): Could not determine threshold for quantile {quantile}.")
        return empty_res

    significant_mask = (val_vol_divergence_score > threshold) & valid_z_mask
    if not significant_mask.any():
        return empty_res

    result_df = df.loc[significant_mask].copy()
    result_df['method_score'] = val_vol_divergence_score[significant_mask]
    result_df['methodology'] = 'Value-Volume Divergence'
    return result_df[['strike', 'method_score', 'methodology']].drop_duplicates(subset=['strike']).reset_index(drop=True)


def run_all_darkpool_methodologies(df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
    """
    Runs all seven darkpool analysis methodologies and concatenates their results.

    Args:
        df (pd.DataFrame): The input options data (e.g., final_metric_rich_df_obj).
                           Assumes 'strike' column exists.
        config (Optional[Dict]): Configuration dictionary for column names and quantiles.
                                  Example: {'gxoi_col': 'custom_gxoi', 'default_quantile': 0.95}

    Returns:
        pd.DataFrame: A DataFrame with columns ['strike', 'method_score', 'methodology'],
                      containing all significant strikes identified by any methodology.
                      Returns an empty DataFrame if no methodologies identify any significant strikes
                      or if critical input data is missing or df is None/empty.
    """
    if not _utils_imported_successfully:
        print("Error (run_all): darkpool_analytics_utils not imported. Cannot run methodologies.")
        return pd.DataFrame(columns=['strike', 'method_score', 'methodology'])

    if df is None or df.empty:
        # print("Warning (run_all): Input DataFrame is None or empty.")
        return pd.DataFrame(columns=['strike', 'method_score', 'methodology'])

    if 'strike' not in df.columns:
        # print("Error (run_all): Input DataFrame must contain a 'strike' column.")
        return pd.DataFrame(columns=['strike', 'method_score', 'methodology'])

    cfg = config if config else {}
    q = cfg.get('default_quantile', 0.9) # Default quantile if not specified

    # Define column names based on config or use defaults
    cols = {
        'gxoi': cfg.get('gxoi_col', 'gxoi'),
        'dxoi': cfg.get('dxoi_col', 'dxoi'),
        'volmbs_15m': cfg.get('volmbs_15m_col', 'volmbs_15m'),
        'volmbs_60m': cfg.get('volmbs_60m_col', 'volmbs_60m'),
        'vannaxoi': cfg.get('vannaxoi_col', 'vannaxoi'),
        'vommaxoi': cfg.get('vommaxoi_col', 'vommaxoi'),
        'charmxoi': cfg.get('charmxoi_col', 'charmxoi'),
        'gxvolm': cfg.get('gxvolm_col', 'gxvolm'),
        'value_bs': cfg.get('value_bs_col', 'value_bs'),
    }

    all_results: List[pd.DataFrame] = []

    method_functions = [
        (analyze_high_gamma_imbalance, {'gxoi_col': cols['gxoi'], 'quantile': q}),
        (analyze_delta_gamma_divergence, {'dxoi_col': cols['dxoi'], 'gxoi_col': cols['gxoi'], 'quantile': q}),
        (analyze_flow_anomaly, {'volmbs_15m_col': cols['volmbs_15m'], 'volmbs_60m_col': cols['volmbs_60m'], 'quantile': q}),
        (analyze_volatility_sensitivity, {'vannaxoi_col': cols['vannaxoi'], 'vommaxoi_col': cols['vommaxoi'], 'quantile': q}),
        (analyze_charm_adjusted_gamma, {'gxoi_col': cols['gxoi'], 'charmxoi_col': cols['charmxoi'], 'quantile': q}),
        (analyze_active_hedging_detection, {'gxoi_col': cols['gxoi'], 'gxvolm_col': cols['gxvolm'], 'quantile': q}),
        (analyze_value_volume_divergence, {'value_bs_col': cols['value_bs'], 'volmbs_15m_col': cols['volmbs_15m'], 'quantile': q})
    ]

    for func, f_kwargs in method_functions:
        try:
            # Ensure all necessary columns for the current method are present before calling
            required_cols_for_method = [val for key, val in f_kwargs.items() if key.endswith('_col')]
            missing_cols = [col for col in required_cols_for_method if col not in df.columns]
            if missing_cols:
                # print(f"Warning (run_all): Skipping {func.__name__} due to missing columns: {missing_cols}")
                continue

            res_df = func(df, **f_kwargs)
            if isinstance(res_df, pd.DataFrame) and not res_df.empty:
                all_results.append(res_df)
        except Exception as e:
            print(f"Error running {func.__name__}: {e}") # Log error and continue

    if not all_results:
        return pd.DataFrame(columns=['strike', 'method_score', 'methodology'])

    combined_df = pd.concat(all_results, ignore_index=True)

    # Sort by strike and then by method_score descending to see most important first for each strike
    if not combined_df.empty:
        combined_df = combined_df.sort_values(by=['strike', 'method_score'], ascending=[True, False])

    return combined_df.reset_index(drop=True)

# (Existing imports and methodology functions from previous step remain at the top)
# ...

# New helper for strike-level aggregation
def _aggregate_metrics_by_strike(options_df: pd.DataFrame, metric_cols: List[str], strike_col: str = 'strike') -> pd.DataFrame:
    """ Aggregates specified metrics by strike using sum. """
    if strike_col not in options_df.columns:
        print(f"Warning (_aggregate_metrics_by_strike): Strike column '{strike_col}' not found.")
        return pd.DataFrame(columns=[strike_col] + metric_cols)

    valid_metric_cols = [col for col in metric_cols if col in options_df.columns]
    if not valid_metric_cols:
        print("Warning (_aggregate_metrics_by_strike): No valid metric columns found in DataFrame.")
        return pd.DataFrame(columns=[strike_col] + metric_cols)

    # Ensure metrics are numeric before aggregation
    temp_df = options_df[[strike_col] + valid_metric_cols].copy()
    for col in valid_metric_cols:
        temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0) # Fill NaNs with 0 for sum

    agg_funcs = {metric: 'sum' for metric in valid_metric_cols}
    strike_aggregated_df = temp_df.groupby(strike_col).agg(agg_funcs).reset_index()
    return strike_aggregated_df

# Function to calculate individual ranking factors
def _calculate_ranking_factors(
    options_df: pd.DataFrame,
    identified_strikes_df: pd.DataFrame,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Calculates the six ranking factors for identified strikes.
    identified_strikes_df has columns: ['strike', 'method_score', 'methodology']
    options_df is the raw, per-contract options data.
    """
    if identified_strikes_df.empty:
        return pd.DataFrame(columns=['strike', 'methodology_diversity_score', 'gamma_concentration_factor',
                                     'flow_consistency_factor', 'delta_gamma_alignment_factor',
                                     'volatility_sensitivity_factor', 'time_decay_sensitivity_factor'])
    cfg = config if config else {}
    cols = { # From config or defaults, matching those in run_all_darkpool_methodologies
        'gxoi': cfg.get('gxoi_col', 'gxoi'), 'dxoi': cfg.get('dxoi_col', 'dxoi'),
        'volmbs_15m': cfg.get('volmbs_15m_col', 'volmbs_15m'),
        'vannaxoi': cfg.get('vannaxoi_col', 'vannaxoi'), 'vommaxoi': cfg.get('vommaxoi_col', 'vommaxoi'),
        'charmxoi': cfg.get('charmxoi_col', 'charmxoi'),
    }

    # 1. Methodology Diversity Score
    methodology_diversity = identified_strikes_df.groupby('strike')['methodology'].nunique().reset_index(name='methodology_count')
    methodology_diversity['methodology_diversity_score'] = methodology_diversity['methodology_count'] / 7.0

    # Prepare strike-aggregated metrics for normalization factors
    metrics_to_aggregate = [cols['gxoi'], cols['dxoi'], cols['volmbs_15m'], cols['vannaxoi'], cols['vommaxoi'], cols['charmxoi']]
    strike_agg_metrics_df = _aggregate_metrics_by_strike(options_df, metrics_to_aggregate)

    if strike_agg_metrics_df.empty:
        print("Warning (_calculate_ranking_factors): Strike aggregated metrics are empty. Factors will be 0 or NaN.")
        # Create an empty df with all factor columns to merge, ensuring columns exist
        empty_factors_cols = ['strike', 'methodology_diversity_score', 'gamma_concentration_factor',
                              'flow_consistency_factor', 'delta_gamma_alignment_factor',
                              'volatility_sensitivity_factor', 'time_decay_sensitivity_factor', 'methodology_count']

        # If methodology_diversity is also empty (e.g. identified_strikes_df was empty)
        if methodology_diversity.empty:
            return pd.DataFrame(columns=empty_factors_cols)

        # Merge diversity with empty factors template
        # Create a template with just 'strike' and factor columns for merging
        factor_names = [
            'gamma_concentration_factor', 'flow_consistency_factor', 'delta_gamma_alignment_factor',
            'volatility_sensitivity_factor', 'time_decay_sensitivity_factor'
        ]
        empty_factors_for_merge = pd.DataFrame(columns=['strike'] + factor_names)
        ranked_df = pd.merge(methodology_diversity, empty_factors_for_merge, on='strike', how='left')
        for col_name in factor_names: # Fill NaN factors that would result
             ranked_df[col_name] = ranked_df[col_name].fillna(0.0)
        return ranked_df[empty_factors_cols]


    # Merge aggregated metrics with the diversity scores
    factors_df = pd.merge(methodology_diversity, strike_agg_metrics_df, on='strike', how='left').fillna(0.0)

    # 2. Gamma Concentration Factor: abs(gxoi_strike) / max(abs(gxoi_all_strikes))
    abs_gxoi = factors_df[cols['gxoi']].abs()
    max_abs_gxoi = abs_gxoi.max() if not abs_gxoi.empty else 0 # Handle empty series case for max
    factors_df['gamma_concentration_factor'] = (abs_gxoi / max_abs_gxoi).fillna(0.0) if max_abs_gxoi != 0 else 0.0

    # 3. Flow Consistency Factor: abs(volmbs_15m_strike) / max(abs(volmbs_15m_all_strikes))
    abs_volmbs15m = factors_df[cols['volmbs_15m']].abs()
    max_abs_volmbs15m = abs_volmbs15m.max() if not abs_volmbs15m.empty else 0
    factors_df['flow_consistency_factor'] = (abs_volmbs15m / max_abs_volmbs15m).fillna(0.0) if max_abs_volmbs15m != 0 else 0.0

    # 4. Delta-Gamma Alignment Factor: 1 - abs(gxoi * dxoi) / (abs(gxoi) * abs(dxoi) + 1)
    gxoi_s = factors_df[cols['gxoi']]
    dxoi_s = factors_df[cols['dxoi']]
    numerator_dga = (gxoi_s * dxoi_s).abs()
    denominator_dga = (gxoi_s.abs() * dxoi_s.abs()) + 1
    factors_df['delta_gamma_alignment_factor'] = (1 - numerator_dga / denominator_dga).fillna(0.0)

    # 5. Volatility Sensitivity Factor
    abs_vannaxoi = factors_df[cols['vannaxoi']].abs()
    abs_vommaxoi = factors_df[cols['vommaxoi']].abs()
    vol_sens_sum = abs_vannaxoi + abs_vommaxoi
    max_vol_sens_sum = vol_sens_sum.max() if not vol_sens_sum.empty else 0
    factors_df['volatility_sensitivity_factor'] = (vol_sens_sum / max_vol_sens_sum).fillna(0.0) if max_vol_sens_sum != 0 else 0.0

    # 6. Time Decay Sensitivity Factor
    abs_charmxoi = factors_df[cols['charmxoi']].abs()
    max_abs_charmxoi = abs_charmxoi.max() if not abs_charmxoi.empty else 0
    factors_df['time_decay_sensitivity_factor'] = (abs_charmxoi / max_abs_charmxoi).fillna(0.0) if max_abs_charmxoi != 0 else 0.0

    return factors_df[['strike', 'methodology_diversity_score', 'gamma_concentration_factor',
                       'flow_consistency_factor', 'delta_gamma_alignment_factor',
                       'volatility_sensitivity_factor', 'time_decay_sensitivity_factor', 'methodology_count']]


# Function to calculate composite plausibility score
def calculate_composite_plausibility(ranking_factors_df: pd.DataFrame) -> pd.DataFrame:
    """ Calculates the composite plausibility score based on weighted factors. """
    if ranking_factors_df.empty: # Check if essential factor columns are present
        return pd.DataFrame()

    # Ensure all factor columns exist, fill with 0 if not (though _calculate_ranking_factors should provide them)
    weights = {
        'methodology_diversity_score': 0.40,
        'gamma_concentration_factor': 0.20,
        'flow_consistency_factor': 0.15,
        'delta_gamma_alignment_factor': 0.10,
        'volatility_sensitivity_factor': 0.10,
        'time_decay_sensitivity_factor': 0.05
    }
    df = ranking_factors_df.copy()

    # Initialize score
    df['composite_plausibility_score'] = 0.0

    for factor_col, weight in weights.items():
        if factor_col in df.columns:
            df['composite_plausibility_score'] += df[factor_col].fillna(0.0) * weight
        else:
            print(f"Warning (calculate_composite_plausibility): Factor column '{factor_col}' not found. It won't contribute to the score.")

    return df

# Main orchestrator for ranking
def get_ranked_darkpool_levels(
    options_df: pd.DataFrame,
    identified_strikes_by_method_df: pd.DataFrame,
    config: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Orchestrates the calculation of ranking factors and composite plausibility score.

    Args:
        options_df (pd.DataFrame): Raw options data (per-contract) for metric aggregation.
        identified_strikes_by_method_df (pd.DataFrame): Output from run_all_darkpool_methodologies.
                                                       Cols: ['strike', 'method_score', 'methodology']
        config (Optional[Dict]): Configuration for column names.

    Returns:
        pd.DataFrame: DataFrame with strikes, factor scores, methodology count,
                      and composite_plausibility_score, sorted by plausibility.
                      Returns empty DataFrame if critical steps fail.
    """
    if not _utils_imported_successfully:
        print("Error (get_ranked_darkpool_levels): darkpool_analytics_utils not imported. Cannot rank levels.")
        return pd.DataFrame()

    if identified_strikes_by_method_df is None or identified_strikes_by_method_df.empty:
        print("Warning (get_ranked_darkpool_levels): No strikes identified by methodologies. Returning empty ranking.")
        return pd.DataFrame()

    if options_df is None or options_df.empty:
        print("Warning (get_ranked_darkpool_levels): Raw options_df is empty. Cannot calculate factors. Returning empty ranking.")
        return pd.DataFrame()

    # Calculate ranking factors
    ranking_factors_df = _calculate_ranking_factors(options_df, identified_strikes_by_method_df, config)
    if ranking_factors_df.empty:
        print("Warning (get_ranked_darkpool_levels): Ranking factors calculation returned empty. Returning empty ranking.")
        # Ensure specific columns are present even for an empty DataFrame for consistency
        return pd.DataFrame(columns=['strike', 'methodology_diversity_score', 'gamma_concentration_factor',
                                     'flow_consistency_factor', 'delta_gamma_alignment_factor',
                                     'volatility_sensitivity_factor', 'time_decay_sensitivity_factor',
                                     'methodology_count', 'composite_plausibility_score', 'contributing_methods'])


    # Calculate composite plausibility score
    final_ranked_df = calculate_composite_plausibility(ranking_factors_df)
    if final_ranked_df.empty: # Should not happen if ranking_factors_df was not empty
        print("Warning (get_ranked_darkpool_levels): Composite plausibility calculation returned empty. Returning empty ranking.")
        return pd.DataFrame(columns=['strike', 'methodology_diversity_score', 'gamma_concentration_factor',
                                     'flow_consistency_factor', 'delta_gamma_alignment_factor',
                                     'volatility_sensitivity_factor', 'time_decay_sensitivity_factor',
                                     'methodology_count', 'composite_plausibility_score', 'contributing_methods'])

    # Merge method details back for full info
    if 'strike' in identified_strikes_by_method_df.columns and 'methodology' in identified_strikes_by_method_df.columns:
        methods_per_strike = identified_strikes_by_method_df.groupby('strike')['methodology'].apply(lambda x: ', '.join(sorted(list(set(x))))).reset_index(name='contributing_methods')
        final_ranked_df = pd.merge(final_ranked_df, methods_per_strike, on='strike', how='left')
        final_ranked_df['contributing_methods'] = final_ranked_df['contributing_methods'].fillna('') # Ensure no NaN in text column
    else:
        final_ranked_df['contributing_methods'] = ''


    return final_ranked_df.sort_values(by='composite_plausibility_score', ascending=False).reset_index(drop=True)


if __name__ == '__main__':
    print("--- Testing Darkpool Analyzer Methodologies & Ranking ---")
    if not _utils_imported_successfully:
        print("Skipping tests as darkpool_analytics_utils could not be imported.")
    else:
        num_rows = 200
        sample_strikes_test = np.linspace(90, 110, num_rows // 10).repeat(10)
        if len(sample_strikes_test) < num_rows: sample_strikes_test = np.append(sample_strikes_test, [sample_strikes_test[-1]]*(num_rows - len(sample_strikes_test)))

        test_data = {
            'strike': sample_strikes_test[:num_rows],
            'opt_kind': np.random.choice(['call', 'put'], num_rows),
            'gxoi': np.random.rand(num_rows) * 1e6 + (np.sin(np.linspace(0, 5*np.pi, num_rows)) * 0.5e6),
            'dxoi': np.random.rand(num_rows) * 5e5 - 2.5e5 + (np.cos(np.linspace(0, 5*np.pi, num_rows)) * 1e5),
            'volmbs_15m': np.random.randint(-1000, 1000, num_rows).astype(float),
            'volmbs_60m': np.random.randint(-3000, 3000, num_rows).astype(float),
            'vannaxoi': np.random.rand(num_rows) * 1e4,
            'vommaxoi': np.random.rand(num_rows) * 1e3,
            'charmxoi': np.random.rand(num_rows) * 5e3 - 2.5e3,
            'gxvolm': np.random.rand(num_rows) * 1e5,
            'value_bs': np.random.randint(-50000, 50000, num_rows).astype(float)
        }
        for col in test_data:
            if col not in ['strike', 'opt_kind']:
                nan_indices = np.random.choice(num_rows, size=num_rows//20, replace=False)
                test_data[col][nan_indices] = np.nan
        sample_options_data_df = pd.DataFrame(test_data)

        print("\n--- Running all methodologies (Test) ---")
        all_identified_df = run_all_darkpool_methodologies(sample_options_data_df.copy())

        if not all_identified_df.empty:
            print(f"Total significant strike instances found: {len(all_identified_df)}")
            print("Breakdown by methodology:")
            print(all_identified_df['methodology'].value_counts())

            print("\n--- Calculating Ranking and Plausibility (Test) ---")
            # Pass a copy of sample_options_data_df as it might be modified by _aggregate_metrics_by_strike if not careful (though it makes a copy now)
            ranked_levels_df = get_ranked_darkpool_levels(sample_options_data_df.copy(), all_identified_df.copy())

            if not ranked_levels_df.empty:
                print("Top 10 Ranked Darkpool Levels:")
                print(ranked_levels_df.head(10).to_string())

                expected_cols = [
                    'strike', 'methodology_diversity_score', 'gamma_concentration_factor',
                    'flow_consistency_factor', 'delta_gamma_alignment_factor',
                    'volatility_sensitivity_factor', 'time_decay_sensitivity_factor',
                    'methodology_count', 'composite_plausibility_score', 'contributing_methods'
                ]
                missing_cols = [col for col in expected_cols if col not in ranked_levels_df.columns]
                if missing_cols:
                    print(f"\nERROR: Missing expected columns in final ranked_levels_df: {missing_cols}")
                else:
                    print("\nAll expected columns are present in the final ranked DataFrame.")
            else:
                print("Ranking process resulted in an empty DataFrame.")
        else:
            print("No significant strikes identified by any methodology in the sample run. Ranking skipped.")

        print("\n--- Test get_ranked_darkpool_levels with empty identified_strikes_df ---")
        empty_ranked_result = get_ranked_darkpool_levels(sample_options_data_df.copy(), pd.DataFrame())
        if empty_ranked_result.empty:
            print("Correctly returned empty DataFrame for empty identified_strikes_df.")
        else:
            print(f"ERROR: Expected empty DataFrame for empty identified_strikes_df, got {len(empty_ranked_result)} rows.")

        print("\n--- Test get_ranked_darkpool_levels with empty options_df ---")
        empty_options_result = get_ranked_darkpool_levels(pd.DataFrame(), all_identified_df.copy() if not all_identified_df.empty else pd.DataFrame({'strike':[100], 'methodology':['Test'], 'method_score':[1]}))
        if empty_options_result.empty:
            print("Correctly returned empty DataFrame for empty options_df.")
        else:
             print(f"ERROR: Expected empty DataFrame for empty options_df, got {len(empty_options_result)} rows.")
```
