# /home/ubuntu/dashboard_v2/elite_darkpool_analyzer.py
# -*- coding: utf-8 -*-
"""
Enhanced Darkpool Analyzer based on 'Elite' principles.
Implements seven methodologies with regime adjustments and proximity weighting.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field

try:
    from .darkpool_analytics_utils import (
        calculate_zscore,
        get_quantile_threshold,
        determine_simplified_regime
    )
    _utils_imported_successfully_eda = True
except ImportError:
    _utils_imported_successfully_eda = False
    try:
        from darkpool_analytics_utils import (
            calculate_zscore, get_quantile_threshold,
            determine_simplified_regime
        )
        print("Warning: elite_darkpool_analyzer.py using fallback import for darkpool_analytics_utils.")
        _utils_imported_successfully_eda = True
    except ImportError:
        print("CRITICAL ERROR: elite_darkpool_analyzer.py cannot import from darkpool_analytics_utils.")
        def calculate_zscore(series): return pd.Series([np.nan] * len(series), index=series.index if isinstance(series, pd.Series) else None)
        def get_quantile_threshold(series, quantile_val): return np.nan
        def determine_simplified_regime(value, thresholds): return "Undefined"

# --- Threshold Configuration ---
@dataclass
class ThresholdConfig:
    type: str = "relative_percentile"  # E.g., "relative_percentile", "absolute_value", "z_score", "relative_mean_factor"
    percentile: Optional[float] = None # For relative_percentile
    value: Optional[float] = None      # For absolute_value
    z_score_value: Optional[float] = None # For z_score
    factor: Optional[float] = None     # For relative_mean_factor
    fallback_value: Optional[float] = 0.0 # Generic fallback

# --- Regime Specific Overrides ---
@dataclass
class RegimeOverrideMethodologyConfig:
    threshold_config: Optional[ThresholdConfig] = None
    proximity_influence_factor: Optional[float] = None
    # Potentially other methodology-specific params can be overridden

@dataclass
class RegimeOverrideRankingFactorConfig:
    weight: Optional[float] = None
    normalization_method: Optional[str] = None
    # Potentially other factor-specific params

# --- Component Configurations ---
@dataclass
class InputColumnMapConfig:
    strike: str = "strike"
    opt_kind: str = "opt_kind"
    gxoi: str = "gxoi"
    dxoi: str = "dxoi"
    volmbs_15m: str = "volmbs_15m"
    volmbs_60m: str = "volmbs_60m"
    vannaxoi: str = "vannaxoi"
    vommaxoi: str = "vommaxoi"
    charmxoi: str = "charmxoi"
    gxvolm: str = "gxvolm"
    value_bs: str = "value_bs"
    volatility: str = "volatility"
    underlying_price_col_options_df: str = "underlying_price" # if price is per-row in options_df

@dataclass
class RegimeThresholdsMapConfig:
    low_medium_boundary: float = 0.15
    medium_high_boundary: float = 0.30

@dataclass
class RegimeAdjustmentsConfig: # Per-regime adjustments
    default_quantile_modifier: float = 0.0
    default_proximity_strength: float = 1.0
    # Add other generic adjustments if needed, e.g., generic_impact_weight

@dataclass
class RegimeDefinitionConfig:
    enabled: bool = True
    metric_source: str = "hv_20d" # E.g., "vix", "hv_20d", "realized_vol_10d"
    thresholds_map: RegimeThresholdsMapConfig = field(default_factory=RegimeThresholdsMapConfig)
    adjustments: Dict[str, RegimeAdjustmentsConfig] = field(default_factory=lambda: {
        "Low": RegimeAdjustmentsConfig(default_quantile_modifier=0.05, default_proximity_strength=0.8),
        "Medium": RegimeAdjustmentsConfig(default_quantile_modifier=0.0, default_proximity_strength=1.0),
        "High": RegimeAdjustmentsConfig(default_quantile_modifier=-0.05, default_proximity_strength=1.2),
    })

@dataclass
class ProximityConfig:
    enabled: bool = True
    exp_decay_factor: float = 2.0
    volatility_influence: float = 0.3 # How much contract IV influences proximity calc

@dataclass
class MethodologySetting:
    enabled: bool = True
    threshold_config: ThresholdConfig = field(default_factory=ThresholdConfig)
    proximity_influence_factor: float = 0.5 # How much general proximity score influences this method
    regime_overrides: Optional[Dict[str, RegimeOverrideMethodologyConfig]] = field(default_factory=dict)

@dataclass
class RankingFactorSetting:
    weight: float = 0.0
    normalization_method: str = "max_abs" # E.g., "max_abs", "iqr", "tanh", "z_score", "none"
    tanh_scale_factor_source: Optional[Union[str, float]] = None # E.g., "mean_abs", "std_dev", or a fixed float
    iqr_clip_range: Optional[Tuple[float, float]] = None # E.g., [0, 2] for 0 to 2x IQR
    regime_overrides: Optional[Dict[str, RegimeOverrideRankingFactorConfig]] = field(default_factory=dict)

@dataclass
class RankingFactorsConfig:
    methodology_diversity: RankingFactorSetting = field(default_factory=lambda: RankingFactorSetting(weight=0.40, normalization_method="none"))
    gamma_concentration: RankingFactorSetting = field(default_factory=lambda: RankingFactorSetting(weight=0.20, normalization_method="max_abs"))
    flow_consistency: RankingFactorSetting = field(default_factory=lambda: RankingFactorSetting(weight=0.15, normalization_method="tanh", tanh_scale_factor_source="mean_abs"))
    delta_gamma_alignment: RankingFactorSetting = field(default_factory=lambda: RankingFactorSetting(weight=0.10, normalization_method="none")) # Assumes score is already -1 to 1 or similar
    volatility_sensitivity: RankingFactorSetting = field(default_factory=lambda: RankingFactorSetting(weight=0.10, normalization_method="iqr", iqr_clip_range=(0.0,2.0))) # Ensure tuple is float
    time_decay_sensitivity: RankingFactorSetting = field(default_factory=lambda: RankingFactorSetting(weight=0.05, normalization_method="max_abs"))
    # Optional: Global regime overrides for weights, if not per-factor
    # global_regime_overrides: Optional[Dict[str, Dict[str, float]]] = field(default_factory=dict) # E.g. {"High": {"gamma_concentration_weight": 0.25}}

@dataclass
class SRLogicConfig:
    plausibility_min_threshold_for_sr: float = 0.3
    delta_significance_config: ThresholdConfig = field(default_factory=lambda: ThresholdConfig(type="relative_mean_factor", factor=0.5, fallback_value=10000.0))
    flow_significance_config: ThresholdConfig = field(default_factory=lambda: ThresholdConfig(type="relative_mean_factor", factor=0.5, fallback_value=500.0))

@dataclass
class ReportColumnConfig: # Keep report column names configurable but separate for clarity
    strike: str = 'strike'
    methodology: str = 'methodology'
    raw_score: str = 'method_raw_score'
    adjusted_score: str = 'method_adjusted_score'
    proximity_factor: str = 'proximity_factor'
    regime: str = 'market_regime'
    methodology_diversity_score: str = 'methodology_diversity_score'
    gamma_concentration_factor: str = 'gamma_concentration_factor'
    flow_consistency_factor: str = 'flow_consistency_factor'
    delta_gamma_alignment_factor: str = 'delta_gamma_alignment_factor'
    volatility_sensitivity_factor: str = 'volatility_sensitivity_factor'
    time_decay_sensitivity_factor: str = 'time_decay_sensitivity_factor'
    composite_plausibility_score: str = 'composite_plausibility_score'
    contributing_methods: str = 'contributing_methods'
    methodology_count: str = 'methodology_count'
    level_type: str = 'level_type'
    sr_rationale: str = 'sr_rationale'

# --- Main Configuration Dataclass ---
@dataclass
class DarkpoolAnalyticsConfig:
    input_column_map: InputColumnMapConfig = field(default_factory=InputColumnMapConfig)
    report_column_map: ReportColumnConfig = field(default_factory=ReportColumnConfig) # New sub-dataclass for this

    regime_definition_config: RegimeDefinitionConfig = field(default_factory=RegimeDefinitionConfig)
    proximity_config: ProximityConfig = field(default_factory=ProximityConfig)

    methodologies: Dict[str, MethodologySetting] = field(default_factory=lambda: {
        "high_gamma_imbalance": MethodologySetting(threshold_config=ThresholdConfig(percentile=0.90, fallback_value=1.5)),
        "delta_gamma_divergence": MethodologySetting(threshold_config=ThresholdConfig(percentile=0.90, fallback_value=1.0)),
        "flow_anomaly": MethodologySetting(threshold_config=ThresholdConfig(percentile=0.90, fallback_value=1.2)),
        "volatility_sensitivity": MethodologySetting(threshold_config=ThresholdConfig(percentile=0.90, fallback_value=1.0)),
        "charm_adjusted_gamma": MethodologySetting(threshold_config=ThresholdConfig(percentile=0.90, fallback_value=1.5)),
        "active_hedging_detection": MethodologySetting(threshold_config=ThresholdConfig(percentile=0.90, fallback_value=1.0)),
        "value_volume_divergence": MethodologySetting(threshold_config=ThresholdConfig(percentile=0.90, fallback_value=1.0)),
    })

    ranking_factors_config: RankingFactorsConfig = field(default_factory=RankingFactorsConfig)
    sr_logic_config: SRLogicConfig = field(default_factory=SRLogicConfig)

    # General settings (can be moved to specific configs if needed)
    default_quantile_for_significance: float = 0.90 # General fallback if not in ThresholdConfig
    strike_clustering_pct_threshold: float = 0.01 # Might be part of a "PostProcessingConfig" later

    # Example of how versioning could be handled if config structure changes often
    config_version: str = "2.0.0"

class EliteDarkpoolAnalyzer:
    def __init__(self,
                 options_df: pd.DataFrame,
                 underlying_price: float,
                 config: Optional[DarkpoolAnalyticsConfig] = None,
                 market_regime_metric_value: Optional[float] = None):

        if not _utils_imported_successfully_eda:
            raise ImportError("EliteDarkpoolAnalyzer cannot operate: darkpool_analytics_utils failed to import.")

        self.config = config if config else DarkpoolAnalyticsConfig()
        self.underlying_price = float(underlying_price) if underlying_price is not None else 0.0

        self.input_df = options_df.copy() if options_df is not None else pd.DataFrame()
        if self.input_df.empty:
            print("Warning (EliteDarkpoolAnalyzer): Initializing with empty options_df.")

        self._prepare_input_df() # Uses self.config.input_column_map

        self.current_regime: str = "Medium"
        if self.config.regime_definition_config.enabled: # Updated
            if market_regime_metric_value is not None:
                self.current_regime = determine_simplified_regime(
                    float(market_regime_metric_value),
                    self.config.regime_definition_config.thresholds_map # Updated (pass the map object)
                )
            else:
                print("Warning (EliteDarkpoolAnalyzer): Regime is enabled but market_regime_metric_value is None. Defaulting to Medium.")

        # Proximity factor calculation
        proximity_factor_col_name = self.config.report_column_map.proximity_factor # Updated
        self.input_df[proximity_factor_col_name] = 1.0
        if self.config.proximity_config.enabled and not self.input_df.empty: # Updated
            strike_col_name = self.config.input_column_map.strike # Updated
            if strike_col_name in self.input_df.columns:
                self.input_df[proximity_factor_col_name] = self._calculate_proximity_factor(
                    self.input_df[strike_col_name]
                )
            else:
                print(f"Warning (EliteDarkpoolAnalyzer): Strike column '{strike_col_name}' not found for proximity calculation. Factors set to 1.0.")

        # Aggregating metrics
        cols_in_map = self.config.input_column_map # Updated
        metrics_to_agg_keys = ['gxoi', 'dxoi', 'volmbs_15m', 'vannaxoi', 'vommaxoi', 'charmxoi']
        metrics_to_agg = [getattr(cols_in_map, k) for k in metrics_to_agg_keys if hasattr(cols_in_map, k) and getattr(cols_in_map, k) in self.input_df.columns]

        self.strike_aggregated_metrics = self._aggregate_metrics_by_strike_internal(metrics_to_agg)

    def _aggregate_metrics_by_strike_internal(self, metric_cols: List[str]) -> pd.DataFrame:
        strike_col = self.config.input_column_map.strike # Updated
        if strike_col not in self.input_df.columns:
            return pd.DataFrame(columns=[strike_col] + metric_cols)

        valid_metric_cols = [col for col in metric_cols if col in self.input_df.columns] # This part is fine
        if not valid_metric_cols:
            return pd.DataFrame(columns=[strike_col] + metric_cols)

        temp_df = self.input_df[[strike_col] + valid_metric_cols].copy()
        for col in valid_metric_cols:
            temp_df[col] = temp_df[col].fillna(0)

        agg_funcs = {metric: 'sum' for metric in valid_metric_cols}
        if not agg_funcs:
            return pd.DataFrame(columns=[strike_col] + metric_cols)

        strike_aggregated_df = temp_df.groupby(strike_col).agg(agg_funcs).reset_index()
        return strike_aggregated_df

    def _prepare_input_df(self):
        if self.input_df.empty: return
        # Iterate through attributes of InputColumnMapConfig for type conversion
        for key in self.config.input_column_map.__annotations__.keys(): # Updated
            col_name = getattr(self.config.input_column_map, key)
            if col_name in self.input_df.columns:
                if key not in ['opt_kind', 'strike']: # These are typically not numeric or handled differently
                    self.input_df[col_name] = pd.to_numeric(self.input_df[col_name], errors='coerce')

    def _get_current_regime_adjustment_config(self) -> RegimeAdjustmentsConfig:
        """Helper to get the RegimeAdjustmentsConfig for the current regime."""
        if not self.config.regime_definition_config.enabled:
            return RegimeAdjustmentsConfig() # Return default (neutral) adjustments

        return self.config.regime_definition_config.adjustments.get(
            self.current_regime, RegimeAdjustmentsConfig() # Fallback to default if current_regime key is missing
        )

    def _calculate_proximity_factor(self, strike_series: pd.Series) -> pd.Series:
        prox_cfg = self.config.proximity_config # Updated
        if not prox_cfg.enabled or self.underlying_price <= 0 or strike_series.empty:
            return pd.Series(1.0, index=strike_series.index)

        numeric_strikes = pd.to_numeric(strike_series, errors='coerce')
        strike_distance_pct = (numeric_strikes - self.underlying_price).abs() / self.underlying_price

        # Get regime-specific proximity strength
        current_regime_adjustments = self._get_current_regime_adjustment_config() # Updated
        prox_strength = current_regime_adjustments.default_proximity_strength

        vol_col = self.config.input_column_map.volatility # Updated
        vol_influence_setting = prox_cfg.volatility_influence # Updated
        vol_influence_factor = pd.Series(1.0, index=strike_series.index)

        if vol_col and vol_col in self.input_df.columns and vol_influence_setting > 0:
            contract_vol = pd.to_numeric(self.input_df[vol_col], errors='coerce').fillna(0.20)
            vol_influence_factor = 1.0 + contract_vol * vol_influence_setting

        decay_factor = prox_cfg.exp_decay_factor * prox_strength # Updated

        # Ensure vol_influence_factor aligns with strike_distance_pct if they have different indexes (should not happen with reindex)
        # Or if input_df was filtered and strike_series is a subset.
        vol_influence_factor_aligned = vol_influence_factor.reindex(strike_distance_pct.index, fill_value=1.0)

        proximity_calc_base = strike_distance_pct / vol_influence_factor_aligned
        proximity = np.exp(-decay_factor * proximity_calc_base)

        return proximity.fillna(0.01)

    def _calculate_dynamic_threshold(self, data_series: pd.Series, config: ThresholdConfig, higher_is_better: bool = True) -> float:
        """
        Calculates a dynamic threshold based on the provided configuration.
        """
        # Ensure data_series is numeric and drop NaNs for calculations
        numeric_series = pd.to_numeric(data_series, errors='coerce').dropna()

        if numeric_series.empty:
            return config.fallback_value if config.fallback_value is not None else 0.0

        threshold: Optional[float] = None
        if config.type == "relative_percentile":
            percentile_val = config.percentile if config.percentile is not None else (0.90 if higher_is_better else 0.10)
            actual_quantile = percentile_val if higher_is_better else (1.0 - percentile_val)
            threshold = get_quantile_threshold(numeric_series, actual_quantile)
        elif config.type == "absolute_value":
            threshold = config.value
        elif config.type == "z_score":
            # This implies the input data_series should ideally be z-scores,
            # or this threshold is an absolute value on a z-score transformed series.
            # For simplicity, if this type is chosen, the calling context should ensure data_series is appropriate.
            threshold = config.z_score_value if config.z_score_value is not None else (2.0 if higher_is_better else -2.0)
        elif config.type == "relative_mean_factor":
            mean_val = numeric_series.mean()
            std_val = numeric_series.std()
            factor = config.factor if config.factor is not None else 1.0

            # If std_val is NaN (e.g. series has 0 variance) or 0, this can lead to issues or threshold = mean.
            if pd.isna(std_val) or std_val == 0:
                 # Fallback: if std is 0, deviation from mean is not meaningful in this context.
                 # Use mean itself or a factor of mean, or fallback_value.
                 # For now, let's use mean + (factor * mean) as a heuristic if std is zero.
                 # This part might need more domain-specific logic.
                 # A simple approach: if std is 0, any value different from mean is "significant".
                 # Or, more robustly, rely on fallback or absolute value if std is 0.
                 # For S/R (delta/flow), we care about magnitude.
                 # Let's assume that if std is 0, the threshold is mean itself or fallback.
                 # If factor is for magnitude (e.g. delta > factor * mean_delta), then use mean * factor.
                 # This is tricky. Let's stick to mean +/- factor * std for now, and handle NaN/0 std below.
                  threshold = mean_val # Default to mean if std is zero/NaN
            else:
                if higher_is_better: # E.g. score > mean + N*std
                    threshold = mean_val + factor * std_val
                else: # E.g. score < mean - N*std (for negative values being significant)
                    threshold = mean_val - factor * std_val
        else:
            # Unknown type, will fall through to None/NaN check
            pass

        if threshold is None or pd.isna(threshold):
            # print(f"Warning: Threshold calculation for type '{config.type}' resulted in None/NaN. Using fallback: {config.fallback_value}")
            return config.fallback_value if config.fallback_value is not None else 0.0

        return float(threshold)

    # def _get_adjusted_quantile_threshold(self) -> float: # To be removed/commented
    #     base_quantile = self.config.default_quantile_for_significance # Updated old reference
    #     modifier = self._get_regime_adjustment('quantile_modifier') # This helper also needs update
    #     return min(max(base_quantile + modifier, 0.01), 0.99)

    def _analyze_method_template(self, raw_score_series: pd.Series, methodology_name: str) -> pd.DataFrame:
        cfg_cols_rep = self.config.report_column_map # Updated
        strike_col_input = self.config.input_column_map.strike # Updated

        empty_df_columns = [cfg_cols_rep.strike, cfg_cols_rep.raw_score,
                            cfg_cols_rep.adjusted_score, cfg_cols_rep.methodology]
        empty_df_result = pd.DataFrame(columns=empty_df_columns)

        if self.input_df.empty or raw_score_series.empty: return empty_df_result
        if strike_col_input not in self.input_df.columns: return empty_df_result

        method_setting = self.config.methodologies.get(methodology_name)
        if not method_setting or not method_setting.enabled:
            return empty_df_result

        current_threshold_config = method_setting.threshold_config
        current_prox_influence_factor = method_setting.proximity_influence_factor

        # Apply regime overrides if applicable
        if self.config.regime_definition_config.enabled and method_setting.regime_overrides:
            regime_override = method_setting.regime_overrides.get(self.current_regime)
            if regime_override:
                if regime_override.threshold_config:
                    current_threshold_config = regime_override.threshold_config
                if regime_override.proximity_influence_factor is not None:
                    current_prox_influence_factor = regime_override.proximity_influence_factor

        # Calculate threshold (higher score is better for methodologies by default)
        threshold = self._calculate_dynamic_threshold(raw_score_series, current_threshold_config, higher_is_better=True)

        adjusted_score_series = raw_score_series.copy()
        if self.config.proximity_config.enabled:
            # Proximity factor is already pre-calculated on self.input_df
            # Need to align it with raw_score_series index (which should match input_df index)
            proximity_factors = self.input_df[cfg_cols_rep.proximity_factor].reindex(raw_score_series.index).fillna(1.0)

            # Apply methodology-specific proximity influence
            # Score = RawScore * (1 + (ProximityFactor - 1) * InfluenceFactor)
            # This means if ProximityFactor is 1 (no effect), score remains RawScore.
            # If ProximityFactor is high (e.g. 1.5) and Influence is 0.5, effect is (1 + 0.5 * 0.5) = 1.25
            # If ProximityFactor is low (e.g. 0.5) and Influence is 0.5, effect is (1 + (-0.5) * 0.5) = 0.75
            effective_proximity_adjustment = (proximity_factors - 1.0) * current_prox_influence_factor
            adjusted_score_series = raw_score_series * (1.0 + effective_proximity_adjustment)


        if threshold is None or pd.isna(threshold): # Should be handled by _calculate_dynamic_threshold fallback
            return empty_df_result

        significant_mask = (adjusted_score_series > threshold) & adjusted_score_series.notna()
        if not significant_mask.any():
            return empty_df_result

        result_df = self.input_df.loc[significant_mask, [strike_col_input]].copy()
        result_df.rename(columns={strike_col_input: cfg_cols_rep.strike}, inplace=True)

        result_df[cfg_cols_rep.raw_score] = raw_score_series[significant_mask]
        result_df[cfg_cols_rep.adjusted_score] = adjusted_score_series[significant_mask]
        result_df[cfg_cols_rep.methodology] = methodology_name

        return result_df.drop_duplicates(subset=[cfg_cols_rep.strike]).reset_index(drop=True)

    def _analyze_high_gamma_imbalance(self) -> pd.DataFrame:
        col = self.config.input_column_map.gxoi # Updated
        if col not in self.input_df.columns: return self._analyze_method_template(pd.Series(dtype=float), "high_gamma_imbalance")
        metric = pd.to_numeric(self.input_df[col], errors='coerce').fillna(0)
        # Methodologies typically use z-scores of raw metrics as their input 'raw_score_series'
        return self._analyze_method_template(calculate_zscore(metric), "high_gamma_imbalance")

    def _analyze_delta_gamma_divergence(self) -> pd.DataFrame:
        c_dx, c_gx = self.config.input_column_map.dxoi, self.config.input_column_map.gxoi # Updated
        if c_dx not in self.input_df.columns or c_gx not in self.input_df.columns: return self._analyze_method_template(pd.Series(dtype=float),"delta_gamma_divergence")
        dz = calculate_zscore(pd.to_numeric(self.input_df[c_dx], errors='coerce').fillna(0))
        gz = calculate_zscore(pd.to_numeric(self.input_df[c_gx], errors='coerce').fillna(0))
        return self._analyze_method_template((gz - dz).abs(), "delta_gamma_divergence")

    def _analyze_flow_anomaly(self) -> pd.DataFrame:
        c15, c60 = self.config.input_column_map.volmbs_15m, self.config.input_column_map.volmbs_60m # Updated
        if c15 not in self.input_df.columns or c60 not in self.input_df.columns: return self._analyze_method_template(pd.Series(dtype=float),"flow_anomaly")
        z15 = calculate_zscore(pd.to_numeric(self.input_df[c15], errors='coerce').fillna(0))
        z60 = calculate_zscore(pd.to_numeric(self.input_df[c60], errors='coerce').fillna(0))
        return self._analyze_method_template(z15.abs() + (z15 - z60).abs(), "flow_anomaly")

    def _analyze_volatility_sensitivity(self) -> pd.DataFrame:
        c_van, c_vom = self.config.input_column_map.vannaxoi, self.config.input_column_map.vommaxoi # Updated
        if c_van not in self.input_df.columns or c_vom not in self.input_df.columns: return self._analyze_method_template(pd.Series(dtype=float),"volatility_sensitivity")
        van_s = pd.to_numeric(self.input_df[c_van], errors='coerce').fillna(0)
        vom_s = pd.to_numeric(self.input_df[c_vom], errors='coerce').fillna(0)
        raw_scores_z = calculate_zscore(van_s.abs() + vom_s.abs())
        return self._analyze_method_template(raw_scores_z, "volatility_sensitivity")

    def _analyze_charm_adjusted_gamma(self) -> pd.DataFrame:
        c_gx, c_ch = self.config.input_column_map.gxoi, self.config.input_column_map.charmxoi # Updated
        if c_gx not in self.input_df.columns or c_ch not in self.input_df.columns: return self._analyze_method_template(pd.Series(dtype=float),"charm_adjusted_gamma")
        gz = calculate_zscore(pd.to_numeric(self.input_df[c_gx], errors='coerce').fillna(0))
        cz = calculate_zscore(pd.to_numeric(self.input_df[c_ch], errors='coerce').fillna(0))
        return self._analyze_method_template(gz * (1 + cz.abs()), "charm_adjusted_gamma")

    def _analyze_active_hedging_detection(self) -> pd.DataFrame:
        c_gx, c_gxvolm = self.config.input_column_map.gxoi, self.config.input_column_map.gxvolm # Updated
        if c_gx not in self.input_df.columns or c_gxvolm not in self.input_df.columns: return self._analyze_method_template(pd.Series(dtype=float),"active_hedging_detection")
        gz = calculate_zscore(pd.to_numeric(self.input_df[c_gx], errors='coerce').fillna(0))
        gzvol = calculate_zscore(pd.to_numeric(self.input_df[c_gxvolm], errors='coerce').fillna(0))
        return self._analyze_method_template(gz * gzvol, "active_hedging_detection")

    def _analyze_value_volume_divergence(self) -> pd.DataFrame:
        c_val, c_vol = self.config.input_column_map.value_bs, self.config.input_column_map.volmbs_15m # Updated
        if c_val not in self.input_df.columns or c_vol not in self.input_df.columns: return self._analyze_method_template(pd.Series(dtype=float),"value_volume_divergence")
        val_z = calculate_zscore(pd.to_numeric(self.input_df[c_val], errors='coerce').fillna(0))
        vol_z = calculate_zscore(pd.to_numeric(self.input_df[c_vol], errors='coerce').fillna(0))
        return self._analyze_method_template((val_z - vol_z).abs(), "value_volume_divergence")

    def run_all_methodologies(self) -> pd.DataFrame:
        all_results = []
        # Methodology names must match keys in config.methodologies
        method_map = {
            "high_gamma_imbalance": self._analyze_high_gamma_imbalance,
            "delta_gamma_divergence": self._analyze_delta_gamma_divergence,
            "flow_anomaly": self._analyze_flow_anomaly,
            "volatility_sensitivity": self._analyze_volatility_sensitivity,
            "charm_adjusted_gamma": self._analyze_charm_adjusted_gamma,
            "active_hedging_detection": self._analyze_active_hedging_detection,
            "value_volume_divergence": self._analyze_value_volume_divergence
        }

        for method_name, func in method_map.items():
            method_setting = self.config.methodologies.get(method_name)
            if method_setting and method_setting.enabled:
                try:
                    res_df = func() # func now implicitly uses its name to get config inside _analyze_method_template
                    if isinstance(res_df, pd.DataFrame) and not res_df.empty:
                        all_results.append(res_df)
                except Exception as e:
                    print(f"Error running methodology {method_name}: {e}")
            else:
                print(f"Methodology {method_name} is disabled or not configured.")


        if not all_results:
            return pd.DataFrame(columns=[self.config.report_column_map.strike,
                                         self.config.report_column_map.raw_score,
                                         self.config.report_column_map.adjusted_score,
                                         self.config.report_column_map.methodology])

        combined_df = pd.concat(all_results, ignore_index=True)
        return combined_df

    def _calculate_ranking_factors(self, all_methodologies_result_df: pd.DataFrame) -> pd.DataFrame:
        cfg_cols_in = self.config.input_column_map # Updated
        cfg_cols_rep = self.config.report_column_map # Updated
        strike_col_report = cfg_cols_rep.strike

        factor_output_columns = [
            strike_col_report, cfg_cols_rep.methodology_count, cfg_cols_rep.methodology_diversity_score,
            cfg_cols_rep.gamma_concentration_factor, cfg_cols_rep.flow_consistency_factor,
            cfg_cols_rep.delta_gamma_alignment_factor, cfg_cols_rep.volatility_sensitivity_factor,
            cfg_cols_rep.time_decay_sensitivity_factor
        ]
        # Ensure all expected output columns are initialized if not calculated
        empty_factors_df = pd.DataFrame(columns=factor_output_columns)


        if all_methodologies_result_df.empty:
            return empty_factors_df

        methodology_diversity = all_methodologies_result_df.groupby(strike_col_report)[cfg_cols_rep.methodology].nunique().reset_index(name=cfg_cols_rep.methodology_count)

        # Normalize methodology_count to get methodology_diversity_score
        num_total_methodologies = len([m for m in self.config.methodologies.values() if m.enabled])
        num_total_methodologies = max(1, num_total_methodologies) # Avoid division by zero

        methodology_diversity[cfg_cols_rep.methodology_diversity_score] = methodology_diversity[cfg_cols_rep.methodology_count] / float(num_total_methodologies)

        strike_agg_df = self.strike_aggregated_metrics
        if strike_agg_df.empty:
            print("Warning (_calculate_ranking_factors): Pre-calculated strike_aggregated_metrics is empty.")
            factors_df = methodology_diversity.copy()
            # Initialize other factor columns to 0.0
            for factor_col_key in self.config.ranking_factors_config.__annotations__.keys():
                if factor_col_key == "methodology_diversity": continue # Already handled
                factor_col_name = getattr(cfg_cols_rep, factor_col_key + "_factor", None) # e.g. gamma_concentration_factor
                if factor_col_name is None and factor_col_key.endswith("_score"): # For methodology_diversity_score
                     factor_col_name = getattr(cfg_cols_rep, factor_col_key, None)

                if factor_col_name and factor_col_name not in factors_df.columns:
                    factors_df[factor_col_name] = 0.0

            # Ensure all columns from factor_output_columns are present
            for col in factor_output_columns:
                if col not in factors_df.columns:
                    factors_df[col] = 0.0 if "score" in col or "factor" in col or "count" in col else ""
            return factors_df.reindex(columns=factor_output_columns).fillna(0.0)


        factors_df = pd.merge(methodology_diversity, strike_agg_df, on=strike_col_report, how='left').fillna(0.0)

        # Helper for normalization
        def normalize_series(series: pd.Series, method: str, tanh_scale_src: Optional[Union[str, float]] = None, iqr_clip: Optional[Tuple[float,float]] = None) -> pd.Series:
            if series.empty: return series
            s_clean = series.dropna()
            if s_clean.empty: return series.fillna(0.0)

            if method == "max_abs":
                max_val = s_clean.abs().max()
                return (s_clean / max_val if max_val != 0 else 0.0).fillna(0.0)
            elif method == "z_score":
                return calculate_zscore(s_clean).fillna(0.0)
            elif method == "iqr":
                q1 = s_clean.quantile(0.25)
                q3 = s_clean.quantile(0.75)
                iqr_val = q3 - q1
                if iqr_val == 0: return pd.Series(0.5, index=s_clean.index) # Neutral if no variance by IQR

                normalized = (s_clean - q1) / iqr_val
                if iqr_clip:
                    normalized = normalized.clip(iqr_clip[0], iqr_clip[1])
                    # Rescale to 0-1 after clipping
                    min_clip, max_clip = iqr_clip
                    if max_clip > min_clip:
                         normalized = (normalized - min_clip) / (max_clip - min_clip)
                    else: # if clip range is tiny, default to 0.5 or 0
                         normalized = pd.Series(0.5 if min_clip == max_clip else 0.0, index=s_clean.index)
                return normalized.fillna(0.0)
            elif method == "tanh":
                scale_factor = 1.0
                if isinstance(tanh_scale_src, float):
                    scale_factor = tanh_scale_src
                elif isinstance(tanh_scale_src, str):
                    if tanh_scale_src == "mean_abs":
                        scale_factor = s_clean.abs().mean()
                    elif tanh_scale_src == "std_dev":
                        scale_factor = s_clean.std()
                if scale_factor == 0 or pd.isna(scale_factor): scale_factor = 1.0 # Avoid division by zero
                return np.tanh(s_clean / scale_factor).fillna(0.0)
            elif method == "none": # No normalization, use raw values (ensure they are sensible 0-1 or -1 to 1)
                return s_clean
            return series.fillna(0.0) # Default for unknown method

        # --- Gamma Concentration Factor ---
        factor_cfg = self.config.ranking_factors_config.gamma_concentration
        col_name = cfg_cols_rep.gamma_concentration_factor
        raw_metric_col = cfg_cols_in.gxoi
        if raw_metric_col in factors_df.columns:
            raw_series = factors_df[raw_metric_col].abs() # Typically use absolute for concentration
            factors_df[col_name] = normalize_series(raw_series, factor_cfg.normalization_method,
                                                   factor_cfg.tanh_scale_factor_source, factor_cfg.iqr_clip_range)
        else: factors_df[col_name] = 0.0

        # --- Flow Consistency Factor ---
        factor_cfg = self.config.ranking_factors_config.flow_consistency
        col_name = cfg_cols_rep.flow_consistency_factor
        raw_metric_col = cfg_cols_in.volmbs_15m # Example, could be ratio or difference
        if raw_metric_col in factors_df.columns:
            # Example: consistency could be ratio of 15m to 60m, or just magnitude of 15m if that's the proxy
            raw_series = factors_df[raw_metric_col].abs()
            factors_df[col_name] = normalize_series(raw_series, factor_cfg.normalization_method,
                                                   factor_cfg.tanh_scale_factor_source, factor_cfg.iqr_clip_range)
        else: factors_df[col_name] = 0.0

        # --- Delta-Gamma Alignment Factor ---
        # Assumes score is already -1 to 1 or 0 to 1 from calculation (e.g. 1 - abs(norm_delta - norm_gamma))
        # Or, if it's based on (gxoi * dxoi).abs() / (gxoi.abs() * dxoi.abs() + eps), it's already 0-1.
        # The current calculation: (1 - numerator_dga / denominator_dga) is 0-1. "none" normalization is fine.
        factor_cfg = self.config.ranking_factors_config.delta_gamma_alignment
        col_name = cfg_cols_rep.delta_gamma_alignment_factor
        gxoi_agg_col = cfg_cols_in.gxoi
        dxoi_agg_col = cfg_cols_in.dxoi
        if gxoi_agg_col in factors_df.columns and dxoi_agg_col in factors_df.columns:
            gxoi_s = factors_df[gxoi_agg_col]
            dxoi_s = factors_df[dxoi_agg_col]
            # Alignment: positive if signs are same, negative if different. Max value 1.
            # A simple approach: sign(gxoi) * sign(dxoi). This gives -1 or 1. Normalize to 0-1 if needed: (val + 1)/2
            # Current: (1 - abs(gxoi*dxoi) / (abs(gxoi)*abs(dxoi)+eps)) - this is more like a measure of magnitude agreement
            # Let's use a direct sign alignment: (np.sign(gxoi_s) * np.sign(dxoi_s) + 1) / 2 for 0-1 score
            raw_series = (np.sign(gxoi_s) * np.sign(dxoi_s) + 1.0) / 2.0
            factors_df[col_name] = normalize_series(raw_series.fillna(0.5), factor_cfg.normalization_method, #fillna 0.5 for neutral alignment
                                             factor_cfg.tanh_scale_factor_source, factor_cfg.iqr_clip_range)

        else: factors_df[col_name] = 0.0


        # --- Volatility Sensitivity Factor ---
        factor_cfg = self.config.ranking_factors_config.volatility_sensitivity
        col_name = cfg_cols_rep.volatility_sensitivity_factor
        vannaxoi_agg_col, vommaxoi_agg_col = cfg_cols_in.vannaxoi, cfg_cols_in.vommaxoi
        if vannaxoi_agg_col in factors_df.columns and vommaxoi_agg_col in factors_df.columns:
            abs_vannaxoi = factors_df[vannaxoi_agg_col].abs()
            abs_vommaxoi = factors_df[vommaxoi_agg_col].abs()
            raw_series = abs_vannaxoi + abs_vommaxoi
            factors_df[col_name] = normalize_series(raw_series, factor_cfg.normalization_method,
                                                   factor_cfg.tanh_scale_factor_source, factor_cfg.iqr_clip_range)
        else: factors_df[col_name] = 0.0

        # --- Time Decay Sensitivity Factor ---
        factor_cfg = self.config.ranking_factors_config.time_decay_sensitivity
        col_name = cfg_cols_rep.time_decay_sensitivity_factor
        charmxoi_agg_col = cfg_cols_in.charmxoi
        if charmxoi_agg_col in factors_df.columns:
            raw_series = factors_df[charmxoi_agg_col].abs() # Charm magnitude
            factors_df[col_name] = normalize_series(raw_series, factor_cfg.normalization_method,
                                                   factor_cfg.tanh_scale_factor_source, factor_cfg.iqr_clip_range)
        else: factors_df[col_name] = 0.0

        # Ensure all factor columns from factor_output_columns are present, fill with 0 if missing
        for col in factor_output_columns:
            if col not in factors_df.columns: # Ensure column exists
                factors_df[col] = 0.0
            else: # Ensure column has no NaNs if it exists
                factors_df[col] = factors_df[col].fillna(0.0)

        return factors_df[factor_output_columns] # Return with NaNs filled for all expected cols

    def _calculate_composite_plausibility(self, ranking_factors_df: pd.DataFrame) -> pd.DataFrame:
        cfg_cols_rep = self.config.report_column_map
        composite_score_col = cfg_cols_rep.composite_plausibility_score

        if ranking_factors_df.empty:
            # Create an empty DataFrame with the composite score column if input is empty
            return pd.DataFrame(columns=ranking_factors_df.columns.tolist() + [composite_score_col])

        df_copy = ranking_factors_df.copy()
        df_copy[composite_score_col] = 0.0

        ranking_cfg = self.config.ranking_factors_config

        for factor_attr_name in ranking_cfg.__annotations__.keys():
            factor_setting = getattr(ranking_cfg, factor_attr_name)
            if not isinstance(factor_setting, RankingFactorSetting):
                continue

            # Determine the actual column name in the DataFrame for this factor
            # (e.g., methodology_diversity -> methodology_diversity_score, gamma_concentration -> gamma_concentration_factor)
            factor_df_col_name = ""
            if hasattr(cfg_cols_rep, f"{factor_attr_name}_score"):
                factor_df_col_name = getattr(cfg_cols_rep, f"{factor_attr_name}_score")
            elif hasattr(cfg_cols_rep, f"{factor_attr_name}_factor"):
                 factor_df_col_name = getattr(cfg_cols_rep, f"{factor_attr_name}_factor")
            else: # Should not happen if ReportColumnConfig is comprehensive
                 print(f"Warning (composite_plausibility): No matching report column for factor '{factor_attr_name}'. Skipping.")
                 continue

            if factor_df_col_name in df_copy.columns:
                current_weight = factor_setting.weight
                # Apply regime override for weight if exists
                if self.config.regime_definition_config.enabled and factor_setting.regime_overrides:
                    regime_override_for_factor = factor_setting.regime_overrides.get(self.current_regime)
                    if regime_override_for_factor and regime_override_for_factor.weight is not None:
                        current_weight = regime_override_for_factor.weight

                df_copy[composite_score_col] += df_copy[factor_df_col_name].fillna(0.0) * current_weight
            else:
                print(f"Warning (calculate_composite_plausibility): Factor column '{factor_df_col_name}' for attribute '{factor_attr_name}' not found in DataFrame. It won't contribute to the score.")

        return df_copy

    def _filter_by_strike_clustering(self, ranked_df: pd.DataFrame) -> pd.DataFrame:
        # This method's logic might need update if strike_clustering_pct_threshold moves into a sub-config
        # Current config: self.config.strike_clustering_pct_threshold
        return ranked_df

    def _determine_support_resistance(self, ranked_df: pd.DataFrame) -> pd.DataFrame:
        cfg_cols_rep = self.config.report_column_map # Updated
        cfg_cols_in = self.config.input_column_map   # Updated
        sr_cfg = self.config.sr_logic_config       # Updated
        strike_col_df = cfg_cols_rep.strike

        # Ensure result columns exist even if we return early
        cols_to_add = [cfg_cols_rep.level_type, cfg_cols_rep.sr_rationale]
        result_df = ranked_df.copy() # Work on a copy
        for col in cols_to_add:
            if col not in result_df.columns:
                 result_df[col] = "Uncertain" if col == cfg_cols_rep.level_type else "S/R Not Determined"

        if result_df.empty or self.strike_aggregated_metrics.empty:
            if not result_df.empty:
                result_df[cfg_cols_rep.level_type] = "Uncertain"
                result_df[cfg_cols_rep.sr_rationale] = "Missing aggregated metrics for S/R."
            return result_df

        agg_strike_col = cfg_cols_in.strike
        if agg_strike_col not in self.strike_aggregated_metrics.columns:
            print(f"Error (_determine_support_resistance): Strike column '{agg_strike_col}' not found in self.strike_aggregated_metrics.")
            return result_df

        df_for_sr = pd.merge(result_df, self.strike_aggregated_metrics,
                             left_on=strike_col_df, right_on=agg_strike_col,
                             how='left')

        if strike_col_df != agg_strike_col and agg_strike_col in df_for_sr.columns:
             df_for_sr.drop(columns=[agg_strike_col], inplace=True)

        plausibility_threshold_for_sr = sr_cfg.plausibility_min_threshold_for_sr

        dxoi_col_agg_name = cfg_cols_in.dxoi
        volmbs15_col_agg_name = cfg_cols_in.volmbs_15m

        delta_series = self.strike_aggregated_metrics.get(dxoi_col_agg_name, pd.Series(dtype=float))
        flow_series = self.strike_aggregated_metrics.get(volmbs15_col_agg_name, pd.Series(dtype=float))

        # For S/R, typically higher absolute delta/flow is significant.
        # So higher_is_better=True for positive values, higher_is_better=False for negative values (more negative is more significant)
        # This is tricky. _calculate_dynamic_threshold needs to be used carefully.
        # Alternative: calculate threshold on abs values, then check sign.

        # For delta, positive (support) and negative (resistance) are both of interest.
        # Let's calculate threshold on absolute values for magnitude, then check original sign.
        abs_delta_threshold = self._calculate_dynamic_threshold(delta_series.abs(), sr_cfg.delta_significance_config, higher_is_better=True)
        abs_flow_threshold = self._calculate_dynamic_threshold(flow_series.abs(), sr_cfg.flow_significance_config, higher_is_better=True)

        level_types = []
        rationales = []

        for _, row in df_for_sr.iterrows():
            level_type = "Uncertain"
            rationale = "Default."

            plausibility = row.get(cfg_cols_rep.composite_plausibility_score, 0)
            strike_price = row[strike_col_df]

            if plausibility >= plausibility_threshold_for_sr:
                net_delta = row.get(dxoi_col_agg_name, 0)
                net_flow = row.get(volmbs15_col_agg_name, 0)

                is_below_price = strike_price < self.underlying_price
                is_above_price = strike_price > self.underlying_price

                # Significance based on absolute thresholds
                strong_pos_delta = net_delta > abs_delta_threshold
                strong_neg_delta = net_delta < -abs_delta_threshold # More negative than negative of threshold
                strong_pos_flow = net_flow > abs_flow_threshold
                strong_neg_flow = net_flow < -abs_flow_threshold

                current_rationale_parts = [f"P={plausibility:.2f}"]

                if is_below_price:
                    current_rationale_parts.append("BelowPx")
                    if strong_pos_delta and strong_pos_flow: level_type = "Support"; current_rationale_parts.append("++Δ&Flow")
                    elif strong_pos_delta: level_type = "Support"; current_rationale_parts.append("+Δ")
                    elif strong_pos_flow: level_type = "Support"; current_rationale_parts.append("+Flow")
                    elif strong_neg_delta or strong_neg_flow: level_type = "Contested"; current_rationale_parts.append("--Δ/Flow vs Support intent")
                    else: level_type = "Potential Support"; current_rationale_parts.append("NeutralΔ/Flow")
                elif is_above_price:
                    current_rationale_parts.append("AbovePx")
                    if strong_neg_delta and strong_neg_flow: level_type = "Resistance"; current_rationale_parts.append("--Δ&Flow")
                    elif strong_neg_delta: level_type = "Resistance"; current_rationale_parts.append("-Δ")
                    elif strong_neg_flow: level_type = "Resistance"; current_rationale_parts.append("-Flow")
                    elif strong_pos_delta or strong_pos_flow: level_type = "Contested"; current_rationale_parts.append("++Δ/Flow vs Resist intent")
                    else: level_type = "Potential Resistance"; current_rationale_parts.append("NeutralΔ/Flow")
                else:
                    current_rationale_parts.append("AtPx")
                    if strong_pos_delta and strong_pos_flow: level_type = "Support"; current_rationale_parts.append("++Δ&Flow")
                    elif strong_neg_delta and strong_neg_flow: level_type = "Resistance"; current_rationale_parts.append("--Δ&Flow")
                    else: level_type = "Contested"; current_rationale_parts.append("MixedΔ/Flow")
                rationale = ", ".join(current_rationale_parts)
            else:
                rationale = f"Low Plaus. ({plausibility:.2f})"

            level_types.append(level_type)
            rationales.append(rationale)

        df_for_sr[cfg_cols_rep.level_type] = level_types
        df_for_sr[cfg_cols_rep.sr_rationale] = rationales

        final_output_cols = result_df.columns.tolist()
        # Ensure new S/R columns are in the final output if they were added to df_for_sr
        if cfg_cols_rep.level_type not in final_output_cols: final_output_cols.append(cfg_cols_rep.level_type)
        if cfg_cols_rep.sr_rationale not in final_output_cols: final_output_cols.append(cfg_cols_rep.sr_rationale)

        for col in final_output_cols: # Ensure all columns actually exist in df_for_sr before selecting
            if col not in df_for_sr.columns:
                 df_for_sr[col] = "ErrorSR"
        return df_for_sr[final_output_cols]


    def analyze(self) -> pd.DataFrame:
        cfg_cols_rep = self.config.report_column_map # Updated
        strike_col_report = cfg_cols_rep.strike

        all_methodologies_df = self.run_all_methodologies()
        if all_methodologies_df.empty:
            # Return empty DF with all expected report columns
            return pd.DataFrame(columns=[getattr(cfg_cols_rep, attr) for attr in cfg_cols_rep.__annotations__])


        ranking_factors_df = self._calculate_ranking_factors(all_methodologies_df)
        if ranking_factors_df.empty:
             return pd.DataFrame(columns=[getattr(cfg_cols_rep, attr) for attr in cfg_cols_rep.__annotations__])


        ranked_df_with_plausibility = self._calculate_composite_plausibility(ranking_factors_df)

        if strike_col_report in all_methodologies_df.columns and cfg_cols_rep.methodology in all_methodologies_df.columns:
            methods_per_strike = all_methodologies_df.groupby(strike_col_report)[cfg_cols_rep.methodology].apply(
                lambda x: ', '.join(sorted(list(set(x))))
            ).reset_index(name=cfg_cols_rep.contributing_methods)

            final_df = pd.merge(ranked_df_with_plausibility, methods_per_strike, on=strike_col_report, how='left')
            final_df[cfg_cols_rep.contributing_methods] = final_df[cfg_cols_rep.contributing_methods].fillna('')
        else:
            final_df = ranked_df_with_plausibility.copy()
            final_df[cfg_cols_rep.contributing_methods] = ''

        final_df = self._determine_support_resistance(final_df)

        # Ensure all report columns are present
        for report_col_key in cfg_cols_rep.__annotations__:
            report_col_name = getattr(cfg_cols_rep, report_col_key)
            if report_col_name not in final_df.columns:
                if 'score' in report_col_key or 'factor' in report_col_key: final_df[report_col_name] = 0.0
                elif 'count' in report_col_key: final_df[report_col_name] = 0
                else: final_df[report_col_name] = "N/A"

        desired_column_order = [
            cfg_cols_rep.strike, cfg_cols_rep.composite_plausibility_score,
            cfg_cols_rep.level_type, cfg_cols_rep.sr_rationale,
            cfg_cols_rep.methodology_count, cfg_cols_rep.methodology_diversity_score,
            cfg_cols_rep.gamma_concentration_factor, cfg_cols_rep.flow_consistency_factor,
            cfg_cols_rep.delta_gamma_alignment_factor, cfg_cols_rep.volatility_sensitivity_factor,
            cfg_cols_rep.time_decay_sensitivity_factor, cfg_cols_rep.contributing_methods,
            cfg_cols_rep.proximity_factor
        ]
        ordered_report_cols = [col for col in desired_column_order if col in final_df.columns]
        other_cols = [col for col in final_df.columns if col not in ordered_report_cols]

        return final_df[ordered_report_cols + other_cols].sort_values(
            by=cfg_cols_rep.composite_plausibility_score, ascending=False
        ).reset_index(drop=True)

# --- Update __main__ test block ---
if __name__ == '__main__':
    print("--- Testing EliteDarkpoolAnalyzer ---")
    if not _utils_imported_successfully_eda:
        print("CRITICAL: Skipping EliteDarkpoolAnalyzer tests as darkpool_analytics_utils could not be imported.")
    else:
        num_rows = 200
        sample_strikes_test = np.linspace(90, 110, num_rows // 10).repeat(10)
        if len(sample_strikes_test) < num_rows:
            sample_strikes_test = np.append(sample_strikes_test, [sample_strikes_test[-1]]*(num_rows - len(sample_strikes_test)))

        test_data = {
            'strike': sample_strikes_test[:num_rows], 'opt_kind': ['call', 'put'] * (num_rows // 2),
            'gxoi': np.random.rand(num_rows) * 1e6, 'dxoi': np.random.randn(num_rows) * 2e5,
            'volmbs_15m': np.random.randn(num_rows) * 500,
            'volmbs_60m': np.random.randn(num_rows) * 1000,
            'vannaxoi': np.random.rand(num_rows) * 1e4, 'vommaxoi': np.random.rand(num_rows) * 1e3,
            'charmxoi': np.random.rand(num_rows) * 5e3, 'gxvolm': np.random.rand(num_rows) * 1e5,
            'value_bs': np.random.randn(num_rows) * 50000,
            'volatility': np.random.uniform(0.1, 0.5, num_rows)
        }
        sample_options_data_df = pd.DataFrame(test_data)
        current_price_test = 100.5
        market_metric_test_value = 0.25 # Example: Medium Volatility

        # Instantiate the new detailed config
        analyzer_config = DarkpoolAnalyticsConfig(
            # Example: Override a methodology's threshold config for testing
            methodologies={
                **DarkpoolAnalyticsConfig().methodologies, # Start with defaults
                "high_gamma_imbalance": MethodologySetting(
                    enabled=True,
                    threshold_config=ThresholdConfig(type="relative_percentile", percentile=0.85, fallback_value=1.0),
                    proximity_influence_factor=0.6
                )
            },
            ranking_factors_config=RankingFactorsConfig(
                gamma_concentration=RankingFactorSetting(weight=0.25, normalization_method="max_abs"),
                # Keep others default or specify as needed
                methodology_diversity=RankingFactorSetting(weight=0.35, normalization_method="none"),
                flow_consistency=RankingFactorSetting(weight=0.15, normalization_method="tanh", tanh_scale_factor_source="std_dev"),
                delta_gamma_alignment=RankingFactorSetting(weight=0.10, normalization_method="none"),
                volatility_sensitivity=RankingFactorSetting(weight=0.10, normalization_method="iqr", iqr_clip_range=(0.0, 2.0)),
                time_decay_sensitivity=RankingFactorSetting(weight=0.05, normalization_method="max_abs")

            ),
            sr_logic_config=SRLogicConfig(
                plausibility_min_threshold_for_sr=0.25,
                delta_significance_config=ThresholdConfig(type="relative_mean_factor", factor=0.75, fallback_value=5000.0),
                flow_significance_config=ThresholdConfig(type="relative_mean_factor", factor=0.75, fallback_value=250.0)
            )
            # Can further customize input_column_map, report_column_map, etc.
        )

        analyzer = EliteDarkpoolAnalyzer(
            options_df=sample_options_data_df.copy(),
            underlying_price=current_price_test,
            config=analyzer_config, # Pass the new config object
            market_regime_metric_value=market_metric_test_value
        )

        print(f"Analyzer initialized. Current Regime: {analyzer.current_regime}")
        print(f"Using config version: {analyzer.config.config_version}")

        print("\n--- Running full analyze() method (Elite - with S/R logic) ---")
        final_analysis_df = analyzer.analyze()

        if not final_analysis_df.empty:
            print("Final analysis df (Top 10 ranked by plausibility with S/R):")
            # Use new report_column_map for accessing column names
            cfg_report = analyzer_config.report_column_map
            cols_to_show = [
                cfg_report.strike,
                cfg_report.composite_plausibility_score,
                cfg_report.level_type,
                cfg_report.sr_rationale,
                cfg_report.contributing_methods,
                cfg_report.methodology_count
            ]
            cols_to_show_existing = [col for col in cols_to_show if col in final_analysis_df.columns]
            print(final_analysis_df[cols_to_show_existing].head(10).to_string())

            sr_cols_expected = [cfg_report.level_type, cfg_report.sr_rationale]
            missing_sr_cols = [col for col in sr_cols_expected if col not in final_analysis_df.columns]
            if missing_sr_cols:
                print(f"\nERROR: Missing S/R columns in final_analysis_df: {missing_sr_cols}")
            else:
                print("\nS/R columns ('level_type', 'sr_rationale') are present in the final_analysis_df.")

            print("\nChecking value counts for 'level_type':")
            if cfg_report.level_type in final_analysis_df.columns:
                print(final_analysis_df[cfg_report.level_type].value_counts(dropna=False))
            else:
                print("'level_type' column not found.")
        else:
            print("Full analysis resulted in an empty DataFrame.")

class EliteDarkpoolAnalyzer:
        if self.input_df.empty or raw_score_series.empty: return empty_df_result
        if strike_col_input not in self.input_df.columns: return empty_df_result

        adjusted_score_series = raw_score_series.copy()
        if self.config.proximity_settings['enabled']:
            proximity_factors = self.input_df[cfg_cols_rep['proximity_factor']].reindex(raw_score_series.index)
            adjusted_score_series = raw_score_series * proximity_factors.fillna(1.0)

        final_quantile_val = self._get_adjusted_quantile_threshold()
        threshold = get_quantile_threshold(adjusted_score_series.dropna(), final_quantile_val)

        if threshold is None or pd.isna(threshold):
            return empty_df_result

        significant_mask = (adjusted_score_series > threshold) & adjusted_score_series.notna()
        if not significant_mask.any():
            return empty_df_result

        result_df = self.input_df.loc[significant_mask, [strike_col_input]].copy()
        result_df.rename(columns={strike_col_input: cfg_cols_rep['strike']}, inplace=True)

        result_df[cfg_cols_rep['raw_score']] = raw_score_series[significant_mask]
        result_df[cfg_cols_rep['adjusted_score']] = adjusted_score_series[significant_mask]
        result_df[cfg_cols_rep['methodology']] = methodology_name

        return result_df.drop_duplicates(subset=[cfg_cols_rep['strike']]).reset_index(drop=True)

    def _analyze_high_gamma_imbalance(self) -> pd.DataFrame:
        col = self.config.cols_input['gxoi']
        if col not in self.input_df.columns: return self._analyze_method_template(pd.Series(dtype=float), "High Gamma Imbalance")
        metric = pd.to_numeric(self.input_df[col], errors='coerce').fillna(0)
        return self._analyze_method_template(calculate_zscore(metric), "High Gamma Imbalance")

    def _analyze_delta_gamma_divergence(self) -> pd.DataFrame:
        c_dx, c_gx = self.config.cols_input['dxoi'], self.config.cols_input['gxoi']
        if c_dx not in self.input_df.columns or c_gx not in self.input_df.columns: return self._analyze_method_template(pd.Series(dtype=float),"Delta-Gamma Divergence")
        dz = calculate_zscore(pd.to_numeric(self.input_df[c_dx], errors='coerce').fillna(0))
        gz = calculate_zscore(pd.to_numeric(self.input_df[c_gx], errors='coerce').fillna(0))
        return self._analyze_method_template((gz - dz).abs(), "Delta-Gamma Divergence")

    def _analyze_flow_anomaly(self) -> pd.DataFrame:
        c15, c60 = self.config.cols_input['volmbs_15m'], self.config.cols_input['volmbs_60m']
        if c15 not in self.input_df.columns or c60 not in self.input_df.columns: return self._analyze_method_template(pd.Series(dtype=float),"Flow Anomaly")
        z15 = calculate_zscore(pd.to_numeric(self.input_df[c15], errors='coerce').fillna(0))
        z60 = calculate_zscore(pd.to_numeric(self.input_df[c60], errors='coerce').fillna(0))
        return self._analyze_method_template(z15.abs() + (z15 - z60).abs(), "Flow Anomaly")

    def _analyze_volatility_sensitivity(self) -> pd.DataFrame:
        c_van, c_vom = self.config.cols_input['vannaxoi'], self.config.cols_input['vommaxoi']
        if c_van not in self.input_df.columns or c_vom not in self.input_df.columns: return self._analyze_method_template(pd.Series(dtype=float),"Volatility Sensitivity")
        van_s = pd.to_numeric(self.input_df[c_van], errors='coerce').fillna(0)
        vom_s = pd.to_numeric(self.input_df[c_vom], errors='coerce').fillna(0)
        raw_scores_z = calculate_zscore(van_s.abs() + vom_s.abs())
        return self._analyze_method_template(raw_scores_z, "Volatility Sensitivity")

    def _analyze_charm_adjusted_gamma(self) -> pd.DataFrame:
        c_gx, c_ch = self.config.cols_input['gxoi'], self.config.cols_input['charmxoi']
        if c_gx not in self.input_df.columns or c_ch not in self.input_df.columns: return self._analyze_method_template(pd.Series(dtype=float),"Charm-Adjusted Gamma")
        gz = calculate_zscore(pd.to_numeric(self.input_df[c_gx], errors='coerce').fillna(0))
        cz = calculate_zscore(pd.to_numeric(self.input_df[c_ch], errors='coerce').fillna(0))
        return self._analyze_method_template(gz * (1 + cz.abs()), "Charm-Adjusted Gamma")

    def _analyze_active_hedging_detection(self) -> pd.DataFrame:
        c_gx, c_gxvolm = self.config.cols_input['gxoi'], self.config.cols_input['gxvolm']
        if c_gx not in self.input_df.columns or c_gxvolm not in self.input_df.columns: return self._analyze_method_template(pd.Series(dtype=float),"Active Hedging Detection")
        gz = calculate_zscore(pd.to_numeric(self.input_df[c_gx], errors='coerce').fillna(0))
        gzvol = calculate_zscore(pd.to_numeric(self.input_df[c_gxvolm], errors='coerce').fillna(0))
        return self._analyze_method_template(gz * gzvol, "Active Hedging Detection")

    def _analyze_value_volume_divergence(self) -> pd.DataFrame:
        c_val, c_vol = self.config.cols_input['value_bs'], self.config.cols_input['volmbs_15m']
        if c_val not in self.input_df.columns or c_vol not in self.input_df.columns: return self._analyze_method_template(pd.Series(dtype=float),"Value-Volume Divergence")
        val_z = calculate_zscore(pd.to_numeric(self.input_df[c_val], errors='coerce').fillna(0))
        vol_z = calculate_zscore(pd.to_numeric(self.input_df[c_vol], errors='coerce').fillna(0))
        return self._analyze_method_template((val_z - vol_z).abs(), "Value-Volume Divergence")

    def run_all_methodologies(self) -> pd.DataFrame:
        all_results = []
        method_functions = [
            self._analyze_high_gamma_imbalance, self._analyze_delta_gamma_divergence,
            self._analyze_flow_anomaly, self._analyze_volatility_sensitivity,
            self._analyze_charm_adjusted_gamma, self._analyze_active_hedging_detection,
            self._analyze_value_volume_divergence
        ]
        for func in method_functions:
            try:
                res_df = func()
                if isinstance(res_df, pd.DataFrame) and not res_df.empty:
                    all_results.append(res_df)
            except Exception as e:
                print(f"Error running methodology {func.__name__}: {e}")

        if not all_results:
            return pd.DataFrame(columns=[self.config.cols_report['strike'], self.config.cols_report['raw_score'],
                                         self.config.cols_report['adjusted_score'], self.config.cols_report['methodology']])

        combined_df = pd.concat(all_results, ignore_index=True)
        return combined_df

    def _calculate_ranking_factors(self, all_methodologies_result_df: pd.DataFrame) -> pd.DataFrame:
        cfg_cols_in = self.config.cols_input
        cfg_cols_rep = self.config.cols_report
        strike_col_report = cfg_cols_rep['strike']

        factor_output_columns = [
            strike_col_report, cfg_cols_rep['methodology_count'], cfg_cols_rep['methodology_diversity_score'],
            cfg_cols_rep['gamma_concentration_factor'], cfg_cols_rep['flow_consistency_factor'],
            cfg_cols_rep['delta_gamma_alignment_factor'], cfg_cols_rep['volatility_sensitivity_factor'],
            cfg_cols_rep['time_decay_sensitivity_factor']
        ]

        if all_methodologies_result_df.empty:
            return pd.DataFrame(columns=factor_output_columns)

        methodology_diversity = all_methodologies_result_df.groupby(strike_col_report)[cfg_cols_rep['methodology']].nunique().reset_index(name=cfg_cols_rep['methodology_count'])
        methodology_diversity[cfg_cols_rep['methodology_diversity_score']] = methodology_diversity[cfg_cols_rep['methodology_count']] / 7.0

        strike_agg_df = self.strike_aggregated_metrics
        if strike_agg_df.empty:
            print("Warning (_calculate_ranking_factors): Pre-calculated strike_aggregated_metrics is empty.")
            factors_df = methodology_diversity.copy()
            for factor_col_key in ['gamma_concentration_factor', 'flow_consistency_factor',
                                   'delta_gamma_alignment_factor', 'volatility_sensitivity_factor',
                                   'time_decay_sensitivity_factor']:
                factors_df[cfg_cols_rep[factor_col_key]] = 0.0
            return factors_df.reindex(columns=factor_output_columns).fillna(0.0)

        factors_df = pd.merge(methodology_diversity, strike_agg_df, on=strike_col_report, how='left').fillna(0.0)

        gxoi_agg_col = cfg_cols_in.get('gxoi', 'gxoi')
        if gxoi_agg_col in factors_df.columns:
            abs_gxoi = factors_df[gxoi_agg_col].abs()
            max_abs_gxoi = abs_gxoi.max() if not abs_gxoi.empty else 0
            factors_df[cfg_cols_rep['gamma_concentration_factor']] = (abs_gxoi / max_abs_gxoi if max_abs_gxoi != 0 else 0.0).fillna(0.0)
        else: factors_df[cfg_cols_rep['gamma_concentration_factor']] = 0.0

        volmbs15_agg_col = cfg_cols_in.get('volmbs_15m', 'volmbs_15m')
        if volmbs15_agg_col in factors_df.columns:
            abs_volmbs15m = factors_df[volmbs15_agg_col].abs()
            max_abs_volmbs15m = abs_volmbs15m.max() if not abs_volmbs15m.empty else 0
            factors_df[cfg_cols_rep['flow_consistency_factor']] = (abs_volmbs15m / max_abs_volmbs15m if max_abs_volmbs15m != 0 else 0.0).fillna(0.0)
        else: factors_df[cfg_cols_rep['flow_consistency_factor']] = 0.0

        dxoi_agg_col = cfg_cols_in.get('dxoi', 'dxoi')
        if gxoi_agg_col in factors_df.columns and dxoi_agg_col in factors_df.columns:
            gxoi_s = factors_df[gxoi_agg_col]
            dxoi_s = factors_df[dxoi_agg_col]
            numerator_dga = (gxoi_s * dxoi_s).abs()
            denominator_dga = (gxoi_s.abs() * dxoi_s.abs()) + 1e-9
            factors_df[cfg_cols_rep['delta_gamma_alignment_factor']] = (1 - numerator_dga / denominator_dga).fillna(0.0)
        else: factors_df[cfg_cols_rep['delta_gamma_alignment_factor']] = 0.0

        vannaxoi_agg_col, vommaxoi_agg_col = cfg_cols_in.get('vannaxoi', 'vannaxoi'), cfg_cols_in.get('vommaxoi', 'vommaxoi')
        if vannaxoi_agg_col in factors_df.columns and vommaxoi_agg_col in factors_df.columns:
            abs_vannaxoi = factors_df[vannaxoi_agg_col].abs()
            abs_vommaxoi = factors_df[vommaxoi_agg_col].abs()
            vol_sens_sum = abs_vannaxoi + abs_vommaxoi
            max_vol_sens_sum = vol_sens_sum.max() if not vol_sens_sum.empty else 0
            factors_df[cfg_cols_rep['volatility_sensitivity_factor']] = (vol_sens_sum / max_vol_sens_sum if max_vol_sens_sum != 0 else 0.0).fillna(0.0)
        else: factors_df[cfg_cols_rep['volatility_sensitivity_factor']] = 0.0

        charmxoi_agg_col = cfg_cols_in.get('charmxoi', 'charmxoi')
        if charmxoi_agg_col in factors_df.columns:
            abs_charmxoi = factors_df[charmxoi_agg_col].abs()
            max_abs_charmxoi = abs_charmxoi.max() if not abs_charmxoi.empty else 0
            factors_df[cfg_cols_rep['time_decay_sensitivity_factor']] = (abs_charmxoi / max_abs_charmxoi if max_abs_charmxoi != 0 else 0.0).fillna(0.0)
        else: factors_df[cfg_cols_rep['time_decay_sensitivity_factor']] = 0.0

        for col_key in factor_output_columns:
            if col_key not in factors_df.columns:
                factors_df[col_key] = 0.0

        return factors_df[factor_output_columns]

    def _calculate_composite_plausibility(self, ranking_factors_df: pd.DataFrame) -> pd.DataFrame:
        cfg_cols_rep = self.config.cols_report
        if ranking_factors_df.empty:
            return pd.DataFrame(columns=ranking_factors_df.columns.tolist() + [cfg_cols_rep['composite_plausibility_score']])

        df_copy = ranking_factors_df.copy()
        weights = self.config.ranking_weights

        df_copy[cfg_cols_rep['composite_plausibility_score']] = 0.0

        for factor_config_key, weight in weights.items():
            factor_col_name = cfg_cols_rep.get(factor_config_key)
            if factor_col_name and factor_col_name in df_copy.columns:
                df_copy[cfg_cols_rep['composite_plausibility_score']] += df_copy[factor_col_name].fillna(0.0) * weight
            else:
                print(f"Warning (calculate_composite_plausibility): Factor column for key '{factor_config_key}' (expected name: '{factor_col_name}') not found. It won't contribute to the score.")

        return df_copy

    def _filter_by_strike_clustering(self, ranked_df: pd.DataFrame) -> pd.DataFrame:
        # Current config: self.config.strike_clustering_pct_threshold
        return ranked_df

    def _determine_support_resistance(self, ranked_df: pd.DataFrame) -> pd.DataFrame:
        cfg_cols_rep = self.config.report_column_map
        cfg_cols_in = self.config.input_column_map
        sr_cfg = self.config.sr_logic_config
        strike_col_df = cfg_cols_rep.strike # This is the strike col name in ranked_df

        result_df = ranked_df.copy()
        # Ensure S/R columns exist, initialize if not
        if cfg_cols_rep.level_type not in result_df.columns:
            result_df[cfg_cols_rep.level_type] = "Uncertain"
        if cfg_cols_rep.sr_rationale not in result_df.columns:
            result_df[cfg_cols_rep.sr_rationale] = "S/R Not Determined"

        if result_df.empty or self.strike_aggregated_metrics.empty:
            if not result_df.empty:
                result_df[cfg_cols_rep.level_type] = "Uncertain"
                result_df[cfg_cols_rep.sr_rationale] = "Missing aggregated metrics for S/R."
            return result_df

        # Strike column in strike_aggregated_metrics is from input_column_map
        agg_strike_col = cfg_cols_in.strike
        if agg_strike_col not in self.strike_aggregated_metrics.columns:
            # This check is important if strike_aggregated_metrics might not have the expected strike col
            print(f"Error (_determine_support_resistance): Strike column '{agg_strike_col}' not found in self.strike_aggregated_metrics.")
            result_df[cfg_cols_rep.sr_rationale] = f"Agg strike col '{agg_strike_col}' missing."
            return result_df

        # Merge aggregated metrics. ranked_df uses report_column_map.strike (aliased as strike_col_df here)
        # strike_aggregated_metrics uses input_column_map.strike (aliased as agg_strike_col here)
        df_for_sr = pd.merge(result_df, self.strike_aggregated_metrics,
                             left_on=strike_col_df, right_on=agg_strike_col,
                             how='left')

        # If merge created two strike columns due to different names and agg_strike_col was not same as strike_col_df
        if strike_col_df != agg_strike_col and agg_strike_col in df_for_sr.columns:
             df_for_sr.drop(columns=[agg_strike_col], inplace=True) # Keep the one from ranked_df (strike_col_df)

        plausibility_threshold_for_sr = sr_cfg.plausibility_min_threshold_for_sr

        # Column names for delta and flow in the merged DataFrame (they came from strike_aggregated_metrics)
        dxoi_col_name_in_merged_df = cfg_cols_in.dxoi
        volmbs15_col_name_in_merged_df = cfg_cols_in.volmbs_15m

        # Get the original series from strike_aggregated_metrics for threshold calculation
        delta_series_for_threshold_calc = self.strike_aggregated_metrics.get(dxoi_col_name_in_merged_df, pd.Series(dtype=float))
        flow_series_for_threshold_calc = self.strike_aggregated_metrics.get(volmbs15_col_name_in_merged_df, pd.Series(dtype=float))

        # Calculate thresholds based on the absolute values of the original series
        abs_delta_threshold = self._calculate_dynamic_threshold(delta_series_for_threshold_calc.abs(), sr_cfg.delta_significance_config, higher_is_better=True)
        abs_flow_threshold = self._calculate_dynamic_threshold(flow_series_for_threshold_calc.abs(), sr_cfg.flow_significance_config, higher_is_better=True)

        level_types = []
        rationales = []

        for _, row in df_for_sr.iterrows():
            level_type = "Uncertain" # Default for this row
            rationale_parts = []     # Default for this row

            plausibility = row.get(cfg_cols_rep.composite_plausibility_score, 0)
            # strike_price already refers to the correct strike column from ranked_df (cfg_cols_rep.strike)
            strike_price = row[cfg_cols_rep.strike]

            if plausibility >= plausibility_threshold_for_sr:
                rationale_parts.append(f"P={plausibility:.2f}")
                net_delta = row.get(dxoi_col_name_in_merged_df, 0)
                net_flow = row.get(volmbs15_col_name_in_merged_df, 0)

                is_below_price = strike_price < self.underlying_price
                is_above_price = strike_price > self.underlying_price

                strong_pos_delta = net_delta > abs_delta_threshold
                strong_neg_delta = net_delta < -abs_delta_threshold
                strong_pos_flow = net_flow > abs_flow_threshold
                strong_neg_flow = net_flow < -abs_flow_threshold

                if is_below_price:
                    rationale_parts.append("BelowPx")
                    if strong_pos_delta and strong_pos_flow: level_type = "Support"; rationale_parts.append("++Δ&Flow")
                    elif strong_pos_delta: level_type = "Support"; rationale_parts.append("+Δ")
                    elif strong_pos_flow: level_type = "Support"; rationale_parts.append("+Flow")
                    elif strong_neg_delta or strong_neg_flow: level_type = "Contested"; rationale_parts.append("--Δ/Flow vs S")
                    else: level_type = "Potential Support"; rationale_parts.append("NeutralΔ/Flow")
                elif is_above_price:
                    rationale_parts.append("AbovePx")
                    if strong_neg_delta and strong_neg_flow: level_type = "Resistance"; rationale_parts.append("--Δ&Flow")
                    elif strong_neg_delta: level_type = "Resistance"; rationale_parts.append("-Δ")
                    elif strong_neg_flow: level_type = "Resistance"; rationale_parts.append("-Flow")
                    elif strong_pos_delta or strong_pos_flow: level_type = "Contested"; rationale_parts.append("++Δ/Flow vs R")
                    else: level_type = "Potential Resistance"; rationale_parts.append("NeutralΔ/Flow")
                else: # At Price
                    rationale_parts.append("AtPx")
                    if strong_pos_delta and strong_pos_flow: level_type = "Support"; rationale_parts.append("++Δ&Flow")
                    elif strong_neg_delta and strong_neg_flow: level_type = "Resistance"; rationale_parts.append("--Δ&Flow")
                    else: level_type = "Contested"; rationale_parts.append("MixedΔ/Flow")

                rationale = ", ".join(rationale_parts) if rationale_parts else "Threshold met, neutral signals."
            else:
                rationale = f"Low Plaus. ({plausibility:.2f})"

            level_types.append(level_type)
            rationales.append(rationale)

        df_for_sr[cfg_cols_rep.level_type] = level_types
        df_for_sr[cfg_cols_rep.sr_rationale] = rationales

        # Select only columns that were in the original ranked_df plus the new S/R columns
        # This ensures no extra columns from strike_aggregated_metrics (like _agg suffixed ones if merge keys were different) persist.
        final_cols = ranked_df.columns.tolist() # Start with original columns
        if cfg_cols_rep.level_type not in final_cols: final_cols.append(cfg_cols_rep.level_type)
        if cfg_cols_rep.sr_rationale not in final_cols: final_cols.append(cfg_cols_rep.sr_rationale)

        # Ensure all desired final columns actually exist in df_for_sr before selecting
        # This can happen if ranked_df was empty initially and columns were added.
        for col in final_cols:
            if col not in df_for_sr.columns:
                # This case should ideally not happen if result_df was initialized correctly
                df_for_sr[col] = "ErrorMissingCol" if col in [cfg_cols_rep.level_type, cfg_cols_rep.sr_rationale] else 0.0


        return df_for_sr[final_cols]


    def analyze(self) -> pd.DataFrame:
        cfg_cols_rep = self.config.report_column_map # Updated

        all_methodologies_df = self.run_all_methodologies()

        expected_report_cols = [getattr(cfg_cols_rep, attr) for attr in cfg_cols_rep.__annotations__]
        if all_methodologies_df.empty:
            return pd.DataFrame(columns=expected_report_cols)

        ranking_factors_df = self._calculate_ranking_factors(all_methodologies_df)
        if ranking_factors_df.empty:
             return pd.DataFrame(columns=expected_report_cols)

        ranked_df_with_plausibility = self._calculate_composite_plausibility(ranking_factors_df)

        # Ensure strike column from report_column_map is used for merging and grouping if different from input
        # all_methodologies_df should have used report_column_map.strike already
        if cfg_cols_rep.strike in all_methodologies_df.columns and \
           cfg_cols_rep.methodology in all_methodologies_df.columns:

            methods_per_strike = all_methodologies_df.groupby(cfg_cols_rep.strike)[cfg_cols_rep.methodology].apply(
                lambda x: ', '.join(sorted(list(set(x))))
            ).reset_index(name=cfg_cols_rep.contributing_methods)

            final_df = pd.merge(ranked_df_with_plausibility, methods_per_strike,
                                on=cfg_cols_rep.strike, # Both should use the report strike col name here
                                how='left')
            final_df[cfg_cols_rep.contributing_methods] = final_df[cfg_cols_rep.contributing_methods].fillna('')
        else:
            final_df = ranked_df_with_plausibility.copy()
            final_df[cfg_cols_rep.contributing_methods] = '' # Ensure column exists

        final_df = self._determine_support_resistance(final_df)

        # Ensure all report columns are present as per ReportColumnConfig
        for report_col_attr_name in cfg_cols_rep.__annotations__:
            report_col_name = getattr(cfg_cols_rep, report_col_attr_name)
            if report_col_name not in final_df.columns:
                # Provide default values based on typical content of the column
                if 'score' in report_col_attr_name or 'factor' in report_col_attr_name:
                    final_df[report_col_name] = 0.0
                elif 'count' in report_col_attr_name:
                    final_df[report_col_name] = 0
                else: # Strings like methodology, rationale, level_type, strike
                    final_df[report_col_name] = "N/A" if report_col_attr_name != 'strike' else 0.0

        # Define the desired order of columns for the final report using ReportColumnConfig attributes
        desired_column_order = [
            cfg_cols_rep.strike, cfg_cols_rep.composite_plausibility_score,
            cfg_cols_rep.level_type, cfg_cols_rep.sr_rationale,
            cfg_cols_rep.methodology_count, cfg_cols_rep.methodology_diversity_score,
            cfg_cols_rep.gamma_concentration_factor, cfg_cols_rep.flow_consistency_factor,
            cfg_cols_rep.delta_gamma_alignment_factor, cfg_cols_rep.volatility_sensitivity_factor,
            cfg_cols_rep.time_decay_sensitivity_factor, cfg_cols_rep.contributing_methods,
        ]
        # Add proximity_factor if it's defined in report_column_map and not already included
        if hasattr(cfg_cols_rep, 'proximity_factor') and cfg_cols_rep.proximity_factor not in desired_column_order:
             desired_column_order.append(cfg_cols_rep.proximity_factor)

        ordered_report_cols = [col for col in desired_column_order if col in final_df.columns]
        # Include any other columns that might have been generated but are not in desired_column_order (e.g. raw scores from methodologies)
        other_cols = [col for col in final_df.columns if col not in ordered_report_cols]

        final_ordered_df = final_df[ordered_report_cols + other_cols]

        return final_ordered_df.sort_values(
            by=cfg_cols_rep.composite_plausibility_score, ascending=False
        ).reset_index(drop=True)

# --- Update __main__ test block ---
if __name__ == '__main__':
    print("--- Testing EliteDarkpoolAnalyzer ---")
    if not _utils_imported_successfully_eda:
        print("CRITICAL: Skipping EliteDarkpoolAnalyzer tests as darkpool_analytics_utils could not be imported.")
    else:
        num_rows = 200
        sample_strikes_test = np.linspace(90, 110, num_rows // 10).repeat(10)
        if len(sample_strikes_test) < num_rows:
            sample_strikes_test = np.append(sample_strikes_test, [sample_strikes_test[-1]]*(num_rows - len(sample_strikes_test)))

        test_data = {
            'strike': sample_strikes_test[:num_rows], 'opt_kind': ['call', 'put'] * (num_rows // 2),
            'gxoi': np.random.rand(num_rows) * 1e6, 'dxoi': np.random.randn(num_rows) * 2e5,
            'volmbs_15m': np.random.randn(num_rows) * 500,
            'volmbs_60m': np.random.randn(num_rows) * 1000,
            'vannaxoi': np.random.rand(num_rows) * 1e4, 'vommaxoi': np.random.rand(num_rows) * 1e3,
            'charmxoi': np.random.rand(num_rows) * 5e3, 'gxvolm': np.random.rand(num_rows) * 1e5,
            'value_bs': np.random.randn(num_rows) * 50000,
            'volatility': np.random.uniform(0.1, 0.5, num_rows)
        }
        sample_options_data_df = pd.DataFrame(test_data)
        current_price_test = 100.5
        market_metric_test_value = 0.25 # Example: Medium Volatility

        # Instantiate the new detailed config
        analyzer_config = DarkpoolAnalyticsConfig(
            report_column_map=ReportColumnConfig(strike="StrikePrice", composite_plausibility_score="Plausibility"), # Example of customizing report names
            methodologies={
                **DarkpoolAnalyticsConfig().methodologies,
                "high_gamma_imbalance": MethodologySetting(
                    enabled=True,
                    threshold_config=ThresholdConfig(type="relative_percentile", percentile=0.85, fallback_value=1.0),
                    proximity_influence_factor=0.6
                )
            },
            ranking_factors_config=RankingFactorsConfig(
                gamma_concentration=RankingFactorSetting(weight=0.25, normalization_method="max_abs"),
                methodology_diversity=RankingFactorSetting(weight=0.35, normalization_method="none"),
                flow_consistency=RankingFactorSetting(weight=0.15, normalization_method="tanh", tanh_scale_factor_source="std_dev"),
                delta_gamma_alignment=RankingFactorSetting(weight=0.10, normalization_method="none"),
                volatility_sensitivity=RankingFactorSetting(weight=0.10, normalization_method="iqr", iqr_clip_range=(0.0, 2.0)),
                time_decay_sensitivity=RankingFactorSetting(weight=0.05, normalization_method="max_abs")
            ),
            sr_logic_config=SRLogicConfig(
                plausibility_min_threshold_for_sr=0.25,
                delta_significance_config=ThresholdConfig(type="relative_mean_factor", factor=0.75, fallback_value=5000.0),
                flow_significance_config=ThresholdConfig(type="relative_mean_factor", factor=0.75, fallback_value=250.0)
            )
        )

        analyzer = EliteDarkpoolAnalyzer(
            options_df=sample_options_data_df.copy(),
            underlying_price=current_price_test,
            config=analyzer_config,
            market_regime_metric_value=market_metric_test_value
        )

        print(f"Analyzer initialized. Current Regime: {analyzer.current_regime}")
        print(f"Using config version: {analyzer.config.config_version}")
        print(f"Report strike column name: {analyzer.config.report_column_map.strike}") # Example of accessing new config

        print("\n--- Running full analyze() method (Elite - with S/R logic) ---")
        final_analysis_df = analyzer.analyze()

        if not final_analysis_df.empty:
            print("Final analysis df (Top 10 ranked by plausibility with S/R):")
            cfg_report = analyzer.config.report_column_map # Use the instance's config
            cols_to_show = [
                cfg_report.strike,
                cfg_report.composite_plausibility_score,
                cfg_report.level_type,
                cfg_report.sr_rationale,
                cfg_report.contributing_methods,
                cfg_report.methodology_count
            ]
            # Ensure all columns in cols_to_show actually exist in the DataFrame
            cols_to_show_existing = [col for col in cols_to_show if col in final_analysis_df.columns]
            print(final_analysis_df[cols_to_show_existing].head(10).to_string())

            sr_cols_expected = [cfg_report.level_type, cfg_report.sr_rationale]
            missing_sr_cols = [col for col in sr_cols_expected if col not in final_analysis_df.columns]
            if missing_sr_cols:
                print(f"\nERROR: Missing S/R columns in final_analysis_df: {missing_sr_cols}")
            else:
                print("\nS/R columns ('level_type', 'sr_rationale') are present in the final_analysis_df.")

            print("\nChecking value counts for 'level_type':")
            if cfg_report.level_type in final_analysis_df.columns:
                print(final_analysis_df[cfg_report.level_type].value_counts(dropna=False))
            else:
                print(f"'{cfg_report.level_type}' column not found.")
        else:
            print("Full analysis resulted in an empty DataFrame.")
```
