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


@dataclass
class DarkpoolAnalyticsConfig:
    cols_input: Dict[str, str] = field(default_factory=lambda: {
        'strike': 'strike', 'opt_kind': 'opt_kind', 'gxoi': 'gxoi', 'dxoi': 'dxoi',
        'volmbs_15m': 'volmbs_15m', 'volmbs_60m': 'volmbs_60m',
        'vannaxoi': 'vannaxoi', 'vommaxoi': 'vommaxoi',
        'charmxoi': 'charmxoi', 'gxvolm': 'gxvolm', 'value_bs': 'value_bs',
        'volatility': 'volatility',
        'underlying_price_col_options_df': 'underlying_price'
    })

    cols_report: Dict[str, str] = field(default_factory=lambda: {
        'strike': 'strike', 'methodology': 'methodology',
        'raw_score': 'method_raw_score', 'adjusted_score': 'method_adjusted_score',
        'proximity_factor': 'proximity_factor', 'regime': 'market_regime',
        'methodology_diversity_score': 'methodology_diversity_score',
        'gamma_concentration_factor': 'gamma_concentration_factor',
        'flow_consistency_factor': 'flow_consistency_factor',
        'delta_gamma_alignment_factor': 'delta_gamma_alignment_factor',
        'volatility_sensitivity_factor': 'volatility_sensitivity_factor',
        'time_decay_sensitivity_factor': 'time_decay_sensitivity_factor',
        'composite_plausibility_score': 'composite_plausibility_score',
        'contributing_methods': 'contributing_methods',
        'methodology_count': 'methodology_count',
        'level_type': 'level_type',
        'sr_rationale': 'sr_rationale'
    })

    default_quantile: float = 0.9

    regime_settings: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True, 'metric_source': 'hv_20d',
        'thresholds': {'low_medium_boundary': 0.15, 'medium_high_boundary': 0.30},
        'adjustments': {
            'Low': {'quantile_modifier': 0.05, 'proximity_strength_factor': 0.8, 'generic_impact_weight': 0.9, 'vol_sensitivity_factor_weight': 0.8},
            'Medium': {'quantile_modifier': 0.0, 'proximity_strength_factor': 1.0, 'generic_impact_weight': 1.0, 'vol_sensitivity_factor_weight': 1.0},
            'High': {'quantile_modifier': -0.05, 'proximity_strength_factor': 1.2, 'generic_impact_weight': 1.1, 'vol_sensitivity_factor_weight': 1.2}
        }
    })

    proximity_settings: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True, 'exp_decay_factor': 2.0, 'volatility_influence': 0.5
    })

    ranking_weights: Dict[str, float] = field(default_factory=lambda: {
        'methodology_diversity_score': 0.40, 'gamma_concentration_factor': 0.20,
        'flow_consistency_factor': 0.15, 'delta_gamma_alignment_factor': 0.10,
        'volatility_sensitivity_factor': 0.10, 'time_decay_sensitivity_factor': 0.05
    })
    strike_clustering_pct_threshold: float = 0.01

    # New field for S/R determination settings
    sr_settings: Dict[str, float] = field(default_factory=lambda: {
        'plausibility_min_threshold': 0.3, # Minimum plausibility to be considered for S/R
        'delta_std_factor': 0.5,         # Std deviation factor for significant delta
        'flow_std_factor': 0.5           # Std deviation factor for significant flow
    })


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

        self._prepare_input_df()

        self.current_regime: str = "Medium"
        if self.config.regime_settings['enabled']:
            if market_regime_metric_value is not None:
                self.current_regime = determine_simplified_regime(
                    float(market_regime_metric_value),
                    self.config.regime_settings['thresholds']
                )
            else:
                print("Warning (EliteDarkpoolAnalyzer): Regime is enabled but market_regime_metric_value is None. Defaulting to Medium.")

        self.input_df[self.config.cols_report['proximity_factor']] = 1.0
        if self.config.proximity_settings['enabled'] and not self.input_df.empty:
            strike_col_name = self.config.cols_input.get('strike', 'strike')
            if strike_col_name in self.input_df.columns:
                self.input_df[self.config.cols_report['proximity_factor']] = self._calculate_proximity_factor(
                    self.input_df[strike_col_name]
                )
            else:
                print(f"Warning (EliteDarkpoolAnalyzer): Strike column '{strike_col_name}' not found for proximity calculation. Factors set to 1.0.")

        metrics_to_agg = [
            self.config.cols_input[k] for k in
            ['gxoi', 'dxoi', 'volmbs_15m', 'vannaxoi', 'vommaxoi', 'charmxoi']
            if self.config.cols_input.get(k) in self.input_df.columns
        ]
        self.strike_aggregated_metrics = self._aggregate_metrics_by_strike_internal(metrics_to_agg)

    def _aggregate_metrics_by_strike_internal(self, metric_cols: List[str]) -> pd.DataFrame:
        strike_col = self.config.cols_input['strike']
        if strike_col not in self.input_df.columns:
            return pd.DataFrame(columns=[strike_col] + metric_cols)

        valid_metric_cols = [col for col in metric_cols if col in self.input_df.columns]
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
        for key, col_name in self.config.cols_input.items():
            if col_name in self.input_df.columns:
                if key not in ['opt_kind', 'strike']:
                    self.input_df[col_name] = pd.to_numeric(self.input_df[col_name], errors='coerce')

    def _get_regime_adjustment(self, adjustment_key: str) -> float:
        default_value = 0.0 if 'modifier' in adjustment_key else 1.0
        if not self.config.regime_settings['enabled']:
            return default_value

        regime_adjustments = self.config.regime_settings['adjustments'].get(self.current_regime, {})
        return regime_adjustments.get(adjustment_key, default_value)

    def _calculate_proximity_factor(self, strike_series: pd.Series) -> pd.Series:
        if not self.config.proximity_settings['enabled'] or self.underlying_price <= 0 or strike_series.empty:
            return pd.Series(1.0, index=strike_series.index)

        numeric_strikes = pd.to_numeric(strike_series, errors='coerce')
        strike_distance_pct = (numeric_strikes - self.underlying_price).abs() / self.underlying_price

        prox_strength = self._get_regime_adjustment('proximity_strength_factor')

        vol_col = self.config.cols_input.get('volatility')
        vol_influence_setting = self.config.proximity_settings.get('volatility_influence', 0)
        vol_influence_factor = pd.Series(1.0, index=strike_series.index)

        if vol_col and vol_col in self.input_df.columns and vol_influence_setting > 0:
            contract_vol = pd.to_numeric(self.input_df[vol_col], errors='coerce').fillna(0.20)
            vol_influence_factor = 1.0 + contract_vol * vol_influence_setting

        decay_factor = self.config.proximity_settings.get('exp_decay_factor', 2.0) * prox_strength

        proximity_calc_base = strike_distance_pct / vol_influence_factor.reindex(strike_distance_pct.index).fillna(1.0)
        proximity = np.exp(-decay_factor * proximity_calc_base)

        return proximity.fillna(0.01)

    def _get_adjusted_quantile_threshold(self) -> float:
        base_quantile = self.config.default_quantile
        modifier = self._get_regime_adjustment('quantile_modifier')
        return min(max(base_quantile + modifier, 0.01), 0.99)

    def _analyze_method_template(self, raw_score_series: pd.Series, methodology_name: str) -> pd.DataFrame:
        cfg_cols_rep = self.config.cols_report
        strike_col_input = self.config.cols_input['strike']
        empty_df_result = pd.DataFrame(columns=[cfg_cols_rep['strike'], cfg_cols_rep['raw_score'],
                                                cfg_cols_rep['adjusted_score'], cfg_cols_rep['methodology']])
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
        return ranked_df

    def _determine_support_resistance(self, ranked_df: pd.DataFrame) -> pd.DataFrame:
        cfg_cols_rep = self.config.cols_report
        cfg_cols_in = self.config.cols_input
        strike_col_df = cfg_cols_rep['strike']

        # Ensure result columns exist even if we return early
        cols_to_add = [cfg_cols_rep['level_type'], cfg_cols_rep['sr_rationale']]
        for col in cols_to_add:
            if col not in ranked_df.columns:
                 ranked_df[col] = "Uncertain" if col == cfg_cols_rep['level_type'] else "S/R Not Determined"


        if ranked_df.empty or self.strike_aggregated_metrics.empty:
            if not ranked_df.empty: # Add default values if ranked_df is not empty but metrics are
                ranked_df[cfg_cols_rep['level_type']] = "Uncertain"
                ranked_df[cfg_cols_rep['sr_rationale']] = "Missing aggregated metrics for S/R."
            return ranked_df

        # Determine the correct strike column name from strike_aggregated_metrics
        # It should be self.config.cols_input['strike']
        agg_strike_col = self.config.cols_input['strike']
        if agg_strike_col not in self.strike_aggregated_metrics.columns:
            print(f"Error (_determine_support_resistance): Strike column '{agg_strike_col}' not found in self.strike_aggregated_metrics.")
            return ranked_df # Cannot proceed without strike in aggregated metrics

        # Merge. Ensure 'strike' in ranked_df (which is cfg_cols_rep['strike']) is used for merging.
        df_for_sr = pd.merge(ranked_df, self.strike_aggregated_metrics,
                             left_on=strike_col_df, right_on=agg_strike_col,
                             how='left')

        # If merge created two strike columns due to different names (e.g. 'strike' vs 'STRIKE')
        # and original ranked_df strike col name was different from agg_strike_col
        if strike_col_df != agg_strike_col and agg_strike_col in df_for_sr.columns:
             df_for_sr.drop(columns=[agg_strike_col], inplace=True)


        plausibility_threshold_for_sr = self.config.sr_settings.get('plausibility_min_threshold', 0.3)
        delta_std_factor = self.config.sr_settings.get('delta_std_factor', 0.5)
        flow_std_factor = self.config.sr_settings.get('flow_std_factor', 0.5)

        dxoi_col_agg_name = cfg_cols_in['dxoi']
        volmbs15_col_agg_name = cfg_cols_in['volmbs_15m']

        delta_threshold = 0
        if dxoi_col_agg_name in self.strike_aggregated_metrics.columns and not self.strike_aggregated_metrics[dxoi_col_agg_name].empty:
            delta_threshold = self.strike_aggregated_metrics[dxoi_col_agg_name].std() * delta_std_factor

        flow_threshold = 0
        if volmbs15_col_agg_name in self.strike_aggregated_metrics.columns and not self.strike_aggregated_metrics[volmbs15_col_agg_name].empty:
            flow_threshold = self.strike_aggregated_metrics[volmbs15_col_agg_name].std() * flow_std_factor

        # Ensure thresholds are not NaN (can happen if std is NaN, e.g. only one unique value in series)
        delta_threshold = 0 if pd.isna(delta_threshold) else delta_threshold
        flow_threshold = 0 if pd.isna(flow_threshold) else flow_threshold

        level_types = []
        rationales = []

        for _, row in df_for_sr.iterrows():
            level_type = "Uncertain"
            rationale = "Default."

            plausibility = row.get(cfg_cols_rep['composite_plausibility_score'], 0)
            strike_price = row[strike_col_df]

            if plausibility >= plausibility_threshold_for_sr:
                net_delta = row.get(dxoi_col_agg_name, 0) # Get from merged df
                net_flow = row.get(volmbs15_col_agg_name, 0) # Get from merged df

                is_below_price = strike_price < self.underlying_price
                is_above_price = strike_price > self.underlying_price

                strong_pos_delta = net_delta > delta_threshold
                strong_neg_delta = net_delta < -delta_threshold
                strong_pos_flow = net_flow > flow_threshold
                strong_neg_flow = net_flow < -flow_threshold

                current_rationale_parts = [f"P={plausibility:.2f}"]

                if is_below_price:
                    current_rationale_parts.append("BelowPx")
                    if strong_pos_delta and strong_pos_flow: level_type = "Support"; current_rationale_parts.append("++Δ&Flow")
                    elif strong_pos_delta: level_type = "Support"; current_rationale_parts.append("+Δ")
                    elif strong_pos_flow: level_type = "Support"; current_rationale_parts.append("+Flow")
                    elif strong_neg_delta or strong_neg_flow: level_type = "Contested"; current_rationale_parts.append("--Δ/Flow")
                    else: level_type = "Potential Support"; current_rationale_parts.append("NeutralΔ/Flow")
                elif is_above_price:
                    current_rationale_parts.append("AbovePx")
                    if strong_neg_delta and strong_neg_flow: level_type = "Resistance"; current_rationale_parts.append("--Δ&Flow")
                    elif strong_neg_delta: level_type = "Resistance"; current_rationale_parts.append("-Δ")
                    elif strong_neg_flow: level_type = "Resistance"; current_rationale_parts.append("-Flow")
                    elif strong_pos_delta or strong_pos_flow: level_type = "Contested"; current_rationale_parts.append("++Δ/Flow")
                    else: level_type = "Potential Resistance"; current_rationale_parts.append("NeutralΔ/Flow")
                else: # At price
                    current_rationale_parts.append("AtPx")
                    if strong_pos_delta and strong_pos_flow: level_type = "Support"; current_rationale_parts.append("++Δ&Flow")
                    elif strong_neg_delta and strong_neg_flow: level_type = "Resistance"; current_rationale_parts.append("--Δ&Flow")
                    else: level_type = "Contested"; current_rationale_parts.append("MixedΔ/Flow")
                rationale = ", ".join(current_rationale_parts)
            else:
                rationale = f"Low Plaus. ({plausibility:.2f})"

            level_types.append(level_type)
            rationales.append(rationale)

        df_for_sr[cfg_cols_rep['level_type']] = level_types
        df_for_sr[cfg_cols_rep['sr_rationale']] = rationales

        # Construct list of columns that should be in the final output
        # Start with columns from original ranked_df
        final_output_cols = ranked_df.columns.tolist()
        # Add new S/R columns if they are not already there (they shouldn't be)
        if cfg_cols_rep['level_type'] not in final_output_cols:
            final_output_cols.append(cfg_cols_rep['level_type'])
        if cfg_cols_rep['sr_rationale'] not in final_output_cols:
            final_output_cols.append(cfg_cols_rep['sr_rationale'])

        # Ensure df_for_sr has all these columns before selecting
        for col in final_output_cols:
            if col not in df_for_sr.columns:
                 df_for_sr[col] = "Error" # Or np.nan, indicates logic error if this happens

        return df_for_sr[final_output_cols]


    def analyze(self) -> pd.DataFrame:
        cfg_cols_rep = self.config.cols_report
        strike_col_report = cfg_cols_rep['strike']

        all_methodologies_df = self.run_all_methodologies()
        if all_methodologies_df.empty:
            return pd.DataFrame(columns=list(cfg_cols_rep.values()))

        ranking_factors_df = self._calculate_ranking_factors(all_methodologies_df)
        if ranking_factors_df.empty:
             return pd.DataFrame(columns=list(cfg_cols_rep.values()))

        ranked_df_with_plausibility = self._calculate_composite_plausibility(ranking_factors_df)

        if strike_col_report in all_methodologies_df.columns and cfg_cols_rep['methodology'] in all_methodologies_df.columns:
            methods_per_strike = all_methodologies_df.groupby(strike_col_report)[cfg_cols_rep['methodology']].apply(
                lambda x: ', '.join(sorted(list(set(x))))
            ).reset_index(name=cfg_cols_rep['contributing_methods'])

            final_df = pd.merge(ranked_df_with_plausibility, methods_per_strike, on=strike_col_report, how='left')
            final_df[cfg_cols_rep['contributing_methods']] = final_df[cfg_cols_rep['contributing_methods']].fillna('')
        else:
            final_df = ranked_df_with_plausibility.copy()
            final_df[cfg_cols_rep['contributing_methods']] = ''

        # final_df = self._filter_by_strike_clustering(final_df) # Pass-through for now
        final_df = self._determine_support_resistance(final_df)

        for report_col_key, report_col_name in self.config.cols_report.items():
            if report_col_name not in final_df.columns:
                if 'score' in report_col_key or 'factor' in report_col_key: final_df[report_col_name] = 0.0
                elif 'count' in report_col_key: final_df[report_col_name] = 0
                else: final_df[report_col_name] = "N/A"

        # Define the desired order of columns for the final report
        desired_column_order = [
            cfg_cols_rep['strike'], cfg_cols_rep['composite_plausibility_score'],
            cfg_cols_rep['level_type'], cfg_cols_rep['sr_rationale'],
            cfg_cols_rep['methodology_count'], cfg_cols_rep['methodology_diversity_score'],
            cfg_cols_rep['gamma_concentration_factor'], cfg_cols_rep['flow_consistency_factor'],
            cfg_cols_rep['delta_gamma_alignment_factor'], cfg_cols_rep['volatility_sensitivity_factor'],
            cfg_cols_rep['time_decay_sensitivity_factor'], cfg_cols_rep['contributing_methods'],
            cfg_cols_rep['proximity_factor'] # Include proximity factor used in adjusted scores
        ]
        # Filter this list to include only columns that actually exist in final_df
        ordered_report_cols = [col for col in desired_column_order if col in final_df.columns]
        # Include any other columns that might have been generated but are not in desired_column_order
        other_cols = [col for col in final_df.columns if col not in ordered_report_cols]

        return final_df[ordered_report_cols + other_cols].sort_values(
            by=cfg_cols_rep['composite_plausibility_score'], ascending=False
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

        analyzer_config = DarkpoolAnalyticsConfig()
        # Ensure sr_settings is part of the config for the test
        analyzer_config.sr_settings = {'plausibility_min_threshold': 0.3, 'delta_std_factor': 0.5, 'flow_std_factor': 0.5}


        analyzer = EliteDarkpoolAnalyzer(
            options_df=sample_options_data_df.copy(),
            underlying_price=current_price_test,
            config=analyzer_config,
            market_regime_metric_value=market_metric_test_value
        )

        print(f"Analyzer initialized. Current Regime: {analyzer.current_regime}")

        print("\n--- Running full analyze() method (Elite - with S/R logic) ---")
        final_analysis_df = analyzer.analyze()

        if not final_analysis_df.empty:
            print("Final analysis df (Top 10 ranked by plausibility with S/R):")
            cols_to_show = [
                analyzer_config.cols_report['strike'],
                analyzer_config.cols_report['composite_plausibility_score'],
                analyzer_config.cols_report['level_type'],
                analyzer_config.cols_report['sr_rationale'],
                analyzer_config.cols_report['contributing_methods'],
                analyzer_config.cols_report['methodology_count']
            ]
            cols_to_show_existing = [col for col in cols_to_show if col in final_analysis_df.columns]
            print(final_analysis_df[cols_to_show_existing].head(10).to_string())

            sr_cols_expected = [analyzer_config.cols_report['level_type'], analyzer_config.cols_report['sr_rationale']]
            missing_sr_cols = [col for col in sr_cols_expected if col not in final_analysis_df.columns]
            if missing_sr_cols:
                print(f"\nERROR: Missing S/R columns in final_analysis_df: {missing_sr_cols}")
            else:
                print("\nS/R columns ('level_type', 'sr_rationale') are present in the final_analysis_df.")

            print("\nChecking value counts for 'level_type':")
            if analyzer_config.cols_report['level_type'] in final_analysis_df.columns:
                print(final_analysis_df[analyzer_config.cols_report['level_type']].value_counts(dropna=False))
            else:
                print("'level_type' column not found.")

        else:
            print("Full analysis resulted in an empty DataFrame.")
```
