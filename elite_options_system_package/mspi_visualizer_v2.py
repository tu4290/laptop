# Ensure these imports are at the top of your mspi_visualizer_v2.py file
import os
import json
import traceback
import logging
from datetime import datetime, date, time # Keep for type hints
import time as pytime # Alias to avoid conflict with datetime.time
from typing import Optional, Dict, Any, List, Tuple, Union, Deque, Callable # Keep for type hints
from collections import deque # Keep for type hints
import inspect # Keep for checking function args in other methods

# Third-Party Imports
import pandas as pd # Keep for type hints
import numpy as np # Keep for type hints
import plotly.graph_objects as go # Keep for type hints
import plotly.express as px # Keep for type hints (though not directly used in default config)
from plotly.subplots import make_subplots # Keep for type hints
from dateutil import parser as date_parser # Keep for type hints
import plotly.colors # Keep for _parse_color_string

# --- Default Visualizer Configuration (Full Version from your script) ---
DEFAULT_VISUALIZER_CONFIG: Dict[str, Any] = {
    "log_level": "INFO",
    "output_dir": "mspi_visualizations_v2_default",
    "save_charts_as_html": False,
    "save_charts_as_png": False,
    "default_chart_height": 600,
    "plotly_template": "plotly_dark",
    "plot_order_history": ["T-5", "T-4", "T-3", "T-2", "T-1", "T-B", "T-A", "Now"],
    "rolling_intervals": ["5m", "15m", "30m", "60m"],
    "min_normalization_denominator": 1e-9,
    "min_value_for_ratio": 1e-6,
    "colorscales": {
        "mspi_heatmap": [[0.0, "rgb(139,0,0)"], [0.5, "rgb(240,240,240)"], [1.0, "rgb(0,0,139)"]],
        "net_value_heatmap": [[0.0, "rgb(160,0,0)"], [0.5, "rgb(240,240,240)"], [1.0, "rgb(0,160,0)"]],
        "net_volume_pressure_heatmap": [[0.0, "rgb(214,47,39)"], [0.5, "rgb(245,245,245)"], [1.0, "rgb(69,117,180)"]], # Added for Net Volume Pressure
        "net_delta_heuristic_heatmap": [ [ 0.0, "rgb(200,100,0)" ], [ 0.5, "rgb(240,240,240)" ], [ 1.0, "rgb(0,100,200)" ] ],
        "net_gamma_flow_heatmap": [ [ 0.0, "rgb(180,0,180)" ], [ 0.5, "rgb(240,240,240)" ], [ 1.0, "rgb(0,180,180)" ] ],
        "net_vega_flow_heatmap": [ [ 0.0, "rgb(255,120,0)" ], [ 0.5, "rgb(240,240,240)" ], [ 1.0, "rgb(0,120,255)" ] ],
        "net_theta_exposure_heatmap": [ [ 0.0, "rgb(255,0,0)" ], [ 0.5, "rgb(240,240,240)" ], [ 1.0, "rgb(0,200,0)" ] ]
    },
    "key_level_markers": {
        "Support": {"symbol": "triangle-up", "color": "rgb(34,139,34)", "name": "Support"},
        "Resistance": {"symbol": "triangle-down", "color": "rgb(220,20,60)", "name": "Resistance"},
        "High Conviction": {"symbol": "diamond", "color": "rgb(255,215,0)", "name": "High Conviction"},
        "Structure Change": {"symbol": "cross", "color": "rgb(0,191,255)", "name": "Structure Change"}
    },
    "signal_styles": {
        "bullish": {"color": "lime", "symbol": "triangle-up"}, "bearish": {"color": "red", "symbol": "triangle-down"},
        "sdag_bullish": {"color": "gold", "symbol": "diamond-wide"}, "sdag_bearish": {"color": "purple", "symbol": "diamond-wide-dot"},
        "expansion": {"color": "cyan", "symbol": "diamond-open"}, "contraction": {"color": "magenta", "symbol": "square-open"},
        "pin_risk": {"color": "yellow", "symbol": "star"}, "charm_cascade": {"color": "orange", "symbol": "hourglass"},
        "structure_change": {"color": "white", "symbol": "cross-thin"}, "flow_divergence": {"color": "lightblue", "symbol": "x-thin"},
        "default": {"color": "grey", "symbol": "circle"}
    },
    "column_names": { 
        "net_volume_pressure": "net_volume_pressure", 
        "net_value_pressure": "net_value_pressure",   
        "mspi": "mspi",
        "strike": "strike",
        "option_kind": "opt_kind",
        "heuristic_net_delta_pressure": "heuristic_net_delta_pressure",
        "net_gamma_flow_at_strike": "net_gamma_flow", 
        "net_vega_flow_at_strike": "net_vega_flow",   
        "net_theta_exposure_at_strike": "net_theta_exposure", 
        "net_delta_flow_total": "net_delta_flow_total",
        "true_net_volume_flow": "true_net_volume_flow",
        "true_net_value_flow": "true_net_value_flow"
    },
    "hover_settings": { 
        "show_overview_metrics_default": True, "show_oi_structure_default": True, "show_details_section_default": True,
        "overview_metrics_config": [
          { "key": "mspi", "label": "MSPI", "precision": 3, "is_currency": False }, { "key": "sai", "label": "SAI", "precision": 3, "is_currency": False },
          { "key": "ssi", "label": "SSI", "precision": 3, "is_currency": False }, { "key": "cfi", "label": "ARFI", "precision": 3, "is_currency": False },
          # Elite Impact Metrics
          { "key": "elite_impact_score", "label": "Elite Score", "precision": 2, "is_currency": False },
          { "key": "prediction_confidence", "label": "Confidence", "precision": 2, "is_currency": False },
          { "key": "signal_strength", "label": "Signal Str.", "precision": 2, "is_currency": False },
          # Original MSPI Components
          { "key": "dag_custom", "label": "DAG(C)", "precision": 0, "is_currency": False }, { "key": "tdpi", "label": "TDPI", "precision": 0, "is_currency": False },
          { "key": "vri", "label": "VRI", "precision": 0, "is_currency": False },
          { "key": "sdag_multiplicative", "label": "SDAG(M)", "precision":0, "is_currency": False }, { "key": "sdag_directional", "label": "SDAG(D)", "precision":0, "is_currency": False },
          { "key": "sdag_weighted", "label": "SDAG(W)", "precision":0, "is_currency": False },  { "key": "sdag_volatility_focused", "label": "SDAG(VF)", "precision":0, "is_currency": False },
          { "key": "net_volume_pressure", "label": "Net Vol P (H)", "precision": 0, "is_currency": False }, { "key": "net_value_pressure", "label": "Net Val P (H)", "precision": 0, "is_currency": True },
          { "key": "heuristic_net_delta_pressure", "label": "Net Delta P (H)", "precision": 0, "is_currency": False },
          { "key": "net_gamma_flow_at_strike", "label": "Net Γ Flow", "precision": 0, "is_currency": False }, 
          { "key": "net_vega_flow_at_strike", "label": "Net ν Flow", "precision": 0, "is_currency": False }, 
          { "key": "net_theta_exposure_at_strike", "label": "Net θ Exp", "precision": 0, "is_currency": False }, 
          { "key": "net_delta_flow_total", "label": "True Net Δ Flow", "precision": 0, "is_currency": False },
          { "key": "true_net_volume_flow", "label": "True Net Vol Flow", "precision": 0, "is_currency": False },
          { "key": "true_net_value_flow", "label": "True Net Val Flow", "precision": 0, "is_currency": True }
        ],
        "oi_structure_metrics_config": [
          { "base_key": "dxoi", "label": "DxOI" }, { "base_key": "gxoi", "label": "GxOI" },
          { "base_key": "txoi", "label": "TxOI" }, { "base_key": "vxoi", "label": "VxOI" }
        ],
        "details_section_keys": [ "level_category", "conviction", "strategy", "rationale", "type", "agree_count", "exit_reason", "status_update" ], 
        "chart_specific_hover": { 
            "default": {"sections": ["base_info", "mspi_value"]},
            "mspi_heatmap": {"sections": ["base_info", "mspi_value", "core_indices"], "core_indices_keys": ["sai", "ssi"]},
            "net_value_heatmap": {"sections": ["base_info", "net_pressures"]},
            "net_volume_pressure_heatmap": {"sections": ["base_info", "net_pressures"]}, # Added for new heatmap
            "net_greek_flow_heatmap": {"sections": ["base_info", "selected_greek_flow", "overview_metrics"]},
            "mspi_components": {"sections": ["base_info", "overview_metrics", "oi_structure"]},
            "sdag": {"sections": ["base_info", "sdag_specific_value", "core_indices"], "core_indices_keys": ["mspi", "sai"]},
            "sdag_net": {"sections": ["base_info", "sdag_specific_value"]},
            "tdpi": {"sections": ["base_info", "tdpi_specific_values"]},
            "vri": {"sections": ["base_info", "vri_specific_values"]},
            "key_levels": {"sections": ["base_info", "core_metrics_context", "details_section"]},
            "trading_signals": {"sections": ["base_info", "core_metrics_context", "details_section"]},
            "elite_score_display": {
                "sections": ["base_info", "elite_score_details"],
                "elite_score_details_keys": [
                    "elite_impact_score",
                    "prediction_confidence",
                    "signal_strength"
                ]
            }
        }
    },
    "chart_specific_params": { 
        "raw_greek_charts_price_range_pct": 7.5,
        "combined_flow_chart_price_range_pct": 12.0,
        "mspi_components_bar_colors": {
            "mspi": { "pos": "darkblue", "neg": "darkred" }, "dag_custom_norm": { "pos": "rgb(255,215,0)", "neg": "rgb(128,0,128)" },
            "tdpi_norm": { "pos": "green", "neg": "red", "is_border": True }, "vri_norm": { "pos": "cyan", "neg": "magenta" },
            "sdag_multiplicative_norm": { "pos": "#FFA07A", "neg": "#6A5ACD" }, "sdag_directional_norm": { "pos": "#FFD700", "neg": "#8A2BE2" },
            "sdag_weighted_norm": { "pos": "#98FB98", "neg": "#FF6347" }, "sdag_volatility_focused_norm": { "pos": "#AFEEEE", "neg": "#DA70D6" }
        },
        "show_net_sdag_trace": True, "net_sdag_trace_default_visibility": "legendonly",
        "net_sdag_marker_style": { "symbol": "diamond", "color": "rgba(255, 255, 255, 0.7)", "size": 8, "line": { "color": "white", "width": 1 } },
        "component_comparison_height": 600, "volval_comparison_height": 600, "key_levels_height": 600, "trading_signals_height": 600, "recommendations_table_height": 650,
        "recommendations_table_column_display_map": {
            "id": "ID", "Category": "Category", "direction_label": "Bias/Type", "strike": "Strike", "strategy": "Strategy / Note",
            "conviction_stars": "Conv★", "raw_conviction_score": "Score", "status": "Status",
            "entry_ideal": "Entry", "target_1": "T1", "target_2": "T2", "stop_loss": "SL",
            "rationale": "Rationale", "target_rationale": "Tgt. Logic",
            "mspi": "MSPI", "sai": "SAI", "ssi": "SSI", "arfi": "ARFI",
            "issued_ts": "Issued", "last_adjusted_ts": "Adjusted", "exit_reason":"Exit Info", "type": "Signal Src", "status_update": "Last Update"
        },
        "combined_rolling_flow_chart_barmode": "overlay",
        "rolling_flow_customization": {
          "5m": {"volume_positive_color": "#2ca02c", "volume_negative_color": "#d62728", "volume_opacity": 0.8, "value_positive_fill_color": "rgba(44,160,44,0.15)", "value_negative_fill_color": "rgba(214,39,40,0.15)", "value_positive_line_color": "rgba(44,160,44,0.6)", "value_negative_line_color": "rgba(214,39,40,0.6)"},
          "15m": {"volume_positive_color": "#98df8a", "volume_negative_color": "#ff9896", "volume_opacity": 0.75, "value_positive_fill_color": "rgba(152,223,138,0.15)", "value_negative_fill_color": "rgba(255,152,150,0.15)", "value_positive_line_color": "rgba(152,223,138,0.55)", "value_negative_line_color": "rgba(255,152,150,0.55)"},
          "30m": {"volume_positive_color": "#1f77b4", "volume_negative_color": "#ff7f0e", "volume_opacity": 0.7, "value_positive_fill_color": "rgba(31,119,180,0.1)", "value_negative_fill_color": "rgba(255,127,14,0.1)", "value_positive_line_color": "rgba(31,119,180,0.5)", "value_negative_line_color": "rgba(255,127,14,0.5)"},
          "60m": {"volume_positive_color": "#aec7e8", "volume_negative_color": "#ffbb78", "volume_opacity": 0.65, "value_positive_fill_color": "rgba(174,199,232,0.1)", "value_negative_fill_color": "rgba(255,187,120,0.1)", "value_positive_line_color": "rgba(174,199,232,0.45)", "value_negative_line_color": "rgba(255,187,120,0.45)"},
           "defaults": {"volume_positive_color": "#cccccc", "volume_negative_color": "#777777", "volume_opacity": 0.7, "value_positive_fill_color": "rgba(204,204,204,0.1)", "value_negative_fill_color": "rgba(119,119,119,0.1)", "value_positive_line_color": "rgba(204,204,204,0.5)", "value_negative_line_color": "rgba(119,119,119,0.5)"}
        }
    },
    "legend_settings": { "orientation": "v", "y_anchor": "top", "y_pos": 1, "x_anchor": "left", "x_pos": 1.02, "trace_order": "reversed" }
}


# --- Logging Setup ---
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] (%(name)s:%(lineno)d) %(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
logger = logging.getLogger(__name__)

class MSPIVisualizerV2:
    def __init__(self, config_path: Optional[str] = None, config_data: Optional[Dict[str, Any]] = None):
        self.instance_logger = logger.getChild(self.__class__.__name__)
        self.instance_logger.info("MSPIVisualizerV2 Initializing (V2.4.1 - Greek Flow Heatmap Prep)...")

        self.full_app_config: Dict[str, Any] = {}
        if isinstance(config_data, dict):
            self.full_app_config = json.loads(json.dumps(config_data)) 
            self.instance_logger.debug("Visualizer initialized with provided 'config_data' (full app config).")
        
        self.config = self._load_visualizer_specific_config(config_path, self.full_app_config)
        
        self._setup_logging()

        self.output_dir: Optional[str] = self.config.get("output_dir") # From viz-specific config
        if self.output_dir and isinstance(self.output_dir, str):
            try:
                os.makedirs(self.output_dir, exist_ok=True)
                self.instance_logger.info(f"Output directory for saved charts ensured: {os.path.abspath(self.output_dir)}")
            except OSError as e_dir:
                self.instance_logger.warning(f"Could not create visualizer output directory '{self.output_dir}': {e_dir}. Saving charts might fail.")
                self.output_dir = None
        else:
            self.instance_logger.info("No output directory specified or configured for visualizer. Charts will not be saved to disk.")
            self.output_dir = None

        column_names_config = self.config.get("column_names", {}) # From viz-specific config
        if not isinstance(column_names_config, dict):
            column_names_config = DEFAULT_VISUALIZER_CONFIG.get("column_names", {})

        self.col_strike: str = column_names_config.get("strike", "strike")
        self.col_opt_kind: str = column_names_config.get("option_kind", "opt_kind")
        self.col_mspi: str = column_names_config.get("mspi", "mspi")
        self.col_net_vol_p: str = column_names_config.get("net_volume_pressure", "net_volume_pressure")
        self.col_net_val_p: str = column_names_config.get("net_value_pressure", "net_value_pressure")
        self.col_heuristic_net_delta_pressure: str = column_names_config.get("heuristic_net_delta_pressure", "heuristic_net_delta_pressure")
        self.col_net_gamma_flow: str = column_names_config.get("net_gamma_flow_at_strike", "net_gamma_flow")
        self.col_net_vega_flow: str = column_names_config.get("net_vega_flow_at_strike", "net_vega_flow")
        self.col_net_theta_exposure: str = column_names_config.get("net_theta_exposure_at_strike", "net_theta_exposure")
        
        self.instance_logger.info(
            f"Visualizer Column Names Configured: Strike='{self.col_strike}', OptKind='{self.col_opt_kind}', "
            f"MSPI='{self.col_mspi}', HeuristicNetVolP='{self.col_net_vol_p}', HeuristicNetValP='{self.col_net_val_p}', "
            f"HeuristicNetDeltaP='{self.col_heuristic_net_delta_pressure}', NetGammaF='{self.col_net_gamma_flow}', "
            f"NetVegaF='{self.col_net_vega_flow}', NetThetaExp='{self.col_net_theta_exposure}'"
        )
        self.instance_logger.info("MSPIVisualizerV2 Initialized successfully.")

    def _deep_merge_dicts(self, base: Dict, updates: Dict) -> Dict:
        merged = base.copy()
        for key, value in updates.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = self._deep_merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    def _load_visualizer_specific_config(self, config_path: Optional[str], full_app_config_from_init: Dict[str, Any]) -> Dict[str, Any]:
        load_config_logger = self.instance_logger.getChild("LoadVisualizerSpecificConfig")
        
        visualizer_config_section_from_default = DEFAULT_VISUALIZER_CONFIG.copy() 
        
        temp_full_app_config_to_use = {}
        source_description_full_app = "Defaults only (no path or init data)"

        if full_app_config_from_init:
            temp_full_app_config_to_use = json.loads(json.dumps(full_app_config_from_init)) 
            source_description_full_app = "Provided 'config_data' to __init__"
        elif config_path:
            abs_path_to_check = config_path
            if not os.path.isabs(config_path):
                try: script_dir_viz = os.path.dirname(os.path.abspath(__file__))
                except NameError: script_dir_viz = os.getcwd()
                path_from_script_dir = os.path.join(script_dir_viz, config_path)
                path_from_cwd = os.path.join(os.getcwd(), config_path)
                if os.path.exists(path_from_cwd): abs_path_to_check = path_from_cwd
                elif os.path.exists(path_from_script_dir): abs_path_to_check = path_from_script_dir
                else: abs_path_to_check = path_from_script_dir
            
            try:
                if os.path.exists(abs_path_to_check):
                    with open(abs_path_to_check, 'r', encoding='utf-8') as f_main_cfg:
                        temp_full_app_config_to_use = json.load(f_main_cfg)
                    source_description_full_app = f"File ('{config_path}')"
                else:
                    load_config_logger.warning(f"Full app config file '{abs_path_to_check}' not found. Visualizer will use its defaults.")
            except Exception as e_main_load_viz:
                load_config_logger.error(f"Error loading full app config from '{abs_path_to_check}': {e_main_load_viz}. Visualizer will use its defaults.", exc_info=True)
        
        if temp_full_app_config_to_use:
             self.full_app_config = temp_full_app_config_to_use 
        
        user_visualizer_section = self.full_app_config.get("visualization_settings", {}).get("mspi_visualizer", {})
        
        final_visualizer_config = visualizer_config_section_from_default 
        if isinstance(user_visualizer_section, dict) and user_visualizer_section:
            final_visualizer_config = self._deep_merge_dicts(final_visualizer_config, user_visualizer_section)
            load_config_logger.info(f"Visualizer-specific settings merged from '{source_description_full_app}'.")
        else:
            load_config_logger.info(f"No specific user visualizer settings found in '{source_description_full_app}'. Using visualizer defaults.")
            
        return final_visualizer_config 

    def _get_config_value(self, path: List[str], default_override: Any = None) -> Any:
        config_get_logger = self.instance_logger.getChild("GetConfigValue")
        
        current_level_full = self.full_app_config
        try:
            for key_item in path:
                if isinstance(current_level_full, dict):
                    current_level_full = current_level_full[key_item]
                else: 
                    raise KeyError(f"Path broken at '{key_item}' during traversal in full_app_config")
            config_get_logger.debug(f"Found path '{'.'.join(path)}' in self.full_app_config.")
            return current_level_full
        except (KeyError, TypeError):
            config_get_logger.debug(f"Path '{'.'.join(path)}' not found or TypeError in self.full_app_config. Trying self.config (visualizer-specific).")
            current_level_viz = self.config
            try:
                relative_path_for_viz_config = path
                if path and path[0] == "visualization_settings":
                    if len(path) > 1 and path[1] == "mspi_visualizer":
                        relative_path_for_viz_config = path[2:] 
                    else: 
                        config_get_logger.debug(f"Path '{'.'.join(path)}' is too general for self.config. Returning default.")
                        return default_override
                
                if not relative_path_for_viz_config: 
                    if path == ["visualization_settings", "mspi_visualizer"]: 
                        return current_level_viz 
                    else: 
                         config_get_logger.debug(f"Path '{'.'.join(path)}' became empty or too general for self.config. Returning default.")
                         return default_override

                for key_item_viz in relative_path_for_viz_config:
                    if isinstance(current_level_viz, dict):
                        current_level_viz = current_level_viz[key_item_viz] 
                    else: 
                         raise KeyError(f"Path broken at '{key_item_viz}' during traversal in self.config")
                config_get_logger.debug(f"Found relative path '{'.'.join(relative_path_for_viz_config)}' in self.config.")
                return current_level_viz
            except (KeyError, TypeError):
                config_get_logger.debug(f"Path '{'.'.join(path)}' (relative: '{'.'.join(relative_path_for_viz_config if 'relative_path_for_viz_config' in locals() else path)}') also not found or TypeError in self.config. Returning default: {default_override}")
                return default_override

    def _setup_logging(self):
        log_level_str_from_viz_config = self.config.get("log_level", "INFO").upper()
        try:
            log_level_to_set = getattr(logging, log_level_str_from_viz_config)
            self.instance_logger.setLevel(log_level_to_set)
        except AttributeError:
            log_level_to_set = logging.INFO
            self.instance_logger.setLevel(log_level_to_set)
            self.instance_logger.warning(f"Invalid log level '{log_level_str_from_viz_config}' in visualizer config. Logger for MSPIVisualizerV2 instance defaulting to INFO.")
        self.instance_logger.info(f"MSPIVisualizerV2 instance logger level set to {logging.getLevelName(self.instance_logger.getEffectiveLevel())} from its config.")

    def _create_empty_figure(self, title: str = "Waiting for data...", height: Optional[int] = None, reason: str = "N/A") -> go.Figure:
        fig_height_actual = height if height is not None else self.config.get("default_chart_height", 600)
        plotly_template_to_use = self.config.get("plotly_template", "plotly_dark") 
        self.instance_logger.debug(f"Creating empty figure: '{title}', Reason: '{reason}', Template: {plotly_template_to_use}")
        fig = go.Figure()
        full_title_with_reason = f"<i>{title}<br><small style='color:grey'>({reason})</small></i>"
        fig.update_layout(
            title={'text': full_title_with_reason, 'y':0.5, 'x':0.5, 'xanchor': 'center', 'yanchor': 'middle', 'font': {'color': 'grey', 'size': 16}},
            template=plotly_template_to_use,
            height=fig_height_actual,
            xaxis={'visible': False, 'showgrid': False, 'zeroline': False},
            yaxis={'visible': False, 'showgrid': False, 'zeroline': False},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        return fig

    def _format_hover_value(self, value: Any, precision: int = 0, is_currency: bool = False) -> str:
        if value is None or value == '' or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
            return "N/A"
        try:
            val_f = float(value)
            prefix = "$" if is_currency else ""
            suffix = ""
            divisor = 1.0
            final_precision = precision
            if abs(val_f) >= 1_000_000_000: suffix = "B"; divisor = 1_000_000_000.0; final_precision = 2
            elif abs(val_f) >= 1_000_000: suffix = "M"; divisor = 1_000_000.0; final_precision = 2
            elif abs(val_f) >= 1_000: suffix = "k"; divisor = 1_000.0; final_precision = 1
            elif abs(val_f) < 10 and final_precision == 0 and not is_currency : final_precision = 2
            elif is_currency and final_precision == 0 : final_precision = 2
            return f"{prefix}{{num:,.{final_precision}f}}{{suffix}}".format(num=val_f / divisor, suffix=suffix)
        except (ValueError, TypeError, OverflowError): return str(value)

    def _create_hover_text(self, row: Union[pd.Series, Dict[str, Any]], chart_type: str = "default", extra_context: Optional[Dict[str, Any]] = None) -> str:
        hover_logger = self.instance_logger.getChild("CreateHoverText")
        if not isinstance(row, (pd.Series, dict)):
            hover_logger.warning(f"Cannot create hover text for non-Series/dict input: {type(row)}. Returning empty hover.")
            return "<extra></extra>"

        hover_main_config = self.config.get("hover_settings", {})
        overview_metrics_config_list = hover_main_config.get("overview_metrics_config", [])
        oi_structure_config_list = hover_main_config.get("oi_structure_metrics_config", [])
        details_keys_config_list = hover_main_config.get("details_section_keys", [])
        chart_specific_hover_config = hover_main_config.get("chart_specific_hover", {})
        current_chart_hover_rules = chart_specific_hover_config.get(chart_type, chart_specific_hover_config.get("default", {}))
        sections_to_display = current_chart_hover_rules.get("sections", ["base_info"])

        def safe_get_from_row(key: str, default_val: Any = None) -> Any:
            if isinstance(row, pd.Series): return row.get(key, default_val)
            elif isinstance(row, dict): return row.get(key, default_val)
            return default_val

        strike_value_raw = safe_get_from_row(self.col_strike)
        strike_display_text = self._format_hover_value(strike_value_raw, 2)
        
        hover_text_parts: List[str] = [f"<b>Strike: {strike_display_text}</b>"]
        if pd.notna(strike_value_raw) and strike_display_text != str(strike_value_raw):
            hover_text_parts[0] += f" <i style='font-size:0.8em; color:grey'>(Raw: {strike_value_raw})</i>"

        option_type_from_context = extra_context.get('Option Type') if extra_context else None
        if option_type_from_context:
            hover_text_parts.append(f"Type: {str(option_type_from_context).capitalize()}")

        for section_key_name in sections_to_display:
            current_section_text_parts: List[str] = []
            if section_key_name == "base_info": continue
            elif section_key_name == "mspi_value":
                mspi_value_data = safe_get_from_row(self.col_mspi)
                if mspi_value_data is not None: current_section_text_parts.append(f"<b>{self.col_mspi.upper()}: {self._format_hover_value(mspi_value_data, 3)}</b>")
            elif section_key_name == "core_indices":
                core_indices_keys_to_show = current_chart_hover_rules.get("core_indices_keys", ['sai', 'ssi', 'cfi'])
                for core_key in core_indices_keys_to_show:
                    label_text = core_key.upper() if core_key != 'cfi' else 'ARFI'
                    value_data = safe_get_from_row(core_key)
                    if value_data is not None: current_section_text_parts.append(f"{label_text}: {self._format_hover_value(value_data, 3)}")
            elif section_key_name == "net_pressures":
                if safe_get_from_row(self.col_net_vol_p) is not None: current_section_text_parts.append(f"Net Vol P (H): {self._format_hover_value(safe_get_from_row(self.col_net_vol_p), 0)}")
                if safe_get_from_row(self.col_net_val_p) is not None: current_section_text_parts.append(f"<b>Net Val P (H): {self._format_hover_value(safe_get_from_row(self.col_net_val_p), 0, True)}</b>")
            elif section_key_name == "overview_metrics" and hover_main_config.get("show_overview_metrics_default", True):
                temp_overview_list = []
                for metric_item_config in overview_metrics_config_list:
                    key_name = metric_item_config.get("key"); label_text = metric_item_config.get("label", key_name.upper() if key_name else "N/A"); precision_val = metric_item_config.get("precision", 0); is_curr_val = metric_item_config.get("is_currency", False)
                    value_data = safe_get_from_row(key_name) if key_name else None
                    if value_data is not None and pd.notna(value_data):
                        display_label = f"<b>{label_text}</b>" if key_name == self.col_mspi else label_text
                        temp_overview_list.append(f"{display_label}: {self._format_hover_value(value_data, precision_val, is_curr_val)}")
                if temp_overview_list: current_section_text_parts = ["--- Overview Metrics ---"] + temp_overview_list
            elif section_key_name == "oi_structure" and hover_main_config.get("show_oi_structure_default", True):
                temp_oi_list = []
                for oi_metric_item_config in oi_structure_config_list:
                     base_key_name = oi_metric_item_config.get("base_key"); label_text = oi_metric_item_config.get("label", base_key_name.upper() if base_key_name else "N/A")
                     if not base_key_name: continue
                     call_val_data = safe_get_from_row(f"call_{base_key_name}"); put_val_data = safe_get_from_row(f"put_{base_key_name}")
                     if (call_val_data is not None and pd.notna(call_val_data)) or \
                        (put_val_data is not None and pd.notna(put_val_data)):
                         temp_oi_list.append(f"{label_text}: {self._format_hover_value(call_val_data,0)} | {self._format_hover_value(put_val_data,0)}")
                if temp_oi_list: current_section_text_parts = ["--- OI Structure (Call | Put) ---"] + temp_oi_list
            elif section_key_name == "sdag_specific_value" and extra_context:
                sdag_method_display_name = extra_context.get("SDAG Method", "SDAG Value")
                sdag_column_name_actual = extra_context.get("sdag_col_name")
                sdag_value_data = safe_get_from_row(sdag_column_name_actual) if sdag_column_name_actual else None
                if sdag_value_data is not None: current_section_text_parts.append(f"<b>{sdag_method_display_name}: {self._format_hover_value(sdag_value_data, 0)}</b>")
            elif section_key_name == "selected_greek_flow" and extra_context:
                metric_label = extra_context.get("metric_label", "Selected Flow")
                metric_col_name = extra_context.get("metric_col_name")
                metric_value = safe_get_from_row(metric_col_name) if metric_col_name else None
                is_currency_flag = extra_context.get("is_currency", False)
                precision_val = extra_context.get("precision", 0)
                if metric_value is not None:
                    current_section_text_parts.append(f"<b>{metric_label}: {self._format_hover_value(metric_value, precision_val, is_currency_flag)}</b>")
            elif section_key_name == "tdpi_specific_values":
                if safe_get_from_row('tdpi') is not None: current_section_text_parts.append(f"<b>TDPI: {self._format_hover_value(safe_get_from_row('tdpi'), 0)}</b>")
                if safe_get_from_row('ctr') is not None: current_section_text_parts.append(f"CTR: {self._format_hover_value(safe_get_from_row('ctr'), 3)}")
                if safe_get_from_row('tdfi') is not None: current_section_text_parts.append(f"TDFI: {self._format_hover_value(safe_get_from_row('tdfi'), 3)}")
            elif section_key_name == "vri_specific_values":
                if safe_get_from_row('vri') is not None: current_section_text_parts.append(f"<b>VRI: {self._format_hover_value(safe_get_from_row('vri'), 0)}</b>")
                if safe_get_from_row('vvr') is not None: current_section_text_parts.append(f"VVR: {self._format_hover_value(safe_get_from_row('vvr'), 3)}")
                if safe_get_from_row('vfi') is not None: current_section_text_parts.append(f"VFI: {self._format_hover_value(safe_get_from_row('vfi'), 3)}")
            elif section_key_name == "core_metrics_context":
                temp_core_list = []
                for core_key, display_label in [(self.col_mspi, 'MSPI'), ('sai', 'SAI'), ('ssi', 'SSI'), ('cfi', 'ARFI')]:
                     value_data = safe_get_from_row(core_key)
                     if value_data is not None: temp_core_list.append(f"{display_label}: {self._format_hover_value(value_data, 3)}")
                if temp_core_list: current_section_text_parts = ["--- Context ---"] + temp_core_list

            elif section_key_name == "elite_score_details":
                temp_elite_details_list = []
                # Get the keys from the chart-specific config
                elite_keys_to_show = current_chart_hover_rules.get("elite_score_details_keys", [])
                for key_name in elite_keys_to_show:
                    # Find the metric's config from the global overview_metrics_config for label, precision etc.
                    metric_config = next((m_cfg for m_cfg in overview_metrics_config_list if m_cfg.get("key") == key_name), None)

                    default_label = key_name.replace('_',' ').title()
                    label_text = metric_config.get("label", default_label) if metric_config else default_label
                    precision_val = metric_config.get("precision", 2) if metric_config else 2
                    is_curr_val = metric_config.get("is_currency", False) if metric_config else False

                    value_data = safe_get_from_row(key_name) # safe_get_from_row is already defined in the method
                    if value_data is not None and pd.notna(value_data):
                        # Make the main score bold
                        display_label = f"<b>{label_text}</b>" if key_name == "elite_impact_score" else label_text
                        temp_elite_details_list.append(f"{display_label}: {self._format_hover_value(value_data, precision_val, is_curr_val)}")
                if temp_elite_details_list:
                    # Add a section title if there are details
                    current_section_text_parts = ["--- Elite Impact ---"] + temp_elite_details_list

            elif section_key_name == "details_section" and hover_main_config.get("show_details_section_default", True):
                temp_details_list = []
                for detail_key in details_keys_config_list:
                      value_data = safe_get_from_row(detail_key)
                      if value_data is not None and pd.notna(value_data) and str(value_data).strip() != '':
                           label_text = detail_key.replace('_', ' ').title()
                           temp_details_list.append(f"{label_text}: {str(value_data).title() if detail_key=='conviction' else str(value_data)}")
                if temp_details_list: current_section_text_parts = ["--- Details ---"] + temp_details_list

            if current_section_text_parts:
                hover_text_parts.extend(current_section_text_parts)

        final_hover_text = "<br>".join(hover_text_parts) + "<extra></extra>"
        hover_logger.debug(f"Generated Hover (Chart Type: {chart_type}) for Strike {strike_value_raw}: {final_hover_text[:350]}...")
        return final_hover_text

    def _add_timestamp_annotation(self, fig: go.Figure, fetch_timestamp: Optional[Union[str, datetime]]) -> go.Figure:
        if fetch_timestamp:
            time_str_display: str = "N/A"; dt_object_parsed: Optional[datetime] = None
            try:
                if isinstance(fetch_timestamp, str): dt_object_parsed = date_parser.isoparse(fetch_timestamp)
                elif isinstance(fetch_timestamp, datetime): dt_object_parsed = fetch_timestamp
                if dt_object_parsed: time_str_display = dt_object_parsed.strftime("%Y-%m-%d %H:%M:%S %Z" if dt_object_parsed.tzinfo else "%Y-%m-%d %H:%M:%S")
                else: time_str_display = str(fetch_timestamp)
                fig.add_annotation(text=f"Data As Of: {time_str_display}", align='right',showarrow=False,xref='paper',yref='paper',x=0.99,y=0.01,xanchor='right',yanchor='bottom',font=dict(color="grey",size=10))
            except Exception as e_ts_anno: self.instance_logger.warning(f"Failed to parse or add timestamp annotation ('{fetch_timestamp}'): {e_ts_anno}")
        return fig

    def _add_price_line(self, fig: go.Figure, current_price: Optional[Union[float, str, int]], orientation: str = 'vertical', **kwargs) -> go.Figure:
        price_line_logger = self.instance_logger.getChild("AddPriceLine")
        
        user_annotation_settings = kwargs.get('annotation', {})
        anno_params_final = dict(showarrow=False, font=dict(color="lightgrey", size=10))
        anno_params_final.update(user_annotation_settings) 

        can_draw_line = False
        numeric_price_for_line_drawing: Optional[float] = None

        if isinstance(current_price, (int, float)) and pd.notna(current_price):
            numeric_price_for_line_drawing = float(current_price)
            if (orientation == 'vertical' and numeric_price_for_line_drawing > 0) or \
               (orientation == 'horizontal'): 
                 can_draw_line = True
        elif isinstance(current_price, str) and orientation == 'horizontal': 
            can_draw_line = True
            price_line_logger.debug(f"Price line value '{current_price}' is a string (category) for horizontal line.")
        
        if can_draw_line:
            line_params = dict(line_width=1, line_dash="dash", line_color="lightgrey")
            line_params.update(kwargs.get('line', {}))
            
            row_val, col_val = kwargs.get('row'), kwargs.get('col')

            try:
                if orientation == 'vertical':
                    if not isinstance(numeric_price_for_line_drawing, (int, float)) or not (numeric_price_for_line_drawing > 0) : 
                        price_line_logger.error(f"For vertical price line, current_price value must be numeric and positive. Got '{current_price}'. Skipping line.")
                        return fig
                    fig.add_vline(x=numeric_price_for_line_drawing, row=row_val, col=col_val, **line_params)
                    
                    if 'x' not in anno_params_final: anno_params_final['x'] = numeric_price_for_line_drawing
                    if 'text' not in anno_params_final: anno_params_final['text'] = f"Current: {numeric_price_for_line_drawing:.2f}"
                    anno_params_final.setdefault('y', 1.01); anno_params_final.setdefault('yref', "paper")
                    anno_params_final.setdefault('xanchor', "left"); anno_params_final.setdefault('yanchor', "bottom")
                    fig.add_annotation(row=row_val, col=col_val, **anno_params_final)
                
                elif orientation == 'horizontal':
                    fig.add_hline(y=current_price, row=row_val, col=col_val, **line_params)
                    
                    if 'y' not in anno_params_final: anno_params_final['y'] = current_price
                    if 'text' not in anno_params_final:
                        if isinstance(numeric_price_for_line_drawing, (int, float)): 
                             anno_params_final['text'] = f"Current: {numeric_price_for_line_drawing:.2f}"
                        elif isinstance(current_price, str): 
                             anno_params_final['text'] = f"Level: {current_price}"

                    anno_params_final.setdefault('x', 1.0); anno_params_final.setdefault('xref', "paper")
                    anno_params_final.setdefault('yanchor', "bottom"); anno_params_final.setdefault('xanchor', "right")
                    fig.add_annotation(row=row_val, col=col_val, **anno_params_final)
                else:
                    price_line_logger.warning(f"Invalid price line orientation specified: {orientation}")
            except Exception as e_price_line_draw:
                price_line_logger.warning(f"Failed adding price line (Value: {current_price}, Orientation: {orientation}): {e_price_line_draw}", exc_info=True)
        else:
            price_line_logger.debug(f"Skipping price line: current_price value ('{current_price}') is not valid for drawing with orientation '{orientation}'.")
        return fig

    def _build_volval_hovertemplate(self, hover_df: pd.DataFrame, plotted_labels: List[str]) -> str:
        col_map_volval: Dict[str, int] = {name: i for i, name in enumerate(hover_df.columns)}
        strike_idx_volval = col_map_volval.get(self.col_strike)
        if strike_idx_volval is None: self.instance_logger.error(f"VolVal Hovertemplate: Strike column ('{self.col_strike}') missing in hover_df."); return "Error: Strike data missing<extra></extra>"
        template_str: str = f"<b>Strike: %{{customdata[{strike_idx_volval:.0f}]:.2f}}</b><br>-----------"
        history_order_cfg_volval = self.config.get("plot_order_history", DEFAULT_VISUALIZER_CONFIG.get("plot_order_history", []))
        for label_item in history_order_cfg_volval:
            if label_item in plotted_labels:
                vol_col_name_hist = f'vol_{label_item}'; val_col_name_hist = f'val_{label_item}'
                vol_idx_hist = col_map_volval.get(vol_col_name_hist); val_idx_hist = col_map_volval.get(val_col_name_hist)
                if vol_idx_hist is not None or val_idx_hist is not None:
                    template_str += f"<br>--- {label_item} ---"
                    if vol_idx_hist is not None: template_str += f"<br>Net Vol P: %{{customdata[{vol_idx_hist:.0f}]:,.0f}}"
                    if val_idx_hist is not None: template_str += f"<br>Net Val P: $%{{customdata[{val_idx_hist:.0f}]:,.0f}}"
        template_str += "<extra></extra>"
        return template_str

    def _save_figure(self, fig: go.Figure, chart_name: str, symbol: str):
        if not self.output_dir:
            self.instance_logger.debug(f"Chart '{chart_name}' for '{symbol}' not saved as output_dir is not configured.")
            return
        safe_symbol_name = "".join(c if c.isalnum() else "_" for c in str(symbol).strip())
        safe_chart_filename_part = "".join(c if c.isalnum() else "_" for c in str(chart_name).lower().replace(' ', '_'))
        timestamp_filename_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_output_filename = f"{safe_symbol_name}_{safe_chart_filename_part}_{timestamp_filename_str}"
        if self.config.get("save_charts_as_html", False):
            html_file_path = os.path.join(self.output_dir, f"{base_output_filename}.html")
            try: fig.write_html(html_file_path, full_html=False, include_plotlyjs='cdn'); self.instance_logger.info(f"Chart saved as HTML: {html_file_path}")
            except Exception as e_html_save: self.instance_logger.error(f"Failed to save chart as HTML to '{html_file_path}': {e_html_save}", exc_info=True)
        if self.config.get("save_charts_as_png", False):
            png_file_path = os.path.join(self.output_dir, f"{base_output_filename}.png")
            try: fig.write_image(png_file_path, scale=2) ; self.instance_logger.info(f"Chart saved as PNG: {png_file_path}")
            except ValueError as ve_png:
                 if "kaleido" in str(ve_png).lower() or "orca" in str(ve_png).lower(): self.instance_logger.error(f"PNG save failed for '{png_file_path}': Plotly image export engine (Kaleido/Orca) is missing. Please install 'kaleido'. Error: {ve_png}")
                 else: self.instance_logger.error(f"PNG save failed for '{png_file_path}' with ValueError: {ve_png}", exc_info=True)
            except Exception as e_png_save: self.instance_logger.error(f"Failed to save chart as PNG to '{png_file_path}': {e_png_save}", exc_info=True)

    def _parse_color_string(self, color_str: str, default_opacity: float = 1.0) -> str:
        NAMED_COLORS_MAP = {
            "darkblue": "rgb(0,0,139)", "darkred": "rgb(139,0,0)", "green": "rgb(0,128,0)",
            "red": "rgb(255,0,0)", "cyan": "rgb(0,255,255)", "magenta": "rgb(255,0,255)",
            "yellow": "rgb(255,255,0)", "black": "rgb(0,0,0)", "white": "rgb(255,255,255)",
            "grey": "rgb(128,128,128)", "gray": "rgb(128,128,128)", "lime": "rgb(0,255,0)",
            "gold": "rgb(255,215,0)", "purple": "rgb(128,0,128)", "orange": "rgb(255,165,0)",
            "lightblue": "rgb(173,216,230)"
        }
        if not isinstance(color_str, str):
            self.instance_logger.warning(f"Invalid color_str type '{type(color_str)}' in _parse_color_string. Using default grey.")
            return f"rgba(128, 128, 128, {default_opacity})"
        color_str_lower = color_str.lower()
        if color_str_lower in NAMED_COLORS_MAP: color_str = NAMED_COLORS_MAP[color_str_lower]
        try:
            parsed_rgba_floats = plotly.colors.color_parser(color_str, plotly.colors.unlabel_rgb)
            r_int, g_int, b_int = int(parsed_rgba_floats[0]*255), int(parsed_rgba_floats[1]*255), int(parsed_rgba_floats[2]*255)
            final_alpha = parsed_rgba_floats[3] if 'rgba' in color_str.lower() or 'hsla' in color_str.lower() else default_opacity
            return f"rgba({r_int}, {g_int}, {b_int}, {final_alpha})"
        except (ValueError, TypeError, Exception) as e_plotly_parse:
            self.instance_logger.debug(f"Plotly's color_parser failed for '{color_str}': {e_plotly_parse}. Trying simpler parsing.")
            if color_str.startswith("rgb(") and color_str.endswith(")"):
                try:
                    parts = color_str[4:-1].split(',')
                    r, g, b = int(parts[0].strip()), int(parts[1].strip()), int(parts[2].strip())
                    return f"rgba({r},{g},{b},{default_opacity})"
                except Exception: pass
            elif color_str.startswith("#") and (len(color_str) == 7 or len(color_str) == 4):
                try:
                    rgb_tuple = plotly.colors.hex_to_rgb(color_str) 
                    return f"rgba({rgb_tuple[0]},{rgb_tuple[1]},{rgb_tuple[2]},{default_opacity})"
                except Exception: pass
            self.instance_logger.warning(f"Could not parse color string '{color_str}'. Using default 'rgba(128, 128, 128, {default_opacity})'.")
            return f"rgba(128, 128, 128, {default_opacity})"
    
    # --- Chart Creation Methods ---
    def create_mspi_heatmap(self, processed_data: pd.DataFrame, symbol: str = "N/A", fetch_timestamp: Optional[str] = None, **kwargs) -> go.Figure:
        chart_name = "MSPI Heatmap"; chart_title = f"<b>{symbol.upper()}</b> - {chart_name}"; chart_logger = self.instance_logger.getChild(chart_name); chart_logger.info(f"Creating {chart_title}...")
        fig_height = self._get_config_value(["visualization_settings", "mspi_visualizer", "default_chart_height"], 600)
        try:
            if not isinstance(processed_data, pd.DataFrame) or processed_data.empty: return self._create_empty_figure(f"{symbol}-{chart_name}: No Data", height=fig_height, reason="Input DataFrame empty/invalid")
            required_cols = [self.col_strike, self.col_opt_kind, self.col_mspi];
            df_cleaned, _ = self._ensure_columns(processed_data.copy(), required_cols, chart_name) 
            if not all(col in df_cleaned.columns for col in required_cols): missing = [col for col in required_cols if col not in df_cleaned.columns]; return self._create_empty_figure(f"{symbol}-{chart_name}: Missing Columns ({', '.join(missing)})", height=fig_height, reason=f"Missing: {', '.join(missing)}")
            
            df = df_cleaned 
            df[self.col_strike] = pd.to_numeric(df[self.col_strike], errors='coerce'); df[self.col_mspi] = pd.to_numeric(df[self.col_mspi], errors='coerce'); df[self.col_opt_kind] = df[self.col_opt_kind].astype(str).str.lower().fillna('?'); df = df.dropna(subset=[self.col_strike, self.col_mspi, self.col_opt_kind]); df = df[df[self.col_opt_kind].isin(['call', 'put'])];
            if df.empty: return self._create_empty_figure(f"{symbol}-{chart_name}: No Valid Call/Put Data", height=fig_height, reason="No valid call/put data after cleaning")
            
            pivot_df = df.pivot_table(values=self.col_mspi, index=self.col_strike, columns=self.col_opt_kind, aggfunc='sum', fill_value=0).reindex(columns=['put','call'], fill_value=0).sort_index(ascending=False)
            if pivot_df.empty: return self._create_empty_figure(f"{symbol}-{chart_name}: No Pivot Data", height=fig_height, reason="Pivot table empty")
            
            hover_context_keys_cfg = self._get_config_value(["visualization_settings", "mspi_visualizer", "hover_settings","chart_specific_hover","mspi_heatmap","core_indices_keys"], ['sai', 'ssi', 'cfi'])
            agg_hover_logic = {hc: 'first' for hc in hover_context_keys_cfg if hc in df.columns}; df_agg_for_hover = df.groupby([self.col_strike, self.col_opt_kind], as_index=False).agg(agg_hover_logic) if agg_hover_logic else pd.DataFrame(columns=[self.col_strike, self.col_opt_kind])
            for hc in hover_context_keys_cfg:
                if hc not in df_agg_for_hover.columns: df_agg_for_hover[hc] = np.nan
            hover_matrix = []
            for strike_val_iter in pivot_df.index:
                row_hovers = []
                for opt_type_iter in pivot_df.columns:
                    hover_dict_content: Dict[str, Any] = {self.col_strike: strike_val_iter, self.col_mspi: pivot_df.loc[strike_val_iter, opt_type_iter]}
                    context_data = df_agg_for_hover[(df_agg_for_hover[self.col_strike] == strike_val_iter) & (df_agg_for_hover[self.col_opt_kind] == opt_type_iter)]
                    if not context_data.empty: hover_dict_content.update(context_data.iloc[0].to_dict())
                    row_hovers.append(self._create_hover_text(hover_dict_content, chart_type="mspi_heatmap", extra_context={'Option Type': opt_type_iter.capitalize()}))
                hover_matrix.append(row_hovers)
            cs = self._get_config_value(["visualization_settings", "mspi_visualizer", "colorscales", "mspi_heatmap"], "RdBu")
            fig = go.Figure(data=[go.Heatmap(z=pivot_df.values,x=pivot_df.columns.str.capitalize(),y=pivot_df.index.astype(str),colorscale=cs,zmid=0,colorbar=dict(title=self.col_mspi.upper()),hovertext=hover_matrix,hoverinfo='text')])
            fig.update_layout(title=chart_title, xaxis_title='Option Type', yaxis_title='Strike', yaxis=dict(type='category', autorange='reversed', tickfont=dict(size=10)), template=self._get_config_value(["visualization_settings", "mspi_visualizer", "plotly_template"],"plotly_dark"),height=fig_height)
            fig=self._add_timestamp_annotation(fig,fetch_timestamp); self._save_figure(fig,chart_name,symbol)
        except Exception as e: chart_logger.error(f"Error during MSPI Heatmap creation: {e}", exc_info=True); return self._create_empty_figure(f"{symbol}-{chart_name}: Error", height=fig_height, reason=str(e))
        chart_logger.info(f"Chart {chart_name} created successfully for {symbol}."); return fig
        
    def create_net_value_heatmap(self, processed_data: pd.DataFrame, symbol: str = "N/A", fetch_timestamp: Optional[str] = None, **kwargs) -> go.Figure:
        chart_name = "Net Value Pressure Heatmap"; chart_title = f"<b>{symbol.upper()}</b> - {chart_name} (Heuristic)"
        chart_logger = self.instance_logger.getChild(chart_name)
        chart_logger.info(f"Creating {chart_title}...")
        fig_height = self._get_config_value(["visualization_settings", "mspi_visualizer", "default_chart_height"], 600)
        try:
            if not isinstance(processed_data, pd.DataFrame) or processed_data.empty:
                return self._create_empty_figure(f"{symbol}-{chart_name}: No Data", height=fig_height, reason="Input DataFrame empty/invalid")
            
            required_cols = [self.col_strike, self.col_net_val_p]
            if self.col_net_vol_p not in required_cols and self.col_net_vol_p in processed_data.columns:
                 required_cols.append(self.col_net_vol_p)
            
            df_cleaned, _ = self._ensure_columns(processed_data.copy(), required_cols, chart_name)
            if not all(c in df_cleaned.columns for c in [self.col_strike, self.col_net_val_p]):
                missing=[c for c in [self.col_strike, self.col_net_val_p] if c not in df_cleaned.columns]
                chart_logger.error(f"Net Value Heatmap: Missing required columns: {missing}. Available: {list(df_cleaned.columns)}")
                return self._create_empty_figure(f"{symbol}-{chart_name}: Missing Core Data ({', '.join(missing)})", height=fig_height, reason=f"Missing: {', '.join(missing)}")

            df = df_cleaned
            agg_logic_nvp = {col: 'first' for col in required_cols if col != self.col_strike and col in df.columns}
            if not agg_logic_nvp :
                return self._create_empty_figure(f"{symbol}-{chart_name}: No metrics to aggregate", height=fig_height, reason="No valid agg logic")

            overview_metrics_cfg_nvp = self._get_config_value(["visualization_settings", "mspi_visualizer", "hover_settings", "overview_metrics_config"], [])
            hover_context_keys_nvp = [m_cfg.get("key") for m_cfg in overview_metrics_cfg_nvp if m_cfg.get("key") and m_cfg.get("key") not in required_cols]
            for hc_nvp in list(set(hover_context_keys_nvp)):
                 if hc_nvp in df.columns and hc_nvp not in agg_logic_nvp: agg_logic_nvp[hc_nvp] = 'first'

            agg_data = df.groupby(self.col_strike, as_index=False).agg(agg_logic_nvp)
            agg_data[self.col_strike] = pd.to_numeric(agg_data[self.col_strike], errors='coerce')
            agg_data[self.col_net_val_p] = pd.to_numeric(agg_data[self.col_net_val_p], errors='coerce')
            agg_data = agg_data.dropna(subset=[self.col_strike, self.col_net_val_p]).sort_values(self.col_strike, ascending=False)

            if agg_data.empty:
                return self._create_empty_figure(f"{symbol}-{chart_name}: No Aggregated Data", height=fig_height, reason="Aggregated data for Net Value Heatmap is empty")

            hovers_nvp = [self._create_hover_text(r_dict.to_dict(), chart_type="net_value_heatmap") for _, r_dict in agg_data.iterrows()]
            z_values_nvp = agg_data[[self.col_net_val_p]].values
            hover_matrix_nvp = [[h] for h in hovers_nvp]

            colorscale_nvp = self._get_config_value(["visualization_settings", "mspi_visualizer", "colorscales", "net_value_heatmap"], "RdYlGn")
            fig = go.Figure(data=[go.Heatmap(
                z=z_values_nvp, x=['Net Value Pressure (H)'], y=agg_data[self.col_strike].astype(str),
                colorscale=colorscale_nvp, zmid=0, colorbar=dict(title='Net Val P (Heuristic)'),
                hovertext=hover_matrix_nvp, hoverinfo='text'
            )])
            fig.update_layout(
                title=chart_title, yaxis_title='Strike',
                yaxis=dict(type='category', autorange='reversed', tickfont=dict(size=10)),
                xaxis=dict(showticklabels=False, showline=False, zeroline=False),
                template=self._get_config_value(["visualization_settings", "mspi_visualizer", "plotly_template"], "plotly_dark"), height=fig_height
            )
            fig = self._add_timestamp_annotation(fig, fetch_timestamp)
            self._save_figure(fig, chart_name, symbol)
        except Exception as e_nvp_hm:
            chart_logger.error(f"Error during {chart_name} creation for {symbol}: {e_nvp_hm}", exc_info=True)
            return self._create_empty_figure(f"{symbol}-{chart_name}: Plotting Error", height=fig_height, reason=str(e_nvp_hm))
        chart_logger.info(f"Chart '{chart_name}' created successfully for {symbol}.")
        return fig

    def create_net_volume_pressure_heatmap(self, processed_data: pd.DataFrame, symbol: str = "N/A", fetch_timestamp: Optional[str] = None, **kwargs) -> go.Figure:
        """Creates a heatmap for Net Volume Pressure (Heuristic)."""
        chart_name = "Net Volume Pressure Heatmap"; chart_title = f"<b>{symbol.upper()}</b> - {chart_name} (Heuristic)"
        chart_logger = self.instance_logger.getChild(chart_name)
        chart_logger.info(f"Creating {chart_title}...")
        fig_height = self._get_config_value(["visualization_settings", "mspi_visualizer", "default_chart_height"], 600)
        
        try:
            if not isinstance(processed_data, pd.DataFrame) or processed_data.empty:
                return self._create_empty_figure(f"{symbol}-{chart_name}: No Data", height=fig_height, reason="Input DataFrame empty/invalid")
            
            required_cols = [self.col_strike, self.col_net_vol_p] 
            if self.col_net_val_p not in required_cols and self.col_net_val_p in processed_data.columns: 
                 required_cols.append(self.col_net_val_p) 
            
            df_cleaned, _ = self._ensure_columns(processed_data.copy(), required_cols, chart_name)
            if not all(c in df_cleaned.columns for c in [self.col_strike, self.col_net_vol_p]):
                missing=[c for c in [self.col_strike, self.col_net_vol_p] if c not in df_cleaned.columns]
                chart_logger.error(f"{chart_name}: Missing required columns: {missing}. Available: {list(df_cleaned.columns)}")
                return self._create_empty_figure(f"{symbol}-{chart_name}: Missing Core Data ({', '.join(missing)})", height=fig_height, reason=f"Missing: {', '.join(missing)}")

            df = df_cleaned
            agg_logic_nvp = {col: 'first' for col in required_cols if col != self.col_strike and col in df.columns}
            if not agg_logic_nvp :
                return self._create_empty_figure(f"{symbol}-{chart_name}: No metrics to aggregate", height=fig_height, reason="No valid agg logic")

            overview_metrics_cfg_nvp = self._get_config_value(["visualization_settings", "mspi_visualizer", "hover_settings", "overview_metrics_config"], [])
            hover_context_keys_nvp = [m_cfg.get("key") for m_cfg in overview_metrics_cfg_nvp if m_cfg.get("key") and m_cfg.get("key") not in required_cols]
            for hc_nvp in list(set(hover_context_keys_nvp)):
                 if hc_nvp in df.columns and hc_nvp not in agg_logic_nvp: agg_logic_nvp[hc_nvp] = 'first'

            agg_data = df.groupby(self.col_strike, as_index=False).agg(agg_logic_nvp)
            agg_data[self.col_strike] = pd.to_numeric(agg_data[self.col_strike], errors='coerce')
            agg_data[self.col_net_vol_p] = pd.to_numeric(agg_data[self.col_net_vol_p], errors='coerce')
            agg_data = agg_data.dropna(subset=[self.col_strike, self.col_net_vol_p]).sort_values(self.col_strike, ascending=False)

            if agg_data.empty:
                return self._create_empty_figure(f"{symbol}-{chart_name}: No Aggregated Data", height=fig_height, reason=f"Aggregated data for {chart_name} is empty")

            hovers_nvp = [self._create_hover_text(r_dict.to_dict(), chart_type="net_volume_pressure_heatmap") for _, r_dict in agg_data.iterrows()]
            z_values_nvp = agg_data[[self.col_net_vol_p]].values
            hover_matrix_nvp = [[h] for h in hovers_nvp]

            colorscale_key = "net_volume_pressure_heatmap" # Key for colorscales config
            colorscale_nvp = self._get_config_value(["visualization_settings", "mspi_visualizer", "colorscales", colorscale_key], "coolwarm") 
            fig = go.Figure(data=[go.Heatmap(
                z=z_values_nvp, x=['Net Volume Pressure (H)'], y=agg_data[self.col_strike].astype(str),
                colorscale=colorscale_nvp, zmid=0, colorbar=dict(title='Net Vol P (Heuristic)'),
                hovertext=hover_matrix_nvp, hoverinfo='text'
            )])
            fig.update_layout(
                title=chart_title, yaxis_title='Strike',
                yaxis=dict(type='category', autorange='reversed', tickfont=dict(size=10)),
                xaxis=dict(showticklabels=False, showline=False, zeroline=False),
                template=self._get_config_value(["visualization_settings", "mspi_visualizer", "plotly_template"], "plotly_dark"), height=fig_height
            )
            fig = self._add_timestamp_annotation(fig, fetch_timestamp)
            self._save_figure(fig, chart_name, symbol)
        except Exception as e_nvp_hm:
            chart_logger.error(f"Error during {chart_name} creation for {symbol}: {e_nvp_hm}", exc_info=True)
            return self._create_empty_figure(f"{symbol}-{chart_name}: Plotting Error", height=fig_height, reason=str(e_nvp_hm))
        chart_logger.info(f"Chart '{chart_name}' created successfully for {symbol}.")
        return fig
        
    def create_component_comparison(
        self,
        processed_data: pd.DataFrame,
        symbol: str = "N/A",
        current_price: Optional[float] = None,
        fetch_timestamp: Optional[str] = None,
        trace_visibility: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> go.Figure:
        chart_name = "MSPI Components"
        chart_title = f"<b>{symbol.upper()}</b> - {chart_name}"
        chart_logger = self.instance_logger.getChild(chart_name)
        chart_logger.info(f"Creating {chart_title} with MSPI as area overlay and grouped component bars...")

        viz_base_path = ["visualization_settings", "mspi_visualizer"]
        
        fig_height = self._get_config_value(viz_base_path + ["chart_specific_params", "component_comparison_height"], 
                                            self._get_config_value(viz_base_path + ["default_chart_height"], 600))
        plotly_template_to_use = self._get_config_value(viz_base_path + ["plotly_template"], "plotly_dark")
        colors_cfg = self._get_config_value(viz_base_path + ["chart_specific_params", "mspi_components_bar_colors"], {})
        legend_cfg = self._get_config_value(viz_base_path + ["legend_settings"], {})
        dag_method_configs = self._get_config_value(["strategy_settings", "dag_methodologies"], {}) 
        overview_metrics_cfg = self._get_config_value(viz_base_path + ["hover_settings", "overview_metrics_config"], []) 
        oi_structure_cfg = self._get_config_value(viz_base_path + ["hover_settings", "oi_structure_metrics_config"], [])

        try:
            if not isinstance(processed_data, pd.DataFrame) or processed_data.empty:
                return self._create_empty_figure(f"{symbol}-{chart_name}: No Data", height=fig_height, reason="Input DataFrame empty/invalid")

            if trace_visibility is None: trace_visibility = {}

            potential_components_for_plot = [self.col_mspi, 'dag_custom_norm', 'tdpi_norm', 'vri_norm']
            if isinstance(dag_method_configs, dict): 
                for method in dag_method_configs.get("enabled", []):
                    method_cfg = dag_method_configs.get(method, {})
                    if isinstance(method_cfg, dict) and method_cfg.get("weight_in_mspi", 0) > 0:
                        potential_components_for_plot.append(f"sdag_{method}_norm")
            
            present_components_in_df = [col for col in potential_components_for_plot if col in processed_data.columns]
            required_cols_for_chart = [self.col_strike] + [col for col in present_components_in_df if col in processed_data.columns] 
            df_cleaned, _ = self._ensure_columns(processed_data.copy(), required_cols_for_chart, chart_name)

            if df_cleaned.empty or self.col_strike not in df_cleaned.columns:
                 return self._create_empty_figure(f"{symbol}-{chart_name}: No valid data after cleaning", height=fig_height, reason="Strike column missing or data empty post-cleaning")
            
            agg_logic: Dict[str, Any] = {col: 'first' for col in present_components_in_df if col != self.col_strike and col != self.col_mspi and col.endswith("_norm")}
            if self.col_mspi in present_components_in_df: agg_logic[self.col_mspi] = 'sum'
            
            hover_cols_needed = [m_cfg.get("key") for m_cfg in overview_metrics_cfg if m_cfg.get("key") and m_cfg.get("key") not in required_cols_for_chart]
            for m_cfg in oi_structure_cfg:
                base_k = m_cfg.get("base_key")
                if base_k: hover_cols_needed.extend([f"call_{base_k}", f"put_{base_k}"])
            for hc in list(set(hover_cols_needed)):
                if hc in df_cleaned.columns and hc not in agg_logic: agg_logic[hc] = 'first'
            
            final_agg_logic = {k: v for k, v in agg_logic.items() if k in df_cleaned.columns}
            if not final_agg_logic:
                 return self._create_empty_figure(f"{symbol}-{chart_name}: No components to aggregate", height=fig_height, reason="No valid components for aggregation logic")

            agg_data = df_cleaned.groupby(self.col_strike, as_index=False).agg(final_agg_logic)
            agg_data[self.col_strike] = pd.to_numeric(agg_data[self.col_strike], errors='coerce')
            agg_data = agg_data.dropna(subset=[self.col_strike]).sort_values(self.col_strike)

            if agg_data.empty:
                return self._create_empty_figure(f"{symbol}-{chart_name}: No Aggregated Data", height=fig_height, reason="Aggregated data for components is empty")

            fig = go.Figure()
            mspi_hovers = [self._create_hover_text(r.to_dict(), chart_type="mspi_components") for _, r in agg_data.iterrows()]
            mspi_col_name_actual = self.col_mspi

            if mspi_col_name_actual in agg_data.columns:
                mspi_series = pd.to_numeric(agg_data[mspi_col_name_actual], errors='coerce').fillna(0.0)
                mspi_positive = mspi_series.where(mspi_series >= 0, 0)
                mspi_negative = mspi_series.where(mspi_series < 0, 0)
                mspi_color_map_cfg = colors_cfg.get(mspi_col_name_actual, {"pos": "darkblue", "neg": "darkred"})
                mspi_trace_visibility_key = mspi_col_name_actual.upper() 
                mspi_visible_from_state = trace_visibility.get(mspi_trace_visibility_key, True)

                fig.add_trace(go.Scatter(
                    x=agg_data[self.col_strike], y=mspi_positive, name=f'{mspi_col_name_actual.upper()} (+)',
                    mode='lines', line=dict(width=0.5, color=self._parse_color_string(str(mspi_color_map_cfg.get('pos')), 0.7)), 
                    fillcolor=self._parse_color_string(str(mspi_color_map_cfg.get('pos')), 0.3), fill='tozeroy', 
                    hovertext=mspi_hovers, hoverinfo='text', visible=mspi_visible_from_state, legendgroup=mspi_col_name_actual.upper()
                ))
                fig.add_trace(go.Scatter(
                    x=agg_data[self.col_strike], y=mspi_negative, name=f'{mspi_col_name_actual.upper()} (-)',
                    mode='lines', line=dict(width=0.5, color=self._parse_color_string(str(mspi_color_map_cfg.get('neg')), 0.7)), 
                    fillcolor=self._parse_color_string(str(mspi_color_map_cfg.get('neg')), 0.3), fill='tozeroy', 
                    hovertext=mspi_hovers, hoverinfo='text', visible=mspi_visible_from_state, legendgroup=mspi_col_name_actual.upper()
                ))
            else:
                chart_logger.warning(f"MSPI column '{mspi_col_name_actual}' not found. Cannot plot MSPI area.")

            overlay_group1_components = []
            if 'vri_norm' in agg_data.columns: overlay_group1_components.append('vri_norm')
            if 'tdpi_norm' in agg_data.columns: overlay_group1_components.append('tdpi_norm')

            gex_dex_structural_group_components = []
            if 'dag_custom_norm' in agg_data.columns: gex_dex_structural_group_components.append('dag_custom_norm')
            
            if isinstance(dag_method_configs, dict):
                preferred_sdag_order = ['multiplicative', 'directional', 'weighted', 'volatility_focused']
                for sdag_method_name_ordered in preferred_sdag_order:
                    if sdag_method_name_ordered in dag_method_configs.get("enabled", []):
                        method_cfg = dag_method_configs.get(sdag_method_name_ordered, {})
                        if isinstance(method_cfg, dict) and method_cfg.get("weight_in_mspi", 0) > 0:
                            sdag_norm_col = f"sdag_{sdag_method_name_ordered}_norm"
                            if sdag_norm_col in agg_data.columns:
                                gex_dex_structural_group_components.append(sdag_norm_col)
                for method in dag_method_configs.get("enabled", []): 
                    if method not in preferred_sdag_order:
                        method_cfg = dag_method_configs.get(method, {})
                        if isinstance(method_cfg, dict) and method_cfg.get("weight_in_mspi", 0) > 0:
                            sdag_norm_col = f"sdag_{method}_norm"
                            if sdag_norm_col in agg_data.columns and sdag_norm_col not in gex_dex_structural_group_components:
                                gex_dex_structural_group_components.append(sdag_norm_col)
            
            def plot_bar_group(components_list: List[str], group_name_for_offset: Optional[str] = None):
                for y_col_bar in components_list:
                    trace_name_bar_parts = y_col_bar.split('_'); trace_name_bar_display: str
                    if y_col_bar.startswith("sdag_") and y_col_bar.endswith("_norm"):
                        method_short_name = trace_name_bar_parts[1][0].upper() if len(trace_name_bar_parts) > 1 else "X"
                        trace_name_bar_display = f"SDAG({method_short_name})(N)"
                    elif y_col_bar.endswith("_norm"):
                        base_name_parts = trace_name_bar_parts[:-1]
                        if len(base_name_parts) > 1 and base_name_parts[0] == "dag": trace_name_bar_display = f"{base_name_parts[0].upper()}({base_name_parts[1][0].upper()})(N)"
                        else: trace_name_bar_display = f"{base_name_parts[0].upper()}(N)"
                    else: trace_name_bar_display = y_col_bar.upper()
                    
                    visibility_for_bar = trace_visibility.get(trace_name_bar_display, 'legendonly')
                    default_bar_color_map = {'pos': '#cccccc', 'neg': '#777777', 'is_border': False}
                    bar_color_map_cfg = colors_cfg.get(y_col_bar, default_bar_color_map) 
                    
                    bar_values = pd.to_numeric(agg_data[y_col_bar], errors='coerce').fillna(0.0)
                    pos_color_str = str(bar_color_map_cfg.get('pos', default_bar_color_map['pos']))
                    neg_color_str = str(bar_color_map_cfg.get('neg', default_bar_color_map['neg']))
                    current_bar_colors = [self._parse_color_string(pos_color_str, 0.7) if v >= 0 else self._parse_color_string(neg_color_str, 0.7) for v in bar_values] 
                    
                    bar_plot_args: Dict[str, Any] = {'x': agg_data[self.col_strike], 'y': bar_values, 'name': trace_name_bar_display, 'visible': visibility_for_bar, 'hoverinfo': 'skip'}
                    if group_name_for_offset: 
                        bar_plot_args['offsetgroup'] = group_name_for_offset
                    if bar_color_map_cfg.get('is_border', False):
                        bar_plot_args.update({'marker_color': 'rgba(0,0,0,0)', 'marker_line_color': current_bar_colors, 'marker_line_width': 1.5, 'opacity': 1.0})
                    else: bar_plot_args.update({'marker_color': current_bar_colors})
                    
                    fig.add_trace(go.Bar(**bar_plot_args))
                    chart_logger.debug(f"Added BAR trace for component '{y_col_bar}' as '{trace_name_bar_display}'. OffsetGroup: {group_name_for_offset}. Colors used (first): {current_bar_colors[0] if current_bar_colors else 'N/A'}")

            chart_logger.debug(f"Plotting Overlay Group 1 (VRI, TDPI): {overlay_group1_components}")
            plot_bar_group(overlay_group1_components, group_name_for_offset=None) 

            chart_logger.debug(f"Plotting GEX/DEX Structural Group (DAG, SDAGs) - also overlaying: {gex_dex_structural_group_components}")
            plot_bar_group(gex_dex_structural_group_components, group_name_for_offset=None) 

            fig.update_layout(
                title=chart_title, xaxis_title='Strike', yaxis_title='MSPI / Norm Component Value',
                barmode='overlay', 
                template=plotly_template_to_use, height=fig_height,
                legend=dict(orientation=legend_cfg.get("orientation", "v"), yanchor=legend_cfg.get("y_anchor", "top"), y=legend_cfg.get("y_pos", 1), xanchor=legend_cfg.get("x_anchor", "left"), x=legend_cfg.get("x_pos", 1.02), traceorder=legend_cfg.get("trace_order", "reversed")),
                hovermode="x unified",
                yaxis=dict(range=[-1.1, 1.1], tickmode='linear', dtick=0.1, showgrid=True, gridcolor='rgba(255,255,255,0.1)', minor=dict(tickmode='linear', dtick=0.05, showgrid=True, gridcolor='rgba(255,255,255,0.02)', griddash='dot'))
            )
            fig.update_xaxes(tickformat=".2f")
            fig = self._add_price_line(fig, current_price, orientation='vertical')
            fig = self._add_timestamp_annotation(fig, fetch_timestamp)
            self._save_figure(fig, chart_name, symbol)
        except Exception as e_comp_chart:
            chart_logger.error(f"MSPI Components chart creation failed for {symbol}: {e_comp_chart}", exc_info=True)
            return self._create_empty_figure(f"{symbol} - {chart_name}: Plotting Error", height=fig_height, reason=str(e_comp_chart))
        chart_logger.info(f"Chart {chart_name} for {symbol} with MSPI area overlay created successfully.")
        return fig

    # ... (Rest of your MSPIVisualizerV2 class methods, ensure they also use self._get_config_value correctly) ...
    def _ensure_columns(self, df: pd.DataFrame, required_cols: List[str], calculation_name: str) -> Tuple[pd.DataFrame, bool]:
        ensure_logger = self.instance_logger.getChild("EnsureColumnsVisualizer") 
        ensure_logger.debug(f"Visualizer: Ensuring columns for '{calculation_name}'. Required: {required_cols}")
        df_copy = df.copy()
        all_present_and_valid_initially = True
        actions_taken_log: List[str] = []
        string_like_id_cols = ['opt_kind', 'symbol', 'underlying_symbol', 'expiration_date', 'fetch_timestamp', 'level_category', 'level_type_original', 'strategy', 'rationale', 'type', 'exit_reason', 'status_update', 'direction_label', 'Category', 'status'] 
        datetime_cols_special_handling = ['date', 'issued_ts', 'last_adjusted_ts'] 

        for col_name in required_cols:
            if col_name not in df_copy.columns:
                all_present_and_valid_initially = False
                actions_taken_log.append(f"Added missing column '{col_name}'")
                default_val_to_add: Any
                if col_name in string_like_id_cols: default_val_to_add = 'N/A_DEFAULT'
                elif col_name in datetime_cols_special_handling: default_val_to_add = pd.NaT
                else: default_val_to_add = 0.0
                df_copy[col_name] = default_val_to_add
                ensure_logger.warning(f"Visualizer Context: {calculation_name}. Missing column '{col_name}' added with default: {default_val_to_add}.")
            else: 
                if col_name in string_like_id_cols:
                    if not pd.api.types.is_string_dtype(df_copy[col_name]) and not pd.api.types.is_object_dtype(df_copy[col_name]):
                        all_present_and_valid_initially = False; original_dtype_str = str(df_copy[col_name].dtype)
                        df_copy[col_name] = df_copy[col_name].astype(str)
                        actions_taken_log.append(f"Coerced visualizer column '{col_name}' from {original_dtype_str} to string")
                    if df_copy[col_name].isnull().any():
                        if all_present_and_valid_initially: all_present_and_valid_initially = False
                        df_copy[col_name] = df_copy[col_name].fillna('N/A_FILLED')
                        actions_taken_log.append(f"Filled NaNs in visualizer string column '{col_name}' with 'N/A_FILLED'")
                elif col_name in datetime_cols_special_handling:
                    if not pd.api.types.is_datetime64_any_dtype(df_copy[col_name]) and not pd.api.types.is_period_dtype(df_copy[col_name]) and not all(isinstance(x, (date, datetime, pd.Timestamp, type(pd.NaT))) for x in df_copy[col_name].dropna()):
                        original_dtype_dt = str(df_copy[col_name].dtype)
                        try:
                            df_copy[col_name] = pd.to_datetime(df_copy[col_name], errors='coerce')
                            coerced_successfully = pd.api.types.is_datetime64_any_dtype(df_copy[col_name])
                        except Exception: coerced_successfully = False
                        if not coerced_successfully:
                            actions_taken_log.append(f"Failed to coerce visualizer '{col_name}' from {original_dtype_dt} to datetime.")
                            all_present_and_valid_initially = False
                        else:
                            actions_taken_log.append(f"Coerced visualizer column '{col_name}' from {original_dtype_dt} to datetime.")
                            all_present_and_valid_initially = False 
                    if df_copy[col_name].isnull().any():
                        if all_present_and_valid_initially : all_present_and_valid_initially = False
                        actions_taken_log.append(f"Visualizer column '{col_name}' (datetime) has NaNs/NaTs.")
                else: 
                    if not pd.api.types.is_numeric_dtype(df_copy[col_name]):
                        all_present_and_valid_initially = False; original_dtype_str = str(df_copy[col_name].dtype)
                        df_copy[col_name] = pd.to_numeric(df_copy[col_name], errors='coerce')
                        actions_taken_log.append(f"Coerced visualizer column '{col_name}' from {original_dtype_str} to numeric")
                    if df_copy[col_name].isnull().any():
                        if all_present_and_valid_initially : all_present_and_valid_initially = False
                        df_copy[col_name] = df_copy[col_name].fillna(0.0)
                        actions_taken_log.append(f"Filled NaNs in visualizer numeric column '{col_name}' with 0.0")
        if not all_present_and_valid_initially:
            ensure_logger.info(f"Visualizer Context: {calculation_name}. Column integrity actions: {'; '.join(actions_taken_log) if actions_taken_log else 'Type/NaN mods.'}")
        else:
            ensure_logger.debug(f"Visualizer Context: {calculation_name}. All required columns initially present and valid.")
        return df_copy, all_present_and_valid_initially

    def _create_raw_greek_chart(
        self,
        processed_data: pd.DataFrame,
        metric_col: str,
        chart_title_part: str,
        xaxis_title: str,
        call_color: str,
        put_color: str,
        symbol: str,
        current_price: Optional[float], # This is the underlying price
        selected_price_range_pct_override: Optional[float], # This comes from the slider
        fetch_timestamp: Optional[str]
    ) -> go.Figure:
        chart_name = f"Raw {chart_title_part.split('(')[0].strip()} Chart"
        chart_title = f"<b>{symbol.upper()}</b> - {chart_title_part}"
        # Use self.instance_logger for consistency
        chart_logger = self.instance_logger.getChild(chart_name) 
        chart_logger.info(f"Creating {chart_name} for {symbol} (Metric: {metric_col}). Received price_range_override: {selected_price_range_pct_override}, current_price: {current_price}")

        fig_height_key = f"{metric_col.lower().replace('sdag_', 'sdag')}_chart_height"
        fig_height = self.config.get("chart_specific_params", {}).get(fig_height_key, self.config.get("default_chart_height", 700))
        
        try:
            if not isinstance(processed_data, pd.DataFrame) or processed_data.empty:
                return self._create_empty_figure(f"{symbol} - {chart_name}: No Data Provided", height=fig_height, reason="Input DataFrame empty/invalid")

            required_cols_for_chart = [self.col_strike, self.col_opt_kind, metric_col]
            if not all(c in processed_data.columns for c in required_cols_for_chart):
                missing = [c for c in required_cols_for_chart if c not in processed_data.columns]
                chart_logger.error(f"Missing required columns for {chart_name}: {missing}. Available: {list(processed_data.columns)}")
                return self._create_empty_figure(f"{symbol} - {chart_name}: Missing Data ({', '.join(missing)})", height=fig_height, reason=f"Missing: {', '.join(missing)}")

            df = processed_data.copy()
            # Ensure 'strike' is numeric for filtering and pivoting, keep original self.col_strike for display if needed
            df['strike_numeric'] = pd.to_numeric(df[self.col_strike], errors='coerce')
            df[metric_col] = pd.to_numeric(df[metric_col], errors='coerce')
            df[self.col_opt_kind] = df[self.col_opt_kind].astype(str).str.lower().fillna('unknown')
            
            df = df.dropna(subset=['strike_numeric', metric_col, self.col_opt_kind])
            df = df[df[self.col_opt_kind].isin(['call', 'put'])]

            if df.empty:
                return self._create_empty_figure(f"{symbol} - {chart_name}: No Valid Call/Put Data After Cleaning", height=fig_height, reason="No valid call/put data after initial cleaning")

            # --- Price Range Filtering Logic ---
            filtered_df = df.copy() # Start with all valid data
            filter_suffix = ""
            
            default_raw_range_pct_from_config = self.config.get("chart_specific_params", {}).get("raw_greek_charts_price_range_pct", 5.0)
            
            # Use the override if it's a valid number, otherwise use the config default
            if isinstance(selected_price_range_pct_override, (int, float)) and pd.notna(selected_price_range_pct_override) and selected_price_range_pct_override > 0:
                selected_pct_to_use = selected_price_range_pct_override
                chart_logger.debug(f"{chart_name}: Using selected_price_range_pct_override: {selected_pct_to_use}%")
            else:
                selected_pct_to_use = default_raw_range_pct_from_config
                chart_logger.debug(f"{chart_name}: Override was invalid or None ({selected_price_range_pct_override}). Using default from config: {selected_pct_to_use}%")

            if current_price is not None and pd.notna(current_price) and current_price > 0:
                min_s = current_price * (1 - (selected_pct_to_use / 100.0))
                max_s = current_price * (1 + (selected_pct_to_use / 100.0))
                chart_logger.info(f"{chart_name}: Applying strike filter. Current Price: {current_price:.2f}, Range Pct: {selected_pct_to_use:.1f}%. Min Strike: {min_s:.2f}, Max Strike: {max_s:.2f}")
                
                # Ensure filtering happens on the numeric strike column
                filtered_df = df[(df['strike_numeric'] >= min_s) & (df['strike_numeric'] <= max_s)].copy()
                filter_suffix = f" (Strikes +/- {selected_pct_to_use:.1f}%)"
                
                if filtered_df.empty:
                    chart_logger.warning(f"{chart_name}: DataFrame became empty after applying price range filter for {symbol}. Will plot with no data for this range.")
                    # Don't return empty figure here; let it proceed to show an empty plot if no data in range
            else:
                chart_logger.debug(f"{chart_name}: Price range filter NOT applied (current_price invalid: {current_price} or selected_pct_to_use not positive: {selected_pct_to_use}). Using all data.")
            # --- End of Price Range Filtering Logic ---

            if filtered_df.empty: # Check again after filtering
                 return self._create_empty_figure(f"{symbol} - {chart_name}: No Data in Selected Range {filter_suffix}", height=fig_height, reason=f"No data for range {filter_suffix}")

            overview_metrics_cfg = self.config.get("hover_settings", {}).get("overview_metrics_config", [])
            # Use actual column names from the DataFrame for hover logic
            column_names_map = self.config.get("column_names", {})
            hover_context_keys = []
            for m_cfg in overview_metrics_cfg:
                cfg_key = m_cfg.get("key")
                if cfg_key:
                    actual_col_for_hover = column_names_map.get(cfg_key, cfg_key)
                    if actual_col_for_hover in filtered_df.columns and actual_col_for_hover != metric_col:
                        hover_context_keys.append(actual_col_for_hover)
            
            agg_hover_logic_local: Dict[str, Any] = {hc: 'first' for hc in list(set(hover_context_keys))}; # Unique keys
            agg_hover_logic_local[metric_col] = 'sum' # The metric being plotted is summed for calls/puts separately
            
            # Ensure all keys in agg_hover_logic_local are present in filtered_df before groupby
            valid_agg_keys = {k: v for k, v in agg_hover_logic_local.items() if k in filtered_df.columns}
            if not valid_agg_keys: # Should at least have metric_col
                chart_logger.warning(f"{chart_name}: No valid keys for hover aggregation. Hover might be limited.")
                hover_data_agg_source = pd.DataFrame({'strike_numeric': filtered_df['strike_numeric'].dropna().unique()})
            else:
                 hover_data_agg_source = filtered_df.groupby('strike_numeric', as_index=False).agg(valid_agg_keys)

            hover_data_for_lookup = hover_data_agg_source.set_index('strike_numeric')
            
            # Pivot for plotting bars
            pivot_plot = filtered_df.pivot_table(values=metric_col, index='strike_numeric', columns=self.col_opt_kind, aggfunc='sum').fillna(0)
            
            # Get unique strikes from the *filtered* data, sort descending for y-axis display
            unique_strikes_desc = sorted(filtered_df['strike_numeric'].dropna().unique(), reverse=True)

            if not unique_strikes_desc:
                return self._create_empty_figure(f"{symbol} - {chart_name}: No Unique Strikes After Filtering", height=fig_height, reason="No strikes left after filtering/cleaning for plot")

            # Reindex pivot table to ensure all unique strikes are present for calls and puts
            puts_agg = pivot_plot.get('put', pd.Series(dtype=float)).reindex(unique_strikes_desc, fill_value=0.0)
            calls_agg = pivot_plot.get('call', pd.Series(dtype=float)).reindex(unique_strikes_desc, fill_value=0.0)
            
            # Reindex hover data lookup to match the strike order
            hover_data_for_lookup = hover_data_for_lookup.reindex(unique_strikes_desc).fillna(np.nan) # Fill with NaN so _create_hover_text shows N/A

            fig = go.Figure()
            y_indices = list(range(len(unique_strikes_desc))) # Integer indices for y-axis
            y_labels = [f"{s:.2f}" for s in unique_strikes_desc] # String labels for y-axis ticks
            
            strike_map_for_hline = {float(f"{s:.2f}"): i for i, s in enumerate(unique_strikes_desc)} # Map numeric strike to y-index for HLINE

            puts_hovers, calls_hovers = [], []
            hover_chart_type = "default"
            if metric_col == 'tdpi': hover_chart_type = "tdpi"
            elif metric_col == 'vri': hover_chart_type = "vri"
            elif metric_col.startswith('sdag_'): hover_chart_type = "sdag"
            
            for strike_numeric_val in unique_strikes_desc:
                # Base dict for hover, always include the strike value
                base_hover_info = {self.col_strike: strike_numeric_val} 
                if strike_numeric_val in hover_data_for_lookup.index:
                    # Add other aggregated metrics for this strike
                    base_hover_info.update(hover_data_for_lookup.loc[strike_numeric_val].dropna().to_dict())

                put_val_specific = puts_agg.get(strike_numeric_val, 0.0)
                put_hover_data_final = base_hover_info.copy()
                put_hover_data_final[metric_col] = put_val_specific # Add the specific put value for this strike
                extra_ctx_put = {'Option Type': 'Put'}
                if hover_chart_type == "sdag": extra_ctx_put.update({"SDAG Method": metric_col.upper(), "sdag_col_name": metric_col})
                puts_hovers.append(self._create_hover_text(put_hover_data_final, chart_type=hover_chart_type, extra_context=extra_ctx_put))

                call_val_specific = calls_agg.get(strike_numeric_val, 0.0)
                call_hover_data_final = base_hover_info.copy()
                call_hover_data_final[metric_col] = call_val_specific # Add the specific call value
                extra_ctx_call = {'Option Type': 'Call'}
                if hover_chart_type == "sdag": extra_ctx_call.update({"SDAG Method": metric_col.upper(), "sdag_col_name": metric_col})
                calls_hovers.append(self._create_hover_text(call_hover_data_final, chart_type=hover_chart_type, extra_context=extra_ctx_call))

            fig.add_trace(go.Bar(y=y_indices, x=puts_agg.values, name=f'Puts {metric_col.upper()}', orientation='h', marker_color=put_color, hovertext=puts_hovers, hoverinfo='text'))
            fig.add_trace(go.Bar(y=y_indices, x=calls_agg.values, name=f'Calls {metric_col.upper()}', orientation='h', marker_color=call_color, hovertext=calls_hovers, hoverinfo='text'))
            
            show_net_trace_cfg = self.config.get("chart_specific_params", {}).get("show_net_sdag_trace", False)
            if show_net_trace_cfg and metric_col.startswith('sdag_'):
                net_aggregated_values = puts_agg + calls_agg
                net_hovers_for_trace = []
                for strike_numeric_val_net in unique_strikes_desc:
                    net_metric_value = net_aggregated_values.get(strike_numeric_val_net, 0.0)
                    net_hover_data_base_ctx = {self.col_strike: strike_numeric_val_net, metric_col: net_metric_value}
                    if strike_numeric_val_net in hover_data_for_lookup.index:
                        row_series_net_ctx = hover_data_for_lookup.loc[strike_numeric_val_net]
                        net_hover_data_base_ctx.update({k:v for k,v in row_series_net_ctx.dropna().to_dict().items() if k != metric_col})
                    
                    hover_text_for_net_trace = self._create_hover_text(net_hover_data_base_ctx, chart_type="sdag_net", extra_context={"SDAG Method": f"Net {metric_col.upper()}", "sdag_col_name": metric_col})
                    net_hovers_for_trace.append(hover_text_for_net_trace)
                
                net_style_cfg = self.config.get("chart_specific_params", {}).get("net_sdag_marker_style", {})
                net_visibility_cfg = self.config.get("chart_specific_params", {}).get("net_sdag_trace_default_visibility", 'legendonly')
                net_trace_display_name = f"Net {metric_col.upper()}"
                fig.add_trace(go.Scatter(y=y_indices, x=net_aggregated_values.values, mode='markers', name=net_trace_display_name, marker=dict(symbol=net_style_cfg.get('symbol', 'diamond'), color=net_style_cfg.get('color', 'rgba(255, 255, 255, 0.7)'), size=net_style_cfg.get('size', 8), line=net_style_cfg.get('line', dict(color='white', width=1))), hovertext=net_hovers_for_trace, hoverinfo='text', visible=net_visibility_cfg))
                chart_logger.debug(f"Added '{net_trace_display_name}' scatter trace.")
            
            legend_cfg=self.config.get("legend_settings",{})
            fig.update_layout(
                title=f"{chart_title}{filter_suffix}", 
                yaxis_title="Strike", 
                xaxis_title=xaxis_title, 
                barmode='relative', 
                template=self.config.get("plotly_template", "plotly_dark"), 
                height=fig_height, 
                yaxis=dict(
                    tickmode='array', 
                    tickvals=y_indices, 
                    ticktext=y_labels, 
                    autorange='reversed' # High strikes at top because unique_strikes_desc is sorted descending
                ), 
                xaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor='lightgrey'), 
                legend=dict(
                    orientation=legend_cfg.get("orientation","v"), 
                    yanchor=legend_cfg.get("y_anchor","top"), 
                    y=legend_cfg.get("y_pos",1), 
                    xanchor=legend_cfg.get("x_anchor","left"), 
                    x=legend_cfg.get("x_pos",1.02), 
                    traceorder=legend_cfg.get("trace_order","reversed")
                ), 
                hovermode="y unified" 
            )

            if current_price is not None and pd.notna(current_price) and current_price > 0:
                # Find the closest *numeric* strike value that is actually plotted
                if unique_strikes_desc: # Ensure there are strikes to find closest to
                    closest_plotted_strike_numeric = min(unique_strikes_desc, key=lambda s: abs(s - current_price))
                    # Get the y-index for this strike
                    y_price_idx_for_line = strike_map_for_hline.get(closest_plotted_strike_numeric)
                    
                    if y_price_idx_for_line is not None:
                        # Pass the integer y-index to _add_price_line for horizontal line
                        fig = self._add_price_line(fig, current_price=y_price_idx_for_line, orientation='horizontal', 
                                               annotation={'text': f"Current: {current_price:.2f}"})
                        chart_logger.debug(f"Added price line for {chart_name} at y-index {y_price_idx_for_line} (Strike ~{closest_plotted_strike_numeric:.2f}) for underlying price {current_price:.2f}")
                    else:
                        chart_logger.warning(f"Could not find y-index for closest strike {closest_plotted_strike_numeric} in strike_map for {chart_name}")
                else:
                    chart_logger.warning(f"No unique strikes available to draw price line for {chart_name}")


            fig = self._add_timestamp_annotation(fig, fetch_timestamp)
            self._save_figure(fig, chart_name, symbol)

        except Exception as e:
            chart_logger.error(f"Raw Metric Chart ({symbol} - {metric_col}) creation failed: {e}", exc_info=True)
            return self._create_empty_figure(f"{symbol} - {chart_name}: Plotting Error", height=fig_height, reason=str(e))
        
        chart_logger.info(f"{chart_name} for {symbol} created successfully.")
        return fig

    def create_time_decay_visualization(self, processed_data:pd.DataFrame, symbol:str="N/A", current_price:Optional[float]=None, selected_price_range_pct_override:Optional[float]=None, fetch_timestamp:Optional[str]=None, **kwargs) -> go.Figure:
        return self._create_raw_greek_chart( processed_data=processed_data, metric_col='tdpi', chart_title_part="Time Decay (TDPI by Strike)", xaxis_title="Time Decay Pressure Index (TDPI)", call_color='green', put_color='red', symbol=symbol, current_price=current_price, selected_price_range_pct_override=selected_price_range_pct_override, fetch_timestamp=fetch_timestamp )

    def create_volatility_regime_visualization(self, processed_data:pd.DataFrame, symbol:str="N/A", current_price:Optional[float]=None, selected_price_range_pct_override:Optional[float]=None, fetch_timestamp:Optional[str]=None, **kwargs) -> go.Figure:
        return self._create_raw_greek_chart( processed_data=processed_data, metric_col='vri', chart_title_part="Volatility Regime (VRI by Strike)", xaxis_title="Volatility Regime Indicator (VRI)", call_color='cyan', put_color='magenta', symbol=symbol, current_price=current_price, selected_price_range_pct_override=selected_price_range_pct_override, fetch_timestamp=fetch_timestamp )

    def plot_sdag_multiplicative(self, processed_data: pd.DataFrame, symbol: str = "N/A", current_price: Optional[float] = None, fetch_timestamp: Optional[str] = None, selected_price_range_pct_override: Optional[float]=None, **kwargs) -> go.Figure:
        metric = 'sdag_multiplicative';
        if metric not in processed_data.columns: return self._create_empty_figure(f"{symbol} - SDAG Multiplicative: Data Not Available", reason=f"{metric} col missing")
        colors = self.config.get("chart_specific_params",{}).get("mspi_components_bar_colors",{}).get(f"{metric}_norm", {"pos": "#FFA07A", "neg": "#6A5ACD"})
        return self._create_raw_greek_chart(processed_data=processed_data, metric_col=metric, chart_title_part="SDAG Multiplicative", xaxis_title="SDAG (Multiplicative)", call_color=colors['pos'], put_color=colors['neg'], symbol=symbol, current_price=current_price, selected_price_range_pct_override=selected_price_range_pct_override, fetch_timestamp=fetch_timestamp)

    def plot_sdag_directional(self, processed_data: pd.DataFrame, symbol: str = "N/A", current_price: Optional[float] = None, fetch_timestamp: Optional[str] = None, selected_price_range_pct_override: Optional[float]=None, **kwargs) -> go.Figure:
        metric = 'sdag_directional';
        if metric not in processed_data.columns: return self._create_empty_figure(f"{symbol} - SDAG Directional: Data Not Available", reason=f"{metric} col missing")
        colors = self.config.get("chart_specific_params",{}).get("mspi_components_bar_colors",{}).get(f"{metric}_norm", {"pos": "#FFD700", "neg": "#8A2BE2"})
        return self._create_raw_greek_chart(processed_data=processed_data, metric_col=metric, chart_title_part="SDAG Directional", xaxis_title="SDAG (Directional)", call_color=colors['pos'], put_color=colors['neg'], symbol=symbol, current_price=current_price, selected_price_range_pct_override=selected_price_range_pct_override, fetch_timestamp=fetch_timestamp)

    def plot_sdag_weighted(self, processed_data: pd.DataFrame, symbol: str = "N/A", current_price: Optional[float] = None, fetch_timestamp: Optional[str] = None, selected_price_range_pct_override: Optional[float]=None, **kwargs) -> go.Figure:
        metric = 'sdag_weighted';
        if metric not in processed_data.columns: return self._create_empty_figure(f"{symbol} - SDAG Weighted: Data Not Available", reason=f"{metric} col missing")
        colors = self.config.get("chart_specific_params",{}).get("mspi_components_bar_colors",{}).get(f"{metric}_norm", {"pos": "#98FB98", "neg": "#FF6347"})
        return self._create_raw_greek_chart(processed_data=processed_data, metric_col=metric, chart_title_part="SDAG Weighted", xaxis_title="SDAG (Weighted)", call_color=colors['pos'], put_color=colors['neg'], symbol=symbol, current_price=current_price, selected_price_range_pct_override=selected_price_range_pct_override, fetch_timestamp=fetch_timestamp)

    def plot_sdag_volatility_focused(self, processed_data: pd.DataFrame, symbol: str = "N/A", current_price: Optional[float] = None, fetch_timestamp: Optional[str] = None, selected_price_range_pct_override: Optional[float]=None, **kwargs) -> go.Figure:
        metric = 'sdag_volatility_focused';
        if metric not in processed_data.columns: return self._create_empty_figure(f"{symbol} - SDAG Volatility Focused: Data Not Available", reason=f"{metric} col missing")
        colors = self.config.get("chart_specific_params",{}).get("mspi_components_bar_colors",{}).get(f"{metric}_norm", {"pos": "#AFEEEE", "neg": "#DA70D6"})
        return self._create_raw_greek_chart(processed_data=processed_data, metric_col=metric, chart_title_part="SDAG Volatility Focused", xaxis_title="SDAG (Volatility Focused)", call_color=colors['pos'], put_color=colors['neg'], symbol=symbol, current_price=current_price, selected_price_range_pct_override=selected_price_range_pct_override, fetch_timestamp=fetch_timestamp)

    def create_volval_comparison( self, processed_data: pd.DataFrame, component_history: Optional[Deque[Tuple[float, pd.DataFrame]]]=None, symbol: str="N/A", current_price: Optional[float]=None, fetch_timestamp: Optional[str]=None, trace_visibility: Optional[Dict[str,Any]]=None, **kwargs ) -> go.Figure:
        chart_name = "Net Volume vs Value Pressure Comparison"; chart_title = f"<b>{symbol.upper()}</b> - {chart_name}";
        chart_logger = logging.getLogger(__name__ + "." + chart_name); chart_logger.info(f"Creating {chart_title}...")
        fig_height = self.config.get("chart_specific_params",{}).get("volval_comparison_height",600)
        try:
            if not isinstance(processed_data, pd.DataFrame) or processed_data.empty: return self._create_empty_figure(f"{symbol} - {chart_name}: No Current Data", height=fig_height, reason="Input DF empty")
            if trace_visibility is None: trace_visibility = {}
            required_cols = [self.col_strike, self.col_net_vol_p, self.col_net_val_p];
            if not all(col in processed_data.columns for col in required_cols): missing = [col for col in required_cols if col not in processed_data.columns]; return self._create_empty_figure(f"{symbol} - {chart_name}: Missing Data ({', '.join(missing)})", height=fig_height, reason=f"Missing: {missing}")
            df = processed_data.copy(); agg_cols_logic = {col: 'first' for col in required_cols if col != self.col_strike}
            agg_data = df.groupby(self.col_strike, as_index=False).agg(agg_cols_logic); agg_data[self.col_strike] = pd.to_numeric(agg_data[self.col_strike], errors='coerce'); agg_data = agg_data.dropna(subset=[self.col_strike]).sort_values(self.col_strike)
            if agg_data.empty: return self._create_empty_figure(f"{symbol} - {chart_name}: No Aggregated Current Data", height=fig_height, reason="Agg current data empty")
            net_val_p_series = pd.to_numeric(agg_data[self.col_net_val_p], errors='coerce').fillna(0); net_val_p_pos = net_val_p_series.where(net_val_p_series >= 0, 0); net_val_p_neg = net_val_p_series.where(net_val_p_series < 0, 0)
            plot_order_history_cfg = self.config.get("plot_order_history", DEFAULT_VISUALIZER_CONFIG.get("plot_order_history", []))
            history_found: Dict[str, Optional[pd.DataFrame]] = {lbl: None for lbl in plot_order_history_cfg}
            current_unix_ts: Optional[float] = None; current_dt_date: Optional[date] = None
            if fetch_timestamp:
                try: dto = date_parser.isoparse(fetch_timestamp); current_unix_ts = dto.timestamp(); current_dt_date = dto.date()
                except Exception as parse_err: chart_logger.warning(f"TS parse fail '{fetch_timestamp}': {parse_err}")
            if component_history and current_unix_ts and current_dt_date:
                chart_logger.debug(f"Processing {len(component_history)} history entries...")
                target_times_dt: Dict[str, time] = {"T-1":time(9,55),"T-2":time(10,55),"T-3":time(12,35),"T-4":time(13,30),"T-5":time(15,0)}; target_times_ts: Dict[str, float] = {lbl:datetime.combine(current_dt_date, t).timestamp() for lbl, t in target_times_dt.items()}
                lookback_A_sec, lookback_B_sec = 5*60, 15*60; tolerance_fixed_sec, tolerance_relative_sec = 15*60, 2*60
                best_matches: Dict[str, Tuple[float, int, Optional[Tuple[float, pd.DataFrame]]]] = {lbl: (float('inf'), -1, None) for lbl in plot_order_history_cfg}; history_list = list(component_history)
                for idx, entry in enumerate(history_list):
                    if not (isinstance(entry, tuple) and len(entry)==2 and isinstance(entry[1], pd.DataFrame) and all(c in entry[1].columns for c in [self.col_strike, self.col_net_vol_p, self.col_net_val_p])) : continue
                    hist_ts, hist_df = entry
                    try:
                        delta_now = current_unix_ts - hist_ts; diff_A = abs(delta_now - lookback_A_sec); diff_B = abs(delta_now - lookback_B_sec)
                        if diff_A < tolerance_relative_sec and diff_A < best_matches.get("T-A",(float('inf'),-1,None))[0]: best_matches["T-A"]=(diff_A, idx, entry)
                        if diff_B < tolerance_relative_sec and diff_B < best_matches.get("T-B",(float('inf'),-1,None))[0]: best_matches["T-B"]=(diff_B, idx, entry)
                        for slot, target_ts_iter in target_times_ts.items():
                            if slot in best_matches: diff_fixed = abs(hist_ts - target_ts_iter);
                            if diff_fixed < tolerance_fixed_sec and diff_fixed < best_matches[slot][0]: best_matches[slot] = (diff_fixed, idx, entry)
                    except Exception as calc_e: chart_logger.warning(f"Hist Err calc time diff {idx}: {calc_e}")
                found_indices = set()
                for label_iter, match_tuple in best_matches.items():
                    diff_val, idx_val, entry_val = match_tuple
                    if diff_val != float('inf') and idx_val != -1 and idx_val not in found_indices and entry_val is not None:
                        _, hist_df_match = entry_val; hist_agg = hist_df_match.groupby(self.col_strike, as_index=False).agg({self.col_net_vol_p:'first', self.col_net_val_p:'first'}); hist_agg[self.col_strike] = pd.to_numeric(hist_agg[self.col_strike], errors='coerce'); hist_agg = hist_agg.dropna(subset=[self.col_strike]); history_found[label_iter] = hist_agg; found_indices.add(idx_val)
                chart_logger.debug(f"Found hist matches: {[lbl for lbl, e in history_found.items() if e is not None]}")
            hover_df = agg_data[[self.col_strike, self.col_net_vol_p, self.col_net_val_p]].copy(); hover_df.columns = [self.col_strike, 'vol_Now', 'val_Now']; plotted_labels = ["Now"]
            for label_hist in plot_order_history_cfg:
                hist_agg_df = history_found.get(label_hist)
                if hist_agg_df is not None and isinstance(hist_agg_df, pd.DataFrame): temp_hist = hist_agg_df[[self.col_strike, self.col_net_vol_p, self.col_net_val_p]].copy(); temp_hist.columns = [self.col_strike, f'vol_{label_hist}', f'val_{label_hist}']; hover_df = pd.merge(hover_df, temp_hist, on=self.col_strike, how='left'); plotted_labels.append(label_hist)
            hover_df = hover_df.fillna(0); hovertemplate_str = self._build_volval_hovertemplate(hover_df, plotted_labels)
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter( x=agg_data[self.col_strike], y=net_val_p_pos, name='Net Val P (+)', mode='lines', line=dict(width=0.6, color='rgba(0, 150, 0, 0.9)'), fillcolor='rgba(0, 240, 0, 0.05)', fill='tozeroy', visible=trace_visibility.get('Net Val P (+)',True), showlegend=True, hoverinfo='skip' ), secondary_y=True)
            fig.add_trace(go.Scatter( x=agg_data[self.col_strike], y=net_val_p_neg, name='Net Val P (-)', mode='lines', line=dict(width=0.6, color='rgba(150, 0, 0, 0.9)'), fillcolor='rgba(240, 0, 0, 0.05)', fill='tozeroy', visible=trace_visibility.get('Net Val P (-)',True), showlegend=True, hoverinfo='skip' ), secondary_y=True)
            ghost_settings_defaults: Dict[str, Dict[str, Any]] = { "T-5":{'c_p':'rgba(0,255,255,0.35)', 'c_n':'rgba(0,0,139,0.35)', 'o':0.6, 's':'(T-5~3pm)'}, "T-4":{'c_p':'rgba(0,255,255,0.40)', 'c_n':'rgba(0,0,139,0.40)', 'o':0.6, 's':'(T-4~1:30pm)'}, "T-3":{'c_p':'rgba(0,255,255,0.45)', 'c_n':'rgba(0,0,139,0.45)', 'o':0.6, 's':'(T-3~12:35pm)'}, "T-2":{'c_p':'rgba(0,255,255,0.50)', 'c_n':'rgba(0,0,139,0.50)', 'o':0.6, 's':'(T-2~11am)'}, "T-1":{'c_p':'rgba(0,255,255,0.55)', 'c_n':'rgba(0,0,139,0.55)', 'o':0.6, 's':'(T-1~10am)'}, "T-B":{'c_p':'rgba(34,139,34,0.65)', 'c_n':'rgba(178,34,34,0.65)', 'o':0.7, 's':'(T-15min)'}, "T-A":{'c_p':'rgba(0,100,0,0.75)', 'c_n':'rgba(139,0,0,0.75)', 'o':0.8, 's':'(T-5min)'}, "Now":{'c_p':'rgb(0,100,0)', 'c_n':'rgb(139,0,0)', 'o':0.9, 's':' (Now)'} }
            ghost_settings_from_config = self.config.get("chart_specific_params",{}).get("volval_ghost_settings", ghost_settings_defaults)
            for label_plot_order in plot_order_history_cfg:
                settings = ghost_settings_from_config.get(label_plot_order); plot_df_bar: Optional[pd.DataFrame] = None
                if label_plot_order == "Now": plot_df_bar = agg_data
                elif label_plot_order in history_found and history_found[label_plot_order] is not None: plot_df_bar = history_found[label_plot_order]
                if settings is None or plot_df_bar is None: continue
                if not isinstance(plot_df_bar, pd.DataFrame) or self.col_net_vol_p not in plot_df_bar or self.col_strike not in plot_df_bar: continue
                df_trace = plot_df_bar[[self.col_strike, self.col_net_vol_p]].copy(); df_trace[self.col_strike] = pd.to_numeric(df_trace[self.col_strike], errors='coerce'); df_trace[self.col_net_vol_p] = pd.to_numeric(df_trace[self.col_net_vol_p], errors='coerce').fillna(0); df_trace = df_trace.dropna(subset=[self.col_strike])
                if df_trace.empty: continue
                trace_name_bar = f'Net Vol P{settings.get("s","")}'; is_now = (label_plot_order == "Now"); default_visibility = True if is_now else 'legendonly'; trace_visibility_state = trace_visibility.get(trace_name_bar, default_visibility)
                chart_logger.debug(f"Trace '{trace_name_bar}': Visibility from state/default: {trace_visibility_state}")
                bar_colors = [settings['c_p'] if v >= 0 else settings['c_n'] for v in df_trace[self.col_net_vol_p]]; customdata_for_trace = hover_df.values if is_now else None; hovertemplate_for_trace = hovertemplate_str if is_now else None; hoverinfo_setting = 'text' if is_now else 'skip'
                fig.add_trace(go.Bar( x=df_trace[self.col_strike], y=df_trace[self.col_net_vol_p], name=trace_name_bar, marker_color=bar_colors, opacity=settings.get('o', 0.8), visible=trace_visibility_state, customdata=customdata_for_trace, hovertemplate=hovertemplate_for_trace, hoverinfo=hoverinfo_setting, showlegend=True ), secondary_y=False)
            legend_cfg=self.config.get("legend_settings",{});
            fig.update_layout( title=chart_title, xaxis_title='Strike', barmode='group', template=self.config.get("plotly_template", "plotly_dark"), height=fig_height, legend=dict(orientation=legend_cfg.get("orientation","v"), yanchor=legend_cfg.get("y_anchor","top"), y=legend_cfg.get("y_pos",1), xanchor=legend_cfg.get("x_anchor","left"), x=legend_cfg.get("x_pos",1.02), traceorder=legend_cfg.get("trace_order","reversed")), hovermode="x unified" )
            fig.update_xaxes(tickformat=".2f"); fig.update_yaxes(title_text="Net Volume Pressure", secondary_y=False, zeroline=True, zerolinewidth=1, zerolinecolor='gray'); fig.update_yaxes(title_text="Net Value Pressure ($)", secondary_y=True, showgrid=False, zeroline=False)
            fig = self._add_price_line(fig, current_price, orientation='vertical'); fig = self._add_timestamp_annotation(fig, fetch_timestamp); self._save_figure(fig, chart_name, symbol)
        except Exception as e:
            chart_logger.error(f"VolVal Comp creation failed for {symbol}: {e}", exc_info=True); return self._create_empty_figure(f"{symbol} - {chart_name}: Plotting Error", height=fig_height, reason=str(e))
        chart_logger.info(f"Chart {chart_name} for {symbol} created successfully."); return fig

    def create_combined_rolling_flow_chart( self, processed_data: pd.DataFrame, symbol: str = "N/A", current_price: Optional[float] = None, fetch_timestamp: Optional[str] = None, trace_visibility: Optional[Dict[str,Any]] = None, selected_price_range_pct_override: Optional[float] = None, **kwargs ) -> go.Figure:
        chart_name="Combined_Rolling_Flow"; chart_title=f"<b>{symbol.upper()}</b> - Rolling Net Volume & Value Flow";
        chart_logger = logging.getLogger(__name__ + "." + chart_name); chart_logger.info(f"Creating chart for symbol '{symbol}'...")
        fig_height=self.config.get("chart_specific_params", {}).get("combined_flow_chart_height", self.config.get("default_chart_height",700))
        try:
            if not isinstance(processed_data, pd.DataFrame) or processed_data.empty: return self._create_empty_figure(f"{symbol}-{chart_name}: No Data", height=fig_height, reason="Input DataFrame empty/invalid")
            if trace_visibility is None: trace_visibility={}
            df=processed_data.copy(); intervals=self.config.get("rolling_intervals",["5m","15m","30m","60m"])
            vol_cols=[f"volmbs_{i}" for i in intervals]; val_cols=[f"valuebs_{i}" for i in intervals]
            act_vol=[m for m in vol_cols if m in df.columns]; act_val=[m for m in val_cols if m in df.columns]
            if self.col_strike not in df.columns: return self._create_empty_figure(f"{symbol}-{chart_name}: Missing Strike Column", height=fig_height, reason=f"'{self.col_strike}' missing")
            if not (act_vol or act_val): return self._create_empty_figure(f"{symbol}-{chart_name}: Missing Rolling Data Columns", height=fig_height, reason="No rolling vol/val cols")
            agg_logic={c:'sum' for c in act_vol+act_val}; agg=df.groupby(self.col_strike,as_index=False).agg(agg_logic)
            agg[self.col_strike]=pd.to_numeric(agg[self.col_strike],errors='coerce'); agg=agg.dropna(subset=[self.col_strike]).sort_values(self.col_strike)
            if agg.empty: return self._create_empty_figure(f"{symbol}-{chart_name}: No Agg Data", height=fig_height, reason="Aggregated data empty")
            for c in act_vol+act_val: agg[c]=pd.to_numeric(agg[c],errors='coerce').fillna(0)
            norm_vol_cols, norm_val_cols = [], []; min_denom=self.config.get("min_normalization_denominator",1e-9)
            max_abs_vol=max((agg[mc].abs().max() for mc in act_vol), default=0.0); max_abs_vol=1.0 if pd.isna(max_abs_vol) or max_abs_vol<min_denom else max_abs_vol
            for mc in act_vol: norm_c=f"{mc}_norm"; agg[norm_c]=agg[mc]/max_abs_vol; norm_vol_cols.append(norm_c)
            max_abs_val=max((agg[mc].abs().max() for mc in act_val), default=0.0); max_abs_val=1.0 if pd.isna(max_abs_val) or max_abs_val<min_denom else max_abs_val
            for mc in act_val: norm_c=f"{mc}_norm"; agg[norm_c]=agg[mc]/max_abs_val; norm_val_cols.append(norm_c)
            fig=go.Figure(); # Initialize with go.Figure for single y-axis approach
            leg_cfg=self.config.get("legend_settings",{}); chart_params_cfg = self.config.get("chart_specific_params", {}); flow_custom_cfg = chart_params_cfg.get("rolling_flow_customization", {}); defaults_style = flow_custom_cfg.get("defaults", {})
            hover_data_cols = [self.col_strike]; vol_cols_for_hover = []; val_cols_for_hover = []
            for iv_label in intervals:
                vol_orig_col = f"volmbs_{iv_label}"; val_orig_col = f"valuebs_{iv_label}"
                if vol_orig_col in agg.columns: vol_cols_for_hover.append(vol_orig_col)
                if val_orig_col in agg.columns: val_cols_for_hover.append(val_orig_col)
            hover_data_cols.extend(vol_cols_for_hover); hover_data_cols.extend(val_cols_for_hover)
            chart_logger.debug(f"Columns selected for custom hoverdata (horizontal): {hover_data_cols}")
            hover_df = agg[hover_data_cols].copy(); customdata_array = hover_df.values
            chart_logger.debug(f"Customdata Array Shape (horizontal): {customdata_array.shape}")
            hovertemplate = f"<b>Strike: %{{customdata[0]:.2f}}</b>"; col_idx = 1; vol_line = "<br><b>Volume:</b>"; vol_data_added = False
            for i, interval in enumerate(intervals):
                 vol_col = f"volmbs_{interval}"
                 if vol_col in vol_cols_for_hover: separator = " | " if vol_data_added else " "; vol_line += f"{separator}{interval}: %{{customdata[{col_idx}]:,.0f}}"; col_idx += 1; vol_data_added = True
            if vol_data_added: hovertemplate += vol_line
            val_line = "<br><b>Value:</b> "; val_data_added = False
            for i, interval in enumerate(intervals):
                 val_col = f"valuebs_{interval}"
                 if val_col in val_cols_for_hover: separator = " | " if val_data_added else " "; val_line += f"{separator}{interval}: $%{{customdata[{col_idx}]:,.0f}}"; col_idx += 1; val_data_added = True
            if val_data_added: hovertemplate += val_line
            hovertemplate += "<extra></extra>"
            chart_logger.debug(f"Generated Horizontal Hovertemplate: {hovertemplate}")
            num_bar_traces_added = 0
            for i, interval_label_bar in enumerate(intervals):
                 orig_vol_col_bar = f"volmbs_{interval_label_bar}"; norm_vol_col_bar = f"{orig_vol_col_bar}_norm"
                 if norm_vol_col_bar not in agg.columns: continue
                 interval_cfg = flow_custom_cfg.get(interval_label_bar, defaults_style); trace_name_bar = f"Net Vol ({interval_label_bar})"; vis_bar = trace_visibility.get(trace_name_bar,True)
                 pos_clr = interval_cfg.get("volume_positive_color", "#cccccc"); neg_clr = interval_cfg.get("volume_negative_color", "#777777"); opac = interval_cfg.get("volume_opacity", 0.7)
                 bar_colors = [pos_clr if v>=0 else neg_clr for v in agg[orig_vol_col_bar]]
                 fig.add_trace(go.Bar(x=agg[self.col_strike], y=agg[norm_vol_col_bar], name=trace_name_bar, marker_color=bar_colors, visible=vis_bar, customdata=customdata_array, hovertemplate=hovertemplate, hoverinfo="text", opacity=opac, offsetgroup=interval_label_bar))
                 num_bar_traces_added += 1
            chart_logger.info(f"Total BAR traces added to figure: {num_bar_traces_added}")
            num_value_traces_added = 0
            for i, interval_label_val_area in enumerate(intervals):
                 orig_val_col_area = f"valuebs_{interval_label_val_area}"; norm_val_col_area = f"{orig_val_col_area}_norm"
                 if norm_val_col_area not in agg.columns: continue
                 interval_cfg_area = flow_custom_cfg.get(interval_label_val_area, defaults_style); val_norm_series=agg[norm_val_col_area].fillna(0);
                 val_norm_pos=val_norm_series.where(val_norm_series>=0,0); val_norm_neg=val_norm_series.where(val_norm_series<0,0)
                 pos_fill = interval_cfg_area.get("value_positive_fill_color", "rgba(204,204,204,0.1)"); neg_fill = interval_cfg_area.get("value_negative_fill_color", "rgba(119,119,119,0.1)")
                 pos_line = interval_cfg_area.get("value_positive_line_color", "rgba(204,204,204,0.5)"); neg_line = interval_cfg_area.get("value_negative_line_color", "rgba(119,119,119,0.5)")
                 tnp=f"Net Val ({interval_label_val_area}) (+)"; vip=trace_visibility.get(tnp,'legendonly');
                 fig.add_trace(go.Scatter(x=agg[self.col_strike],y=val_norm_pos,name=tnp,mode='lines',line=dict(width=1,color=pos_line),fillcolor=pos_fill,fill='tozeroy',visible=vip,showlegend=True,hoverinfo='skip')); num_value_traces_added +=1
                 tnn=f"Net Val ({interval_label_val_area}) (-)"; vin=trace_visibility.get(tnn,'legendonly');
                 fig.add_trace(go.Scatter(x=agg[self.col_strike],y=val_norm_neg,name=tnn,mode='lines',line=dict(width=1,color=neg_line),fillcolor=neg_fill,fill='tozeroy',visible=vin,showlegend=True,hoverinfo='skip')); num_value_traces_added +=1
            chart_logger.info(f"Total VALUE area traces added to figure: {num_value_traces_added}")
            barmode_cfg = chart_params_cfg.get("combined_rolling_flow_chart_barmode", "group"); x_r=None; filter_suffix="";
            sel_pct = selected_price_range_pct_override if selected_price_range_pct_override is not None else chart_params_cfg.get("combined_flow_chart_price_range_pct", 10.0)
            if current_price is not None and pd.notna(current_price) and current_price>0 and sel_pct>0: min_x=current_price*(1-(sel_pct/100.0)); max_x=current_price*(1+(sel_pct/100.0)); x_r=[min_x, max_x]; filter_suffix=f" (+/- {sel_pct:.0f}%)"
            fig.update_layout(title=f"{chart_title}{filter_suffix}",xaxis_title='Strike', barmode=barmode_cfg, template=self.config.get("plotly_template","plotly_dark"), height=fig_height, legend=dict(orientation=leg_cfg.get("orientation","h"),yanchor=leg_cfg.get("y_anchor","bottom"),y=leg_cfg.get("y_pos",1.02),xanchor=leg_cfg.get("x_anchor","right"),x=leg_cfg.get("x_pos",1),traceorder=leg_cfg.get("trace_order","normal")), hovermode="x unified", xaxis_range=x_r)
            fig.update_xaxes(tickformat=".2f");
            fig.update_yaxes(
                title_text="Normalized Net Flow (Vol & Val)", zeroline=True, zerolinewidth=1, zerolinecolor='gray',
                range=[-1.1, 1.1], tickmode='linear', dtick=0.1, showgrid=True, gridcolor='rgba(255,255,255,0.1)',
                minor=dict(tickmode='linear', dtick=0.05, showgrid=True, gridcolor='rgba(255,255,255,0.05)', griddash='dot')
            );
            if current_price is not None and pd.notna(current_price): fig=self._add_price_line(fig,current_price,orientation='vertical')
            fig=self._add_timestamp_annotation(fig,fetch_timestamp); self._save_figure(fig,chart_name,symbol)
        except Exception as e: chart_logger.error(f"Error during {chart_name} creation: {e}",exc_info=True); return self._create_empty_figure(f"{symbol}-{chart_name}: Plot Error",height=fig_height, reason=str(e))
        chart_logger.info(f"Chart {chart_name} created successfully for {symbol}."); return fig

    def create_key_levels_visualization(self, key_levels_data: Dict[str, List[Dict]], symbol: str="N/A", current_price: Optional[float]=None, fetch_timestamp: Optional[str]=None, **kwargs) -> go.Figure:
        chart_name="Key Levels"; chart_title=f"<b>{symbol.upper()}</b> - {chart_name}";
        chart_logger = logging.getLogger(__name__ + "." + chart_name); chart_logger.info(f"Creating {chart_title}...")
        fig_height = self.config.get("chart_specific_params",{}).get("key_levels_height",600)
        try:
            if not isinstance(key_levels_data,dict) or not any(isinstance(v,list) and v for v in key_levels_data.values()): return self._create_empty_figure(f"{symbol}-{chart_name}: No Data", height=fig_height, reason="Input key_levels_data invalid/empty")
            all_lvls=[]; markers_cfg_from_config=self.config.get("key_level_markers", DEFAULT_VISUALIZER_CONFIG.get("key_level_markers", {}))
            for key, d_list in key_levels_data.items():
                mk_lkup=key.replace('_',' ').title(); mk_cfg=None; mk_name=mk_lkup
                for mk_cat,mk_conf_iter in markers_cfg_from_config.items():
                    if mk_cat.lower()==mk_lkup.lower(): mk_cfg=mk_conf_iter; mk_name=mk_conf_iter.get('name',mk_cat); break
                if not mk_cfg or not isinstance(d_list,list): chart_logger.warning(f"No marker/invalid data for level: {key}"); continue
                for rec in d_list:
                    if isinstance(rec,dict) and self.col_strike in rec:
                        rc=rec.copy(); rc[self.col_strike]=pd.to_numeric(rc.get(self.col_strike),errors='coerce');
                        if pd.notna(rc[self.col_strike]): rc['level_category']=mk_name; rc['level_type_original']=key; all_lvls.append(rc)
                    else: chart_logger.warning(f"Skip invalid level record '{key}': {rec}")
            if not all_lvls: return self._create_empty_figure(f"{symbol}-{chart_name}: No Valid Levels", height=fig_height, reason="No valid levels after processing")
            lvls_df=pd.DataFrame(all_lvls);
            overview_metrics_cfg = self.config.get("hover_settings", {}).get("overview_metrics_config", [])
            details_keys_cfg = self.config.get("hover_settings", {}).get("details_section_keys", [])
            hover_needed=[m_cfg.get("key") for m_cfg in overview_metrics_cfg if m_cfg.get("key")] + [k for k in details_keys_cfg if k!='level_category']
            for hc in list(set(hover_needed)):
                 if hc not in lvls_df.columns: lvls_df[hc]=np.nan
            fig=go.Figure(); plotted=False; y_cats_plotted = set();
            y_order_from_markers = [cfg.get("name",key) for key,cfg in markers_cfg_from_config.items()]
            unique_cats_in_data = lvls_df['level_category'].unique()
            ordered_cats_to_plot = [cat for cat in y_order_from_markers if cat in unique_cats_in_data]
            ordered_cats_to_plot.extend([cat for cat in unique_cats_in_data if cat not in ordered_cats_to_plot])
            for cat_name in ordered_cats_to_plot:
                style=None;
                for mk_cat_iter,mk_conf_iter in markers_cfg_from_config.items():
                     if mk_conf_iter.get('name',mk_cat_iter)==cat_name: style=mk_conf_iter; break
                if not style: chart_logger.warning(f"No style found for level category '{cat_name}'. Skipping."); continue
                subset=lvls_df[lvls_df['level_category']==cat_name].copy()
                if subset.empty: continue
                plotted=True; y_cats_plotted.add(cat_name)
                subset[self.col_mspi]=pd.to_numeric(subset.get(self.col_mspi),errors='coerce').fillna(0); subset['mspi_abs']=subset[self.col_mspi].abs()
                max_abs=subset['mspi_abs'].replace([0,np.inf,-np.inf],np.nan).max(); min_sz,max_sz=8,23;
                min_denom_local = self.config.get("min_normalization_denominator", 1e-9)
                sizes=min_sz+((subset['mspi_abs']/(max_abs if pd.notna(max_abs) and max_abs>min_denom_local else 1))*(max_sz-min_sz))
                sizes=pd.to_numeric(sizes,errors='coerce').fillna(min_sz).clip(lower=min_sz,upper=max_sz)
                hovers=[self._create_hover_text(r_dict, chart_type="key_levels") for r_dict in subset.to_dict('records')]
                fig.add_trace(go.Scatter( x=subset[self.col_strike], y=subset['level_category'], mode='markers', name=style['name'], marker=dict(symbol=style['symbol'], color=style['color'], size=sizes, opacity=0.85, line=dict(width=1, color='rgba(255,255,255,0.6)')), hovertext=hovers, hoverinfo='text'))
            if not plotted: return self._create_empty_figure(f"{symbol}-{chart_name}: No Levels Plotted", height=fig_height, reason="No levels to plot after filtering")
            final_y_cats = [cat for cat in ordered_cats_to_plot if cat in y_cats_plotted]
            legend_cfg=self.config.get("legend_settings",{});
            fig.update_layout( title=chart_title, xaxis_title="Strike", yaxis_title="Level Type", yaxis=dict(type='category', categoryorder='array', categoryarray=final_y_cats), template=self.config.get("plotly_template","plotly_dark"), height=fig_height, legend_title="Level Types", legend=dict(orientation=legend_cfg.get("orientation","v"),yanchor=legend_cfg.get("y_anchor","top"),y=legend_cfg.get("y_pos",1),xanchor=legend_cfg.get("x_anchor","left"),x=legend_cfg.get("x_pos",1.02)), hovermode='closest')
            fig.update_xaxes(tickformat=".2f"); fig=self._add_price_line(fig, current_price, orientation='vertical'); fig=self._add_timestamp_annotation(fig, fetch_timestamp); self._save_figure(fig, chart_name, symbol)
        except Exception as e: chart_logger.error(f"{chart_name} Error: {e}", exc_info=True); return self._create_empty_figure(f"{symbol}-{chart_name}: Plot Error", height=fig_height, reason=str(e))
        chart_logger.info(f"{chart_name} OK: {symbol}"); return fig

    def create_trading_signals_visualization(self, trading_signals_data: Dict[str, Dict[str, List[Dict]]], symbol: str="N/A", current_price: Optional[float]=None, fetch_timestamp: Optional[str]=None, **kwargs) -> go.Figure:
        chart_name="Trading Signals"; chart_title=f"<b>{symbol.upper()}</b> - {chart_name}";
        chart_logger = logging.getLogger(__name__ + "." + chart_name); chart_logger.info(f"Creating {chart_title}...")
        fig_height = self.config.get("chart_specific_params",{}).get("trading_signals_height",600)
        try:
            if not isinstance(trading_signals_data,dict) or not trading_signals_data: return self._create_empty_figure(f"{symbol}-{chart_name}: No Signal Data", height=fig_height, reason="Input trading_signals_data empty/invalid")
            fig=go.Figure(); plotted=False; y_cats_plotted = []
            cat_order=['directional','complex','time_decay','volatility'];
            styles_from_config=self.config.get("signal_styles", DEFAULT_VISUALIZER_CONFIG.get("signal_styles", {}))
            default_style_from_config=styles_from_config.get("default", DEFAULT_VISUALIZER_CONFIG.get("signal_styles", {}).get("default", {"color":"grey", "symbol":"circle"}))
            for sig_cat in cat_order:
                types_dict=trading_signals_data.get(sig_cat)
                if not isinstance(types_dict,dict): continue
                for sig_type,data_list in types_dict.items():
                    style=styles_from_config.get(sig_type, default_style_from_config);
                    if not isinstance(data_list,list) or not data_list: continue
                    try: df_sub=pd.DataFrame(data_list)
                    except Exception as df_err: chart_logger.warning(f"Signals({symbol}): DF fail {sig_cat}.{sig_type}: {df_err}"); continue
                    if self.col_strike not in df_sub.columns: continue
                    df_sub[self.col_strike]=pd.to_numeric(df_sub[self.col_strike],errors='coerce'); df_sub=df_sub.dropna(subset=[self.col_strike])
                    if df_sub.empty: continue
                    plotted=True; y_lbl=f"{sig_cat.replace('_',' ').title()}: {sig_type.replace('_',' ').title()}"
                    if y_lbl not in y_cats_plotted: y_cats_plotted.append(y_lbl)
                    overview_metrics_cfg = self.config.get("hover_settings", {}).get("overview_metrics_config", [])
                    details_keys_cfg = self.config.get("hover_settings", {}).get("details_section_keys", [])
                    hover_needed=[m_cfg.get("key") for m_cfg in overview_metrics_cfg if m_cfg.get("key")] + [k for k in details_keys_cfg if k!='type']
                    for hc in list(set(hover_needed)):
                        if hc not in df_sub.columns: df_sub[hc]=np.nan

                    # Use conviction_stars from signal payload for size/opacity if available
                    if 'conviction_stars' in df_sub.columns:
                        sizes = df_sub['conviction_stars'].apply(lambda x: 10 + x * 2 if isinstance(x, (int, float)) and pd.notna(x) else 10).fillna(10).clip(lower=8, upper=25)
                        opacities = df_sub['conviction_stars'].apply(lambda x: 0.6 + x * 0.08 if isinstance(x, (int, float)) and pd.notna(x) else 0.6).fillna(0.6).clip(lower=0.5, upper=1.0)
                    else: # Fallback if no conviction_stars in payload
                        sizes = pd.Series([12] * len(df_sub))
                        opacities = pd.Series([0.75] * len(df_sub))

                    hovers=[self._create_hover_text(r_dict,chart_type="trading_signals",extra_context={'Signal Type':sig_type.replace('_',' ').title()}) for r_dict in df_sub.to_dict('records')]
                    fig.add_trace(go.Scatter( x=df_sub[self.col_strike], y=[y_lbl]*len(df_sub), mode='markers', name=y_lbl, marker=dict(size=sizes,color=style.get('color','grey'),symbol=style.get('symbol','circle'),opacity=opacities,line=dict(width=1,color='rgba(255,255,255,0.6)')), hovertext=hovers, hoverinfo='text'))
            if not plotted: return self._create_empty_figure(f"{symbol}-{chart_name}: No Signals Plotted", height=fig_height, reason="No signals to plot after filtering")
            legend_cfg=self.config.get("legend_settings",{});
            fig.update_layout( title=chart_title, xaxis_title="Strike", yaxis_title="Signal Type", yaxis=dict(type='category', categoryorder='array', categoryarray=y_cats_plotted), template=self.config.get("plotly_template","plotly_dark"), height=fig_height, legend_title="Signal Types", legend=dict(orientation=legend_cfg.get("orientation","v"),yanchor=legend_cfg.get("y_anchor","top"),y=legend_cfg.get("y_pos",1),xanchor=legend_cfg.get("x_anchor","left"),x=legend_cfg.get("x_pos",1.02)), hovermode='closest')
            fig.update_xaxes(tickformat=".2f"); fig=self._add_price_line(fig, current_price, orientation='vertical'); fig=self._add_timestamp_annotation(fig, fetch_timestamp); self._save_figure(fig, chart_name, symbol)
        except Exception as e:
            chart_logger.error(f"{chart_name} Error: {e}", exc_info=True); return self._create_empty_figure(f"{symbol}-{chart_name}: Plot Error", height=fig_height, reason=str(e))
        chart_logger.info(f"{chart_name} OK: {symbol}"); return fig

    def create_strategy_recommendations_table(
        self,
        recommendations_list: List[Dict], # Changed from recommendations_df to accept flat list directly
        symbol: str = "N/A",
        fetch_timestamp: Optional[str] = None,
        **kwargs
    ) -> go.Figure:
        chart_name = "Strategy Insights Table"
        chart_title = f"<b>{symbol.upper()}</b> - {chart_name}"
        chart_logger = logging.getLogger(__name__ + "." + chart_name)
        chart_logger.info(f"Creating {chart_title}...")
        fig_height = self.config.get("chart_specific_params", {}).get("recommendations_table_height", 600)

        try:
            if not isinstance(recommendations_list, list) or not recommendations_list:
                return self._create_empty_figure(f"{symbol}-{chart_name}: No Recommendations Data", height=fig_height, reason="No recommendations provided or list is empty")

            recommendations_df = pd.DataFrame.from_records(recommendations_list)

            if recommendations_df.empty:
                 return self._create_empty_figure(f"{symbol}-{chart_name}: No Valid Recommendation Data", height=fig_height, reason="DataFrame empty after creation from list")

            default_display_map = self.config.get("chart_specific_params", {}).get("recommendations_table_column_display_map", {
                'Category': "Category", 'direction_label': "Bias/Type", 'strike': "Strike",
                'strategy': "Strategy / Note", 'conviction_stars': "Conv★",
                'raw_conviction_score': "Score", 'status': 'Status',
                'entry_ideal': "Entry", 'target_1': "T1", 'target_2': "T2", 'stop_loss': "SL",
                'rationale': "Rationale", 'target_rationale': "Tgt. Logic",
                'mspi': 'MSPI', 'sai': 'SAI', 'ssi': 'SSI', 'arfi': 'ARFI',
                'issued_ts': 'Issued', 'last_adjusted_ts': 'Adjusted', 'exit_reason':'Exit Info',
                'type': "Signal Src", 'id': "ID"
            })

            cols_to_display_ordered = [
                'id', 'Category', 'direction_label', 'strike', 'strategy',
                'conviction_stars', 'raw_conviction_score', 'status',
                'entry_ideal', 'target_1', 'target_2', 'stop_loss',
                'rationale', 'target_rationale',
                'mspi', 'sai', 'ssi', 'arfi',
                'issued_ts', 'last_adjusted_ts', 'exit_reason', 'type'
            ]

            actual_cols_to_display = [col for col in cols_to_display_ordered if col in recommendations_df.columns]

            if not actual_cols_to_display:
                return self._create_empty_figure(f"{symbol}-{chart_name}: No displayable columns in data", height=fig_height, reason="No configured columns found in recommendation data")

            df_display = recommendations_df[actual_cols_to_display].copy()

            if "strike" in df_display.columns:
                 df_display["strike"] = pd.to_numeric(df_display["strike"], errors='coerce').apply(lambda x: f"{x:.2f}" if pd.notna(x) and isinstance(x, (int,float)) else (str(x) if pd.notna(x) else "N/A"))
            for target_col in ['entry_ideal', 'target_1', 'target_2', 'stop_loss', 'mspi', 'sai', 'ssi', 'arfi', 'raw_conviction_score']:
                if target_col in df_display.columns:
                    df_display[target_col] = pd.to_numeric(df_display[target_col], errors='coerce').apply(lambda x: f"{x:.2f}" if pd.notna(x) else "---")

            if "conviction_stars" in df_display.columns:
                df_display["conviction_stars"] = pd.to_numeric(df_display["conviction_stars"], errors='coerce').fillna(0).astype(int).apply(lambda x: "★" * x + "☆" * (5 - x) if 0 <= x <= 5 else "N/A")

            for col in ['type', 'direction_label', 'Category', 'status', 'exit_reason', 'rationale', 'target_rationale', 'strategy']:
                if col in df_display.columns:
                    df_display[col] = df_display[col].astype(str).fillna("N/A")

            for ts_col in ['issued_ts', 'last_adjusted_ts']: # Original timestamp in 'timestamp'
                if ts_col == 'issued_ts' and 'timestamp' in recommendations_df.columns: # Use original timestamp for issued_ts
                     df_display[ts_col] = recommendations_df['timestamp'].apply(lambda x: datetime.fromisoformat(x).strftime('%H:%M:%S') if pd.notna(x) and isinstance(x, str) and x else "N/A")
                elif ts_col in df_display.columns: # For 'last_adjusted_ts' or if 'timestamp' is missing
                     df_display[ts_col] = df_display[ts_col].apply(lambda x: datetime.fromisoformat(x).strftime('%H:%M:%S') if pd.notna(x) and isinstance(x, str) and x else "N/A")


            table_header_values = [default_display_map.get(col, col.replace('_',' ').title()) for col in actual_cols_to_display]
            table_cells_values = [df_display[col_name].tolist() for col_name in actual_cols_to_display]

            font_colors = [(['#EAEAEA'] * len(df_display)) for _ in actual_cols_to_display]

            # Color 'direction_label'
            if 'direction_label' in actual_cols_to_display:
                dir_label_idx = actual_cols_to_display.index('direction_label')
                colors_for_dir_label_col = []
                for label_val_str in df_display['direction_label'].tolist():
                    label_val = str(label_val_str).lower()
                    color = '#EAEAEA' # Default
                    if 'bullish' in label_val: color = 'lime'
                    elif 'bearish' in label_val: color = 'red'
                    elif 'expansion' in label_val: color = 'cyan'
                    elif 'contraction' in label_val: color = '#DA70D6'
                    elif 'caution' in label_val: color = 'orange'
                    elif 'pin risk' in label_val: color = 'yellow'
                    elif 'neutral' in label_val: color = 'lightgrey'
                    colors_for_dir_label_col.append(color)
                font_colors[dir_label_idx] = colors_for_dir_label_col
            
            # Color 'conviction_stars'
            if 'conviction_stars' in actual_cols_to_display:
                conv_col_idx = actual_cols_to_display.index('conviction_stars')
                colors_for_conv_col = []
                for star_str in df_display['conviction_stars'].tolist():
                    num_stars = star_str.count("★")
                    if num_stars == 5: colors_for_conv_col.append('gold')
                    elif num_stars == 4: colors_for_conv_col.append('#FFD700')
                    elif num_stars == 3: colors_for_conv_col.append('lightskyblue')
                    elif num_stars == 2: colors_for_conv_col.append('lightgreen')
                    else: colors_for_conv_col.append('darkgrey')
                font_colors[conv_col_idx] = colors_for_conv_col

            # Color 'status'
            if 'status' in actual_cols_to_display:
                status_col_idx = actual_cols_to_display.index('status')
                colors_for_status_col = []
                
                # Iterate through the original recommendation dictionaries to get 'timestamp' and 'last_updated_ts'
                for rec_dict in recommendations_df.to_dict('records'):
                    status_val = str(rec_dict.get('status', '')).upper()
                    issued_ts_str = rec_dict.get('timestamp') # Original timestamp when rec was first generated
                    last_updated_ts_str = rec_dict.get('last_updated_ts') # Timestamp of last parameter adjustment

                    color = '#EAEAEA' # Default
                    if 'ACTIVE_NEW' in status_val or 'NEW' in status_val :
                        color = 'lightgreen'
                    elif 'ACTIVE' in status_val :
                         # Check if it was adjusted
                        if issued_ts_str and last_updated_ts_str and last_updated_ts_str != issued_ts_str:
                            color = 'skyblue' # Adjusted active
                        else:
                            color = '#66FF99' # Standard active
                    elif 'EXITED' in status_val:
                        color = 'orangered'
                    elif 'ADJUSTED' in status_val: # Explicit 'ADJUSTED' status
                        color = 'skyblue'
                    elif 'NOTE' in status_val:
                        color = 'lightgrey'
                    colors_for_status_col.append(color)
                font_colors[status_col_idx] = colors_for_status_col


            table = go.Table(
                header=dict(values=table_header_values, fill_color='rgb(30, 30, 30)', align='left', font=dict(color='white', size=13, family="Arial Black, sans-serif"), line_color='rgb(60,60,60)', height=40),
                cells=dict(values=table_cells_values, fill_color='rgb(45, 45, 45)', align='left', font=dict(color=font_colors, size=12, family="Arial, sans-serif"), line_color='rgb(60,60,60)', height=30)
            )
            fig = go.Figure(data=[table])
            fig.update_layout(title=chart_title, template=self.config.get("plotly_template","plotly_dark"), height=fig_height, margin=dict(l=15,r=15,t=70,b=20))
            self._save_figure(fig, chart_name, symbol) # Usually not saved for dashboard use
        except Exception as e:
            chart_logger.error(f"{chart_name} creation failed for {symbol}: {e}", exc_info=True)
            return self._create_empty_figure(f"{symbol} - {chart_name}: Table Creation Error", height=fig_height, reason=str(e))
        chart_logger.info(f"{chart_name} for {symbol} created successfully.")
        return fig
        
    def create_net_greek_flow_heatmap(
        self,
        processed_data: pd.DataFrame,
        metric_column_to_plot: str,
        chart_main_title_prefix: str,
        colorscale_config_key: str,
        colorbar_title_text: str,
        symbol: str = "N/A",
        fetch_timestamp: Optional[str] = None,
        current_price: Optional[float] = None,
        **kwargs
    ) -> go.Figure:
        chart_name = f"{chart_main_title_prefix.replace(' ', '_')}_Heatmap"
        full_chart_title = f"<b>{symbol.upper()}</b> - {chart_main_title_prefix}"
        chart_logger = self.instance_logger.getChild(chart_name)
        chart_logger.info(f"Creating {full_chart_title} using metric column: '{metric_column_to_plot}'...")

        fig_height = self.config.get("default_chart_height", 700)
        plotly_template = self.config.get("plotly_template", "plotly_dark")

        try:
            if not isinstance(processed_data, pd.DataFrame) or processed_data.empty:
                return self._create_empty_figure(f"{symbol}-{chart_name}: No Data", height=fig_height, reason="Input DataFrame empty/invalid")

            required_cols = [self.col_strike, metric_column_to_plot]
            if not all(col in processed_data.columns for col in required_cols):
                missing = [col for col in required_cols if col not in processed_data.columns]
                chart_logger.error(f"{chart_name}: Missing required columns: {missing}. Available: {list(processed_data.columns)}")
                return self._create_empty_figure(f"{symbol}-{chart_name}: Missing Data ({', '.join(missing)})", height=fig_height, reason=f"Missing columns: {', '.join(missing)}")

            df_viz = processed_data.copy()
            df_viz[self.col_strike] = pd.to_numeric(df_viz[self.col_strike], errors='coerce')
            df_viz[metric_column_to_plot] = pd.to_numeric(df_viz[metric_column_to_plot], errors='coerce')

            overview_metrics_cfg_list = self.config.get("hover_settings", {}).get("overview_metrics_config", [])
            column_names_map = self.config.get("column_names", {})
            agg_logic_heatmap: Dict[str, Any] = {}

            if metric_column_to_plot in df_viz.columns:
                 agg_logic_heatmap[metric_column_to_plot] = 'first'

            for cfg_key in [m_cfg.get("key") for m_cfg in overview_metrics_cfg_list if m_cfg.get("key")]:
                actual_df_col = column_names_map.get(cfg_key, cfg_key)
                if actual_df_col in df_viz.columns and actual_df_col != self.col_strike and actual_df_col not in agg_logic_heatmap:
                    agg_logic_heatmap[actual_df_col] = 'first'

            if self.col_strike in agg_logic_heatmap: del agg_logic_heatmap[self.col_strike]
            if not metric_column_to_plot in agg_logic_heatmap:
                 return self._create_empty_figure(f"{symbol}-{chart_name}: Aggregation Logic Error", height=fig_height, reason=f"Metric '{metric_column_to_plot}' missing from agg logic.")

            agg_data_for_heatmap = df_viz.groupby(self.col_strike, as_index=False).agg(agg_logic_heatmap)
            agg_data_for_heatmap = agg_data_for_heatmap.dropna(subset=[self.col_strike, metric_column_to_plot])

            # MODIFICATION: Sort strikes numerically descending (High strikes first)
            agg_data_for_heatmap = agg_data_for_heatmap.sort_values(by=self.col_strike, ascending=False)

            if agg_data_for_heatmap.empty:
                return self._create_empty_figure(f"{symbol}-{chart_name}: No Aggregated Data", height=fig_height, reason="Aggregated data for this heatmap is empty after cleaning.")

            # MODIFICATION: Create numerical indices for y-axis and corresponding text labels
            y_axis_tick_labels = agg_data_for_heatmap[self.col_strike].apply(lambda x: f"{x:.2f}").tolist() # e.g., ['595.00', '594.00', ..., '585.00']
            y_axis_tick_indices = list(range(len(y_axis_tick_labels))) # e.g., [0, 1, ..., N-1]

            z_values = agg_data_for_heatmap[[metric_column_to_plot]].values
            x_axis_label = [chart_main_title_prefix]

            chart_logger.debug(f"Heatmap y-axis tick labels (count: {len(y_axis_tick_labels)}): {y_axis_tick_labels[:3]}...{y_axis_tick_labels[-3:]}")
            chart_logger.debug(f"Heatmap z-values shape: {z_values.shape}, first 3 z-values: {z_values[:3].tolist()}")

            hover_texts_matrix = []
            # Loop based on the (now descending-strike sorted) agg_data_for_heatmap
            for _, row_data in agg_data_for_heatmap.iterrows():
                hover_extra_context = {
                    'metric_label': chart_main_title_prefix,
                    'metric_col_name': metric_column_to_plot,
                    'is_currency': "value" in metric_column_to_plot.lower() or "val" in metric_column_to_plot.lower(),
                    'precision': 0 if "volm" in metric_column_to_plot.lower() else 2
                }
                row_dict_for_hover = row_data.to_dict()
                hover_texts_matrix.append([self._create_hover_text(row_dict_for_hover, chart_type="net_greek_flow_heatmap", extra_context=hover_extra_context)])

            colorscale_value = self.config.get("colorscales", {}).get(colorscale_config_key, "RdBu")
            zmid_value = 0.0

            fig = go.Figure(data=[go.Heatmap(
                z=z_values,
                x=x_axis_label,
                y=y_axis_tick_indices, # MODIFICATION: Use numerical indices for y-data
                colorscale=colorscale_value,
                zmid=zmid_value,
                colorbar=dict(title=colorbar_title_text),
                hovertext=hover_texts_matrix,
                hoverinfo='text'
            )])

            fig.update_layout(
                title=full_chart_title,
                yaxis_title='Strike',
                yaxis=dict(
                    # MODIFICATION: Configure y-axis with explicit ticks and labels
                    tickmode='array',
                    tickvals=y_axis_tick_indices,
                    ticktext=y_axis_tick_labels, # ticktext[0] is now HighStrike
                    autorange='reversed', # This will display tickvals[0] (HighStrike) at the top
                    tickfont=dict(size=9)
                ),
                xaxis=dict(showticklabels=False, showline=False, zeroline=False, ticks=""),
                template=plotly_template,
                height=fig_height,
                autosize=False,
                margin=dict(l=70, r=50, t=80, b=50)
            )

            # REMOVED: fig.update_yaxes(nticks=...) as it's not ideal for categorical/array-ticked axes

            fig = self._add_timestamp_annotation(fig, fetch_timestamp)
            if current_price is not None and pd.notna(current_price) and current_price > 0:
                try:
                    # Map current_price to the closest y_axis_tick_index
                    numeric_strike_values_on_axis = [float(s_val) for s_val in y_axis_tick_labels] # these are already sorted high-to-low
                    closest_numeric_strike_on_axis = min(numeric_strike_values_on_axis, key=lambda s_val: abs(s_val - current_price))
                    # Find the index corresponding to this strike label
                    index_of_closest_strike = y_axis_tick_labels.index(f"{closest_numeric_strike_on_axis:.2f}")
                    y_index_for_line = y_axis_tick_indices[index_of_closest_strike]


                    if y_index_for_line is not None: # Check if a valid index was found
                        fig = self._add_price_line(
                            fig,
                            current_price=y_index_for_line, # Pass the numerical index for hline's y
                            orientation='horizontal',
                            annotation={'text': f"ATM ~{current_price:.2f}",
                                        'x':0.5, 'xref':"paper", 'xanchor':'center',
                                        'y': y_index_for_line, 'yref': 'y', 'yanchor':'bottom', # y refers to the data coord (index)
                                        'bgcolor':"rgba(50,50,50,0.7)", 'borderpad':2}
                        )
                        chart_logger.debug(f"Added ATM price line at y-index {y_index_for_line} (Strike ~{closest_numeric_strike_on_axis:.2f}) for current price {current_price:.2f}")
                    else:
                        chart_logger.warning(f"Could not find y-index for closest strike {closest_numeric_strike_on_axis} for price line in {chart_name}.")
                except ValueError:
                    chart_logger.warning(f"Could not convert strikes to numeric for price line on {chart_name}.")
                except Exception as e_price_line_heatmap:
                    chart_logger.error(f"Error adding price line/annotation to {chart_name}: {e_price_line_heatmap}", exc_info=True)

            self._save_figure(fig, chart_name, symbol)

        except Exception as e_greek_hm:
            chart_logger.error(f"Error during {chart_name} creation for {symbol} ({metric_column_to_plot}): {e_greek_hm}", exc_info=True)
            return self._create_empty_figure(f"{symbol}-{chart_name}: Plotting Error", height=fig_height, reason=str(e_greek_hm))

        chart_logger.info(f"Chart '{chart_name}' for {symbol} ({metric_column_to_plot}) created successfully.")
        return fig

    def create_elite_impact_score_chart(
        self,
        processed_data: pd.DataFrame, # Expects strike-level data from processor
        symbol: str = "N/A",
        current_price: Optional[float] = None,
        fetch_timestamp: Optional[str] = None,
        selected_price_range_pct_override: Optional[float] = None,
        **kwargs
    ) -> go.Figure:
        """
        Generates a bar chart for Elite Impact Scores, with bars categorized and colored
        by score magnitude, and opacity/vividness adjusted by signal strength.
        Uses multiple traces for an interactive legend. Sourced from config.
        """
        chart_name = "Elite Impact Score Chart"
        chart_title_part = "Elite Impact Score by Strike (Configurable)"
        full_chart_title = f"<b>{symbol.upper()}</b> - {chart_title_part}"
        chart_logger = self.instance_logger.getChild(chart_name)
        chart_logger.info(f"Creating {full_chart_title} for {symbol}.")

        fig_height = self.config.get("default_chart_height", 700) # Or a specific height for this chart
        plotly_template = self.config.get("plotly_template", "plotly_dark")

        # --- Configuration Loading ---
        # Default config structure for this chart, to be merged with main config
        DEFAULT_ELITE_SCORE_CHART_CONFIG_METHOD_LEVEL = {
            "score_categories": {
                "Strong Positive": {"min_score": 0.75, "base_color_rgb_str": "0,180,0", "legend_name": "Strong Pos (Score >= 0.75)"},
                "Moderate Positive": {"min_score": 0.25, "base_color_rgb_str": "144,238,144", "legend_name": "Mod Pos (0.25 <= Score < 0.75)"},
                "Weak Positive": {"min_score": 0.0, "base_color_rgb_str": "135,206,250", "legend_name": "Weak Pos (0.0 <= Score < 0.25)"}, # Mapped from Neutral Positive
                "Weak Negative": {"min_score": -0.25, "base_color_rgb_str": "255,165,0", "legend_name": "Weak Neg (-0.25 <= Score < 0.0)"}, # Mapped from Neutral Negative
                "Moderate Negative": {"min_score": -0.75, "base_color_rgb_str": "250,128,114", "legend_name": "Mod Neg (-0.75 <= Score < -0.25)"},
                "Strong Negative": {"min_score": -float('inf'), "base_color_rgb_str": "220,20,60", "legend_name": "Strong Neg (Score < -0.75)"}
            },
            "visual_adjustments": { # Renamed from opacity_signal_strength_scaling and color_vividness_signal_strength_scaling
                "opacity_by_signal_strength": {"min_opacity": 0.3, "max_opacity": 1.0},
                "vividness_by_signal_strength": {"pale_intensity_factor": 0.4, "target_pale_color_rgb": "220,220,220"} # Factor for base color, target for interpolation
            },
            "column_names": { # Added to make it self-contained if needed, though processor usually aligns
                "elite_score": "elite_impact_score",
                "signal_strength": "signal_strength"
                # 'prediction_confidence' is no longer directly used for primary visual encoding but can be in hover
            },
            "price_range_filter_default_pct": 10.0 # Default for this chart if not overridden
        }

        # Use _get_config_value to fetch 'elite_score_chart_config' from the main application config
        # The path should be relative to the root of the application config structure.
        # Example: if main config has {"charts": {"elite_score_settings": {...}}}, path is ["charts", "elite_score_settings"]
        # For this example, assuming it's at top level: ["elite_score_chart_config"]

        # First, try to get the whole 'elite_score_chart_config' block
        loaded_chart_config_block = self._get_config_value(["elite_score_chart_config"], default_override=None)
        chart_logger.info(f"EliteScoreChart DEBUG: Raw visual_settings from config: {loaded_chart_config_block}") # Renamed visual_settings to loaded_chart_config_block for clarity before merge
        # Log the specific positive color strings from config
        if isinstance(loaded_chart_config_block, dict): # Check if it's a dict before .get
            chart_logger.info(f"EliteScoreChart DEBUG: Config strong_positive_color_rgb: {loaded_chart_config_block.get('score_categories', {}).get('Strong Positive', {}).get('base_color_rgba')}")
            chart_logger.info(f"EliteScoreChart DEBUG: Config moderate_positive_color_rgb: {loaded_chart_config_block.get('score_categories', {}).get('Moderate Positive', {}).get('base_color_rgba')}")
            chart_logger.info(f"EliteScoreChart DEBUG: Config neutral_positive_color_rgb: {loaded_chart_config_block.get('score_categories', {}).get('Weak Positive', {}).get('base_color_rgba')}") # Assuming Weak Positive is the neutral positive for >0 scores
        else:
            chart_logger.info("EliteScoreChart DEBUG: loaded_chart_config_block is not a dict, cannot log specific color strings from it.")


        # Deep merge the loaded config with the method-level default
        if isinstance(loaded_chart_config_block, dict):
            final_chart_config = self._deep_merge_dicts(DEFAULT_ELITE_SCORE_CHART_CONFIG_METHOD_LEVEL.copy(), loaded_chart_config_block)
            chart_logger.info("Successfully merged 'elite_score_chart_config' from application config with method defaults.")
        else:
            final_chart_config = DEFAULT_ELITE_SCORE_CHART_CONFIG_METHOD_LEVEL.copy()
            chart_logger.info("Using method-level default 'elite_score_chart_config' as it was not found or invalid in application config.")

        SCORE_CATEGORIES = final_chart_config.get("score_categories", DEFAULT_ELITE_SCORE_CHART_CONFIG_METHOD_LEVEL["score_categories"])
        VISUAL_ADJUSTMENTS = final_chart_config.get("visual_adjustments", DEFAULT_ELITE_SCORE_CHART_CONFIG_METHOD_LEVEL["visual_adjustments"])
        OPACITY_SETTINGS = VISUAL_ADJUSTMENTS.get("opacity_by_signal_strength", DEFAULT_ELITE_SCORE_CHART_CONFIG_METHOD_LEVEL["visual_adjustments"]["opacity_by_signal_strength"])
        VIVIDNESS_SETTINGS = VISUAL_ADJUSTMENTS.get("vividness_by_signal_strength", DEFAULT_ELITE_SCORE_CHART_CONFIG_METHOD_LEVEL["visual_adjustments"]["vividness_by_signal_strength"])
        CHART_COL_NAMES = final_chart_config.get("column_names", DEFAULT_ELITE_SCORE_CHART_CONFIG_METHOD_LEVEL["column_names"])
        PRICE_FILTER_DEFAULT_PCT = final_chart_config.get("price_range_filter_default_pct", DEFAULT_ELITE_SCORE_CHART_CONFIG_METHOD_LEVEL["price_range_filter_default_pct"])

        col_elite_score = CHART_COL_NAMES.get("elite_score", "elite_impact_score")
        col_signal_strength = CHART_COL_NAMES.get("signal_strength", "signal_strength")
        # self.col_strike is already defined in __init__

        # Log parsed RGB tuples (moved slightly down after SCORE_CATEGORIES is fully defined by merge)
        # This requires SCORE_CATEGORIES to be defined from final_chart_config first.
        # Let's log them after SCORE_CATEGORIES, OPACITY_SETTINGS, etc. are set.
        chart_logger.info(f"EliteScoreChart DEBUG: Final SCORE_CATEGORIES after merge: {SCORE_CATEGORIES}")
        chart_logger.info(f"EliteScoreChart DEBUG: Parsed STRONG_POS_VIVID_GREEN_RGB (from Strong Positive category): {SCORE_CATEGORIES.get('Strong Positive', {}).get('base_color_rgba')}")
        chart_logger.info(f"EliteScoreChart DEBUG: Parsed MODERATE_POS_PALE_GREEN_RGB (from Moderate Positive category): {SCORE_CATEGORIES.get('Moderate Positive', {}).get('base_color_rgba')}")
        chart_logger.info(f"EliteScoreChart DEBUG: Parsed NEUTRAL_POS_BLUE_RGB (from Weak Positive category): {SCORE_CATEGORIES.get('Weak Positive', {}).get('base_color_rgba')}")
        chart_logger.info(f"EliteScoreChart DEBUG: Parsed NEUTRAL_NEG_ORANGE_RGB (from Weak Negative category): {SCORE_CATEGORIES.get('Weak Negative', {}).get('base_color_rgba')}")
        chart_logger.info(f"EliteScoreChart DEBUG: Parsed MODERATE_NEG_PALE_RED_RGB (from Moderate Negative category): {SCORE_CATEGORIES.get('Moderate Negative', {}).get('base_color_rgba')}")
        chart_logger.info(f"EliteScoreChart DEBUG: Parsed STRONG_NEG_VIVID_RED_RGB (from Strong Negative category): {SCORE_CATEGORIES.get('Strong Negative', {}).get('base_color_rgba')}")

        try:
            if not isinstance(processed_data, pd.DataFrame) or processed_data.empty:
                return self._create_empty_figure(f"{symbol} - {chart_name}: No Data Provided", height=fig_height, reason="Input DataFrame empty/invalid")

            # Ensure required columns exist (strike from self, score & strength from config)
            # Note: 'prediction_confidence' is no longer essential for core visuals but useful for hover
            required_cols_for_chart = [self.col_strike, col_elite_score, col_signal_strength]
            if 'prediction_confidence' in processed_data.columns: # Optional for hover
                required_cols_for_chart.append('prediction_confidence')

            df_cleaned, _ = self._ensure_columns(processed_data.copy(), required_cols_for_chart, chart_name)

            if not all(c in df_cleaned.columns for c in [self.col_strike, col_elite_score]):
                missing = [c for c in [self.col_strike, col_elite_score] if c not in df_cleaned.columns]
                chart_logger.error(f"Missing required columns for {chart_name}: {missing}.")
                return self._create_empty_figure(f"{symbol} - {chart_name}: Missing Core Data ({', '.join(missing)})", height=fig_height, reason=f"Missing: {', '.join(missing)}")

            df = df_cleaned
            df['strike_numeric'] = pd.to_numeric(df[self.col_strike], errors='coerce')
            df[col_elite_score] = pd.to_numeric(df[col_elite_score], errors='coerce')

            # Handle signal_strength: fillna with 0, then clip to [0,1] after min-max scaling
            if col_signal_strength not in df.columns:
                chart_logger.warning(f"'{col_signal_strength}' column missing. Using 0.5 for all signal strengths.")
                df[col_signal_strength] = 0.5 # Default if missing
            df[col_signal_strength] = pd.to_numeric(df[col_signal_strength], errors='coerce').fillna(0.0) # Fill NA with 0 before scaling

            df = df.dropna(subset=['strike_numeric', col_elite_score])
            if df.empty:
                return self._create_empty_figure(f"{symbol} - {chart_name}: No Valid Data After Cleaning", height=fig_height, reason="No valid data after initial cleaning")

            # Aggregate scores per strike if multiple options per strike (usually not the case for elite scores)
            agg_functions: Dict[str, Any] = {col_elite_score: 'mean', col_signal_strength: 'mean'}
            if 'prediction_confidence' in df.columns: agg_functions['prediction_confidence'] = 'mean'

            # Add other hover context columns to agg_functions (from general hover config)
            overview_metrics_cfg = self.config.get("hover_settings", {}).get("overview_metrics_config", [])
            app_col_names_map = self.config.get("column_names", {}) # App-level column name mapping
            for m_cfg in overview_metrics_cfg:
                cfg_key = m_cfg.get("key")
                if cfg_key:
                    actual_col_for_hover = app_col_names_map.get(cfg_key, cfg_key)
                    if actual_col_for_hover in df.columns and actual_col_for_hover not in agg_functions and actual_col_for_hover != 'strike_numeric':
                        agg_functions[actual_col_for_hover] = 'first'
            if self.col_opt_kind in df.columns and self.col_opt_kind not in agg_functions:
                 agg_functions[self.col_opt_kind] = 'first' # For hover context

            agg_data = df.groupby('strike_numeric', as_index=False).agg(agg_functions)

            # Scale signal_strength to [0, 1] for opacity and color intensity adjustments
            min_raw_strength = agg_data[col_signal_strength].min()
            max_raw_strength = agg_data[col_signal_strength].max()
            if max_raw_strength == min_raw_strength: # Avoid division by zero
                agg_data['signal_strength_scaled'] = 0.5 if max_raw_strength is not None else 0.0
            else:
                agg_data['signal_strength_scaled'] = (agg_data[col_signal_strength] - min_raw_strength) / (max_raw_strength - min_raw_strength)
            agg_data['signal_strength_scaled'] = agg_data['signal_strength_scaled'].fillna(0.0).clip(0.0, 1.0)
            chart_logger.debug(f"Elite Impact Score chart: First 5 scaled signal_strengths: {agg_data['signal_strength_scaled'].head().tolist()}")


            # --- Price Range Filtering ---
            filtered_df = agg_data.copy()
            filter_suffix = ""
            selected_pct_to_use = selected_price_range_pct_override if isinstance(selected_price_range_pct_override, (int, float)) and pd.notna(selected_price_range_pct_override) and selected_price_range_pct_override > 0 else PRICE_FILTER_DEFAULT_PCT

            if current_price is not None and pd.notna(current_price) and current_price > 0:
                min_s = current_price * (1 - (selected_pct_to_use / 100.0))
                max_s = current_price * (1 + (selected_pct_to_use / 100.0))
                filtered_df = agg_data[(agg_data['strike_numeric'] >= min_s) & (agg_data['strike_numeric'] <= max_s)].copy()
                filter_suffix = f" (Strikes +/- {selected_pct_to_use:.1f}%)"
                if filtered_df.empty: chart_logger.warning(f"{chart_name}: DataFrame empty after price range filter for {symbol}.")

            if filtered_df.empty:
                 return self._create_empty_figure(f"{symbol} - {chart_name}: No Data in Selected Range {filter_suffix}", height=fig_height, reason=f"No data for range {filter_suffix}")

            unique_strikes_desc = sorted(filtered_df['strike_numeric'].dropna().unique(), reverse=True)
            if not unique_strikes_desc:
                return self._create_empty_figure(f"{symbol} - {chart_name}: No Unique Strikes After Filtering", height=fig_height, reason="No strikes left for plot")

            plot_df = filtered_df.set_index('strike_numeric').reindex(unique_strikes_desc).reset_index()
            chart_logger.info(f"EliteScoreChart: plot_df head for coloring (first 3 rows):\n{plot_df[[col_elite_score, 'signal_strength_scaled']].head(3).to_string()}")

            # --- Categorize scores ---
            def get_score_category(score_val):
                # Iterate in defined order (e.g., strong pos to strong neg)
                # Assumes SCORE_CATEGORIES is ordered appropriately if iteration order matters for assignment
                # For multiple matches (e.g. score = 0.0 might match "Weak Positive" min_score:0.0 and "Weak Negative" min_score:-0.25 if logic is score >= min_score)
                # The first category in SCORE_CATEGORIES that satisfies score >= min_score will be chosen.
                # Consider sorting SCORE_CATEGORIES by min_score descending to ensure correct categorization.

                # Sort categories by min_score descending to ensure correct bucketing for "greater than or equal to" logic
                sorted_categories = sorted(SCORE_CATEGORIES.items(), key=lambda item: item[1]['min_score'], reverse=True)

                for cat_name, props in sorted_categories:
                    if score_val >= props["min_score"]:
                        return cat_name
                # Fallback if no category matches (should ideally not happen with a -inf category)
                return list(SCORE_CATEGORIES.keys())[-1] if SCORE_CATEGORIES else "Uncategorized"


            plot_df['score_category'] = plot_df[col_elite_score].apply(get_score_category)
            chart_logger.debug(f"Elite Impact Score chart: Value counts for score_category:\n{plot_df['score_category'].value_counts()}")

            # --- Helper function for color and opacity adjustments ---
            # This function is now more of a wrapper since the base color is selected before calling it in the loop.
            # However, the actual color string parsing and adjustments are still done here.
            def get_adjusted_color_for_bar(base_rgb_tuple_param: Tuple[int,int,int], s_strength_scaled: float, category_name_for_debug: str) -> str:
                # category_props = SCORE_CATEGORIES.get(category_name) # Not needed if base_rgb_tuple is passed directly
                # if not category_props: return "rgba(128,128,128,0.5)"

                # base_color_str = category_props.get("base_color_rgba", "rgba(128,128,128,1)") # Base color already selected
                r_base, g_base, b_base = base_rgb_tuple_param # Unpack the pre-selected base RGB tuple

                # try: # Parse base RGBA string (e.g., "rgba(0,100,0,1)")
                #     parsed_color = self._parse_color_string(base_color_str, default_opacity=1.0) # Use existing helper
                #     r_base, g_base, b_base, a_base_original = map(float, parsed_color.strip('rgba()').split(','))
                # except Exception as e_parse:
                #     chart_logger.error(f"Could not parse base_color_rgba: '{base_color_str}' for category '{category_name}'. Error: {e_parse}. Using fallback black.")
                #     r_base, g_base, b_base, a_base_original = 0,0,0,1

                # 1. Adjust opacity based on signal_strength_scaled
                min_op = OPACITY_SETTINGS.get("min_opacity", 0.3)
                max_op = OPACITY_SETTINGS.get("max_opacity", 1.0)
                # Opacity: High strength (1.0) => max_op; Low strength (0.0) => min_op
                final_opacity = min_op + (max_op - min_op) * s_strength_scaled
                final_opacity = max(0.0, min(1.0, final_opacity))

                # 2. Adjust color vividness based on signal_strength_scaled
                pale_intensity_factor = VIVIDNESS_SETTINGS.get("pale_intensity_factor", 0.4) # How much of original color at low strength
                target_pale_rgb_str = VIVIDNESS_SETTINGS.get("target_pale_color_rgb", "220,220,220") # Target for paleness
                try:
                    r_pale_target, g_pale_target, b_pale_target = map(int, target_pale_rgb_str.split(','))
                except ValueError:
                    chart_logger.warning(f"Invalid target_pale_color_rgb '{target_pale_rgb_str}'. Using (220,220,220).")
                    r_pale_target, g_pale_target, b_pale_target = 220,220,220

                # Effective strength for vividness: interp_strength goes from pale_intensity_factor (at strength_scaled=0) to 1.0 (at strength_scaled=1)
                # This means at strength_scaled=0, color is pale_intensity_factor * base + (1-pale_intensity_factor) * pale_target
                # At strength_scaled=1, color is 1.0 * base + 0 * pale_target = base
                vividness_interp_factor = pale_intensity_factor + (1.0 - pale_intensity_factor) * s_strength_scaled

                r_final_adj = int(r_base * vividness_interp_factor + r_pale_target * (1 - vividness_interp_factor))
                g_final_adj = int(g_base * vividness_interp_factor + g_pale_target * (1 - vividness_interp_factor))
                b_final_adj = int(b_base * vividness_interp_factor + b_pale_target * (1 - vividness_interp_factor))

                r_final_adj, g_final_adj, b_final_adj = max(0,min(255,r_final_adj)), max(0,min(255,g_final_adj)), max(0,min(255,b_final_adj))
                return f'rgba({r_final_adj},{g_final_adj},{b_final_adj},{final_opacity:.2f})'

            # --- Create Figure and Traces ---
            fig = go.Figure()
            y_indices = list(range(len(unique_strikes_desc)))
            y_labels = [f"{s:.2f}" for s in unique_strikes_desc]
            strike_map_for_hline = {float(label): index for index, label in zip(y_indices, y_labels)}

            # Iterate through categories to add traces - ensures legend order from config if desired
            # Or, iterate unique categories in plot_df['score_category'] for efficiency if exact legend order isn't critical

            # Sort categories by min_score descending for legend order (Strong Pos at top)
            sorted_legend_categories = sorted(SCORE_CATEGORIES.items(), key=lambda item: item[1]['min_score'], reverse=True)

            # Pre-calculate all bar colors and details for logging before creating traces
            # This is a change from the previous structure to facilitate the detailed logging requested.
            all_bars_details = []
            for i in range(len(plot_df)):
                score = plot_df[col_elite_score].iloc[i]
                s_strength = plot_df['signal_strength_scaled'].iloc[i]
                current_category_name = plot_df['score_category'].iloc[i] # Category determined earlier

                # Determine base RGB from SCORE_CATEGORIES based on current_category_name
                # This part is slightly redundant with get_score_category but needed for direct RGB tuple access
                # And for logging the *chosen* base RGB before adjustments.

                base_rgb_tuple_for_this_bar = (128, 128, 128) # Default grey
                log_category_descriptor = "Unknown/DefaultGrey"

                # Simplified: Use the already determined 'current_category_name' to fetch base color string
                # Then parse it to an RGB tuple.
                category_props_for_bar = SCORE_CATEGORIES.get(current_category_name)
                if category_props_for_bar:
                    base_color_rgb_str_val = category_props_for_bar.get("base_color_rgb_str", "128,128,128") # Get new key, new default format
                    try:
                        # Direct parsing for "R,G,B"
                        rgb_parts = list(map(int, base_color_rgb_str_val.split(',')))
                        if len(rgb_parts) == 3: # Ensure it's R,G,B
                            base_rgb_tuple_for_this_bar = (rgb_parts[0], rgb_parts[1], rgb_parts[2])
                            log_category_descriptor = current_category_name
                        else:
                            chart_logger.error(f"Invalid RGB string format '{base_color_rgb_str_val}' for category '{current_category_name}'. Expected 3 parts.")
                            # base_rgb_tuple_for_this_bar remains grey
                    except Exception as e_base_parse:
                        chart_logger.error(f"Error parsing base_color_rgb_str '{base_color_rgb_str_val}' for category '{current_category_name}': {e_base_parse}")
                        # base_rgb_tuple_for_this_bar remains grey
                else:
                    chart_logger.warning(f"Category '{current_category_name}' not found in SCORE_CATEGORIES for bar {i}. Using default grey.")

                # Now, call get_adjusted_color_for_bar with the determined base_rgb_tuple and strength
                final_rgba_str = get_adjusted_color_for_bar(base_rgb_tuple_for_this_bar, s_strength, current_category_name)

                all_bars_details.append({
                    'strike_numeric': plot_df['strike_numeric'].iloc[i],
                    'score': score,
                    'signal_strength_scaled': s_strength,
                    'category_name': current_category_name,
                    'base_rgb_selected': base_rgb_tuple_for_this_bar,
                    'final_rgba': final_rgba_str,
                    'y_index': strike_map_for_hline.get(plot_df['strike_numeric'].iloc[i]) # Get y_index for this bar
                })

                if i < 5: # Log for first 5 bars
                    # For logging interpolated RGB, we need to parse final_rgba_str or re-calculate parts of get_adjusted_color_for_bar here.
                    # For simplicity, the detailed log will now show the final RGBA and the base RGB selected.
                    # The internal workings of get_adjusted_color_for_bar (interpolation) are implicitly tested.
                    chart_logger.info(
                        f"EliteScoreChart VisualDEBUG Bar {i} (Strike: {plot_df['strike_numeric'].iloc[i]:.2f}): "
                        f"Score={score:.4f}, SignalStr={s_strength:.2f} -> Category='{current_category_name}', BaseRGBSelected={base_rgb_tuple_for_this_bar} -> "
                        # InterpolatedRGB parts are inside get_adjusted_color_for_bar
                        f"FinalRGBA='{final_rgba_str}'"
                    )

            for category_name_legend, category_props_legend in sorted_legend_categories:
                # Filter `all_bars_details` for the current category to build the trace
                bars_for_this_category_trace = [bar_detail for bar_detail in all_bars_details if bar_detail['category_name'] == category_name_legend]

                if not bars_for_this_category_trace:
                    # For dummy traces (when a category has no bars but we want it in the legend)
                    # Use its configured base color, but make it semi-transparent.
                    base_rgb_str_dummy = category_props_legend.get("base_color_rgb_str", "128,128,128")
                    dummy_marker_color = "rgba(128,128,128,0.5)" # Default fallback
                    try:
                        r_dummy, g_dummy, b_dummy = map(int, base_rgb_str_dummy.split(','))
                        dummy_marker_color = f'rgba({r_dummy},{g_dummy},{b_dummy},0.5)'
                    except ValueError as e_dummy_parse: # More specific exception
                        chart_logger.warning(f"Invalid RGB string format or content in '{base_rgb_str_dummy}' for dummy trace '{category_name_legend}': {e_dummy_parse}. Using fallback.")
                    except Exception as e_dummy_generic: # Catch other unexpected errors
                        chart_logger.warning(f"Generic error parsing base_color_rgb_str '{base_rgb_str_dummy}' for dummy trace '{category_name_legend}': {e_dummy_generic}. Using fallback.")

                    fig.add_trace(go.Bar(
                        y=[None], x=[None], name=category_props_legend.get("legend_name", category_name_legend),
                        orientation='h', marker_color=dummy_marker_color,
                        legendgroup=category_name_legend
                    ))
                    continue

                # Extract data for the trace
                trace_y_indices = [bar['y_index'] for bar in bars_for_this_category_trace]
                trace_x_scores = [bar['score'] for bar in bars_for_this_category_trace]
                trace_marker_colors = [bar['final_rgba'] for bar in bars_for_this_category_trace]

                # Create hover data for this trace's bars
                # Need to reconstruct a "row-like" dict for _create_hover_text for each bar in this trace
                # This means fetching the full row from plot_df that corresponds to each bar in bars_for_this_category_trace
                original_rows_for_trace_hover = []
                for bar_detail_hover in bars_for_this_category_trace:
                    # Find the original row in plot_df that matches this bar's strike_numeric
                    # This is a bit inefficient but necessary if _create_hover_text needs many columns
                    matching_rows = plot_df[plot_df['strike_numeric'] == bar_detail_hover['strike_numeric']]
                    if not matching_rows.empty:
                        original_rows_for_trace_hover.append(matching_rows.iloc[0].to_dict())
                    else: # Should not happen if logic is correct
                        original_rows_for_trace_hover.append({'strike_numeric': bar_detail_hover['strike_numeric'], col_elite_score: bar_detail_hover['score']})


                trace_hovers = [self._create_hover_text(r, chart_type="elite_score_display") for r in original_rows_for_trace_hover]
                # Customdata can be simplified if hovertext is pre-generated and has all info.
                # Or, pass specific fields needed by a potentially simplified hovertemplate if not using _create_hover_text directly per bar.
                # For now, let's assume _create_hover_text is robust. We need to provide it with enough data.
                # The `customdata` field in go.Bar is often used with a `hovertemplate` string directly in `fig.add_trace`.
                # If `hovertext` is used, `customdata` might be redundant unless hovertemplate is also constructed.
                # The current _create_hover_text generates the full HTML-like string.

                fig.add_trace(go.Bar(
                    y=trace_y_indices,
                    x=trace_x_scores,
                    name=category_props_legend.get("legend_name", category_name_legend),
                    orientation='h',
                    marker_color=trace_marker_colors,
                    # customdata=... # If needed for a specific hovertemplate string
                    hovertext=trace_hovers,
                    hoverinfo='text', # Indicates that `hovertext` provides the content
                    legendgroup=category_name_legend
                ))

            # --- Layout and Final Touches ---
            legend_cfg = self.config.get("legend_settings", {}) # General legend settings from main config
            fig.update_layout(
                title=f"{full_chart_title}{filter_suffix}",
                    # This is now done in the main loop before calling this helper.
                    # r_base, g_base, b_base are passed as base_rgb_tuple_param.
                    pass # Original parsing logic removed as base RGB tuple is now an argument.

                # 1. Adjust opacity based on signal_strength_scaled
                min_op = OPACITY_SETTINGS.get("min_opacity", 0.3)
                max_op = OPACITY_SETTINGS.get("max_opacity", 1.0)
                # Opacity: High strength (1.0) => max_op; Low strength (0.0) => min_op
                final_opacity = min_op + (max_op - min_op) * strength_scaled
                final_opacity = max(0.0, min(1.0, final_opacity))

                # 2. Adjust color vividness based on signal_strength_scaled
                pale_intensity_factor = VIVIDNESS_SETTINGS.get("pale_intensity_factor", 0.4) # How much of original color at low strength
                target_pale_rgb_str = VIVIDNESS_SETTINGS.get("target_pale_color_rgb", "220,220,220") # Target for paleness
                try:
                    r_pale_target, g_pale_target, b_pale_target = map(int, target_pale_rgb_str.split(','))
                except ValueError:
                    chart_logger.warning(f"Invalid target_pale_color_rgb '{target_pale_rgb_str}'. Using (220,220,220).")
                    r_pale_target, g_pale_target, b_pale_target = 220,220,220

                # Effective strength for vividness: interp_strength goes from pale_intensity_factor (at strength_scaled=0) to 1.0 (at strength_scaled=1)
                # This means at strength_scaled=0, color is pale_intensity_factor * base + (1-pale_intensity_factor) * pale_target
                # At strength_scaled=1, color is 1.0 * base + 0 * pale_target = base
                vividness_interp_factor = pale_intensity_factor + (1.0 - pale_intensity_factor) * strength_scaled

                r_final_adj = int(r_base * vividness_interp_factor + r_pale_target * (1 - vividness_interp_factor))
                g_final_adj = int(g_base * vividness_interp_factor + g_pale_target * (1 - vividness_interp_factor))
                b_final_adj = int(b_base * vividness_interp_factor + b_pale_target * (1 - vividness_interp_factor))

                r_final_adj, g_final_adj, b_final_adj = max(0,min(255,r_final_adj)), max(0,min(255,g_final_adj)), max(0,min(255,b_final_adj))
                return f'rgba({r_final_adj},{g_final_adj},{b_final_adj},{final_opacity:.2f})'

            # --- Create Figure and Traces ---
            # The main loop for processing bars for logging and color calculation is now above this section.
            # This section now focuses on creating traces from `all_bars_details`.

            fig = go.Figure() # Already initialized earlier if this part is moved

            for category_name_legend, category_props_legend in sorted_legend_categories:
                # Filter `all_bars_details` for the current category to build the trace
                bars_for_this_category_trace = [bar_detail for bar_detail in all_bars_details if bar_detail['category_name'] == category_name_legend]

                if not bars_for_this_category_trace:
                    fig.add_trace(go.Bar(
                        y=[None], x=[None], name=category_props_legend.get("legend_name", category_name_legend),
                        orientation='h', marker_color=category_props_legend.get("base_color_rgba","rgba(128,128,128,0.5)"),
                        legendgroup=category_name_legend
                    ))
                    continue

                # Extract data for the trace
                trace_y_indices = [bar['y_index'] for bar in bars_for_this_category_trace]
                trace_x_scores = [bar['score'] for bar in bars_for_this_category_trace]
                trace_marker_colors = [bar['final_rgba'] for bar in bars_for_this_category_trace]

                original_rows_for_trace_hover = []
                for bar_detail_hover in bars_for_this_category_trace:
                    matching_rows = plot_df[plot_df['strike_numeric'] == bar_detail_hover['strike_numeric']]
                    if not matching_rows.empty:
                        original_rows_for_trace_hover.append(matching_rows.iloc[0].to_dict())
                    else:
                        original_rows_for_trace_hover.append({'strike_numeric': bar_detail_hover['strike_numeric'], col_elite_score: bar_detail_hover['score']})

                trace_hovers = [self._create_hover_text(r, chart_type="elite_score_display") for r in original_rows_for_trace_hover]

                fig.add_trace(go.Bar(
                    y=trace_y_indices,
                    x=trace_x_scores,
                    name=category_props_legend.get("legend_name", category_name_legend),
                    orientation='h',
                    marker_color=trace_marker_colors,
                    hovertext=trace_hovers,
                    hoverinfo='text',
                    legendgroup=category_name_legend
                ))

            # --- Layout and Final Touches ---
            legend_cfg = self.config.get("legend_settings", {}) # General legend settings from main config
            fig.update_layout(
                title=f"{full_chart_title}{filter_suffix}",
                yaxis_title="Strike",
                xaxis_title="Elite Impact Score",
                barmode='relative', # Scores are mutually exclusive per strike, 'relative' or 'stack' works. 'relative' is like 'overlay' for single bars.
                template=plotly_template,
                height=fig_height,
                yaxis=dict(tickmode='array', tickvals=y_indices, ticktext=y_labels, autorange='reversed'), # High strikes at top
                xaxis=dict(zeroline=True, zerolinewidth=1.5, zerolinecolor='lightgrey'),
                legend=dict(
                    title_text="Score Categories",
                    orientation=legend_cfg.get("orientation", "v"), yanchor=legend_cfg.get("y_anchor", "top"),
                    y=legend_cfg.get("y_pos", 1), xanchor=legend_cfg.get("x_anchor", "left"),
                    x=legend_cfg.get("x_pos", 1.02), traceorder=legend_cfg.get("trace_order", "reversed")
                ),
                hovermode="y unified" # Shows hover for all traces at a given y (strike)
            )

            if current_price is not None and pd.notna(current_price) and current_price > 0:
                if unique_strikes_desc: # Ensure there are strikes to find closest to
                    closest_plotted_strike_numeric = min(unique_strikes_desc, key=lambda s: abs(s - current_price))
                    y_price_idx_for_line = strike_map_for_hline.get(float(f"{closest_plotted_strike_numeric:.2f}")) # Use formatted key
                    if y_price_idx_for_line is not None:
                        fig = self._add_price_line(fig, current_price=y_price_idx_for_line, orientation='horizontal',
                                               annotation={'text': f"Current: {current_price:.2f}"})

            fig = self._add_timestamp_annotation(fig, fetch_timestamp)
            self._save_figure(fig, chart_name, symbol) # Save if configured

        except Exception as e:
            chart_logger.error(f"{chart_name} ({symbol}) creation failed: {e}", exc_info=True)
            return self._create_empty_figure(f"{symbol} - {chart_name}: Plotting Error", height=fig_height, reason=str(e))

        chart_logger.info(f"{chart_name} for {symbol} created successfully.")
        return fig

# --- E. Standalone Test Block (`if __name__ == '__main__':`) ---
if __name__ == '__main__':
        
        # --- E. Standalone Test Block (`if __name__ == '__main__':`) ---
    if __name__ == '__main__':
        # Setup basic logging if this script is run directly
        if not logging.getLogger().hasHandlers() or not any(isinstance(h, logging.StreamHandler) for h in logging.getLogger().handlers):
            logging.basicConfig(
                level=logging.DEBUG, # Use DEBUG for detailed test output
                format='[%(levelname)s] (%(module)s-%(funcName)s:%(lineno)d) %(asctime)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        main_test_logger = logging.getLogger(__name__ + ".__main__") # Specific logger for this test block
        main_test_logger.setLevel(logging.DEBUG)
        # Also set the module-level logger for MSPIVisualizerV2 to DEBUG for more verbosity from its methods
        logger.setLevel(logging.DEBUG)

        main_test_logger.info("--- MSPIVisualizerV2 Test Run (V2.4.1 - Greek Flow Heatmap & New Data) --- ")

        # 1. Load main config (or create dummy if not found)
        test_config_path = "config_v2.json" # Assumes config_v2.json is in the same directory or accessible
        main_app_config_for_test = {}
        abs_test_config_path = test_config_path
        if not os.path.isabs(test_config_path):
            try: script_dir_test_viz = os.path.dirname(os.path.abspath(__file__))
            except NameError: script_dir_test_viz = os.getcwd()
            abs_test_config_path = os.path.join(script_dir_test_viz, test_config_path)

        try:
            if os.path.exists(abs_test_config_path):
                with open(abs_test_config_path, 'r') as f_cfg_viz_test:
                    main_app_config_for_test = json.load(f_cfg_viz_test)
                main_test_logger.info(f"Loaded main application config for visualizer test run from {abs_test_config_path}")
            else:
                main_test_logger.warning(f"Main config file '{abs_test_config_path}' not found for visualizer test. Using visualizer's internal DEFAULT_VISUALIZER_CONFIG only.")
                # MSPIVisualizerV2's __init__ will use its DEFAULT_VISUALIZER_CONFIG if main_app_config_for_test is empty or lacks the specific section
        except Exception as e_cfg_load_test:
            main_test_logger.error(f"Could not load '{abs_test_config_path}' for visualizer test run: {e_cfg_load_test}. Visualizer will use its defaults.")

        # 2. Instantiate MSPIVisualizerV2
        # Pass the full app config; the visualizer's __init__ will extract its relevant section.
        visualizer_test_instance = MSPIVisualizerV2(config_data=main_app_config_for_test)
        # Ensure instance logger level is also DEBUG for detailed output from visualizer methods
        visualizer_test_instance.instance_logger.setLevel(logging.DEBUG)
        for handler in visualizer_test_instance.instance_logger.handlers: handler.setLevel(logging.DEBUG)

        main_test_logger.info("MSPIVisualizerV2 instance created for testing.")

        # 3. Create sample `processed_data` DataFrame (Simulating output of EnhancedDataProcessor v2.0.7)
        main_test_logger.info("--- Creating Sample Processed Data for Visualizer Tests ---")
        sample_strikes_viz = np.arange(585, 596, 1).astype(float) # Tighter range for focused testing
        test_symbol_viz = 'SPY_TEST'
        current_underlying_price_viz = 592.0
        fetch_timestamp_viz = datetime.now().isoformat()

        per_contract_data_list = []
        for strike_val_viz in sample_strikes_viz:
            # --- Generate Strike-Level Aggregated Metrics (as processor would) ---
            # These will be the same for all contracts at the same strike
            strike_level_heuristic_net_delta_pressure = np.random.uniform(-50000, 50000)
            strike_level_net_gamma_flow = np.random.uniform(-5000, 15000) if strike_val_viz % 3 == 0 else np.random.uniform(-1000,1000) # Make some more variable
            strike_level_net_vega_flow = np.random.uniform(-200000, 200000)
            strike_level_net_theta_exposure = np.random.uniform(-100000, 50000)
            strike_level_true_net_vol_flow = np.random.randint(-5000, 5000)
            strike_level_true_net_val_flow = np.random.randint(-2000000, 2000000)
            strike_level_net_delta_flow_total = np.random.uniform(-150000, 150000)
            strike_level_net_delta_flow_calls = strike_level_net_delta_flow_total * np.random.uniform(0.3, 0.7) if strike_level_net_delta_flow_total > 0 else strike_level_net_delta_flow_total * np.random.uniform(1.3, 1.7)
            strike_level_net_delta_flow_puts = strike_level_net_delta_flow_total - strike_level_net_delta_flow_calls
            strike_level_heuristic_net_value_pressure = np.random.randint(-3000000, 3000000)
            strike_level_heuristic_net_volume_pressure = np.random.randint(-10000, 10000)

            for opt_type_viz in ['call', 'put']:
                # --- Per-Contract Metrics (as ITS would calculate) ---
                mspi_val = np.random.uniform(-0.95, 0.95)
                contract_data = {
                    # Base fields
                    'strike': strike_val_viz, 'opt_kind': opt_type_viz, 'symbol': f".{test_symbol_viz}...",
                    'underlying_symbol': test_symbol_viz, 'price': current_underlying_price_viz,
                    # Per-contract ITS calculated metrics
                    'mspi': mspi_val,
                    'sai': np.random.uniform(-1,1),
                    'ssi': np.random.uniform(0.1, 0.9),
                    'cfi': np.random.uniform(0, 3), # ARFI
                    'dag_custom': mspi_val * np.random.uniform(0.5e5, 1.5e5),
                    'tdpi': np.random.uniform(-1e5, 1e5),
                    'vri': np.random.uniform(-1e5, 1e5),
                    'sdag_multiplicative': mspi_val * np.random.uniform(0.8e5, 1.2e5),
                    'sdag_directional': mspi_val * np.random.uniform(0.8e5, 1.2e5),
                    'sdag_weighted': mspi_val * np.random.uniform(0.8e5, 1.2e5),
                    'sdag_volatility_focused': mspi_val * np.random.uniform(0.8e5, 1.2e5),
                    # Include normalized versions for MSPI Components chart
                    'dag_custom_norm': np.random.uniform(-1,1) if mspi_val !=0 else 0,
                    'tdpi_norm': np.random.uniform(-1,1) if mspi_val !=0 else 0,
                    'vri_norm': np.random.uniform(-1,1) if mspi_val !=0 else 0,
                    'sdag_multiplicative_norm': np.random.uniform(-1,1) if mspi_val !=0 else 0,
                    'sdag_weighted_norm': np.random.uniform(-1,1) if mspi_val !=0 else 0, # Example for weighted
                    'sdag_volatility_focused_norm': np.random.uniform(-1,1) if mspi_val !=0 else 0, # Example
                    # Other data that might be in hovertext
                    'oi': np.random.randint(10,500),
                    'dxoi': np.random.uniform(-20000,20000), 'gxoi': np.random.uniform(1000,30000),
                    'txoi': np.random.uniform(-5000, -500), 'vxoi': np.random.uniform(500,10000),
                    # Strike-level aggregated flow metrics (broadcasted by processor)
                    'heuristic_net_delta_pressure': strike_level_heuristic_net_delta_pressure,
                    'net_gamma_flow_at_strike': strike_level_net_gamma_flow, # Column name from config
                    'net_vega_flow_at_strike': strike_level_net_vega_flow,     # Column name from config
                    'net_theta_exposure_at_strike': strike_level_net_theta_exposure, # Column name from config
                    'true_net_volume_flow': strike_level_true_net_vol_flow,
                    'true_net_value_flow': strike_level_true_net_val_flow,
                    'net_delta_flow_total': strike_level_net_delta_flow_total,
                    # Your original heuristic pressures (now calculated by processor at strike level)
                    'net_value_pressure': strike_level_heuristic_net_value_pressure,
                    'net_volume_pressure': strike_level_heuristic_net_volume_pressure,
                    # Rolling flows for combined flow chart test
                    'volmbs_5m': np.random.randint(-100,100) * 10, 'valuebs_5m': np.random.randint(-1000,1000) * 100,
                    'volmbs_15m': np.random.randint(-200,200) * 10, 'valuebs_15m': np.random.randint(-2000,2000) * 100,
                }
                per_contract_data_list.append(contract_data)
        
        sample_processed_data_df = pd.DataFrame(per_contract_data_list)
        main_test_logger.info(f"Sample Processed DataFrame created. Shape: {sample_processed_data_df.shape}, Columns: {sample_processed_data_df.columns.tolist()}")

        # Sample data for other charts that don't take the full options_df
        sample_key_levels_data_viz = {
            'Support': [{'strike': 590.0, 'mspi': 0.8, 'sai': 0.7, 'ssi': 0.6, 'level_category':'Support'}],
            'Resistance': [{'strike': 595.0, 'mspi': -0.75, 'sai': 0.65, 'ssi': 0.65, 'level_category':'Resistance'}],
            'High Conviction': [{'strike': 590.0, 'mspi': 0.8, 'sai': 0.85, 'conviction':'Very High', 'level_category':'High Conviction'}],
            'Structure Change': [{'strike': 593.0, 'ssi': 0.1, 'mspi':0.2, 'level_category':'Structure Change'}]
        }
        sample_trading_signals_data_viz = {
            'directional': {'bullish': [{'strike': 590.0, 'mspi': 0.8, 'sai': 0.75, 'conviction_stars': 4, 'type':'directional'}]},
            'complex': {'sdag_conviction': [{'strike': 590.0, 'type':'sdag_conviction_bullish', 'agree_count':3, 'conviction_stars': 3}]}
        }
        sample_recommendations_list_viz = [
            {'id': 'DREC_TESTVIS_1', 'symbol': test_symbol_viz, 'Category': 'Directional Trades', 'direction_label': 'Bullish', 'strike': 590.0, 'strategy': 'Consider Longs near 590', 'conviction_stars': 4, 'raw_conviction_score': 3.5, 'status': 'ACTIVE_NEW', 'entry_ideal': 590.10, 'target_1': 593.0, 'target_2': 595.0, 'stop_loss': 588.0, 'rationale': 'Strong MSPI & SAI', 'target_rationale': 'ATR Based', 'mspi': 0.8, 'sai': 0.75, 'ssi': 0.6, 'arfi': 1.1, 'timestamp': datetime.now().isoformat(), 'last_updated_ts': None, 'exit_reason':None, 'type':'directional'}
        ]

        # 4. Test the new `create_net_greek_flow_heatmap` method
        main_test_logger.info(f"\n--- Testing New 'create_net_greek_flow_heatmap' ---")
        metrics_for_new_heatmap = [
            {"col": visualizer_test_instance.col_heuristic_net_delta_pressure, "title": "Heuristic Net Delta Pressure", "cs_key": "net_delta_heuristic_heatmap", "cb_title": "Net Delta (H)"},
            {"col": visualizer_test_instance.col_net_gamma_flow, "title": "Net Gamma Flow", "cs_key": "net_gamma_flow_heatmap", "cb_title": "Net Gamma Flow"},
            {"col": visualizer_test_instance.col_net_vega_flow, "title": "Net Vega Flow", "cs_key": "net_vega_flow_heatmap", "cb_title": "Net Vega Flow"},
            {"col": visualizer_test_instance.col_net_theta_exposure, "title": "Net Theta Exposure", "cs_key": "net_theta_exposure_heatmap", "cb_title": "Net Theta Exp."}
        ]

        for metric_info in metrics_for_new_heatmap:
            if metric_info["col"] in sample_processed_data_df.columns:
                main_test_logger.info(f"  Testing heatmap for: {metric_info['title']}")
                fig_greek_heatmap = visualizer_test_instance.create_net_greek_flow_heatmap(
                    processed_data=sample_processed_data_df,
                    metric_column_to_plot=metric_info["col"],
                    chart_main_title_prefix=metric_info["title"],
                    colorscale_config_key=metric_info["cs_key"],
                    colorbar_title_text=metric_info["cb_title"],
                    symbol=test_symbol_viz,
                    fetch_timestamp=fetch_timestamp_viz,
                    current_price=current_underlying_price_viz
                )
                if fig_greek_heatmap and fig_greek_heatmap.data:
                    main_test_logger.info(f"    Successfully generated '{metric_info['title']}' heatmap.")
                    # fig_greek_heatmap.show() # Uncomment to display locally
                else:
                    main_test_logger.error(f"    FAILED to generate '{metric_info['title']}' heatmap.")
            else:
                main_test_logger.warning(f"  Skipping heatmap for '{metric_info['title']}', column '{metric_info['col']}' not in sample data.")

        # 5. Briefly test a few existing chart functions
        main_test_logger.info(f"\n--- Testing a few Existing Chart Functions ---")
        # MSPI Heatmap (will use per-contract MSPI, then visualizer aggregates)
        fig_mspi_hm = visualizer_test_instance.create_mspi_heatmap(sample_processed_data_df, symbol=test_symbol_viz, fetch_timestamp=fetch_timestamp_viz)
        if fig_mspi_hm and fig_mspi_hm.data: main_test_logger.info("  MSPI Heatmap generated.")
        else: main_test_logger.error("  MSPI Heatmap FAILED.")

        # Net Value Pressure Heatmap (will use 'net_value_pressure' column -> heuristic)
        fig_nvp_hm = visualizer_test_instance.create_net_value_heatmap(sample_processed_data_df, symbol=test_symbol_viz, fetch_timestamp=fetch_timestamp_viz)
        if fig_nvp_hm and fig_nvp_hm.data: main_test_logger.info("  Net Value Pressure Heatmap (Heuristic) generated.")
        else: main_test_logger.error("  Net Value Pressure Heatmap (Heuristic) FAILED.")

        # MSPI Components (will use per-contract components, visualizer aggregates for display)
        fig_mspi_comp = visualizer_test_instance.create_component_comparison(sample_processed_data_df, symbol=test_symbol_viz, current_price=current_underlying_price_viz, fetch_timestamp=fetch_timestamp_viz)
        if fig_mspi_comp and fig_mspi_comp.data: main_test_logger.info("  MSPI Components chart generated.")
        else: main_test_logger.error("  MSPI Components chart FAILED.")
        
        # Combined Rolling Flow (will show empty if _5m fields are zero, which they are in this sample)
        fig_roll_flow = visualizer_test_instance.create_combined_rolling_flow_chart(sample_processed_data_df, symbol=test_symbol_viz, current_price=current_underlying_price_viz, fetch_timestamp=fetch_timestamp_viz)
        if fig_roll_flow and fig_roll_flow.data : main_test_logger.info("  Combined Rolling Flow chart generated (might be empty if no rolling data).")
        else: main_test_logger.warning("  Combined Rolling Flow chart FAILED or has no data traces.")


        # 6. Test `create_strategy_recommendations_table`
        main_test_logger.info(f"\n--- Testing Strategy Recommendations Table ---")
        fig_recs_table = visualizer_test_instance.create_strategy_recommendations_table(
            recommendations_list=sample_recommendations_list_viz,
            symbol=test_symbol_viz,
            fetch_timestamp=fetch_timestamp_viz
        )
        if fig_recs_table and fig_recs_table.data:
            main_test_logger.info("  Strategy Recommendations Table generated.")
            # fig_recs_table.show() # Uncomment to display
        else:
            main_test_logger.error("  Strategy Recommendations Table FAILED.")

        main_test_logger.info("--- MSPIVisualizerV2 Test Run Complete ---")