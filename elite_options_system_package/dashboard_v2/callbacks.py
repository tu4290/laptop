# /home/ubuntu/dashboard_v2/callbacks.py
# -*- coding: utf-8 -*-
"""
Defines all the callbacks for the Enhanced Options Dashboard V2.
Connects user interface components (layout.py) to backend logic
(fetcher, processor, stateful ITS, visualizer, TradierFetcher) and manages data flow.
(Version: Canon V2.4.5 - MSPI Card Toggle Integration)
"""

# Standard Library Imports
import logging
import time as pytime 
import re 
from datetime import datetime, time, date, timedelta 
from typing import Dict, Any, Optional, Tuple, List, Union, Deque, Callable 
from collections import deque 
import inspect 
import copy 

# Third-Party Imports
import pandas as pd
import numpy as np 
import dash
from dash import html, dcc, callback, Input, Output, State, no_update, callback_context, ctx 
import plotly.graph_objects as go 
import dash_bootstrap_components as dbc 
from dateutil import parser as date_parser 

# --- Logger for callbacks.py ---
logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers(): 
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] (%(name)s:%(lineno)d) %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# --- Safe Imports with Fallbacks for Core Dashboard Utils & Styling ---
_utils_styling_imported_successfully_cb = False
_PLOTLY_TEMPLATE_DARK_FALLBACK_CB = {"layout": go.Layout(template="plotly_dark")}
PLOTLY_TEMPLATE_DARK_CB: Dict[str, Any] = _PLOTLY_TEMPLATE_DARK_FALLBACK_CB 

def _fallback_create_empty_figure_impl_cb(title: str = "Error", height: Optional[int] = None, reason: str = "Utils Unavailable") -> go.Figure:
    logger.error(f"FALLBACK (callbacks.py): create_empty_figure for '{title}', Reason: '{reason}'")
    fig_height = height if height is not None else 600
    fig = go.Figure()
    fig.update_layout(
        title={'text': f"<i>{title}<br><small style='color:grey'>({reason})</small></i>", 'y':0.5, 'x':0.5, 'xanchor': 'center', 'yanchor': 'middle', 'font': {'color': 'grey', 'size': 16}},
        template=PLOTLY_TEMPLATE_DARK_CB.get("layout", {}).get("template", "plotly_dark"), 
        height=fig_height, xaxis={'visible': False}, yaxis={'visible': False},
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def _fallback_get_config_value_impl_cb(path: List[str], default: Any = None) -> Any:
    logger.warning(f"FALLBACK (callbacks.py): get_config_value for path: {path}")
    if path == ["visualization_settings", "dashboard", "defaults", "range_pct"]: return 5.0
    if path == ["system_settings", "dashboard_cache_timeout_seconds"]: return 300
    if path == ["visualization_settings", "mspi_visualizer", "greek_flow_heatmap_options"]: return [{'label': 'Net Delta P (Heuristic)', 'value': 'heuristic_net_delta_pressure'}]
    if path == ["visualization_settings", "mspi_visualizer", "greek_flow_heatmap_default_metric"]: return "heuristic_net_delta_pressure"
    # Add other fallbacks as needed
    return default

def _fallback_format_status_message_impl_cb(message: str, is_error: bool = False) -> Tuple[html.Div, Dict[str, str]]:
    logger.error(f"FALLBACK (callbacks.py): format_status_message for msg: '{message}'")
    style = {"padding": "10px", "textAlign": "center", "borderRadius": "5px", "fontWeight": "bold", "color": "#FADBD8", "backgroundColor": "#78281F"} if is_error else {"color": "#D4E6F1", "backgroundColor": "#1B4F72", "padding": "10px", "textAlign": "center", "borderRadius": "5px", "fontWeight": "bold"}
    return html.Div(message, className="p-2 text-center rounded fallback-status"), style

def _fallback_generate_cache_key_impl_cb(symbol: str, dte_str: str, range_pct: Optional[Union[int, float]]) -> str:
    logger.error(f"FALLBACK (callbacks.py): generate_cache_key for {symbol}"); ts = datetime.now().strftime('%Y%m%d%H%M'); range_val = int(range_pct) if isinstance(range_pct, (int,float)) else 0; return f"fbk_{symbol}_{dte_str}_{range_val}_{ts}"

def _fallback_get_data_from_server_cache_impl_cb(cache_key: Optional[str], server_cache_ref: Dict) -> Optional[Dict[str, Any]]:
    logger.error(f"FALLBACK (callbacks.py): get_data_from_server_cache for key: {cache_key}")
    if cache_key and isinstance(cache_key, str) and cache_key in server_cache_ref:
        entry = server_cache_ref.get(cache_key)
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], dict):
            stored_ts, stored_bundle = entry
            timeout_val = _fallback_get_config_value_impl_cb(["system_settings", "dashboard_cache_timeout_seconds"], 300)
            if (pytime.time() - stored_ts) > timeout_val:
                logger.warning(f"FALLBACK CACHE: Key '{cache_key}' expired. Returning None.")
                server_cache_ref.pop(cache_key, None)
                return None
            return copy.deepcopy(stored_bundle) 
    return None

def _fallback_store_data_in_server_cache_impl_cb(cache_key: Optional[str], data_bundle: Dict[str, Any], server_cache_ref: Dict) -> None:
    logger.error(f"FALLBACK (callbacks.py): store_data_in_server_cache for key: {cache_key}")
    if cache_key and isinstance(cache_key, str) and isinstance(data_bundle, dict):
        server_cache_ref[cache_key] = (pytime.time(), copy.deepcopy(data_bundle)) 

def _fallback_parse_timestamp_impl_cb(ts_str: Optional[str]) -> Optional[datetime]:
    logger.error(f"FALLBACK (callbacks.py): parse_timestamp for: {ts_str}")
    if ts_str and isinstance(ts_str, str):
        try: return date_parser.isoparse(ts_str)
        except: return None
    return None

create_empty_figure_cb = _fallback_create_empty_figure_impl_cb
get_config_value_cb = _fallback_get_config_value_impl_cb
format_status_message_cb = _fallback_format_status_message_impl_cb
generate_cache_key_cb = _fallback_generate_cache_key_impl_cb
get_data_from_server_cache_cb = _fallback_get_data_from_server_cache_impl_cb
store_data_in_server_cache_cb = _fallback_store_data_in_server_cache_impl_cb
parse_timestamp_cb = _fallback_parse_timestamp_impl_cb

try:
    from .styling import PLOTLY_TEMPLATE_DARK as imported_plotly_template_styling_cb
    PLOTLY_TEMPLATE_DARK_CB = imported_plotly_template_styling_cb
    from .utils import ( 
        create_empty_figure as imported_create_empty_figure_u,
        get_config_value as imported_get_config_value_u,
        format_status_message as imported_format_status_message_u,
        generate_cache_key as imported_generate_cache_key_u,
        get_data_from_server_cache as imported_get_data_from_server_cache_u,
        store_data_in_server_cache as imported_store_data_in_server_cache_u,
        parse_timestamp as imported_parse_timestamp_u
    )
    create_empty_figure_cb = imported_create_empty_figure_u
    get_config_value_cb = imported_get_config_value_u
    format_status_message_cb = imported_format_status_message_u
    generate_cache_key_cb = imported_generate_cache_key_u
    get_data_from_server_cache_cb = imported_get_data_from_server_cache_u
    store_data_in_server_cache_cb = imported_store_data_in_server_cache_u
    parse_timestamp_cb = imported_parse_timestamp_u
    _utils_styling_imported_successfully_cb = True
    logger.info("CALLBACKS.PY: Successfully imported local styling and utils functions.")
except ImportError as _utils_import_err_final_cb:
    logger.critical(f"CALLBACKS.PY CRITICAL: Failed to import from .styling or .utils: {_utils_import_err_final_cb}. Using fallbacks for these functions.", exc_info=True)

_backend_modules_imported_fully_cb = False
try:
    from elite_options_system_package.enhanced_data_fetcher_v2 import EnhancedDataFetcher_v2
    from elite_options_system_package.enhanced_tradier_fetcher_v2 import TradierDataFetcher
    from elite_options_system_package.enhanced_data_processor_v2 import EnhancedDataProcessor
    from elite_options_system_package.integrated_strategies_v2 import IntegratedTradingSystem
    from elite_options_system_package.mspi_visualizer_v2 import MSPIVisualizerV2
    _backend_modules_imported_fully_cb = True
    logger.info("CALLBACKS.PY: Backend module classes imported successfully for type hinting and instance checks.")
except ImportError as _backend_module_import_error_cb_final:
    logger.critical(f"CALLBACKS.PY CRITICAL: Failed to import one or more backend modules: {_backend_module_import_error_cb_final}. Fallback classes will be used for type checking, functionality may be impaired if instances are not provided correctly to register_callbacks.", exc_info=True)
    class EnhancedDataFetcher_v2: pass 
    class TradierDataFetcher: pass 
    class EnhancedDataProcessor: pass 
    class IntegratedTradingSystem: pass 
    class MSPIVisualizerV2: pass 

_layout_ids_imported_successfully_cb = False 
CHART_IDS_CB: List[str] = [] 
ID_SYMBOL_INPUT_CB, ID_EXPIRATION_INPUT_CB, ID_RANGE_SLIDER_CB, ID_INTERVAL_DROPDOWN_CB, \
ID_FETCH_BUTTON_CB, ID_STATUS_DISPLAY_CB, ID_INTERVAL_TIMER_CB, ID_CACHE_STORE_CB, \
ID_CONFIG_STORE_CB, ID_NET_GREEK_FLOW_HEATMAP_CHART_CB, ID_GREEK_FLOW_SELECTOR_IN_CARD_CB, \
ID_MSPI_CHART_TOGGLE_SELECTOR_CB, \
ID_MODE_TABS_CB, ID_MODE_CONTENT_CB, ID_TAB_MAIN_DASHBOARD_CB, ID_TAB_SDAG_DIAGNOSTICS_CB = \
    "symbol-input", "expiration-input", "price-range-slider", "interval-dropdown", \
    "fetch-button", "status-display", "interval-component", "cache-key-store", \
    "app-config-store", "net-greek-flow-heatmap-chart", "greek-flow-selector-in-card", \
    "mspi-chart-toggle-selector", \
    "mode-tabs", "mode-content", "tab-main-dashboard", "tab-sdag-diagnostics" 

_layout_mode_functions_imported_cb = False
get_main_dashboard_mode_layout_cb: Callable[[], html.Div] = lambda: html.Div("Error: Main layout function not loaded.")
get_sdag_diagnostics_mode_layout_cb: Callable[[], html.Div] = lambda: html.Div("Error: SDAG layout function not loaded.")

try:
    from .layout import (
        ALL_CHART_IDS_FOR_FACTORY, 
        ID_SYMBOL_INPUT, ID_EXPIRATION_INPUT, ID_RANGE_SLIDER,
        ID_INTERVAL_DROPDOWN, ID_FETCH_BUTTON, ID_STATUS_DISPLAY,
        ID_INTERVAL_TIMER, ID_CACHE_STORE, ID_CONFIG_STORE,
        ID_NET_GREEK_FLOW_HEATMAP_CHART, ID_GREEK_FLOW_SELECTOR_IN_CARD, 
        ID_MSPI_CHART_TOGGLE_SELECTOR, 
        ID_MODE_TABS, ID_MODE_CONTENT, ID_TAB_MAIN_DASHBOARD, ID_TAB_SDAG_DIAGNOSTICS, 
        get_main_dashboard_mode_layout, get_sdag_diagnostics_mode_layout 
    )
    CHART_IDS_CB = ALL_CHART_IDS_FOR_FACTORY 
    ID_SYMBOL_INPUT_CB, ID_EXPIRATION_INPUT_CB, ID_RANGE_SLIDER_CB, ID_INTERVAL_DROPDOWN_CB, \
    ID_FETCH_BUTTON_CB, ID_STATUS_DISPLAY_CB, ID_INTERVAL_TIMER_CB, ID_CACHE_STORE_CB, \
    ID_CONFIG_STORE_CB, ID_NET_GREEK_FLOW_HEATMAP_CHART_CB, ID_GREEK_FLOW_SELECTOR_IN_CARD_CB, \
    ID_MSPI_CHART_TOGGLE_SELECTOR_CB, \
    ID_MODE_TABS_CB, ID_MODE_CONTENT_CB, ID_TAB_MAIN_DASHBOARD_CB, ID_TAB_SDAG_DIAGNOSTICS_CB = \
        ID_SYMBOL_INPUT, ID_EXPIRATION_INPUT, ID_RANGE_SLIDER, ID_INTERVAL_DROPDOWN, \
        ID_FETCH_BUTTON, ID_STATUS_DISPLAY, ID_INTERVAL_TIMER, ID_CACHE_STORE, \
        ID_CONFIG_STORE, ID_NET_GREEK_FLOW_HEATMAP_CHART, ID_GREEK_FLOW_SELECTOR_IN_CARD, \
        ID_MSPI_CHART_TOGGLE_SELECTOR, \
        ID_MODE_TABS, ID_MODE_CONTENT, ID_TAB_MAIN_DASHBOARD, ID_TAB_SDAG_DIAGNOSTICS
    _layout_ids_imported_successfully_cb = True 
    
    get_main_dashboard_mode_layout_cb = get_main_dashboard_mode_layout
    get_sdag_diagnostics_mode_layout_cb = get_sdag_diagnostics_mode_layout
    _layout_mode_functions_imported_cb = True
    logger.info("CALLBACKS.PY: Successfully imported CHART_IDS, Component IDs, and Mode Layout functions from .layout.")
except ImportError as e_layout_ids_import_cb_final:
    logger.error(f"CALLBACKS.PY: Failed to import from layout: {e_layout_ids_import_cb_final}. Using fallback IDs/functions.", exc_info=True)
    _layout_ids_imported_successfully_cb = False 

CHARTS_NEEDING_FIGURE_STATE_CB: List[str] = get_config_value_cb(
    ["visualization_settings", "mspi_visualizer", "charts_needing_figure_state_in_callbacks"],
    ["mspi_components", "net_volval_comp", "combined_rolling_flow_chart", "sdag_multiplicative",
     "sdag_directional", "sdag_weighted", "sdag_volatility_focused", "volatility_regime", "time_decay"]
)

def register_callbacks(
    app: dash.Dash,
    fetcher_instance: Optional[EnhancedDataFetcher_v2], 
    tradier_fetcher_instance: Optional[TradierDataFetcher],
    processor_instance: Optional[EnhancedDataProcessor],
    its_instance: Optional[IntegratedTradingSystem],
    visualizer_instance: Optional[MSPIVisualizerV2],
    server_cache_ref: Dict[str, Tuple[float, Dict[str, Any]]], 
    component_history_ref: Dict[str, Deque[Tuple[float, Any]]] 
) -> None:
    """ Registers all callbacks for the dashboard application (V2.4.5 - MSPI Card Toggle). """

    if not _utils_styling_imported_successfully_cb: logger.error("CALLBACKS REGISTER: Core utility/styling functions are using FALLBACKS.")
    if not _backend_modules_imported_fully_cb: logger.error("CALLBACKS REGISTER: Core backend module classes are using FALLBACKS.")
    if not _layout_ids_imported_successfully_cb: logger.error("CALLBACKS REGISTER: CHART_IDS/Component IDs from layout are using FALLBACKS.") 
    if not _layout_mode_functions_imported_cb: logger.error("CALLBACKS REGISTER: Mode layout functions from layout are using FALLBACKS.")

    logger.info("Registering dashboard callbacks (V2.4.5 - MSPI Card Toggle)...")

    @app.callback(
        Output(ID_INTERVAL_TIMER_CB, "interval"), 
        Output(ID_INTERVAL_TIMER_CB, "disabled"),
        Input(ID_INTERVAL_DROPDOWN_CB, "value"),
        prevent_initial_call=True
    )
    def update_interval_timer_callback(interval_value_ms_from_dropdown: Optional[Union[int, str]]) -> Tuple[int, bool]:
        callback_logger_uit = logger.getChild("update_interval_timer")
        callback_logger_uit.debug(f"Interval dropdown changed to: {interval_value_ms_from_dropdown}")
        parsed_interval_ms_uit: int = 0
        try:
            if isinstance(interval_value_ms_from_dropdown, str): parsed_interval_ms_uit = int(interval_value_ms_from_dropdown)
            elif isinstance(interval_value_ms_from_dropdown, int): parsed_interval_ms_uit = interval_value_ms_from_dropdown
        except (ValueError, TypeError):
            callback_logger_uit.warning(f"Invalid interval dropdown value '{interval_value_ms_from_dropdown}'. Defaulting to manual.")
            parsed_interval_ms_uit = 0 
        if parsed_interval_ms_uit <= 0:
            callback_logger_uit.info("Setting to Manual Refresh (timer disabled).")
            return 24 * 60 * 60 * 1000, True 
        else:
            callback_logger_uit.info(f"Setting auto-refresh interval to {parsed_interval_ms_uit}ms (timer enabled).")
            return parsed_interval_ms_uit, False


    @app.callback(
        [Output(ID_STATUS_DISPLAY_CB, "children"), Output(ID_STATUS_DISPLAY_CB, "style"), Output(ID_CACHE_STORE_CB, "data")],
        [Input(ID_FETCH_BUTTON_CB, "n_clicks"), Input(ID_INTERVAL_TIMER_CB, "n_intervals")],
        [State(ID_SYMBOL_INPUT_CB, "value"), State(ID_EXPIRATION_INPUT_CB, "value"),
         State(ID_RANGE_SLIDER_CB, "value"), State(ID_INTERVAL_DROPDOWN_CB, "value")],
        prevent_initial_call=True
    )
    def fetch_process_and_cache_data_callback(
        button_n_clicks_main: Optional[int], timer_n_intervals_main: Optional[int],
        symbol_in: Optional[str], dte_str_in: Optional[str],
        range_pct_in: Optional[Union[int, float]],
        interval_setting_in: Optional[Union[int, str]]
    ) -> Tuple[Any, Dict[str, str], Optional[str]]: 
        main_data_cb_logger = logger.getChild("fetch_process_cache_V2.4.5_Modes") 
        start_time_main_cb = pytime.time()
        trigger_id_main_cb = ctx.triggered_id if ctx.triggered and ctx.triggered_id else "Unknown_Trigger"
        main_data_cb_logger.info(f"--- Main Data Orchestration START (Trigger: {trigger_id_main_cb}) ---")

        should_proceed_main_cb = False; current_interval_ms_main_cb = 0
        try: current_interval_ms_main_cb = int(str(interval_setting_in)) if interval_setting_in is not None else 0
        except: pass 
        if trigger_id_main_cb == ID_FETCH_BUTTON_CB and isinstance(button_n_clicks_main, int) and button_n_clicks_main > 0: should_proceed_main_cb = True
        elif trigger_id_main_cb == ID_INTERVAL_TIMER_CB and current_interval_ms_main_cb > 0: should_proceed_main_cb = True
        if not should_proceed_main_cb: main_data_cb_logger.debug("Callback triggered but conditions for fetch/process not met. No update."); return dash.no_update, dash.no_update, dash.no_update

        errors_validation_main_cb: List[str] = []
        symbol_main_cb = str(symbol_in).strip().upper() if isinstance(symbol_in, str) and symbol_in else ""
        if not symbol_main_cb: errors_validation_main_cb.append("Symbol is required.")
        dte_str_main_cb = str(dte_str_in).strip() if isinstance(dte_str_in, str) and dte_str_in else ""
        if not dte_str_main_cb: errors_validation_main_cb.append("DTE string is required.")
        range_pct_main_cb = get_config_value_cb(["visualization_settings", "dashboard", "defaults", "range_pct"], 5.0)
        if isinstance(range_pct_in, (int, float)) and 1 <= range_pct_in <= 20: range_pct_main_cb = float(range_pct_in)
        
        dte_list_main_cb: List[int] = []
        if not errors_validation_main_cb and dte_str_main_cb:
            try: 
                if "-" in dte_str_main_cb: s,e=map(int,dte_str_main_cb.split('-')); dte_list_main_cb=list(range(s,e+1))
                elif "," in dte_str_main_cb: dte_list_main_cb=sorted(list(set(int(d.strip()) for d in dte_str_main_cb.split(',') if d.strip().isdigit())))
                elif dte_str_main_cb.isdigit(): dte_list_main_cb=[int(dte_str_main_cb)]
                else: raise ValueError("Invalid DTE format")
                if not dte_list_main_cb or any(d<0 for d in dte_list_main_cb): raise ValueError("DTEs must be non-negative and list non-empty.")
            except Exception as e_dte: errors_validation_main_cb.append(f"DTE Error: {e_dte}")
        
        if errors_validation_main_cb:
            err_msg_div, err_style_div = format_status_message_cb("Input Error: "+"; ".join(errors_validation_main_cb),True); return err_msg_div, err_style_div, no_update

        cache_key_main_cb = generate_cache_key_cb(symbol_main_cb, dte_str_main_cb, range_pct_main_cb)
        main_data_cb_logger.info(f"Orchestration for: Sym='{symbol_main_cb}', DTE(s)='{dte_list_main_cb}', Range={range_pct_main_cb}%, CacheKey='{cache_key_main_cb}'")
        
        status_messages_overall: List[str] = []; has_critical_error_flag = False
        fetched_options_df_main: Optional[pd.DataFrame] = None; fetched_underlying_data_main: Optional[Dict[str, Any]] = None
        fetched_ohlc_df_main: Optional[pd.DataFrame] = None; combined_volatility_data_main: Dict[str, Any] = {}
        current_timestamp_iso: str = datetime.now().isoformat() 

        if not isinstance(fetcher_instance, EnhancedDataFetcher_v2) or \
           not isinstance(tradier_fetcher_instance, TradierDataFetcher) or \
           not isinstance(processor_instance, EnhancedDataProcessor) or \
           not isinstance(its_instance, IntegratedTradingSystem):
            crit_err_inst = "CRITICAL: One or more backend services are not correctly initialized. Cannot proceed with data fetching and processing."
            main_data_cb_logger.critical(crit_err_inst); status_messages_overall.append(crit_err_inst); has_critical_error_flag = True
            err_msg_inst_div, err_style_inst_div = format_status_message_cb(crit_err_inst, True)
            return err_msg_inst_div, err_style_inst_div, None 

        try:
            main_data_cb_logger.info(f"Fetching ConvexValue data for {symbol_main_cb}...")
            fetched_options_df_main, fetched_underlying_data_main = fetcher_instance.fetch_options_chain(symbol_main_cb, dte_list_main_cb, range_pct_main_cb)
            if isinstance(fetched_underlying_data_main, dict) and fetched_underlying_data_main.get("error"): status_messages_overall.append(f"CV Fetch Err: {fetched_underlying_data_main['error']}")
            if not isinstance(fetched_options_df_main, pd.DataFrame) or fetched_options_df_main.empty: status_messages_overall.append("CV options chain empty/failed.")
            
            if isinstance(fetched_underlying_data_main, dict):
                combined_volatility_data_main = {
                    "current_iv": fetched_underlying_data_main.get("volatility"),
                    "front_volatility": fetched_underlying_data_main.get("front_volatility"),
                    "back_volatility": fetched_underlying_data_main.get("back_volatility"),
                    "iv_percentile_30d": fetched_underlying_data_main.get(get_config_value_cb(["data_processor_settings","iv_context_parameters","iv_percentile"],"iv_percentile_30d"))
                }
            current_timestamp_iso = fetched_underlying_data_main.get("fetch_timestamp", current_timestamp_iso) if isinstance(fetched_underlying_data_main, dict) else current_timestamp_iso
            
            main_data_cb_logger.info(f"Fetching Tradier OHLCV for {symbol_main_cb}...")
            num_days_hist_ohlc = get_config_value_cb(["tradier_api_settings", "ohlcv_num_days_history"], 30)
            fetched_ohlc_df_main = tradier_fetcher_instance.get_ohlcv_data(symbol_main_cb, num_days_history=int(num_days_hist_ohlc))
            if not isinstance(fetched_ohlc_df_main, pd.DataFrame) or fetched_ohlc_df_main.empty: status_messages_overall.append("Tradier OHLCV empty/failed.")
            
            target_dte_iv = get_config_value_cb(["tradier_api_settings", "iv_approx_target_dte"], 5)
            main_data_cb_logger.info(f"Fetching Tradier IV Approx (DTE {target_dte_iv}) for {symbol_main_cb}...")
            iv_approx_tradier = tradier_fetcher_instance.get_iv_approximation(symbol_main_cb, target_dte=int(target_dte_iv))
            if iv_approx_tradier and iv_approx_tradier.get(f"avg_{target_dte_iv}day_iv") is not None:
                combined_volatility_data_main[f"avg_{target_dte_iv}day_iv"] = iv_approx_tradier[f"avg_{target_dte_iv}day_iv"]
                if int(target_dte_iv) == 5: combined_volatility_data_main["avg_5day_iv"] = iv_approx_tradier["avg_5day_iv"] 
            else: status_messages_overall.append(f"Tradier IV{target_dte_iv} approx failed.")

            main_data_cb_logger.info(f"Processing data for {symbol_main_cb}...")
            data_bundle_for_cache = processor_instance.process_data_with_integrated_strategies(
                options_chain_df=fetched_options_df_main,
                underlying_data=fetched_underlying_data_main,
                volatility_data=combined_volatility_data_main,
                historical_ohlc_df=fetched_ohlc_df_main
            ) 
            if isinstance(data_bundle_for_cache, dict) and data_bundle_for_cache.get("error"): status_messages_overall.append(f"Processor: {data_bundle_for_cache['error']}")
            if not isinstance(data_bundle_for_cache, dict) or not isinstance(data_bundle_for_cache.get("final_metric_rich_df_obj"), pd.DataFrame) or data_bundle_for_cache.get("final_metric_rich_df_obj").empty :
                 status_messages_overall.append("Processor returned empty/invalid bundle or DataFrame."); has_critical_error_flag = True
                 if not isinstance(data_bundle_for_cache, dict): data_bundle_for_cache = {}
                 data_bundle_for_cache.setdefault("error", "Processor output invalid.")
                 data_bundle_for_cache.setdefault("symbol", symbol_main_cb)
                 data_bundle_for_cache.setdefault("fetch_timestamp", current_timestamp_iso)
                 data_bundle_for_cache.setdefault("final_metric_rich_df_obj", pd.DataFrame())
                 data_bundle_for_cache.setdefault("processed_data", {"options_chain": []})

            data_bundle_for_cache["historical_ohlc_df_obj"] = fetched_ohlc_df_main 
            data_bundle_for_cache["volatility_context_combined"] = combined_volatility_data_main 

        except Exception as e_orch_main:
            has_critical_error_flag = True; error_text_main_orch = f"Core Orchestration Error: {str(e_orch_main)[:120]}"
            status_messages_overall.append(error_text_main_orch)
            main_data_cb_logger.critical(f"CRITICAL error in main data callback for {symbol_main_cb}: {e_orch_main}", exc_info=True)
            data_bundle_for_cache = {"error": status_messages_overall[0] if status_messages_overall else "Unknown Orchestration Error", "symbol": symbol_main_cb, "fetch_timestamp": current_timestamp_iso, "final_metric_rich_df_obj": pd.DataFrame(), "processed_data": {"options_chain": []}, "historical_ohlc_df_obj": None, "volatility_context_combined": None}
        
        store_data_in_server_cache_cb(cache_key_main_cb, data_bundle_for_cache, server_cache_ref)
        
        if not has_critical_error_flag and isinstance(data_bundle_for_cache.get("final_metric_rich_df_obj"), pd.DataFrame) and not data_bundle_for_cache.get("final_metric_rich_df_obj").empty:
            metric_df_for_hist_main = data_bundle_for_cache["final_metric_rich_df_obj"]
            if symbol_main_cb not in component_history_ref:
                hist_maxlen_main = get_config_value_cb(["system_settings","df_history_maxlen"],10); component_history_ref[symbol_main_cb] = deque(maxlen=int(hist_maxlen_main))
            hist_cols_main = ['strike', get_config_value_cb(["visualization_settings","mspi_visualizer","column_names","net_volume_pressure"],"net_volume_pressure"), get_config_value_cb(["visualization_settings","mspi_visualizer","column_names","net_value_pressure"],"net_value_pressure")] + [c for c in metric_df_for_hist_main.columns if 'volmbs_' in c or 'valuebs_' in c]
            avail_hist_cols_main = [c for c in hist_cols_main if c in metric_df_for_hist_main.columns]
            if avail_hist_cols_main:
                hist_slice_main = metric_df_for_hist_main[avail_hist_cols_main].copy()
                for col_h_main in hist_slice_main.columns:
                    if col_h_main != 'strike': hist_slice_main[col_h_main] = pd.to_numeric(hist_slice_main[col_h_main], errors='coerce').fillna(0.0)
                component_history_ref[symbol_main_cb].appendleft((pytime.time(), hist_slice_main))
                main_data_cb_logger.debug(f"Added data to component history for '{symbol_main_cb}'. Size: {len(component_history_ref[symbol_main_cb])}")
        
        final_status_message_text = f"✓ Data for {symbol_main_cb} ({dte_str_main_cb}) loaded." if not status_messages_overall else f"⚠ Issues for {symbol_main_cb}: {'; '.join(s for s in status_messages_overall if s)}"
        duration_main_cb = pytime.time() - start_time_main_cb
        final_status_message_text += f" ({datetime.now().strftime('%H:%M:%S')} in {duration_main_cb:.2f}s)"
        status_msg_div_final, status_style_final = format_status_message_cb(final_status_message_text, is_error=has_critical_error_flag or bool(status_messages_overall))
        main_data_cb_logger.info(f"Main Data Orchestration END. Duration: {duration_main_cb:.3f}s. Final Status: '{final_status_message_text}'")
        return status_msg_div_final, status_style_final, cache_key_main_cb


    def create_chart_update_callback_factory(chart_id_cb_factory: str):
        chart_factory_instance_logger = logger.getChild(f"chart_factory.{chart_id_cb_factory}")
        chart_display_title_default = chart_id_cb_factory.replace('_', ' ').replace('-chart','').title()
        
        # Special handling for the chart that now has an internal toggle
        if chart_id_cb_factory == "mspi_heatmap": # This is the ID of the dcc.Graph component
            chart_display_title_default = "MSPI View" # Generic title as content can change
        elif chart_id_cb_factory == ID_NET_GREEK_FLOW_HEATMAP_CHART_CB: 
            chart_display_title_default = "Net Greek Flow & Pressure"

        is_fig_state_needed = chart_id_cb_factory in CHARTS_NEEDING_FIGURE_STATE_CB
        
        callback_outputs_list = Output(chart_id_cb_factory, "figure")
        callback_inputs_list = [Input(ID_CACHE_STORE_CB, "data")] 
        callback_states_list = [State(ID_RANGE_SLIDER_CB, "value")] 

        # Add specific Input for charts with internal selectors
        if chart_id_cb_factory == ID_NET_GREEK_FLOW_HEATMAP_CHART_CB: 
            callback_inputs_list.append(Input(ID_GREEK_FLOW_SELECTOR_IN_CARD_CB, "value")) 
        elif chart_id_cb_factory == "mspi_heatmap": # The dcc.Graph for MSPI view
             callback_inputs_list.append(Input(ID_MSPI_CHART_TOGGLE_SELECTOR_CB, "value"))
        
        if is_fig_state_needed:
            callback_states_list.append(State(chart_id_cb_factory, "figure"))

        @app.callback(callback_outputs_list, callback_inputs_list, callback_states_list, prevent_initial_call=True)
        def generated_chart_update_callback(cached_data_key_chart: Optional[str], *dynamic_args: Any) -> go.Figure:
            chart_update_cb_start_time = pytime.time()
            selected_metric_dropdown_value: Optional[str] = None # For Greek heatmap or MSPI toggle
            current_range_slider_val_chart: Optional[Union[int, float]] = None
            previous_fig_state_chart: Optional[Dict] = None
            current_dynamic_arg_index = 0

            # Unpack dynamic_args based on which inputs were added
            if chart_id_cb_factory == ID_NET_GREEK_FLOW_HEATMAP_CHART_CB or chart_id_cb_factory == "mspi_heatmap":
                if len(dynamic_args) > current_dynamic_arg_index:
                    selected_metric_dropdown_value = dynamic_args[current_dynamic_arg_index]
                    current_dynamic_arg_index += 1
                    if not selected_metric_dropdown_value: 
                        if chart_id_cb_factory == ID_NET_GREEK_FLOW_HEATMAP_CHART_CB:
                            default_val = get_config_value_cb(["visualization_settings", "mspi_visualizer", "greek_flow_heatmap_default_metric"], "heuristic_net_delta_pressure")
                        else: # mspi_heatmap
                            default_val = "mspi_heatmap" # Default for MSPI toggle
                        selected_metric_dropdown_value = default_val
                        chart_factory_instance_logger.warning(f"Dropdown for chart {chart_id_cb_factory} has no value. Using default: {default_val}")
                else: 
                    if chart_id_cb_factory == ID_NET_GREEK_FLOW_HEATMAP_CHART_CB:
                        default_val = get_config_value_cb(["visualization_settings", "mspi_visualizer", "greek_flow_heatmap_default_metric"], "heuristic_net_delta_pressure")
                    else: # mspi_heatmap
                        default_val = "mspi_heatmap"
                    selected_metric_dropdown_value = default_val
                    chart_factory_instance_logger.warning(f"Missing expected arg for dropdown (Input) for chart {chart_id_cb_factory}. Using default: {default_val}")

            if len(dynamic_args) > current_dynamic_arg_index:
                current_range_slider_val_chart = dynamic_args[current_dynamic_arg_index]
                current_dynamic_arg_index += 1
            else: 
                chart_factory_instance_logger.warning(f"Missing expected arg for range slider (State) for chart {chart_id_cb_factory}. Using default from config.")
                current_range_slider_val_chart = get_config_value_cb(["visualization_settings","dashboard","defaults","range_pct"], 5.0) 
            
            if is_fig_state_needed:
                if len(dynamic_args) > current_dynamic_arg_index:
                    previous_fig_state_chart = dynamic_args[current_dynamic_arg_index]
            
            chart_factory_instance_logger.info(
                f"Chart Update START for '{chart_id_cb_factory}'. Key: '{cached_data_key_chart}'. "
                f"RangeSliderVal_State: {current_range_slider_val_chart}. " 
                f"SelectedMetric/View: '{selected_metric_dropdown_value}'"
            )

            if not cached_data_key_chart: return create_empty_figure_cb(title=f"{chart_display_title_default} - Waiting for Initial Data")
            data_bundle_chart = get_data_from_server_cache_cb(cached_data_key_chart, server_cache_ref)
            if data_bundle_chart is None: return create_empty_figure_cb(title=f"{chart_display_title_default} - Data Expired/Not Found in Cache")

            error_in_bundle_chart = data_bundle_chart.get("error"); symbol_chart = data_bundle_chart.get("symbol", "N/A")
            if error_in_bundle_chart: return create_empty_figure_cb(title=f"{symbol_chart} - {chart_display_title_default}: Data Error", reason=str(error_in_bundle_chart)[:100]) 
            
            options_df_plot_chart = data_bundle_chart.get("final_metric_rich_df_obj")
            if not isinstance(options_df_plot_chart, pd.DataFrame) or options_df_plot_chart.empty: 
                options_list_plot_chart = data_bundle_chart.get("processed_data", {}).get("options_chain", [])
                if isinstance(options_list_plot_chart, list) and options_list_plot_chart:
                    try: options_df_plot_chart = pd.DataFrame.from_records(options_list_plot_chart)
                    except Exception as e_df_rec_chart: chart_factory_instance_logger.error(f"DF reconstruct error for '{chart_id_cb_factory}': {e_df_rec_chart}", exc_info=True); options_df_plot_chart = pd.DataFrame()
                else: options_df_plot_chart = pd.DataFrame()
            
            if options_df_plot_chart.empty and chart_id_cb_factory != "recommendations_table": return create_empty_figure_cb(f"{symbol_chart} - {chart_display_title_default}: No Chart Data")
            
            if not isinstance(visualizer_instance, MSPIVisualizerV2): 
                chart_factory_instance_logger.error(f"Visualizer instance is not a valid MSPIVisualizerV2 object for chart {chart_id_cb_factory}.")
                return create_empty_figure_cb(f"{chart_display_title_default} - Visualizer Error (Instance Invalid)")

            viz_method_map_final: Dict[str, str] = {
                "mspi_heatmap": "create_mspi_heatmap", 
                "net_volume_pressure_heatmap": "create_net_volume_pressure_heatmap", 
                "net_value_heatmap": "create_net_value_heatmap",
                "mspi_components": "create_component_comparison", "net_volval_comp": "create_volval_comparison",
                "combined_rolling_flow_chart": "create_combined_rolling_flow_chart",
                ID_NET_GREEK_FLOW_HEATMAP_CHART_CB: "create_net_greek_flow_heatmap",
                "volatility_regime": "create_volatility_regime_visualization", "time_decay": "create_time_decay_visualization",
                "sdag_multiplicative": "plot_sdag_multiplicative", "sdag_directional": "plot_sdag_directional",
                "sdag_weighted": "plot_sdag_weighted", "sdag_volatility_focused": "plot_sdag_volatility_focused",
                "key_levels": "create_key_levels_visualization", "trading_signals": "create_trading_signals_visualization",
                "recommendations_table": "create_strategy_recommendations_table"
            }
            
            target_method_name_chart: Optional[str] = None
            if chart_id_cb_factory == "mspi_heatmap": 
                if selected_metric_dropdown_value == "mspi_heatmap":
                    target_method_name_chart = viz_method_map_final["mspi_heatmap"]
                elif selected_metric_dropdown_value == "net_volume_pressure_heatmap":
                    target_method_name_chart = viz_method_map_final["net_volume_pressure_heatmap"]
                else:
                    chart_factory_instance_logger.warning(f"Unknown view '{selected_metric_dropdown_value}' for MSPI card toggle. Defaulting to MSPI heatmap.")
                    target_method_name_chart = viz_method_map_final["mspi_heatmap"]
            else:
                target_method_name_chart = viz_method_map_final.get(chart_id_cb_factory)

            if not target_method_name_chart: 
                chart_factory_instance_logger.error(f"No visualizer method mapped for chart ID '{chart_id_cb_factory}' or selected view '{selected_metric_dropdown_value}'.")
                return create_empty_figure_cb(f"{chart_display_title_default} - Plot Method Not Mapped")
            
            viz_method_to_call = getattr(visualizer_instance, target_method_name_chart, None)
            if not callable(viz_method_to_call):
                chart_factory_instance_logger.error(f"Visualizer method '{target_method_name_chart}' not found or not callable for chart ID '{chart_id_cb_factory}'.")
                return create_empty_figure_cb(f"{chart_display_title_default} - Plot Logic Missing ({target_method_name_chart})")

            args_for_viz_method: Dict[str, Any] = {
                "processed_data": options_df_plot_chart, 
                "symbol": symbol_chart,
                "current_price": data_bundle_chart.get("underlying", {}).get("price"),
                "fetch_timestamp": data_bundle_chart.get("fetch_timestamp"),
                "selected_price_range_pct_override": current_range_slider_val_chart 
            }
            
            if chart_id_cb_factory == ID_NET_GREEK_FLOW_HEATMAP_CHART_CB:
                metric_col_name_actual = selected_metric_dropdown_value 
                if not metric_col_name_actual: 
                    chart_factory_instance_logger.error(f"Net Greek Heatmap: metric_col_name_actual is None or empty for dropdown value '{selected_metric_dropdown_value}'.")
                    return create_empty_figure_cb(f"{chart_display_title_default} - No Greek Metric Column Resolved")
                
                metric_display_config_map = {
                    "heuristic_net_delta_pressure": {"title": "Heuristic Net Delta Pressure", "cs_key": "net_delta_heuristic_heatmap", "cb": "Net Delta (H)"},
                    "net_gamma_flow": {"title": "Net Gamma Flow", "cs_key": "net_gamma_flow_heatmap", "cb": "Net Gamma"},
                    "net_vega_flow": {"title": "Net Vega Flow", "cs_key": "net_vega_flow_heatmap", "cb": "Net Vega"},
                    "net_theta_exposure": {"title": "Net Theta Exposure", "cs_key": "net_theta_exposure_heatmap", "cb": "Net Theta Exp"},
                    "net_delta_flow_total": {"title": "True Net Delta Flow", "cs_key": "true_net_delta_flow_heatmap", "cb": "True Net Delta"},
                    "true_net_volume_flow": {"title": "True Net Volume Flow", "cs_key": "true_net_volume_flow_heatmap", "cb": "True Net Vol"},
                    "true_net_value_flow": {"title": "True Net Value Flow", "cs_key": "true_net_value_flow_heatmap", "cb": "True Net Val"}
                }
                selected_metric_display_details = metric_display_config_map.get(metric_col_name_actual, {"title": metric_col_name_actual.replace("_"," ").title(), "cs_key": "RdBu", "cb": "Value"})
                args_for_viz_method["metric_column_to_plot"] = metric_col_name_actual
                args_for_viz_method["chart_main_title_prefix"] = selected_metric_display_details["title"]
                args_for_viz_method["colorscale_config_key"] = selected_metric_display_details["cs_key"]
                args_for_viz_method["colorbar_title_text"] = selected_metric_display_details["cb"]
            
            elif chart_id_cb_factory == "recommendations_table": args_for_viz_method["recommendations_list"] = data_bundle_chart.get("strategy_recommendations", [])
            elif chart_id_cb_factory == "key_levels": args_for_viz_method["key_levels_data"] = data_bundle_chart.get("key_levels", {})
            elif chart_id_cb_factory == "trading_signals": args_for_viz_method["trading_signals_data"] = data_bundle_chart.get("trading_signals", {})
            elif chart_id_cb_factory == "net_volval_comp": args_for_viz_method["component_history"] = component_history_ref.get(symbol_chart)
            
            if is_fig_state_needed and previous_fig_state_chart:
                trace_visibility_state: Dict[str,Any] = {}
                if isinstance(previous_fig_state_chart,dict) and isinstance(previous_fig_state_chart.get('data'),list):
                    for trace_data_item in previous_fig_state_chart['data']:
                        if isinstance(trace_data_item,dict) and 'name' in trace_data_item: trace_visibility_state[trace_data_item['name']] = trace_data_item.get('visible', True)
                if trace_visibility_state: args_for_viz_method["trace_visibility"] = trace_visibility_state
            
            method_signature_params_final = inspect.signature(viz_method_to_call).parameters
            final_args_to_pass_to_viz = {k:v for k,v in args_for_viz_method.items() if k in method_signature_params_final}

            try: generated_figure_obj = viz_method_to_call(**final_args_to_pass_to_viz)
            except Exception as e_viz_call_final: 
                chart_factory_instance_logger.error(f"Visualizer method '{target_method_name_chart}' failed for {symbol_chart} on chart '{chart_id_cb_factory}': {e_viz_call_final}", exc_info=True)
                return create_empty_figure_cb(f"{symbol_chart} - {chart_display_title_default}: Plot Gen Error", reason=str(e_viz_call_final)[:150]) 
            
            if not isinstance(generated_figure_obj, go.Figure): 
                chart_factory_instance_logger.error(f"Visualizer method '{target_method_name_chart}' for chart '{chart_id_cb_factory}' returned type {type(generated_figure_obj)}, expected go.Figure.")
                return create_empty_figure_cb(f"{chart_display_title_default} - Invalid Plot Output", reason=f"Expected Figure, got {type(generated_figure_obj)}")
            
            chart_factory_instance_logger.info(f"Chart Update END for '{chart_id_cb_factory}'. Duration: {pytime.time() - chart_update_cb_start_time:.3f}s.")
            return generated_figure_obj
        return generated_chart_update_callback 
        
    if isinstance(CHART_IDS_CB, list) and CHART_IDS_CB and _layout_ids_imported_successfully_cb:
        for chart_id_for_registration in CHART_IDS_CB: 
            if isinstance(chart_id_for_registration, str) and chart_id_for_registration:
                 create_chart_update_callback_factory(chart_id_for_registration)
                 logger.debug(f"Dynamic chart update callback registered for: ID='{chart_id_for_registration}'")
            else: logger.error(f"Invalid chart_id ('{chart_id_for_registration}') found in CHART_IDS_CB. Skipping callback registration for it.")
    else: logger.error("CHART_IDS_CB not available from layout, not a list, or empty. Cannot register dynamic chart update callbacks.")

    @app.callback(
        Output(ID_MODE_CONTENT_CB, "children"),
        Input(ID_MODE_TABS_CB, "active_tab")
    )
    def update_active_mode_content(active_tab_id: str) -> html.Div:
        mode_switch_logger = logger.getChild("update_active_mode_content")
        mode_switch_logger.info(f"Mode tab switched to: {active_tab_id}")
        if active_tab_id == ID_TAB_MAIN_DASHBOARD_CB:
            return get_main_dashboard_mode_layout_cb()
        elif active_tab_id == ID_TAB_SDAG_DIAGNOSTICS_CB:
            return get_sdag_diagnostics_mode_layout_cb()
        else:
            mode_switch_logger.warning(f"Unknown tab ID received: {active_tab_id}. Defaulting to main dashboard layout.")
            return get_main_dashboard_mode_layout_cb() 

    logger.info("All dashboard callbacks (V2.4.5 - MSPI Card Toggle) defined and registration process completed.")

