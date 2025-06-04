# /home/ubuntu/dashboard_v2/enhanced_dashboard_v2.py
# -*- coding: utf-8 -*-
"""
Core Dash application definition script for the Enhanced Options Dashboard V2.

This script initializes the Dash application, loads configuration,
instantiates backend services (Fetcher, Processor, Visualizer, Stateful ITS, TradierFetcher),
sets the application layout by calling the layout module, and registers
all interactive callbacks by calling the callbacks module.
(Version: Canon V2.4.2 - Full System Integration - Final Orchestration)
"""

# Standard Library Imports
import os
import logging
import json # For fallback config loading if utils fails
from collections import deque # For COMPONENT_HISTORY_CACHE type hint
from typing import Dict, Any, Optional, Tuple, Deque, Callable, List

# Third-Party Imports
import dash
import dash_bootstrap_components as dbc
from dash import html # For fallback layout
import pandas as pd # For fallback type hints
import plotly.graph_objects as go # For fallback type hints

# --- Initialize flags for successful import of core dashboard utility modules ---
_core_dashboard_modules_loaded_successfully_app: bool = False
_layout_module_available_app: bool = False
_callbacks_module_available_app: bool = False
_utils_module_available_app: bool = False
_styling_module_available_app: bool = False

# --- Define Fallback Functions and Values FIRST (for this script's direct use if utils fails) ---
_DEFAULT_APP_THEME_FALLBACK_APP: Any = dbc.themes.DARKLY # Default if styling.py fails

def _fallback_get_main_layout_app() -> html.Div:
    """Fallback layout function if layout.py fails to import or its function fails."""
    print("ENH_DASH_APP FALLBACK: Using _fallback_get_main_layout_app.")
    return html.Div([
        html.H1("Fatal Error: Dashboard Layout Module Unavailable", style={'color': '#E74C3C', 'textAlign': 'center', 'padding': '20px'}),
        html.P("The dashboard's core layout definition (layout.py) could not be loaded. Check server logs.", style={'textAlign': 'center', 'color': '#BDC3C7'})
    ], style={"padding": "30px", "fontFamily": "Arial, sans-serif", "backgroundColor": "#2C3E50", "minHeight": "100vh"})

def _fallback_register_callbacks_app(app_instance: dash.Dash, *args: Any, **kwargs: Any) -> None:
    """Fallback callback registration if callbacks.py fails to import or its function fails."""
    print("ENH_DASH_APP FALLBACK: Using _fallback_register_callbacks_app. Dashboard will lack interactivity.")
    pass # Minimal fallback, error messages would ideally be handled by callbacks.py's own fallbacks

_APP_CONFIG_FOR_FALLBACK_APP: Dict[str, Any] = {} # Global for fallback config

def _fallback_load_app_config_app(config_file_path: str = "config_v2.json") -> Dict[str, Any]:
    global _APP_CONFIG_FOR_FALLBACK_APP
    print(f"ENH_DASH_APP FALLBACK: Using _fallback_load_app_config_app for '{config_file_path}'.")
    try:
        # Try to resolve path relative to this script first, then project root relative to CWD
        abs_path = config_file_path
        if not os.path.isabs(config_file_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            path_rel_script = os.path.join(script_dir, config_file_path)
            # Heuristic: config_v2.json is usually in the parent of dashboard_v2 package
            path_rel_project_root = os.path.join(os.path.dirname(script_dir), config_file_path)

            if os.path.exists(path_rel_project_root): abs_path = path_rel_project_root
            elif os.path.exists(path_rel_script): abs_path = path_rel_script
            elif os.path.exists(config_file_path): abs_path = os.path.abspath(config_file_path) # CWD relative
            else: raise FileNotFoundError(f"Config not found at various relative paths based on {config_file_path}")

        with open(abs_path, 'r', encoding='utf-8') as f_config:
            _APP_CONFIG_FOR_FALLBACK_APP = json.load(f_config)
            print(f"ENH_DASH_APP FALLBACK: Successfully loaded config from {abs_path}")
            return _APP_CONFIG_FOR_FALLBACK_APP
    except Exception as e_conf_load_fb_app:
        print(f"ENH_DASH_APP FALLBACK: Direct minimal config load failed for '{config_file_path}': {e_conf_load_fb_app}. Returning empty dict.")
        _APP_CONFIG_FOR_FALLBACK_APP = {} # Ensure it's an empty dict on failure
        return _APP_CONFIG_FOR_FALLBACK_APP

def _fallback_get_config_value_app(path_keys_list: List[str], default_return_value: Any = None) -> Any:
    # This fallback is only used if utils.get_config_value itself fails to import.
    # It accesses _APP_CONFIG_FOR_FALLBACK_APP which _fallback_load_app_config_app tries to populate.
    current_config_level = _APP_CONFIG_FOR_FALLBACK_APP
    try:
        for key_segment in path_keys_list:
            if isinstance(current_config_level, dict): current_config_level = current_config_level[key_segment]
            else: return default_return_value
        return current_config_level
    except (KeyError, TypeError): return default_return_value

# --- Attempt to Import Custom Dashboard Modules ---
# These define the app's structure, appearance, and interactive logic.
try:
    from .layout import get_main_layout as imported_get_main_layout
    _layout_module_available_app = True
    get_main_layout_for_app: Callable[[], dbc.Container] = imported_get_main_layout # type: ignore
    print("ENH_DASH_APP: Successfully imported get_main_layout from .layout.")
except ImportError as e_layout_imp:
    print(f"ENH_DASH_APP CRITICAL: Import Error for layout.py: {e_layout_imp}. Using fallback layout.", exc_info=True)
    get_main_layout_for_app = _fallback_get_main_layout_app

try:
    from .callbacks import register_callbacks as imported_register_callbacks
    _callbacks_module_available_app = True
    register_callbacks_for_app: Callable[..., None] = imported_register_callbacks # type: ignore
    print("ENH_DASH_APP: Successfully imported register_callbacks from .callbacks.")
except ImportError as e_callbacks_imp:
    print(f"ENH_DASH_APP CRITICAL: Import Error for callbacks.py: {e_callbacks_imp}. Using fallback callback registration.", exc_info=True)
    register_callbacks_for_app = _fallback_register_callbacks_app

# Utils is critical for config loading used by this script itself.
utils_load_app_config_func: Callable[[str], Dict[str, Any]]
utils_get_config_value_func: Callable[[List[str], Any], Any]
try:
    from .utils import load_app_config, get_config_value
    utils_load_app_config_func = load_app_config
    utils_get_config_value_func = get_config_value
    _utils_module_available_app = True
    print("ENH_DASH_APP: Successfully imported load_app_config and get_config_value from .utils.")
except ImportError as e_utils_imp:
    print(f"ENH_DASH_APP CRITICAL: Import Error for utils.py: {e_utils_imp}. Using fallback config functions for app setup.", exc_info=True)
    utils_load_app_config_func = _fallback_load_app_config_app
    utils_get_config_value_func = _fallback_get_config_value_app

try:
    from .styling import APP_THEME as imported_app_theme
    _styling_module_available_app = True
    APP_THEME_FINAL: Any = imported_app_theme
    print(f"ENH_DASH_APP: Successfully imported APP_THEME from .styling (Theme: {str(APP_THEME_FINAL)}).")
except ImportError as e_styling_imp:
    print(f"ENH_DASH_APP CRITICAL: Import Error for styling.py: {e_styling_imp}. Using fallback theme.", exc_info=True)
    APP_THEME_FINAL = _DEFAULT_APP_THEME_FALLBACK_APP

_core_dashboard_modules_loaded_successfully_app = all([_layout_module_available_app, _callbacks_module_available_app, _utils_module_available_app, _styling_module_available_app])

try:
    from .darkpool_processor import parse_darkpool_report
    _darkpool_processor_available_app = True
    print("ENH_DASH_APP: Successfully imported parse_darkpool_report from .darkpool_processor.")
except ImportError as e_dp_imp:
    print(f"ENH_DASH_APP WARNING: Import Error for darkpool_processor.py: {e_dp_imp}. Darkpool data will be unavailable.", exc_info=True)
    parse_darkpool_report = None # Fallback
    _darkpool_processor_available_app = False

# --- Backend Service Imports (with Fallbacks if modules are missing/broken) ---
_BACKEND_MODULES_LOADED_FULLY_APP = False
_backend_import_error_msg_app: Optional[str] = None

# Define Fallback Classes for type hinting and graceful degradation
class _FallbackFetcher: pass
class _FallbackTradierFetcher: pass
class _FallbackProcessor: pass
class _FallbackITS: pass
class _FallbackVisualizer: pass

EnhancedDataFetcher_v2_Class: Any = _FallbackFetcher
TradierDataFetcher_Class: Any = _FallbackTradierFetcher
EnhancedDataProcessor_Class: Any = _FallbackProcessor
IntegratedTradingSystem_Class: Any = _FallbackITS
MSPIVisualizerV2_Class: Any = _FallbackVisualizer

try:
    from enhanced_data_fetcher_v2 import EnhancedDataFetcher_v2
    from enhanced_tradier_fetcher_v2 import TradierDataFetcher
    from enhanced_data_processor_v2 import EnhancedDataProcessor
    from integrated_strategies_v2 import IntegratedTradingSystem
    from mspi_visualizer_v2 import MSPIVisualizerV2
    EnhancedDataFetcher_v2_Class = EnhancedDataFetcher_v2
    TradierDataFetcher_Class = TradierDataFetcher
    EnhancedDataProcessor_Class = EnhancedDataProcessor
    IntegratedTradingSystem_Class = IntegratedTradingSystem
    MSPIVisualizerV2_Class = MSPIVisualizerV2
    _BACKEND_MODULES_LOADED_FULLY_APP = True
    print("ENH_DASH_APP: All backend service classes imported successfully.")
except ImportError as backend_imp_err:
    _backend_import_error_msg_app = f"ENH_DASH_APP CRITICAL: Failed to import one or more backend service modules: {backend_imp_err}. Fallbacks will be used."
    logging.critical(_backend_import_error_msg_app, exc_info=True)
except Exception as general_backend_imp_err: # Catch any other unexpected error during imports
    _backend_import_error_msg_app = f"ENH_DASH_APP CRITICAL: Unexpected error importing backend service modules: {general_backend_imp_err}. Fallbacks will be used."
    logging.critical(_backend_import_error_msg_app, exc_info=True)


# --- Logging Setup for This Script (Main App) ---
# This logger will be used for messages from this enhanced_dashboard_v2.py script.
# Individual modules (utils, layout, callbacks, backend services) have their own loggers.
dashboard_app_logger = logging.getLogger(__name__) # Use __name__ for module-specific logger

# --- Global Application Settings & Variables ---
dashboard_app_logger.info("Loading application configuration using the (potentially fallback) load_app_config function...")
# Use the (potentially fallback) utils function to load config.
# Default to "config_v2.json" in the parent directory of this script's package.
_default_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_v2.json")
APP_CONFIG: Dict[str, Any] = utils_load_app_config_func(config_path=_default_config_path)
if not APP_CONFIG: # If fallback was used and still failed, or file was empty
    dashboard_app_logger.critical(f"APP_CONFIG is empty after load attempt from '{_default_config_path}'. Dashboard may not function correctly.")

# --- Define Path to Darkpool Report and Parse Data ---
# _default_config_path is: os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config_v2.json")
# This path points to elite_options_system_package/config_v2.json
# So, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) should give the repo root.
_repo_root_path_app = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DARKPOOL_REPORT_PATH_APP = os.path.join(_repo_root_path_app, "darkpool", "Darkpool Analysis Report for SPY.md")
dashboard_app_logger.info(f"Darkpool report path resolved to: {DARKPOOL_REPORT_PATH_APP}")

if _darkpool_processor_available_app and parse_darkpool_report:
    dashboard_app_logger.info(f"Attempting to parse Darkpool report from: {DARKPOOL_REPORT_PATH_APP}")
    darkpool_data_bundle = parse_darkpool_report(DARKPOOL_REPORT_PATH_APP)
    if darkpool_data_bundle:
        APP_CONFIG['darkpool_data'] = darkpool_data_bundle
        dashboard_app_logger.info("Successfully parsed and stored Darkpool report data in APP_CONFIG.")
        # For debugging, print keys or df head if needed
        # if 'ultra_levels_df' in darkpool_data_bundle and darkpool_data_bundle['ultra_levels_df'] is not None:
        #     dashboard_app_logger.debug(f"Darkpool Ultra Levels DF head:\n{darkpool_data_bundle['ultra_levels_df'].head()}")
    else:
        APP_CONFIG['darkpool_data'] = None # Indicate parsing failed or returned None
        dashboard_app_logger.warning("Failed to parse Darkpool report or report was empty. 'darkpool_data' set to None in APP_CONFIG.")
else:
    APP_CONFIG['darkpool_data'] = None # Indicate processor was not available
    dashboard_app_logger.warning("Darkpool processor module not available. 'darkpool_data' set to None in APP_CONFIG.")

# Set logger level for this script based on (potentially fallback) config
log_level_cfg_path_app = ["system_settings", "log_level"]
log_level_default_app = "INFO"
log_level_str_app = utils_get_config_value_func(log_level_cfg_path_app, log_level_default_app)
if not isinstance(log_level_str_app, str): log_level_str_app = log_level_default_app
try:
    final_app_log_level = getattr(logging, log_level_str_app.upper())
    dashboard_app_logger.setLevel(final_app_log_level)
    # Also set root logger level if this script is the main entry point and utils failed
    if not _utils_module_available_app: logging.getLogger().setLevel(final_app_log_level)
except AttributeError:
    dashboard_app_logger.setLevel(logging.INFO)
    if not _utils_module_available_app: logging.getLogger().setLevel(logging.INFO)
    dashboard_app_logger.warning(f"Invalid log level '{log_level_str_app}' in config. Dashboard App logger defaulting to INFO.")
dashboard_app_logger.info(f"Logging level for '{__name__}' (main app script) set to: {logging.getLevelName(dashboard_app_logger.getEffectiveLevel())} from config.")

# --- Initialize Dash App ---
dashboard_app_logger.info("Initializing Dash application instance...")
app_title_cfg_path = ["visualization_settings", "dashboard", "title"]
app_title_default = "Elite Options Dashboard V2 (Canon)"
app_title_final = utils_get_config_value_func(app_title_cfg_path, app_title_default)
if not isinstance(app_title_final, str): app_title_final = app_title_default

# Determine assets folder relative to this script's location
_current_script_abs_path = os.path.abspath(__file__)
_current_script_dir_name = os.path.dirname(_current_script_abs_path)
_assets_folder_abs_path = os.path.join(_current_script_dir_name, 'assets')
dashboard_app_logger.debug(f"Assets folder path resolved to: {_assets_folder_abs_path}")

app = dash.Dash(
    __name__,
    external_stylesheets=[APP_THEME_FINAL, dbc.icons.BOOTSTRAP], # Use final theme
    suppress_callback_exceptions=True, # Common setting for complex apps
    title=app_title_final,
    assets_folder=_assets_folder_abs_path # Crucial for custom CSS
)
server = app.server # Expose server for deployment (e.g., Gunicorn)
dashboard_app_logger.info(f"Dash application initialized. App Title: '{app_title_final}'. Theme: '{str(APP_THEME_FINAL)}'. Assets Folder: '{_assets_folder_abs_path}'")

# --- Global Server-Side Caches (shared across callbacks) ---
SERVER_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {} # Stores [timestamp, data_bundle]
COMPONENT_HISTORY_CACHE: Dict[str, Deque[Tuple[float, Any]]] = {} # Stores {symbol: deque([(ts, df_slice), ...])}
dashboard_app_logger.info("Global server-side caches (SERVER_CACHE for main data, COMPONENT_HISTORY_CACHE for volval) created.")

# --- Instantiate Backend Components ---
# These instances will be passed to the callbacks module.
FETCHER_INSTANCE: Optional[EnhancedDataFetcher_v2_Class] = None
TRADIER_FETCHER_INSTANCE: Optional[TradierDataFetcher_Class] = None
PROCESSOR_INSTANCE: Optional[EnhancedDataProcessor_Class] = None
ITS_INSTANCE: Optional[IntegratedTradingSystem_Class] = None
VISUALIZER_INSTANCE: Optional[MSPIVisualizerV2_Class] = None

if _core_dashboard_modules_loaded_successfully_app and _BACKEND_MODULES_LOADED_FULLY_APP:
    dashboard_app_logger.info("Attempting to initialize instances of ALL backend services using production classes...")
    try:
        FETCHER_INSTANCE = EnhancedDataFetcher_v2_Class(config_path=_default_config_path) # Use resolved config path
        if hasattr(FETCHER_INSTANCE, 'api') and FETCHER_INSTANCE.api is not None:
             dashboard_app_logger.info("ConvexValue Fetcher instance (EnhancedDataFetcher_v2) created successfully.")
        else: dashboard_app_logger.warning("ConvexValue Fetcher instance created, but API connection might have failed or was not attempted during its init.")
    except Exception as e_fetch_inst:
        dashboard_app_logger.critical(f"Failed to instantiate EnhancedDataFetcher_v2: {e_fetch_inst}", exc_info=True)
        FETCHER_INSTANCE = _FallbackFetcher() # Assign fallback instance on error

    try:
        # Pass the entire loaded APP_CONFIG to TradierFetcher, its __init__ will extract tradier_api_settings
        TRADIER_FETCHER_INSTANCE = TradierDataFetcher_Class(config=APP_CONFIG)
        dashboard_app_logger.info("TradierDataFetcher instance created successfully.")
    except Exception as e_tradier_inst:
        dashboard_app_logger.critical(f"Failed to instantiate TradierDataFetcher: {e_tradier_inst}", exc_info=True)
        TRADIER_FETCHER_INSTANCE = _FallbackTradierFetcher()

    try:
        # Processor needs data_directory from config
        data_dir_for_proc_app = utils_get_config_value_func(["system_settings", "data_directory"], "data_processed_default")
        PROCESSOR_INSTANCE = EnhancedDataProcessor_Class(config_path=_default_config_path, data_dir=data_dir_for_proc_app)
        dashboard_app_logger.info("EnhancedDataProcessor instance created successfully.")
    except Exception as e_proc_inst:
        dashboard_app_logger.critical(f"Failed to instantiate EnhancedDataProcessor: {e_proc_inst}", exc_info=True)
        PROCESSOR_INSTANCE = _FallbackProcessor()

    try:
        ITS_INSTANCE = IntegratedTradingSystem_Class(config_path=_default_config_path)
        dashboard_app_logger.info("IntegratedTradingSystem instance created successfully.")
    except Exception as e_its_inst:
        dashboard_app_logger.critical(f"Failed to instantiate IntegratedTradingSystem: {e_its_inst}", exc_info=True)
        ITS_INSTANCE = _FallbackITS()

    try:
        # Visualizer can also take the full APP_CONFIG; its __init__ extracts the relevant part
        VISUALIZER_INSTANCE = MSPIVisualizerV2_Class(config_data=APP_CONFIG)
        dashboard_app_logger.info("MSPIVisualizerV2 instance created successfully.")
    except Exception as e_viz_inst:
        dashboard_app_logger.critical(f"Failed to instantiate MSPIVisualizerV2: {e_viz_inst}", exc_info=True)
        VISUALIZER_INSTANCE = _FallbackVisualizer()
else:
    dashboard_app_logger.critical("Fallback instances will be used for ALL backend services due to critical import failures of either core dashboard modules or backend modules.")
    FETCHER_INSTANCE = _FallbackFetcher()
    TRADIER_FETCHER_INSTANCE = _FallbackTradierFetcher()
    PROCESSOR_INSTANCE = _FallbackProcessor()
    ITS_INSTANCE = _FallbackITS()
    VISUALIZER_INSTANCE = _FallbackVisualizer()

# --- Set App Layout ---
dashboard_app_logger.info("Setting application layout...")
try:
    app.layout = get_main_layout_for_app() # Use the (potentially fallback) layout function
    dashboard_app_logger.info("Application layout set successfully.")
except Exception as e_app_layout:
    dashboard_app_logger.critical(f"Failed to set application layout via get_main_layout_for_app function: {e_app_layout}", exc_info=True)
    app.layout = html.Div([html.H1("Fatal Error: Dashboard Layout Construction Failed", style={'color':'red', 'textAlign':'center'}), html.Pre(f"Error: {e_app_layout}")])

# --- Register Callbacks ---
# This needs to happen after layout is set and all instances are created.
dashboard_app_logger.info("Registering application callbacks...")
try:
    register_callbacks_for_app( # Use the (potentially fallback) register_callbacks function
        app=app,
        fetcher_instance=FETCHER_INSTANCE,
        tradier_fetcher_instance=TRADIER_FETCHER_INSTANCE,
        processor_instance=PROCESSOR_INSTANCE,
        its_instance=ITS_INSTANCE,
        visualizer_instance=VISUALIZER_INSTANCE,
        server_cache_ref=SERVER_CACHE,
        component_history_ref=COMPONENT_HISTORY_CACHE
    )
    dashboard_app_logger.info("Application callbacks registration initiated.")
except Exception as e_callbacks_reg_main:
    dashboard_app_logger.critical(f"Main call to register_callbacks_for_app failed: {e_callbacks_reg_main}", exc_info=True)
    # Attempt to add a visible error to the layout if possible
    if hasattr(app, 'layout') and app.layout is not None and hasattr(app.layout, 'children'):
        error_alert = dbc.Alert("CRITICAL ERROR: Callback registration failed. Dashboard will not be interactive. Check server logs.", color="danger", dismissable=False, className="m-3 p-3 text-center h4")
        if isinstance(app.layout.children, list): app.layout.children.insert(0, error_alert) # Insert at top
        else: app.layout.children = [error_alert, app.layout.children] if app.layout.children else [error_alert]

# --- Cleanup Function (Optional - called by runner script) ---
def cleanup_cache() -> None:
    """ Example cleanup function that could be called on shutdown. """
    dashboard_app_logger.info("Executing cleanup_cache function from enhanced_dashboard_v2...")
    SERVER_CACHE.clear()
    COMPONENT_HISTORY_CACHE.clear()
    dashboard_app_logger.info("Server-side caches cleared.")

# --- Main Execution Block (for running with `python enhanced_dashboard_v2.py`) ---
if __name__ == "__main__":
    dashboard_app_logger.info(f"--- Running '{os.path.basename(__file__)}' directly for development purposes (V2.4.2) ---")

    run_debug_mode_app_default = True # Default for direct run
    run_debug_mode_app_cfg_path = ["system_settings", "dashboard_debug_mode"]
    run_debug_mode_app = utils_get_config_value_func(run_debug_mode_app_cfg_path, run_debug_mode_app_default)
    if not isinstance(run_debug_mode_app, bool): run_debug_mode_app = run_debug_mode_app_default

    run_host_app_default = "127.0.0.1" # Safer default for direct run
    run_host_app_cfg_path = ["system_settings", "dashboard_host"]
    run_host_address_app = utils_get_config_value_func(run_host_app_cfg_path, run_host_app_default)
    if not isinstance(run_host_address_app, str): run_host_address_app = run_host_app_default

    run_port_app_default = 8050
    run_port_app_cfg_path = ["system_settings", "dashboard_port"]
    run_port_number_app = utils_get_config_value_func(run_port_app_cfg_path, run_port_app_default)
    if not isinstance(run_port_number_app, int): run_port_number_app = run_port_app_default

    dashboard_app_logger.info(f"Starting Dash development server (Direct Run) on http://{run_host_address_app}:{run_port_number_app}/")
    dashboard_app_logger.info(f"Direct Run Debug Mode: {run_debug_mode_app}")

    if not _core_dashboard_modules_loaded_successfully_app or not _BACKEND_MODULES_LOADED_FULLY_APP:
        dashboard_app_logger.critical("\n" + "!"*80 +
                                         "\n!!! WARNING: DASHBOARD RUNNING IN DEGRADED STATE (IMPORT OR INSTANTIATION FAILURES) !!!" +
                                         "\n!!! Review console output above for specific IMPORT/INIT ERROR messages." +
                                         "\n" + "!"*80)
    try:
        app.run_server( # Use run_server for more control, equivalent to app.run
            debug=run_debug_mode_app,
            host=run_host_address_app,
            port=run_port_number_app,
            use_reloader=False # Typically False for production or if stateful components are tricky with reloader
        )
    except Exception as e_app_run_main:
        dashboard_app_logger.critical(f"Failed to start Dash server on direct run: {e_app_run_main}", exc_info=True)
    finally:
        cleanup_cache() # Call cleanup when server stops (e.g., Ctrl+C)
        dashboard_app_logger.info("enhanced_dashboard_v2.py direct run finished.")