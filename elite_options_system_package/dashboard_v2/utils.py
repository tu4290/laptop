# /home/ubuntu/dashboard_v2/utils.py
# -*- coding: utf-8 -*-
"""
Utility functions for the Enhanced Options Dashboard V2, including
configuration management, plotting helpers, caching interaction wrappers,
and formatting utilities.
(Version: Utils Rewrite - Canon Directive - Full Integration)
"""

# Standard Library Imports
import logging
import time as pytime
import json
import copy
import os
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union

# Third-Party Imports
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dateutil import parser as date_parser
from dash import html # <<<================ ADD THIS IMPORT (or import dash_html_components as html for older Dash)

# Setup logger for this utility module
logger = logging.getLogger(__name__)
# Basic logging config if not already set by a higher-level script (e.g., runner)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] (%(name)s:%(lineno)d) %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

# --- Attempt to Import Styling for Plotly Template ---
# This is used by create_empty_figure.
# If styling.py is unavailable, a very basic Plotly dark template is used.
_PLOTLY_TEMPLATE_FALLBACK = {"layout": go.Layout(template="plotly_dark")}
PLOTLY_TEMPLATE_DARK: Dict[str, Any]
try:
    from .styling import PLOTLY_TEMPLATE_DARK as imported_plotly_template
    PLOTLY_TEMPLATE_DARK = imported_plotly_template
    logger.debug("UTILS.PY: Successfully imported PLOTLY_TEMPLATE_DARK from .styling.")
except ImportError:
    logger.warning("UTILS.PY: Could not import PLOTLY_TEMPLATE_DARK from .styling. Using basic plotly_dark fallback template for empty figures.")
    PLOTLY_TEMPLATE_DARK = _PLOTLY_TEMPLATE_FALLBACK


# --- Configuration Management ---
CONFIG_CACHE: Optional[Dict[str, Any]] = None # Module-level cache for the application configuration
DEFAULT_CONFIG_FILENAME_UTILS: str = "config_v2.json" # Default name for the main config file

def load_app_config(config_path: str = DEFAULT_CONFIG_FILENAME_UTILS) -> Dict[str, Any]:
    """
    Loads the main application configuration from a JSON file.
    The path is resolved assuming 'utils.py' is in a subdirectory (e.g., 'dashboard_v2'),
    and the config file is in the parent directory of that subdirectory (the project root).
    Caches the loaded configuration globally within this module for subsequent calls.
    """
    global CONFIG_CACHE # Declare intent to modify the global CONFIG_CACHE

    # If config is already loaded and cached, return the cached version
    if CONFIG_CACHE is not None:
        logger.debug("UTILS: Returning already cached application configuration.")
        return CONFIG_CACHE

    # Determine the absolute path to the configuration file
    absolute_config_path_to_load: str
    if os.path.isabs(config_path):
        absolute_config_path_to_load = config_path
    else:
        # Assume utils.py is in something like 'project_root/dashboard_v2/utils.py'
        # We want 'project_root/config_v2.json'
        current_script_directory = os.path.dirname(os.path.abspath(__file__)) # e.g., project_root/dashboard_v2
        project_root_directory = os.path.dirname(current_script_directory)    # e.g., project_root
        absolute_config_path_to_load = os.path.normpath(os.path.join(project_root_directory, config_path))
    
    logger.info(f"UTILS: Attempting to load application configuration from: '{absolute_config_path_to_load}'")

    try:
        with open(absolute_config_path_to_load, "r", encoding="utf-8") as f:
            loaded_config_data = json.load(f)
        if not isinstance(loaded_config_data, dict):
            logger.error(f"UTILS: Config file at '{absolute_config_path_to_load}' did not contain a valid JSON dictionary. Using empty config.")
            CONFIG_CACHE = {}
        else:
            CONFIG_CACHE = loaded_config_data
            logger.info(f"UTILS: Successfully loaded and cached application configuration from '{absolute_config_path_to_load}'.")
    except FileNotFoundError:
        logger.error(f"UTILS: Configuration file not found at '{absolute_config_path_to_load}'. Using empty config.")
        CONFIG_CACHE = {}
    except json.JSONDecodeError as e_json:
        logger.error(f"UTILS: Error decoding JSON from '{absolute_config_path_to_load}': {e_json}. Using empty config.")
        CONFIG_CACHE = {}
    except Exception as e_load:
        logger.error(f"UTILS: Unexpected error loading configuration from '{absolute_config_path_to_load}': {e_load}.", exc_info=True)
        CONFIG_CACHE = {}
    
    return CONFIG_CACHE

def get_config_value(path: List[str], default: Any = None) -> Any:
    """
    Safely retrieves a nested value from the globally cached application configuration.
    Args:
        path (List[str]): A list of keys representing the path to the desired value.
        default (Any, optional): The value to return if the path is not found or an error occurs.
                                 Defaults to None.
    Returns:
        Any: The retrieved configuration value or the default.
    """
    # Ensure CONFIG_CACHE is populated. If it's None, load_app_config() will attempt to load it.
    # This function relies on load_app_config to set the global CONFIG_CACHE.
    if CONFIG_CACHE is None:
        logger.warning("UTILS: get_config_value called before CONFIG_CACHE was populated. Attempting to load default config now.")
        load_app_config() # Attempt to load using default path

    # Proceed with current_level starting from the (now hopefully populated) CONFIG_CACHE
    current_level = CONFIG_CACHE
    if not isinstance(current_level, dict): # Should be a dict after load_app_config, even if empty
        logger.error(f"UTILS: CONFIG_CACHE is not a dictionary (type: {type(current_level)}) after load attempt. Cannot retrieve path: {'.'.join(path)}")
        return default

    try:
        for key_segment in path:
            if isinstance(current_level, dict):
                current_level = current_level[key_segment] # This can raise KeyError if key not found
            else:
                # This case occurs if a segment in the path is expected to be a dict but isn't
                logger.debug(f"UTILS: Path traversal for '{'.'.join(path)}' failed at non-dictionary for segment '{key_segment}'. Current level type: {type(current_level)}.")
                return default
        return current_level
    except KeyError:
        # Key not found at some level of the path
        # logger.debug(f"UTILS: Configuration key path '{'.'.join(path)}' not found. Returning default value: {default}")
        return default
    except TypeError:
        # This can happen if a path tries to index a non-dictionary/non-list element
        logger.warning(f"UTILS: Invalid structure or path encountered for '{'.'.join(path)}'. Path segment might be incorrect. Returning default.")
        return default
    except Exception as e_get_val:
        logger.error(f"UTILS: Unexpected error retrieving config value for path '{'.'.join(path)}': {e_get_val}. Returning default.")
        return default

# --- Plotting Utilities ---
def create_empty_figure(title: str = "Waiting for data...", height: Optional[int] = None, reason: str = "N/A") -> go.Figure:
    """ Creates a standard empty Plotly figure with improved styling, using configured defaults. """
    empty_fig_logger = logger.getChild("CreateEmptyFigure")
    empty_fig_logger.debug(f"Creating empty figure with title: '{title}', Reason: '{reason}'")
    
    fig = go.Figure()

    # Get default height from config if not provided, fallback to a sensible default
    if height is None:
        height = get_config_value(["visualization_settings", "dashboard", "default_graph_height"], 600)
        if not isinstance(height, int) or height <= 0: height = 600 # Ensure valid height

    # Use the Plotly template (imported or fallback)
    # The PLOTLY_TEMPLATE_DARK should ideally contain a go.Layout object or be a string template name
    base_fig_layout = {}
    if isinstance(PLOTLY_TEMPLATE_DARK, dict) and "layout" in PLOTLY_TEMPLATE_DARK:
        base_fig_layout = PLOTLY_TEMPLATE_DARK["layout"]
    elif isinstance(PLOTLY_TEMPLATE_DARK, str): # If it's just a template name string
        base_fig_layout = go.Layout(template=PLOTLY_TEMPLATE_DARK)
    else: # Fallback if PLOTLY_TEMPLATE_DARK is not what we expect
        base_fig_layout = go.Layout(template="plotly_dark")
        empty_fig_logger.warning("PLOTLY_TEMPLATE_DARK from styling was not in expected format. Using basic 'plotly_dark'.")
    
    fig.update_layout(base_fig_layout) # Apply base template

    # Customize for the "empty" state
    title_text_with_reason = f"<i>{title}<br><small style='color:grey; font-size:0.8em;'>({reason})</small></i>"
    fig.update_layout(
        title={
            "text": title_text_with_reason,
            "y": 0.5, "x": 0.5, # Center title
            "xanchor": "center", "yanchor": "middle",
            "font": {"size": 16, "color": "#95a5a6"} # Muted color for empty state
        },
        height=height,
        xaxis={"visible": False, "showgrid": False, "zeroline": False}, 
        yaxis={"visible": False, "showgrid": False, "zeroline": False},
        annotations=[], # Clear any default annotations from template
        paper_bgcolor='rgba(0,0,0,0)', # Ensure transparent background
        plot_bgcolor='rgba(0,0,0,0)'   # Ensure transparent plot area
    )
    return fig

# --- Data Handling & Caching Utilities ---

# CACHE_TIMEOUT_SECONDS will be initialized after load_app_config is called at module level
# This ensures get_config_value can access the loaded config.
# If load_app_config has not been called yet (e.g. during initial import by other modules),
# get_config_value will trigger a load itself.
_cache_timeout_config_path = ["system_settings", "dashboard_cache_timeout_seconds"]
_cache_timeout_default = 600 # Default 10 minutes
CACHE_TIMEOUT_SECONDS: int = get_config_value(_cache_timeout_config_path, _cache_timeout_default)
if not isinstance(CACHE_TIMEOUT_SECONDS, int) or CACHE_TIMEOUT_SECONDS < 0:
    logger.warning(f"Invalid CACHE_TIMEOUT_SECONDS '{CACHE_TIMEOUT_SECONDS}' from config. Defaulting to {_cache_timeout_default}s.")
    CACHE_TIMEOUT_SECONDS = _cache_timeout_default
logger.info(f"UTILS: Cache timeout set to {CACHE_TIMEOUT_SECONDS} seconds.")

def generate_cache_key(symbol: str, dte_str: str, range_pct: Optional[Union[int, float]]) -> str:
    """
    Generates a consistent cache key from input parameters.
    Includes symbol, DTE string, range, and current minute for time-sensitivity.
    """
    range_val_for_key = int(range_pct) if isinstance(range_pct, (int, float)) and pd.notna(range_pct) else 0
    # Key format: SYMBOL_DTE-STRING_RANGEpct_YYYYMMDDHHMM
    # Example: /ES:XCME_0-7_5pct_202505091035
    timestamp_minute_str = datetime.now().strftime('%Y%m%d%H%M')
    cache_key_generated = f"{str(symbol).strip().upper()}_{str(dte_str).strip()}_{range_val_for_key}pct_{timestamp_minute_str}"
    logger.debug(f"UTILS: Generated cache key: '{cache_key_generated}'")
    return cache_key_generated

def get_data_from_server_cache(cache_key: Optional[str], server_cache_ref: Dict[str, Tuple[float, Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """
    Retrieves data from the main server-side cache if the key exists and data hasn't expired.
    Returns a deep copy of the cached bundle to prevent mutation issues.
    Expects the cached data bundle to have 'options_chain' as a list-of-dicts.
    """
    cache_get_logger = logger.getChild("CacheGet")
    if not cache_key or not isinstance(cache_key, str):
        cache_get_logger.debug(f"Invalid cache key provided (type: {type(cache_key)}, value: '{cache_key}'). Cannot retrieve from cache.")
        return None
    if cache_key not in server_cache_ref:
        cache_get_logger.debug(f"Key '{cache_key}' not found in server cache.")
        return None

    # Retrieve the timestamp and the data bundle
    stored_timestamp, stored_data_bundle = server_cache_ref[cache_key]

    # Check for expiration
    current_time_secs = pytime.time()
    age_seconds = current_time_secs - stored_timestamp
    if age_seconds > CACHE_TIMEOUT_SECONDS:
        cache_get_logger.info(f"Key '{cache_key}' has expired (Age: {age_seconds:.1f}s > Timeout: {CACHE_TIMEOUT_SECONDS}s). Removing from cache.")
        server_cache_ref.pop(cache_key, None) # Remove the expired entry
        return None

    # Validate the retrieved bundle before returning
    if not isinstance(stored_data_bundle, dict):
        cache_get_logger.error(f"Corrupted cache entry for key '{cache_key}': Stored data is not a dictionary (Type: {type(stored_data_bundle)}). Removing.")
        server_cache_ref.pop(cache_key, None)
        return None
    
    # --- Point C Logging (Verification of 'options_chain' type from cache) ---
    # This is effectively the same as Point D in callbacks, verifying what's IN the cache.
    options_chain_in_bundle = stored_data_bundle.get("processed_data", {}).get("options_chain")
    if isinstance(options_chain_in_bundle, pd.DataFrame):
         cache_get_logger.error(f"CRITICAL CACHE CORRUPTION for key '{cache_key}': 'options_chain' retrieved from cache is a DataFrame! Should be list of dicts.")
         # To prevent further issues, consider invalidating this cache entry
         # server_cache_ref.pop(cache_key, None)
         # return None # Or attempt to convert, but this indicates upstream error in store_data_in_server_cache or processor
    elif not isinstance(options_chain_in_bundle, list):
        cache_get_logger.warning(f"Cache data for key '{cache_key}': 'options_chain' is not a list (Type: {type(options_chain_in_bundle)}). Plotting might fail if expecting list of records.")

    cache_get_logger.debug(f"Retrieved valid (non-expired) data bundle for key: '{cache_key}'. Age: {age_seconds:.1f}s.")
    # Return a DEEP COPY to prevent modification of the cached object by reference
    try:
        return copy.deepcopy(stored_data_bundle)
    except Exception as e_deepcopy:
        cache_get_logger.error(f"Failed to deep copy cached data for key '{cache_key}': {e_deepcopy}. Returning original reference (MUTABLE).", exc_info=True)
        return stored_data_bundle # Fallback to original, but warn about mutability

def store_data_in_server_cache(cache_key: Optional[str], data_bundle_to_store: Dict[str, Any], server_cache_ref: Dict[str, Tuple[float, Dict[str, Any]]]):
    """
    Stores the data bundle in the main server-side cache with a current timestamp.
    Performs a deep copy of the bundle before storing to ensure cache integrity.
    Includes Point C style logging to check 'options_chain' format before storage.
    """
    cache_store_logger = logger.getChild("CacheStore")
    if not cache_key or not isinstance(cache_key, str):
        cache_store_logger.error(f"Invalid cache key provided (type: {type(cache_key)}, value: '{cache_key}'). Cannot store data.")
        return
    if not isinstance(data_bundle_to_store, dict):
        cache_store_logger.error(f"Cannot store non-dictionary data in cache for key '{cache_key}'. Provided data type: {type(data_bundle_to_store)}")
        return

    # --- Create a deep copy of the bundle to be stored ---
    # This prevents the cached version from being modified if the original object is changed elsewhere.
    try:
        bundle_for_storage = copy.deepcopy(data_bundle_to_store)
    except Exception as e_deepcopy_store:
         cache_store_logger.error(f"Failed to deep copy data bundle before caching for key '{cache_key}': {e_deepcopy_store}. Storing original reference (RISK OF MUTATION).", exc_info=True)
         bundle_for_storage = data_bundle_to_store # Fallback, but this is risky

    # --- Point C Logging (Verification of 'options_chain' type BEFORE storing) ---
    options_chain_data_in_bundle = bundle_for_storage.get("processed_data", {}).get("options_chain")
    if isinstance(options_chain_data_in_bundle, list):
        if options_chain_data_in_bundle: # If list is not empty
            first_record_check = options_chain_data_in_bundle[0]
            if isinstance(first_record_check, dict):
                rolling_cols_for_log = [col for col in first_record_check.keys() if 'volmbs_' in col or 'valuebs_' in col]
                if rolling_cols_for_log:
                    rolling_vals_log = {k: (v, type(v).__name__) for k,v in first_record_check.items() if k in rolling_cols_for_log}
                    cache_store_logger.debug(f"UTILS STORE CHECK ({cache_key}): Rolling vals/types in first rec before store: {rolling_vals_log}")
                    if any(not isinstance(val_type_tuple[0], (int, float, type(None), np.number)) for val_type_tuple in rolling_vals_log.values()):
                         cache_store_logger.error(f"UTILS STORE CHECK ({cache_key}): CRITICAL - Rolling column has non-numeric/None type just before storing!")
                else: cache_store_logger.debug(f"UTILS STORE CHECK ({cache_key}): No 'volmbs_' or 'valuebs_' keys in first record of options_chain.")
            else: cache_store_logger.warning(f"UTILS STORE CHECK ({cache_key}): First item in 'options_chain' list is not a dictionary. Type: {type(first_record_check)}")
        else: cache_store_logger.debug(f"UTILS STORE CHECK ({cache_key}): 'options_chain' is an empty list.")
    elif isinstance(options_chain_data_in_bundle, pd.DataFrame):
         cache_store_logger.error(f"UTILS STORE ERROR ({cache_key}): 'options_chain' is a DataFrame JUST BEFORE STORING! Processor's JSON-safe conversion was bypassed or failed.")
    else:
         cache_store_logger.warning(f"UTILS STORE CHECK ({cache_key}): 'options_chain' data type before storing is '{type(options_chain_data_in_bundle)}'. Expected list of dicts.")
    # --- End of Point C Logging ---

    # Store the deep-copied bundle with the current timestamp
    server_cache_ref[cache_key] = (pytime.time(), bundle_for_storage)
    cache_store_logger.info(f"Stored data bundle with key: '{cache_key}'. Current server cache size: {len(server_cache_ref)}")


# --- Formatting & Display Utilities ---
def format_status_message(message: str, is_error: bool = False) -> Tuple[Union[str, html.Div], Dict[str, str]]:
    """
    Formats a status message string for display and returns an appropriate style dictionary.
    Uses html.Div for consistent component type return.
    """
    # Retrieve styles from config, with hardcoded fallbacks for robustness
    base_style_cfg_path = ["visualization_settings", "dashboard", "styles", "status_display", "base"]
    base_style_default = {"padding": "10px", "textAlign": "center", "borderRadius": "5px", "fontSize": "1em", "minHeight": "30px", "fontWeight":"500"}
    base_style = get_config_value(base_style_cfg_path, base_style_default)
    if not isinstance(base_style, dict): base_style = base_style_default # Ensure dict

    final_style: Dict[str,str]
    if is_error:
        error_style_cfg_path = ["visualization_settings", "dashboard", "styles", "status_display", "error"]
        error_style_default = {"color": "#f8d7da", "backgroundColor": "#721c24", "border": "1px solid #f5c6cb"}
        error_style = get_config_value(error_style_cfg_path, error_style_default)
        if not isinstance(error_style, dict): error_style = error_style_default
        final_style = {**base_style, **error_style}
        prefix = "Error: " if not ("error" in message.lower() or "failed" in message.lower() or "critical" in message.lower()) else ""
        display_message = f"{prefix}{message}"
    else:
        # Default to info style for non-errors, or success style if specific keywords found
        is_success_msg = any(keyword in message.lower() for keyword in ["success", "loaded", "completed", "✓"])
        if is_success_msg:
            success_style_cfg_path = ["visualization_settings", "dashboard", "styles", "status_display", "success"]
            success_style_default = {"color": "#d4edda", "backgroundColor": "#155724", "border": "1px solid #c3e6cb"}
            success_style = get_config_value(success_style_cfg_path, success_style_default)
            if not isinstance(success_style, dict): success_style = success_style_default
            final_style = {**base_style, **success_style}
        else: # Info style for fetching, processing, etc.
            info_style_cfg_path = ["visualization_settings", "dashboard", "styles", "status_display", "info"]
            info_style_default = {"color": "#bee5eb", "backgroundColor": "#0c5460", "border": "1px solid #bee5eb"}
            info_style = get_config_value(info_style_cfg_path, info_style_default)
            if not isinstance(info_style, dict): info_style = info_style_default
            final_style = {**base_style, **info_style}
        display_message = message
    
    # Return an html.Div component for consistent rendering in Dash
    return html.Div(display_message), final_style

def parse_timestamp(timestamp_string: Optional[str]) -> Optional[datetime]:
    """
    Safely parses an ISO format timestamp string (or other common formats)
    into a timezone-aware or naive datetime object using dateutil.parser.
    Returns None if parsing fails or input is invalid.
    """
    if not timestamp_string or not isinstance(timestamp_string, str):
        # logger.debug(f"UTILS: parse_timestamp received invalid input: {timestamp_string}")
        return None
    try:
        # isoparse is generally good for ISO 8601 formats.
        # `parse` from dateutil is more flexible for various formats.
        parsed_datetime_object = date_parser.parse(timestamp_string)
        return parsed_datetime_object
    except (ValueError, TypeError, OverflowError) as e_parse:
        logger.warning(f"UTILS: Could not parse timestamp string '{timestamp_string}': {e_parse}")
        return None
    except Exception as e_unknown_parse: # Catch any other unexpected parsing errors
        logger.error(f"UTILS: Unexpected error parsing timestamp string '{timestamp_string}': {e_unknown_parse}", exc_info=True)
        return None

# Example of how get_config_value is intended to be used if this module is imported:
# app_config_main = load_app_config() # Loads or gets from cache
# some_setting = get_config_value(["path","to","setting"], "default_val")
# No need to pass app_config_main to get_config_value due to module-level CONFIG_CACHE.