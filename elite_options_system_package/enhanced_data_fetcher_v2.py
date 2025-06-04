# enhanced_data_fetcher_v2.py
# (Elite Version 2.0.3 - V2.4 API PARAMS REFINED - Canon Directive Integration Update)

# Standard Library Imports
import os
import traceback
import time
import logging
import json
import random
import re # Added for DTE parsing
from datetime import datetime, date, timedelta # Added date/timedelta
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from functools import wraps

# Third-Party Imports
import pandas as pd
import numpy as np
import requests # Required for specific exception handling during connection

# Specific Third-Party Imports (API Wrapper)
try:
    from convexlib.api import ConvexApi
    CONVEXLIB_AVAILABLE = True
    _api_import_error = None
except ImportError as import_error:
    print(f"FATAL ERROR: Could not import ConvexApi: {import_error}. Ensure 'convexlib' is installed (pip install git+https://github.com/convexvalue/convexlib.git).")
    CONVEXLIB_AVAILABLE = False
    _api_import_error = import_error
    class ConvexApi: pass # Dummy class for type hinting if import fails
except Exception as general_import_err:
    print(f"FATAL ERROR: An unexpected error occurred during ConvexApi import: {general_import_err}")
    CONVEXLIB_AVAILABLE = False
    _api_import_error = general_import_err
    class ConvexApi: pass # Dummy class

# --- Logging Setup ---
log_level_str = os.environ.get("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(
    level=log_level,
    format='[%(levelname)s] (%(name)s:%(lineno)d) %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.info("--- EnhancedDataFetcher_v2 Script Loading (Version 2.0.3 - V2.4 API PARAMS REFINED - Canon Directive Integration Update) ---")
if not CONVEXLIB_AVAILABLE:
    logger.critical(f"Convexlib is not available. Fetcher cannot function. Error: {_api_import_error}")

# --- Constants ---
UNDERLYING_REQUIRED_PARAMS: List[str] = [
    "price", "volatility", "day_volume", "call_gxoi", "put_gxoi",
    "gammas_call_buy", "gammas_call_sell", "gammas_put_buy", "gammas_put_sell",
    "deltas_call_buy", "deltas_call_sell", "deltas_put_buy", "deltas_put_sell",
    "vegas_call_buy", "vegas_call_sell", "vegas_put_buy", "vegas_put_sell",
    "thetas_call_buy", "thetas_call_sell", "thetas_put_buy", "thetas_put_sell",
    "call_vxoi", "put_vxoi", "value_bs", "volm_bs", "deltas_buy", "deltas_sell",
    "vegas_buy", "vegas_sell", "thetas_buy", "thetas_sell", "volm_call_buy",
    "volm_put_buy", "volm_call_sell", "volm_put_sell", "value_call_buy",
    "value_put_buy", "value_call_sell", "value_put_sell", "vflowratio",
    "dxoi", "gxoi", "vxoi", "txoi", "call_dxoi", "put_dxoi"
]
logger.info(f"V2.0.3 (Canon Update): UNDERLYING_REQUIRED_PARAMS set to {len(UNDERLYING_REQUIRED_PARAMS)} items ('day_open_price' EXCLUDED).")

OPTIONS_CHAIN_REQUIRED_PARAMS: List[str] = [
  "price", "volatility", "multiplier", "oi", "delta", "gamma", "theta", "vega",
  "vanna", "vomma", "charm", "dxoi", "gxoi", "vxoi", "txoi", "vannaxoi", "vommaxoi", "charmxoi",
  "dxvolm", "gxvolm", "vxvolm", "txvolm", "vannaxvolm", "vommaxvolm", "charmxvolm",
  "value_bs", "volm_bs", "deltas_buy", "deltas_sell", "gammas_buy", "gammas_sell",
  "vegas_buy", "vegas_sell", "thetas_buy", "thetas_sell",
  "valuebs_5m", "volmbs_5m", "valuebs_15m", "volmbs_15m",
  "valuebs_30m", "volmbs_30m", "valuebs_60m", "volmbs_60m",
  "volm", "volm_buy", "volm_sell", "value_buy", "value_sell"
]
logger.info(f"V2.0.3 (Canon Update): OPTIONS_CHAIN_REQUIRED_PARAMS set to {len(OPTIONS_CHAIN_REQUIRED_PARAMS)} items for V2.4 needs (includes volm, volm_buy/sell, value_buy/sell).")

NUMERIC_COLUMNS_OPTIONS: List[str] = [
    'strike', 'price', 'volatility', 'multiplier', 'oi', 'delta', 'gamma', 'theta', 'vega',
    'vanna', 'vomma', 'charm', 'dxoi', 'gxoi', 'vxoi', 'txoi', 'vannaxoi', 'vommaxoi',
    'charmxoi', 'dxvolm', 'gxvolm', 'vxvolm', 'txvolm',
    'vannaxvolm', 'vommaxvolm', 'charmxvolm',
    'value_bs', 'volm_bs', 'deltas_buy', 'deltas_sell',
    'gammas_buy', 'gammas_sell', 'vegas_buy', 'vegas_sell', 'thetas_buy', 'thetas_sell',
    'valuebs_5m', 'volmbs_5m', 'valuebs_15m', 'volmbs_15m',
    'valuebs_30m', 'volmbs_30m', 'valuebs_60m', 'volmbs_60m',
    'volm', 'volm_buy', 'volm_sell', 'value_buy', 'value_sell'
]
NUMERIC_COLUMNS_UNDERLYING: List[str] = [
    "price", "volatility", "day_volume", "call_gxoi", "put_gxoi",
    "gammas_call_buy", "gammas_call_sell", "gammas_put_buy", "gammas_put_sell",
    "deltas_call_buy", "deltas_call_sell", "deltas_put_buy", "deltas_put_sell",
    "vegas_call_buy", "vegas_call_sell", "vegas_put_buy", "vegas_put_sell",
    "thetas_call_buy", "thetas_call_sell", "thetas_put_buy", "thetas_put_sell",
    "call_vxoi", "put_vxoi", "value_bs", "volm_bs", "deltas_buy", "deltas_sell",
    "vegas_buy", "vegas_sell", "thetas_buy", "thetas_sell", "volm_call_buy",
    "volm_put_buy", "volm_call_sell", "volm_put_sell", "value_call_buy",
    "value_put_buy", "value_call_sell", "value_put_sell", "vflowratio",
    "dxoi", "gxoi", "vxoi", "txoi", "call_dxoi", "put_dxoi"
]
DEFAULT_FETCHER_CONFIG_PATH: str = "config_v2.json"

def _load_config_from_file(config_path: str) -> Dict:
    logger.debug(f"Fetcher Helper: Attempting to load config. Provided path: {config_path}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.isabs(config_path): absolute_config_path = config_path
    else: absolute_config_path = os.path.join(script_dir, config_path)
    logger.debug(f"Fetcher Helper: Resolved absolute config path to: {absolute_config_path}")
    try:
        with open(absolute_config_path, 'r') as f: config_data = json.load(f)
        logger.info(f"Fetcher Helper: Loaded full config from {absolute_config_path}")
        return config_data
    except FileNotFoundError: logger.error(f"Fetcher Helper: Config file not found at '{absolute_config_path}'. Returning empty dict."); return {}
    except json.JSONDecodeError as e: logger.error(f"Fetcher Helper: Error decoding JSON from '{absolute_config_path}': {e}. Returning empty dict."); return {}
    except Exception as e: logger.error(f"Fetcher Helper: Unexpected error loading config: {e}. Returning empty dict."); return {}

def retry_with_backoff(retries: int, base_delay: float, max_delay: float, jitter: bool = True):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0; current_delay = base_delay; last_exception = None
            while attempts < retries:
                try: return func(*args, **kwargs)
                except (requests.exceptions.RequestException, requests.exceptions.Timeout, ConnectionError) as e:
                    attempts += 1; last_exception = e
                    logger.warning(f"Retryable error in {func.__name__} (Attempt {attempts}/{retries}): {type(e).__name__} - {e}")
                    if attempts < retries:
                        sleep_time = current_delay + (random.uniform(0, current_delay * 0.1) if jitter else 0)
                        logger.info(f"Retrying {func.__name__} in {sleep_time:.2f} seconds...")
                        time.sleep(sleep_time)
                        current_delay = min(current_delay * 2, max_delay)
                    else:
                        logger.error(f"Max retries ({retries}) reached for {func.__name__}.")
                        raise last_exception from e
                except Exception as e:
                    logger.error(f"Non-retryable error in {func.__name__}: {e}", exc_info=True)
                    raise
            if last_exception: raise last_exception
            logger.error(f"Exited retry loop for {func.__name__} unexpectedly without raising or returning.");
            return None
        return wrapper
    return decorator

class EnhancedDataFetcher_v2:
    def __init__(self, config_path: str = DEFAULT_FETCHER_CONFIG_PATH):
        logger.info(f"Initializing EnhancedDataFetcher_v2 (Version 2.0.3 - V2.4 API PARAMS REFINED - Canon Directive Integration Update) using config: {config_path}...")
        self.config_path = config_path
        self.config: Dict = _load_config_from_file(self.config_path)

        if not CONVEXLIB_AVAILABLE:
            logger.critical("Convexlib not loaded. Fetcher cannot function. Ensure 'convexlib' is installed.")
            self.api: Optional[ConvexApi] = None
            self._configure_defaults_without_api()
            return

        self.email: Optional[str] = None
        self.password: Optional[str] = None
        self._load_credentials()

        fetcher_settings = self.config.get("data_fetcher_settings", {})
        self.max_retries: int = int(fetcher_settings.get("max_retries", 3))
        self.base_retry_delay: float = float(fetcher_settings.get("base_retry_delay_seconds", 1.0))
        self.max_retry_delay: float = float(fetcher_settings.get("max_retry_delay_seconds", 10.0))
        self.inter_call_delay: float = float(fetcher_settings.get("inter_call_delay_seconds", 0.3))
        self.default_dte_range: List[int] = fetcher_settings.get("default_dte_range", [0, 1, 7])
        self.default_price_range_pct: float = float(fetcher_settings.get("default_price_range_pct", 0.05))

        self.api: Optional[ConvexApi] = None
        if not (self.email and self.password):
            logger.error("API email or password missing. Cannot connect to ConvexApi.")
        else:
            try:
                connect_with_retry_decorated = retry_with_backoff(self.max_retries, self.base_retry_delay, self.max_retry_delay)(self._connect)
                connect_with_retry_decorated()
            except Exception as connect_err:
                logger.error(f"Initial API connection failed after retries: {connect_err}")
                self.api = None

        if self.api:
            logger.info("EnhancedDataFetcher_v2 Initialized and Connected to ConvexApi.")
        else:
            logger.warning("EnhancedDataFetcher_v2 Initialized, but API connection FAILED.")

    def _configure_defaults_without_api(self):
        logger.warning("Configuring fetcher with safe default settings as API library is unavailable.")
        self.email=None; self.password=None; self.max_retries=0;
        self.base_retry_delay=1.0; self.max_retry_delay=1.0; self.inter_call_delay=0.1;
        self.default_dte_range=[0]; self.default_price_range_pct=0.05;

    def _load_credentials(self):
        logger.debug("Loading API credentials...")
        api_creds_config = self.config.get("api_credentials", {})
        email_env_var = api_creds_config.get("email_env_var", "CONVEX_EMAIL")
        password_env_var = api_creds_config.get("password_env_var", "CONVEX_PASSWORD")

        self.email = os.getenv(email_env_var)
        self.password = os.getenv(password_env_var)

        if self.email and self.password:
            logger.info(f"Loaded API credentials from Environment Variables ('{email_env_var}', Password Redacted).")
        else:
            logger.warning(f"API credentials not found in Environment Variables ({email_env_var}, {password_env_var}). Checking config file '{self.config_path}' for fallback.")
            self.email = api_creds_config.get("convex_email")
            self.password = api_creds_config.get("convex_password")
            if self.email and self.password:
                logger.warning("Loaded API credentials from config file. This is less secure than using environment variables.")
            else:
                logger.error("API Credentials NOT FOUND in Environment Variables or config file fallback. API connection will fail.")
                self.email = None; self.password = None

    def _connect(self) -> None:
        if not CONVEXLIB_AVAILABLE:
            raise ConnectionError("ConvexApi library is not available. Cannot connect.")
        if not (self.email and self.password):
            raise ConnectionError("API credentials (email/password) are not configured. Cannot connect.")

        logger.info(f"Attempting API connection for user: {self.email[:3]}***...")
        self.api = ConvexApi(self.email, self.password)
        if self.api is None:
            raise ConnectionError("ConvexApi constructor returned None, connection failed.")
        logger.info("ConvexApi instance created. Connection presumed successful or ready.")

    def _parse_underlying_response(self, symbol: str, raw_response: Dict, required_params_ordered: List[str]) -> Dict[str, Any]:
        symbol_upper = symbol.upper()
        logger.debug(f"Parsing underlying response for '{symbol_upper}' using {len(required_params_ordered)} ordered params.")
        parsed_result: Dict[str, Any] = {'symbol': symbol_upper, 'error': None}

        for param in required_params_ordered:
            parsed_result[param] = None

        if not isinstance(raw_response, dict):
            msg = f"Invalid raw response format for {symbol_upper}. Expected dict, got {type(raw_response)}."
            logger.warning(msg)
            parsed_result['error'] = msg
            return parsed_result

        data_container = raw_response.get('data')
        if not isinstance(data_container, list) or not data_container:
            msg = f"API 'data' for {symbol_upper} is not a list or is empty: {str(raw_response)[:200]}"
            logger.error(msg)
            parsed_result['error'] = msg
            return parsed_result

        symbol_data_values_list: List[Any]
        first_item_in_data_container = data_container[0]

        if isinstance(first_item_in_data_container, list):
            if len(first_item_in_data_container) > 0 and isinstance(first_item_in_data_container[0], list):
                if len(first_item_in_data_container) == 1:
                    symbol_data_values_list = first_item_in_data_container[0]
                    logger.debug(f"Unwrapped doubly-nested list for {symbol_upper}.")
                else:
                    symbol_data_values_list = first_item_in_data_container
            else:
                symbol_data_values_list = first_item_in_data_container
        else:
            msg = f"API 'data[0]' for {symbol_upper} is malformed: expected a list, got {type(first_item_in_data_container)}. Content: {str(first_item_in_data_container)[:200]}"
            logger.error(msg)
            parsed_result['error'] = msg
            return parsed_result

        if not isinstance(symbol_data_values_list, list) or not symbol_data_values_list:
            msg = f"Effective API 'data' row for {symbol_upper} is malformed or empty after unwrapping attempts: {str(symbol_data_values_list)[:200]}"
            logger.error(msg)
            parsed_result['error'] = msg
            return parsed_result

        api_symbol_value = symbol_data_values_list[0]
        if not isinstance(api_symbol_value, str) or api_symbol_value.upper() != symbol_upper:
            msg = (f"Symbol mismatch or format error in 'data' row for {symbol_upper}. "
                   f"Expected '{symbol_upper}', got '{api_symbol_value}'. Full row (first 200 chars): {str(symbol_data_values_list)[:200]}")
            logger.error(msg)
            current_err = parsed_result.get('error')
            parsed_result['error'] = f"{current_err}. {msg}".strip('. ') if current_err else msg

        actual_values_from_api = symbol_data_values_list[1:]
        num_actual_values = len(actual_values_from_api)
        num_expected_params = len(required_params_ordered)

        if num_actual_values != num_expected_params:
            msg = (f"CRITICAL Positional Mapping Warning for {symbol_upper} (get_und): "
                   f"Requested {num_expected_params} params, API returned {num_actual_values} values. "
                   f"Data WILL BE MISALIGNED or incomplete! Requested: {required_params_ordered}")
            logger.error(msg)
            current_err = parsed_result.get('error')
            parsed_result['error'] = f"{current_err}. {msg}".strip('. ') if current_err else msg

        max_len_to_parse = min(num_actual_values, num_expected_params)
        for i in range(max_len_to_parse):
            param_name = required_params_ordered[i]
            raw_val = actual_values_from_api[i]
            try:
                if param_name in NUMERIC_COLUMNS_UNDERLYING:
                    num_val = pd.to_numeric(raw_val, errors='coerce')
                    parsed_result[param_name] = None if pd.isna(num_val) else float(num_val)
                else:
                    parsed_result[param_name] = str(raw_val) if raw_val is not None else None
            except (ValueError, TypeError) as e_conv:
                logger.warning(
                    f"Underlying Parse ({symbol_upper}): Conversion error for '{param_name}', raw value '{raw_val}': {e_conv}. Setting to None.")
                parsed_result[param_name] = None

        if num_expected_params > num_actual_values:
            for i in range(num_actual_values, num_expected_params):
                param_name_missed = required_params_ordered[i]
                logger.warning(
                    f"Underlying Parse ({symbol_upper}): No value returned by API for requested param '{param_name_missed}' (expected at index {i}). It remains None.")
        elif num_actual_values > num_expected_params:
            logger.warning(
                f"Underlying Parse ({symbol_upper}): API returned {num_actual_values - num_expected_params} MORE value(s) than expected. These extra values are ignored."
            )

        price_val = parsed_result.get('price')
        if price_val is None or not isinstance(price_val, (int, float)) or price_val <= 0:
            err_price_msg = f"Essential 'price' for {symbol_upper} is invalid, missing, or non-positive (value: {price_val})."
            logger.error(f"Underlying Parse ({symbol_upper}): {err_price_msg}")
            current_err = parsed_result.get('error')
            parsed_result['error'] = f"{current_err}. {err_price_msg}".strip('. ') if current_err else err_price_msg

        logger.debug(f"Underlying Parse ({symbol_upper}): Complete. Price: {price_val}, Error: '{parsed_result.get('error')}'")
        return parsed_result

    @retry_with_backoff(retries=3, base_delay=0.5, max_delay=3.0)
    def _fetch_underlying_data_internal_with_retry(self, symbol: str, params: List[str]) -> Dict[str, Any]:
        symbol_upper = symbol.upper()
        fetch_timestamp = datetime.now().isoformat()
        logger.info(f"Fetching underlying data for: {symbol_upper} (Params: {params})")

        if not self.api:
            logger.error(f"Fetch Underlying ({symbol_upper}): API not connected. Cannot fetch.")
            return {'symbol': symbol_upper, 'error': "API not connected.", 'fetch_timestamp': fetch_timestamp}

        try:
            raw_data = self.api.get_und(symbols=[symbol_upper], params=params)
            logger.debug(f"Fetch Underlying ({symbol_upper}): Raw API response received: {str(raw_data)[:500]}")

            parsed_data = self._parse_underlying_response(symbol_upper, raw_data, params)
            if not isinstance(parsed_data, dict):
                 logger.error(f"Fetch Underlying ({symbol_upper}): Parser returned non-dict ({type(parsed_data)}). Defaulting error.")
                 parsed_data = {'symbol': symbol_upper, 'error': f"Parsing returned invalid type: {type(parsed_data)}"}

            parsed_data['fetch_timestamp'] = fetch_timestamp
            return parsed_data
        except AttributeError as e_attr:
            logger.error(f"Fetch Underlying ({symbol_upper}): API attribute error: {e_attr}.")
            raise ConnectionError(f"API Attribute Error: {e_attr}") from e_attr
        except Exception as e_unexp:
            logger.error(f"Fetch Underlying ({symbol_upper}): Unexpected error: {e_unexp}", exc_info=True)
            return {
                'symbol': symbol_upper,
                'error': f"Unexpected exception during fetch: {type(e_unexp).__name__} - {e_unexp}",
                'fetch_timestamp': fetch_timestamp
            }

    def fetch_options_chain(self, symbol: str, dte_list: Optional[List[int]] = None, price_range_pct: Optional[float] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        symbol_upper = symbol.upper()
        eff_dte_list = dte_list if dte_list is not None else self.default_dte_range
        eff_price_range_input_pct = price_range_pct if price_range_pct is not None else self.default_price_range_pct * 100
        eff_price_range_decimal_for_api = eff_price_range_input_pct / 100.0

        logger.info(f"Fetching options chain for {symbol_upper}. DTEs: {eff_dte_list}, Price Range: {eff_price_range_input_pct:.1f}% (API decimal: {eff_price_range_decimal_for_api:.3f})")

        if not self.api:
            logger.error(f"Options Chain Fetch ({symbol_upper}): API not connected. Aborting fetch operation.")
            return pd.DataFrame(), {"symbol": symbol_upper, "error": "API not connected."}

        underlying_data_result: Dict[str, Any]
        try:
            underlying_data_result = self._fetch_underlying_data_internal_with_retry(symbol_upper, UNDERLYING_REQUIRED_PARAMS)
        except Exception as e_und_fetch:
            logger.error(f"Options Chain ({symbol_upper}): Failed to fetch underlying data after retries: {e_und_fetch}")
            underlying_data_result = {"symbol": symbol_upper, "error": f"Underlying fetch failed: {e_und_fetch}", "fetch_timestamp": datetime.now().isoformat()}
            return pd.DataFrame(), underlying_data_result

        current_underlying_price = underlying_data_result.get("price")
        if underlying_data_result.get("error") or not (isinstance(current_underlying_price, (int, float)) and current_underlying_price > 0):
            existing_error = underlying_data_result.get("error", "")
            price_error = ""
            if not (isinstance(current_underlying_price, (int, float)) and current_underlying_price > 0):
                price_error = f"Underlying price invalid or missing (price: {current_underlying_price})."

            combined_error = f"{existing_error}. {price_error}".strip(". ")
            logger.error(f"Options Chain ({symbol_upper}): Cannot proceed with chain fetch due to invalid underlying data: {combined_error}")
            underlying_data_result["error"] = combined_error
            return pd.DataFrame(), underlying_data_result

        logger.debug(f"Options Chain ({symbol_upper}): Successfully fetched underlying. Using price: {current_underlying_price:.4f}.")

        try:
            api_chain_params = {"params": OPTIONS_CHAIN_REQUIRED_PARAMS, "exps": eff_dte_list, "rng": eff_price_range_decimal_for_api}
            logger.debug(f"Options Chain Fetch ({symbol_upper}): Calling API.get_chain_as_rows with: {api_chain_params}")

            decorated_chain_fetch = retry_with_backoff(self.max_retries, self.base_retry_delay, self.max_retry_delay)(self.api.get_chain_as_rows)
            raw_options_rows: List[List[Any]] = decorated_chain_fetch(symbol_upper, **api_chain_params)

            if not raw_options_rows or not isinstance(raw_options_rows, list):
                logger.warning(f"Options Chain ({symbol_upper}): No data returned from get_chain_as_rows or unexpected format (type: {type(raw_options_rows)}). Returning empty DataFrame.")
                return pd.DataFrame(), underlying_data_result

            logger.info(f"Options Chain ({symbol_upper}): Received {len(raw_options_rows)} raw option rows from API.")

            prefix_columns = ["symbol_contract_api", "expiration_val_api", "strike_api", "opt_kind_api"]
            df_column_names = prefix_columns + OPTIONS_CHAIN_REQUIRED_PARAMS

            num_expected_total_cols = len(df_column_names)
            processed_option_rows: List[List[Any]] = []

            # Determine indices for logging - this should be done once before the loop
            indices_determined = False
            log_strike_idx, log_vommaxoi_idx, log_charmxoi_idx, log_dxvolm_idx = -1,-1,-1,-1 # Default to -1
            try:
                base_param_idx = len(prefix_columns)
                # Ensure all keys exist in OPTIONS_CHAIN_REQUIRED_PARAMS before calling .index()
                if 'vommaxoi' in OPTIONS_CHAIN_REQUIRED_PARAMS:
                    log_vommaxoi_idx = base_param_idx + OPTIONS_CHAIN_REQUIRED_PARAMS.index('vommaxoi')
                if 'charmxoi' in OPTIONS_CHAIN_REQUIRED_PARAMS:
                    log_charmxoi_idx = base_param_idx + OPTIONS_CHAIN_REQUIRED_PARAMS.index('charmxoi')
                if 'dxvolm' in OPTIONS_CHAIN_REQUIRED_PARAMS:
                    log_dxvolm_idx   = base_param_idx + OPTIONS_CHAIN_REQUIRED_PARAMS.index('dxvolm')
                if 'strike_api' in prefix_columns:
                    log_strike_idx   = prefix_columns.index('strike_api')

                # Check if all critical logging indices were found
                if all(idx != -1 for idx in [log_strike_idx, log_vommaxoi_idx, log_charmxoi_idx, log_dxvolm_idx]):
                    indices_determined = True
                else:
                    missing_indices_for_log = []
                    if log_strike_idx == -1: missing_indices_for_log.append("'strike_api' in prefix_columns")
                    if log_vommaxoi_idx == -1: missing_indices_for_log.append("'vommaxoi' in OPTIONS_CHAIN_REQUIRED_PARAMS")
                    if log_charmxoi_idx == -1: missing_indices_for_log.append("'charmxoi' in OPTIONS_CHAIN_REQUIRED_PARAMS")
                    if log_dxvolm_idx == -1: missing_indices_for_log.append("'dxvolm' in OPTIONS_CHAIN_REQUIRED_PARAMS")
                    logger.error(f"Could not determine all indices for logging raw API data. Missing: {', '.join(missing_indices_for_log)}. Logging will be limited.")

            except ValueError as ve_idx: # Catch if .index() fails
                logger.error(f"ValueError during index determination for logging raw API data: {ve_idx}. Logging will be limited.")
            except Exception as e_idx: # Catch any other error during index determination
                logger.error(f"Unexpected error during index determination for logging raw API data: {e_idx}. Logging will be limited.")


            for i, row_tuple in enumerate(raw_options_rows):
                row_list = list(row_tuple)
                current_row_len = len(row_list)

                # START: Integrated logging block
                if indices_determined:
                    try:
                        # Ensure indices are within the bounds of the current row_list
                        strike_val_raw_log = row_list[log_strike_idx] if log_strike_idx < current_row_len else "N/A_idx_strike"
                        vommaxoi_raw_log = row_list[log_vommaxoi_idx] if log_vommaxoi_idx < current_row_len else "N/A_idx_vomma"
                        charmxoi_raw_log = row_list[log_charmxoi_idx] if log_charmxoi_idx < current_row_len else "N/A_idx_charm"
                        dxvolm_raw_log   = row_list[log_dxvolm_idx] if log_dxvolm_idx < current_row_len else "N/A_idx_dxvolm"

                        logger.debug(
                            f"Raw API Data (Row {i}, Strike: {strike_val_raw_log}): "
                            f"vommaxoi_raw='{vommaxoi_raw_log}' (type: {type(vommaxoi_raw_log).__name__}), "
                            f"CHARMXOI_RAW='{charmxoi_raw_log}' (type: {type(charmxoi_raw_log).__name__}), " # Highlighted
                            f"dxvolm_raw='{dxvolm_raw_log}' (type: {type(dxvolm_raw_log).__name__})"
                        )
                    except IndexError:
                        logger.warning(f"Raw API Data (Row {i}): IndexError during logging, row length {current_row_len} may be less than expected logging indices.")
                    except Exception as e_log_raw:
                        logger.warning(f"Raw API Data (Row {i}): Error during raw data logging: {e_log_raw}")
                else: # Fallback if indices couldn't be determined
                    if i < 5: # Log first few full rows
                         logger.debug(f"Raw API Data (Row {i}, Full, indices undetermined): {row_list}")
                # END: Integrated logging block

                if current_row_len < num_expected_total_cols:
                    row_list.extend([None] * (num_expected_total_cols - current_row_len))
                elif current_row_len > num_expected_total_cols:
                    row_list = row_list[:num_expected_total_cols]
                processed_option_rows.append(row_list)

            if not processed_option_rows:
                logger.warning(f"Options Chain ({symbol_upper}): No valid rows after length adjustment. Returning empty DataFrame.")
                return pd.DataFrame(), underlying_data_result

            options_df = pd.DataFrame(processed_option_rows, columns=df_column_names)
            logger.debug(f"Options Chain ({symbol_upper}): Initial DataFrame created. Shape: {options_df.shape}, Columns: {options_df.columns.tolist()}")

            options_df["fetch_timestamp"] = datetime.now().isoformat()
            options_df["underlying_price_at_fetch"] = current_underlying_price
            options_df["underlying_symbol"] = symbol_upper

            epoch_date = datetime(1970, 1, 1)
            options_df["expiration_val_api"] = pd.to_numeric(options_df["expiration_val_api"], errors='coerce')
            options_df["expiration_date"] = pd.to_timedelta(options_df["expiration_val_api"], unit='D', errors='coerce') + epoch_date
            options_df["expiration_date"] = options_df["expiration_date"].dt.strftime('%Y-%m-%d').fillna('N/A')

            options_df.rename(columns={
                "strike_api": "strike",
                "opt_kind_api": "opt_kind",
                "symbol_contract_api": "symbol"
            }, inplace=True)

            options_df["opt_kind"] = options_df["opt_kind"].astype(str).str.lower().fillna('unknown')
            options_df["symbol"] = options_df["symbol"].astype(str).str.upper().fillna(symbol_upper + "_OPTION_UNKNOWN")

            if 'multiplier' in options_df.columns and not options_df['multiplier'].empty:
                first_valid_multiplier = options_df['multiplier'].dropna().iloc[0] if not options_df['multiplier'].dropna().empty else 100.0
                underlying_data_result['multiplier'] = float(first_valid_multiplier)
                logger.debug(f"Set underlying_data_result['multiplier'] to {underlying_data_result['multiplier']} from chain for {symbol_upper}")
            elif 'multiplier' not in underlying_data_result:
                logger.warning(f"Multiplier not found in chain data for {symbol_upper} and not in underlying_data. Defaulting to 100.0 for underlying_data.")
                underlying_data_result['multiplier'] = 100.0

            logger.debug(f"Options Chain ({symbol_upper}): Cleaning {len(NUMERIC_COLUMNS_OPTIONS)} configured numeric columns...")
            for col_name in NUMERIC_COLUMNS_OPTIONS:
                if col_name in options_df.columns:
                    options_df[col_name] = pd.to_numeric(options_df[col_name], errors='coerce').fillna(0.0)
                else:
                    logger.warning(f"Options Chain Clean ({symbol_upper}): Configured numeric column '{col_name}' was MISSING from fetched data. Adding as 0.0.")
                    options_df[col_name] = 0.0
            logger.debug(f"Options Chain ({symbol_upper}): Numeric column conversion and cleaning complete.")

            rolling_cols_to_log = [c for c in options_df.columns if 'volmbs_' in c or 'valuebs_' in c]
            if rolling_cols_to_log:
                logger.info(f"FETCHER V2.0.3 (Canon Update) ({symbol_upper}): Final Dtypes of rolling flow cols:\n{options_df[rolling_cols_to_log].dtypes.to_string()}")
            else:
                logger.info(f"FETCHER V2.0.3 (Canon Update) ({symbol_upper}): No 'volmbs_' or 'valuebs_' (rolling flow) columns found in final DataFrame.")

            logger.info(f"Options Chain ({symbol_upper}): Successfully processed. Final DataFrame shape: {options_df.shape}")
            return options_df, underlying_data_result

        except Exception as e_chain_fetch:
            logger.error(f"Options Chain ({symbol_upper}): Unexpected error during chain fetch or processing: {e_chain_fetch}", exc_info=True)
            underlying_data_result['error'] = f"Options chain fetch/processing failed: {e_chain_fetch}"
            return pd.DataFrame(), underlying_data_result

    def fetch_market_data(self, symbols: List[str], dte_list: Optional[List[int]] = None, price_range_pct: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        logger.info(f"\n--- Starting Market Data Fetch (Version 2.0.3 - Canon Update) for Symbols: {symbols} ---")
        market_fetch_start_time = time.time()
        market_data_results_bundle: Dict[str, Dict[str, Any]] = {}
        num_symbols_to_fetch = len(symbols)

        effective_dte_list = dte_list if dte_list is not None else self.default_dte_range
        effective_price_range_pct = price_range_pct if price_range_pct is not None else self.default_price_range_pct * 100

        for i, symbol_item in enumerate(symbols):
            current_symbol_upper = symbol_item.strip().upper()
            if not current_symbol_upper:
                logger.warning(f"Skipping empty symbol string at index {i}.")
                continue

            symbol_fetch_start_time = time.time()
            logger.info(f"\nProcessing symbol: '{current_symbol_upper}' ({i+1}/{num_symbols_to_fetch})...")

            options_df_result, underlying_info_result = self.fetch_options_chain(
                current_symbol_upper,
                effective_dte_list,
                effective_price_range_pct
            )

            market_data_results_bundle[current_symbol_upper] = {
                "options_chain": options_df_result,
                "underlying": underlying_info_result,
                "fetch_timestamp": datetime.now().isoformat(),
                "error": underlying_info_result.get("error"),
                "symbol": current_symbol_upper
            }

            status_msg = "Failed" if market_data_results_bundle[current_symbol_upper]["error"] else \
                         ("Empty Chain" if options_df_result.empty else "Success")
            logger.info(f"Finished processing '{current_symbol_upper}' in {time.time() - symbol_fetch_start_time:.3f}s. Status: {status_msg}")

            if i < num_symbols_to_fetch - 1 and self.inter_call_delay > 0:
                logger.debug(f"Applying inter-call delay of {self.inter_call_delay:.2f}s before next symbol.")
                time.sleep(self.inter_call_delay)

        total_market_fetch_duration = time.time() - market_fetch_start_time
        logger.info(f"\n--- Finished All Market Data Fetch in {total_market_fetch_duration:.3f} seconds ---")

        errors_encountered_count = sum(1 for symbol_bundle in market_data_results_bundle.values() if symbol_bundle.get("error"))
        logger.info(f"    Total symbols processed: {len(market_data_results_bundle)} / {num_symbols_to_fetch}. Total errors: {errors_encountered_count}")

        return market_data_results_bundle

if __name__ == "__main__":
    test_logger_main = logging.getLogger()
    if not test_logger_main.hasHandlers():
        console_handler_main = logging.StreamHandler()
        console_handler_main.setLevel(logging.DEBUG)
        formatter_main = logging.Formatter('[%(levelname)s] (%(name)s:%(lineno)d) %(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        console_handler_main.setFormatter(formatter_main)
        test_logger_main.addHandler(console_handler_main)
        test_logger_main.setLevel(logging.DEBUG)

    logger.info("--- EnhancedDataFetcher_v2 Test (Version 2.0.3 - V2.4 API PARAMS REFINED - Canon Update) --- ")

    TEST_CONFIG_PATH_MAIN = DEFAULT_FETCHER_CONFIG_PATH

    if not os.path.exists(TEST_CONFIG_PATH_MAIN):
        logger.error(f"Test config file '{TEST_CONFIG_PATH_MAIN}' not found! Cannot run full test.")
    elif not CONVEXLIB_AVAILABLE:
        logger.critical("Convexlib library is not available. Full test functionality is disabled.")
    else:
        try:
            fetcher_instance_test = EnhancedDataFetcher_v2(config_path=TEST_CONFIG_PATH_MAIN)

            if not fetcher_instance_test.api:
                logger.error("Fetcher API instance not initialized successfully. Aborting further tests.")
            else:
                test_symbol_underlying = "AAPL"
                logger.info(f"\n--- Testing Underlying Data Fetch for: {test_symbol_underlying} (V2.0.3 Fetcher - Canon Update) ---")
                underlying_data_test = fetcher_instance_test._fetch_underlying_data_internal_with_retry(
                    test_symbol_underlying,
                    UNDERLYING_REQUIRED_PARAMS
                )
                if underlying_data_test and not underlying_data_test.get("error"):
                    logger.info(f"OK: Underlying data for {test_symbol_underlying} fetched. "
                                f"Price: {underlying_data_test.get('price')}, "
                                f"Volatility: {underlying_data_test.get('volatility')}, "
                                f"Day Volume: {underlying_data_test.get('day_volume')}")
                else:
                    logger.error(f"FAIL: Underlying data fetch for {test_symbol_underlying}. "
                                 f"Error: {underlying_data_test.get('error') if underlying_data_test else 'Unknown error'}")

                test_symbol_options = "SPY"
                logger.info(f"\n--- Testing Options Chain Fetch for: {test_symbol_options} (V2.0.3 Fetcher - Canon Update) ---")
                options_df_test, underlying_info_test = fetcher_instance_test.fetch_options_chain(
                    test_symbol_options,
                    dte_list=[0],
                    price_range_pct=1.0
                )
                if not options_df_test.empty:
                    logger.info(f"OK: Options chain for {test_symbol_options} fetched. Shape: {options_df_test.shape}.")
                    logger.info(f"Columns: {options_df_test.columns.tolist()}")
                    added_opt_params_check = ["volm", "volm_buy", "value_buy"]
                    for p_check in added_opt_params_check:
                        if p_check in options_df_test.columns: logger.info(f"  Column '{p_check}' present.")
                        else: logger.warning(f"  Column '{p_check}' MISSING from options_df_test.")
                else:
                    logger.error(f"FAIL: Options chain fetch for {test_symbol_options} resulted in an empty DataFrame. "
                                 f"Underlying fetch error (if any): {underlying_info_test.get('error')}")

                if 'multiplier' in underlying_info_test:
                     logger.info(f"  Multiplier from underlying_info_test for {test_symbol_options}: {underlying_info_test['multiplier']}")

        except Exception as e_main_test:
            logger.critical(f"Critical error during EnhancedDataFetcher_v2 test execution: {e_main_test}", exc_info=True)

    logger.info("--- EnhancedDataFetcher_v2 Test (Version 2.0.3 - V2.4 API PARAMS REFINED - Canon Update) Finished ---")
