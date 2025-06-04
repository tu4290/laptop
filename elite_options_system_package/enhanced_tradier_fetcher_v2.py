# data_management/tradier_data_fetcher.py
# (Elite Options Trading System - Tradier Data Fetcher for OHLC & IV5 Approx.)
# Version 2.4.1 - EOTS Integration Focus

# Standard Library Imports
import os
import sys 
import time
import logging
import json
import random
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from functools import wraps

# Third-Party Imports
import pandas as pd
import requests

# --- Module-Specific Logger ---
# Basic configuration if run standalone for testing.
# In a full application, the root logger would be configured.
if not logging.getLogger(__name__).hasHandlers():
    tradier_logger_formatter = logging.Formatter(
        '[%(levelname)s] (%(name)s:%(lineno)d) %(asctime)s - %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    tradier_logger_handler = logging.StreamHandler(sys.stdout) # Use sys.stdout for notebook compatibility
    tradier_logger_handler.setFormatter(tradier_logger_formatter)
    logging.getLogger(__name__).addHandler(tradier_logger_handler)
    logging.getLogger(__name__).setLevel(logging.INFO) 
logger = logging.getLogger(__name__)

# --- Retry Decorator ---
def tradier_retry_api_call(
    retries: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
    jitter: bool = True,
    logger_instance: Optional[logging.Logger] = None,
    expected_response_type: type = dict, 
    func_name_override: Optional[str] = None
):
    """
    Decorator to retry Tradier API calls with exponential backoff and jitter.
    Handles common HTTP errors and network issues.
    """
    log = logger_instance if logger_instance else logging.getLogger(f"{__name__}.tradier_retry_api_call")

    def decorator(func: Callable):
        actual_func_name = func_name_override if func_name_override else func.__name__
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = base_delay_seconds
            last_exception: Optional[BaseException] = None
            response_content_for_error: Optional[str] = None

            while attempts <= retries: 
                try:
                    response_obj = func(*args, **kwargs) 

                    if not isinstance(response_obj, requests.Response):
                        log.error(f"Tradier API call {actual_func_name} did not return a requests.Response object. Got: {type(response_obj)}. Aborting.")
                        # Return type-appropriate error indicator
                        if expected_response_type == dict: return {"error": f"Internal error: {actual_func_name} did not return Response object."}
                        elif expected_response_type == list: return []
                        elif expected_response_type == pd.DataFrame: return pd.DataFrame()
                        return None 

                    response_content_for_error = str(response_obj.text)[:500] 
                    response_obj.raise_for_status() 
                    parsed_response = response_obj.json()
                    
                    # Handle Tradier's specific error structure if present
                    if isinstance(parsed_response, dict) and "errors" in parsed_response and "error" in parsed_response["errors"]:
                        api_errors = parsed_response["errors"]["error"]
                        log.warning(f"Tradier API reported error(s) in {actual_func_name} on attempt {attempts + 1}: {api_errors}")
                        last_exception = RuntimeError(f"API Error(s): {api_errors}")
                        # Check for non-retryable content in error messages
                        if isinstance(api_errors, list) and any("symbol not found" in str(err).lower() for err in api_errors):
                             log.error(f"Tradier API: Symbol not found for {actual_func_name}. Not retrying.")
                             raise last_exception # Will be caught and handled as non-retryable below for this iteration
                        if isinstance(api_errors, str) and "symbol not found" in api_errors.lower():
                             log.error(f"Tradier API: Symbol not found for {actual_func_name}. Not retrying.")
                             raise last_exception
                        raise last_exception # Force retry for other general API errors

                    if not isinstance(parsed_response, expected_response_type) and \
                       not (expected_response_type == pd.DataFrame and isinstance(parsed_response, (dict, list))):
                        log.warning(f"Tradier API call {actual_func_name} returned unexpected JSON data type on attempt {attempts + 1}. "
                                    f"Expected {expected_response_type}, got {type(parsed_response)}. Response: {str(parsed_response)[:200]}")
                        if attempts == retries: # Final attempt failed due to type
                            log.error(f"Tradier API call {actual_func_name} failed due to unexpected JSON data type after max retries.")
                            error_msg_type = f"API returned unexpected JSON data type for {actual_func_name}"
                            if expected_response_type == dict: return {"error": error_msg_type}
                            elif expected_response_type == list: return []
                            elif expected_response_type == pd.DataFrame: return pd.DataFrame()
                            else: return None
                        raise TypeError(f"Unexpected JSON type: {type(parsed_response)}") # Force retry

                    log.debug(f"Tradier API call {actual_func_name} successful on attempt {attempts + 1}.")
                    return parsed_response 

                except requests.exceptions.HTTPError as e_http:
                    status_code = e_http.response.status_code
                    log.warning(f"Tradier API HTTP Error on attempt {attempts + 1} for {actual_func_name}: {status_code} - {response_content_for_error}")
                    last_exception = e_http
                    sleep_duration_http = current_delay # Default for most HTTP errors
                    if status_code in [401, 403]: 
                        log.error(f"Fatal Tradier API authentication/authorization error ({status_code}) for {actual_func_name}. Aborting retries.")
                        error_msg_auth = f"Tradier API Auth Error ({status_code}) for {actual_func_name}"
                        if expected_response_type == dict: return {"error": error_msg_auth}
                        elif expected_response_type == list: return []
                        elif expected_response_type == pd.DataFrame: return pd.DataFrame()
                        else: raise last_exception # Re-raise immediately
                    elif status_code == 404: 
                        log.error(f"Tradier API: Resource not found (404) for {actual_func_name}. Not retrying this specific error.")
                        error_msg_404 = f"Tradier API Resource not found (404) for {actual_func_name}"
                        if expected_response_type == dict: return {"error": error_msg_404}
                        elif expected_response_type == list: return []
                        elif expected_response_type == pd.DataFrame: return pd.DataFrame()
                        else: raise last_exception # Re-raise immediately for 404
                    elif status_code == 429: 
                        log.warning(f"Tradier API Rate Limit Exceeded (429) for {actual_func_name}. Increasing delay.")
                        retry_after_header = e_http.response.headers.get('Retry-After')
                        if retry_after_header and retry_after_header.isdigit():
                            sleep_duration_http = int(retry_after_header)
                            log.info(f"Respecting Retry-After header: sleeping for {sleep_duration_http} seconds.")
                        else: 
                            sleep_duration_http = min(current_delay * 2.5, max_delay_seconds * 2) 
                except requests.exceptions.RequestException as e_req: 
                    log.warning(f"Tradier API Network/Request Error on attempt {attempts + 1} for {actual_func_name}: {type(e_req).__name__} - {str(e_req)[:150]}")
                    last_exception = e_req
                    sleep_duration_http = current_delay 
                except json.JSONDecodeError as e_json:
                    log.warning(f"Tradier API JSONDecodeError on attempt {attempts + 1} for {actual_func_name}: {e_json}. Response text: {response_content_for_error}")
                    last_exception = e_json
                    sleep_duration_http = current_delay
                except TypeError as e_type: 
                    log.warning(f"Tradier API call {actual_func_name} encountered TypeError on attempt {attempts + 1}: {e_type}")
                    last_exception = e_type
                    sleep_duration_http = current_delay
                except RuntimeError as e_runtime: # Catch API errors raised from within the try block
                    log.warning(f"Tradier API reported error (RuntimeError) on attempt {attempts + 1} for {actual_func_name}: {e_runtime}")
                    last_exception = e_runtime
                    sleep_duration_http = current_delay
                    if "symbol not found" in str(e_runtime).lower(): # Non-retryable API error
                        log.error(f"Tradier API: Symbol not found for {actual_func_name} (from API error). Not retrying.")
                        error_msg_sym_nf = f"Tradier API: Symbol not found for {actual_func_name}"
                        if expected_response_type == dict: return {"error": error_msg_sym_nf}
                        elif expected_response_type == list: return []
                        elif expected_response_type == pd.DataFrame: return pd.DataFrame()
                        else: raise last_exception
                except Exception as e_gen: 
                    log.error(f"Unexpected Error during Tradier API call attempt {attempts + 1} for {actual_func_name}: {type(e_gen).__name__} - {e_gen}", exc_info=False)
                    if log.getEffectiveLevel() <= logging.DEBUG: log.debug(f"Full traceback for {actual_func_name}:", exc_info=True)
                    last_exception = e_gen
                    sleep_duration_http = current_delay 

                attempts += 1
                if attempts <= retries:
                    sleep_time_actual = sleep_duration_http + (random.uniform(0, sleep_duration_http * 0.1) if jitter else 0)
                    log.info(f"Retrying Tradier API call {actual_func_name} in {sleep_time_actual:.2f} seconds... (Attempt {attempts}/{retries})")
                    time.sleep(sleep_time_actual)
                    current_delay = min(current_delay * 1.8, max_delay_seconds) 
                else: # Max retries reached
                    log.error(f"Tradier API call {actual_func_name} failed after {retries} retries.")
                    error_message_final = f"Tradier API call {actual_func_name} failed after max retries."
                    if last_exception:
                        error_message_final += f" Last error: {type(last_exception).__name__} - {str(last_exception)[:100]}"
                    
                    if expected_response_type == dict: return {"error": error_message_final}
                    elif expected_response_type == list: return []
                    elif expected_response_type == pd.DataFrame: return pd.DataFrame()
                    if last_exception and not isinstance(last_exception, (requests.exceptions.HTTPError, requests.exceptions.RequestException, json.JSONDecodeError, TypeError, RuntimeError)):
                        raise last_exception # Re-raise if it's an unexpected general exception
                    return None # For handled exceptions after retries
            
            log.critical(f"Fell through Tradier retry loop for {actual_func_name} - this indicates a logic error.")
            if expected_response_type == dict: return {"error": "Retry loop logic error."}
            elif expected_response_type == list: return []
            elif expected_response_type == pd.DataFrame: return pd.DataFrame()
            return None
        return wrapper
    return decorator

class TradierDataFetcher:
    """
    Handles data fetching from the Tradier API for EOTS.
    - Fetches OHLCV data (for ATR calculation).
    - Fetches option chain data for IV5 approximation.
    - Approximates IV5 using ATM option SMV_VOL.
    - Accepts a configuration dictionary for settings and credentials.
    - Implements robust API call retries and logging.
    Version: 2.4.1
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logger.getChild(self.__class__.__name__)
        self.initialization_failed = False
        self.config = config if isinstance(config, dict) else {}

        self._load_config_settings() 
        
        self.headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json"
        }

        if self.initialization_failed or self.access_token == "YOUR_TRADIER_ACCESS_TOKEN_PLACEHOLDER" or not self.access_token:
            self.logger.error("TradierDataFetcher: CRITICAL - Access token is missing, invalid, or placeholder. API calls will fail.")
            self.initialization_failed = True 
        
        self.logger.info(f"TradierDataFetcher initialized. Target API: {self.base_url}. Token Loaded: {'Yes' if self.access_token and self.access_token != 'YOUR_TRADIER_ACCESS_TOKEN_PLACEHOLDER' else 'NO/Placeholder'}")

    def _load_config_settings(self):
        """Loads settings from the provided configuration dictionary."""
        self.logger.debug("Loading TradierDataFetcher configurations...")
        
        # Expects config structure like: {"tradier_api_settings": {"base_url": "...", "access_token_env_var": "...", ...}}
        tradier_settings = self.config.get("tradier_api_settings", {})
        if not isinstance(tradier_settings, dict):
            self.logger.warning("Tradier API settings ('tradier_api_settings' key) not found or invalid in config. Using defaults.")
            tradier_settings = {}

        self.base_url = str(tradier_settings.get("base_url", "https://api.tradier.com/v1/"))
        
        access_token_env_var = str(tradier_settings.get("access_token_env_var", "TRADIER_ACCESS_TOKEN"))
        self.access_token = os.getenv(access_token_env_var)
        if not self.access_token:
            # Fallback to a direct token in config if env var is not set (less secure, for dev/testing)
            self.access_token = str(tradier_settings.get("access_token_direct", "YOUR_TRADIER_ACCESS_TOKEN_PLACEHOLDER"))
            if self.access_token != "YOUR_TRADIER_ACCESS_TOKEN_PLACEHOLDER":
                 self.logger.warning(f"Loaded Tradier access token directly from config key 'access_token_direct'. Environment variable '{access_token_env_var}' was not set.")
            else:
                 self.logger.error(f"Tradier access token not found in environment variable '{access_token_env_var}' and no valid 'access_token_direct' in config.")
                 self.initialization_failed = True

        retry_cfg = tradier_settings.get("retry_config", {})
        if not isinstance(retry_cfg, dict): retry_cfg = {}
        self.max_retries = int(retry_cfg.get("max_retries", 3))
        self.base_retry_delay = float(retry_cfg.get("base_delay_seconds", 1.0))
        self.max_retry_delay = float(retry_cfg.get("max_delay_seconds", 10.0))
        self.retry_jitter = bool(retry_cfg.get("jitter", True))
        
        # Set logger level from main config if passed down (e.g., under "system_settings")
        main_system_settings = self.config.get("system_settings", {})
        if not isinstance(main_system_settings, dict): main_system_settings = {}
        log_level_str = main_system_settings.get("log_level", "INFO") 
        try:
            effective_log_level = getattr(logging, str(log_level_str).upper())
            self.logger.setLevel(effective_log_level)
            logging.getLogger(f"{__name__}.tradier_retry_api_call").setLevel(effective_log_level)
        except (AttributeError, ValueError):
            self.logger.warning(f"Invalid log level '{log_level_str}' from main config. TradierFetcher defaulting to INFO.")
            self.logger.setLevel(logging.INFO)
            logging.getLogger(f"{__name__}.tradier_retry_api_call").setLevel(logging.INFO)

        self.logger.debug(f"Tradier API URL: {self.base_url}, Retries: {self.max_retries}, BaseDelay: {self.base_retry_delay}s")

    def _make_tradier_request(self, endpoint_path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 20) -> requests.Response:
        """Internal helper to make a GET request to a Tradier endpoint. Expected to return raw Response."""
        if self.initialization_failed:
            self.logger.error(f"Attempted API call ({endpoint_path}) while fetcher initialization failed (e.g. no token). Returning mock error response.")
            error_response = requests.Response()
            error_response.status_code = 503 
            error_response.reason = "Fetcher Not Initialized - Token Missing"
            error_response._content = b'{"error": "TradierDataFetcher not properly initialized due to missing token."}'
            return error_response

        full_url = f"{self.base_url.rstrip('/')}/{endpoint_path.lstrip('/')}"
        self.logger.debug(f"Making Tradier request to: {full_url} with params: {params}")
        return requests.get(full_url, headers=self.headers, params=params or {}, timeout=timeout)

    def get_underlying_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetches the current quote for an underlying symbol. Returns quote dict or None on error."""
        self.logger.info(f"Fetching quote for symbol: {symbol}")
        decorator_instance = tradier_retry_api_call(
            retries=self.max_retries, base_delay_seconds=self.base_retry_delay,
            max_delay_seconds=self.max_retry_delay, jitter=self.retry_jitter,
            logger_instance=self.logger, expected_response_type=dict,
            func_name_override=f"get_underlying_quote_{symbol}"
        )
        api_call_func = decorator_instance(self._make_tradier_request)
        response_data = api_call_func(endpoint_path="markets/quotes", params={"symbols": symbol, "greeks": "false"})

        if isinstance(response_data, dict) and not response_data.get("error"):
            if 'quotes' in response_data and response_data['quotes'] and response_data['quotes'] != 'null':
                quote_data_item = response_data['quotes'].get('quote')
                if isinstance(quote_data_item, list) and quote_data_item: return quote_data_item[0]
                elif isinstance(quote_data_item, dict): return quote_data_item
            elif 'fault' in response_data:
                fault_str = response_data.get('fault', {}).get('faultstring', 'Unknown API fault')
                self.logger.error(f"Tradier API fault for {symbol} quote: {fault_str}")
            else:
                self.logger.warning(f"Unexpected quote structure for {symbol}: {str(response_data)[:300]}")
        elif isinstance(response_data, dict) and response_data.get("error"):
            self.logger.error(f"Failed to fetch quote for {symbol}: {response_data.get('error')}")
        return None

    def get_ohlcv_data(self, symbol: str, interval: str = "daily",
                       start_date_str: Optional[str] = None,
                       end_date_str: Optional[str] = None,
                       num_days_history: int = 30) -> pd.DataFrame:
        """
        Fetches historical OHLCV data and returns it as a Pandas DataFrame.
        Standardized column names: ['date', 'open', 'high', 'low', 'close', 'volume']
        'date' column will be datetime.date objects. Returns empty DataFrame on failure.
        """
        self.logger.info(f"Fetching OHLCV for {symbol}, Interval: {interval}, Start: {start_date_str}, End: {end_date_str}, TargetDays: {num_days_history}")
        
        # Date range logic
        if start_date_str is None and end_date_str is None: 
            end_date_dt = datetime.now().date() # Use date object
            # Fetch more calendar days to ensure enough trading days, approx 1.5x + buffer
            start_date_dt = end_date_dt - timedelta(days=int(num_days_history * 1.7) + 7) 
            end_date_str = end_date_dt.strftime('%Y-%m-%d')
            start_date_str = start_date_dt.strftime('%Y-%m-%d')
            self.logger.debug(f"Defaulting OHLCV date range for {symbol}: {start_date_str} to {end_date_str}")
        elif end_date_str is None and start_date_str is not None: 
             end_date_str = datetime.now().strftime('%Y-%m-%d')
        elif start_date_str is None and end_date_str is not None: 
            try:
                end_date_dt_param = datetime.strptime(end_date_str, '%Y-%m-%d').date()
                start_date_dt_param = end_date_dt_param - timedelta(days=int(num_days_history * 1.7) + 7)
                start_date_str = start_date_dt_param.strftime('%Y-%m-%d')
            except ValueError:
                self.logger.error(f"Invalid end_date_str format '{end_date_str}' for OHLCV for {symbol}. Returning empty DataFrame.")
                return pd.DataFrame()

        params = {"symbol": symbol, "interval": interval, "start": start_date_str, "end": end_date_str}
        
        decorator_instance = tradier_retry_api_call(
            retries=self.max_retries, base_delay_seconds=self.base_retry_delay,
            max_delay_seconds=self.max_retry_delay, jitter=self.retry_jitter,
            logger_instance=self.logger, expected_response_type=dict, 
            func_name_override=f"get_ohlcv_data_{symbol}"
        )
        api_call_func = decorator_instance(self._make_tradier_request)
        response_data = api_call_func(endpoint_path="markets/history", params=params)

        if isinstance(response_data, dict) and not response_data.get("error"):
            if response_data.get('history') and response_data['history'] != 'null' and 'day' in response_data['history']:
                days_data = response_data['history']['day']
                days_data_list = [days_data] if isinstance(days_data, dict) else (days_data if isinstance(days_data, list) else [])
                
                if not days_data_list:
                    self.logger.info(f"No historical OHLCV data points returned for {symbol} in range {start_date_str}-{end_date_str}.")
                    return pd.DataFrame()
                try:
                    df = pd.DataFrame(days_data_list)
                    df.rename(columns={'date': 'date_str', 'open': 'open', 'high': 'high', 
                                       'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
                    
                    required_cols = ['date_str', 'open', 'high', 'low', 'close', 'volume']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        self.logger.error(f"OHLCV data for {symbol} is missing required columns after rename: {missing_cols}. Available: {df.columns.tolist()}")
                        return pd.DataFrame()

                    df['date'] = pd.to_datetime(df['date_str']).dt.date # Convert to datetime.date objects
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    df.dropna(subset=required_cols[1:], inplace=True) # Drop rows if numeric conversion failed for OHLCV
                    
                    df.sort_values(by='date', inplace=True)
                    if len(df) > num_days_history: # Trim to actual number of trading days
                        df = df.tail(num_days_history)
                    
                    # Ensure correct final columns
                    final_df = df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
                    self.logger.info(f"Successfully processed {len(final_df)} OHLCV data points for {symbol}.")
                    return final_df
                except Exception as e_df:
                    self.logger.error(f"Error processing OHLCV data into DataFrame for {symbol}: {e_df}", exc_info=True)
            elif response_data.get('history') == 'null' or \
                 (isinstance(response_data.get('history'), dict) and response_data['history'].get('day') is None):
                 self.logger.info(f"No historical OHLCV data found for {symbol} in range {start_date_str}-{end_date_str}. API returned 'null' or empty history.")
            elif 'fault' in response_data:
                fault_str = response_data.get('fault', {}).get('faultstring', 'Unknown API fault')
                self.logger.error(f"Tradier API fault for {symbol} OHLCV: {fault_str}")
            else:
                self.logger.warning(f"Unexpected OHLCV data structure for {symbol}: {str(response_data)[:300]}")
        elif isinstance(response_data, dict) and response_data.get("error"):
            self.logger.error(f"Failed to fetch OHLCV for {symbol}: {response_data.get('error')}")
        
        return pd.DataFrame()

    def get_option_expirations(self, symbol: str) -> List[str]:
        """Fetches option expiration dates. Returns list of 'YYYY-MM-DD' strings or empty list."""
        self.logger.info(f"Fetching option expirations for {symbol}")
        decorator_instance = tradier_retry_api_call(
            retries=self.max_retries, base_delay_seconds=self.base_retry_delay,
            max_delay_seconds=self.max_retry_delay, jitter=self.retry_jitter,
            logger_instance=self.logger, expected_response_type=dict,
            func_name_override=f"get_option_expirations_{symbol}"
        )
        api_call_func = decorator_instance(self._make_tradier_request)
        response_data = api_call_func(endpoint_path="markets/options/expirations", params={"symbol": symbol, "includeAllRoots": "true", "strikes": "false"})

        if isinstance(response_data, dict) and not response_data.get("error"):
            if 'expirations' in response_data and response_data['expirations'] and response_data['expirations'] != 'null' and 'date' in response_data['expirations']:
                dates = response_data['expirations']['date']
                return [dates] if isinstance(dates, str) else (dates if isinstance(dates, list) else [])
            elif response_data.get('expirations') == 'null' or \
                 (isinstance(response_data.get('expirations'), dict) and response_data['expirations'].get('date') is None):
                self.logger.info(f"No option expirations found for {symbol}. API returned 'null' or empty expirations.")
            elif 'fault' in response_data:
                fault_str = response_data.get('fault', {}).get('faultstring', 'Unknown API fault')
                self.logger.error(f"Tradier API fault for {symbol} expirations: {fault_str}")
            else:
                self.logger.warning(f"Unexpected expirations data structure for {symbol}: {str(response_data)[:300]}")
        elif isinstance(response_data, dict) and response_data.get("error"):
            self.logger.error(f"Failed to fetch expirations for {symbol}: {response_data.get('error')}")
        return []

    def get_option_chain(self, symbol: str, expiration_date: str) -> List[Dict[str, Any]]:
        """Fetches option chain for a symbol and expiration. Returns list of option dicts or empty list."""
        self.logger.info(f"Fetching option chain for {symbol}, Expiration: {expiration_date}")
        params = {"symbol": symbol, "expiration": expiration_date, "greeks": "true"}
        decorator_instance = tradier_retry_api_call(
            retries=self.max_retries, base_delay_seconds=self.base_retry_delay,
            max_delay_seconds=self.max_retry_delay, jitter=self.retry_jitter,
            logger_instance=self.logger, expected_response_type=dict,
            func_name_override=f"get_option_chain_{symbol}_{expiration_date}"
        )
        api_call_func = decorator_instance(self._make_tradier_request)
        response_data = api_call_func(endpoint_path="markets/options/chains", params=params)

        if isinstance(response_data, dict) and not response_data.get("error"):
            if 'options' in response_data and response_data['options'] and response_data['options'] != 'null' and 'option' in response_data['options']:
                options_data = response_data['options']['option']
                return options_data if isinstance(options_data, list) else ([options_data] if isinstance(options_data, dict) else [])
            elif response_data.get('options') == 'null' or \
                 (isinstance(response_data.get('options'), dict) and response_data['options'].get('option') is None):
                self.logger.info(f"No option chain data found for {symbol} on {expiration_date}. API returned 'null' or empty options.")
            elif 'fault' in response_data:
                fault_str = response_data.get('fault', {}).get('faultstring', 'Unknown API fault')
                self.logger.error(f"Tradier API fault for {symbol} chain ({expiration_date}): {fault_str}")
            else:
                self.logger.warning(f"Unexpected option chain structure for {symbol} (exp: {expiration_date}): {str(response_data)[:300]}")
        elif isinstance(response_data, dict) and response_data.get("error"):
            self.logger.error(f"Failed to fetch option chain for {symbol}, {expiration_date}: {response_data.get('error')}")
        return []

    def get_iv_approximation(self, symbol: str, target_dte: int = 5) -> Optional[Dict[str, Any]]:
        """
        Approximates IV for a target DTE (e.g., 5-day) using ATM option SMV_VOL.
        Returns a dictionary with 'avg_5day_iv' (float) and context, or None on failure.
        Key 'avg_5day_iv' is used for integration.
        """
        self.logger.info(f"Approximating IV for DTE={target_dte} for {symbol}")
        quote_data = self.get_underlying_quote(symbol) 
        
        current_price: Optional[float] = None
        if quote_data and quote_data.get('last') is not None:
            try:
                current_price = float(quote_data['last'])
                self.logger.info(f"Current {symbol} price for IV{target_dte} approx: {current_price}")
            except (ValueError, TypeError):
                self.logger.error(f"Could not convert last price '{quote_data.get('last')}' to float for {symbol}.")
                return None
        else:
            self.logger.error(f"Cannot get current underlying price for {symbol} to approximate IV{target_dte}.")
            return None

        expirations = self.get_option_expirations(symbol)
        if not expirations:
            self.logger.warning(f"No expirations found for {symbol} for IV{target_dte} approximation.")
            return None

        today = date.today()
        closest_expiration_str: Optional[str] = None
        min_dte_diff = float('inf')
        actual_dte_of_closest_exp = -1

        for exp_str in expirations:
            try:
                exp_date_obj = datetime.strptime(exp_str, '%Y-%m-%d').date()
                if exp_date_obj < today: continue 
                
                dte = (exp_date_obj - today).days
                dte_diff = abs(dte - target_dte)

                if dte_diff < min_dte_diff:
                    min_dte_diff = dte_diff
                    closest_expiration_str = exp_str
                    actual_dte_of_closest_exp = dte
                elif dte_diff == min_dte_diff and (closest_expiration_str is None or dte < actual_dte_of_closest_exp):
                    closest_expiration_str = exp_str 
                    actual_dte_of_closest_exp = dte
            except ValueError:
                self.logger.warning(f"Could not parse expiration date '{exp_str}' during IV{target_dte} approx for {symbol}.")
        
        if closest_expiration_str is None:
            self.logger.warning(f"No suitable future expiration found for {symbol} for IV{target_dte} approx.")
            return None
        
        self.logger.info(f"Selected expiration for {symbol} IV{target_dte} approx: {closest_expiration_str} (Actual DTE: {actual_dte_of_closest_exp})")
        option_chain = self.get_option_chain(symbol, closest_expiration_str)
        if not option_chain:
            self.logger.warning(f"Could not get option chain for {symbol} exp {closest_expiration_str} for IV{target_dte} approx.")
            return None

        # Find ATM call and put SMV vol with more robust filtering
        atm_call = None
        valid_calls = [opt for opt in option_chain if opt and opt.get('option_type') == 'call' and isinstance(opt.get('strike'), (int, float))]
        if valid_calls:
            calls_ge_price = [opt for opt in valid_calls if opt['strike'] >= current_price]
            if calls_ge_price:
                atm_call = min(calls_ge_price, key=lambda x: x['strike'], default=None)
            else: # All calls are OTM below current price, take highest strike call
                atm_call = max(valid_calls, key=lambda x: x['strike'], default=None)
        
        atm_put = None
        valid_puts = [opt for opt in option_chain if opt and opt.get('option_type') == 'put' and isinstance(opt.get('strike'), (int, float))]
        if valid_puts:
            puts_le_price = [opt for opt in valid_puts if opt['strike'] <= current_price]
            if puts_le_price:
                atm_put = max(puts_le_price, key=lambda x: x['strike'], default=None)
            else: # All puts are OTM above current price, take lowest strike put
                atm_put = min(valid_puts, key=lambda x: x['strike'], default=None)

        call_smv_val: Optional[float] = None
        if atm_call and isinstance(atm_call.get('greeks'), dict) and atm_call['greeks'].get('smv_vol') is not None:
            try: call_smv_val = float(atm_call['greeks']['smv_vol'])
            except (ValueError, TypeError): self.logger.warning(f"Could not parse call smv_vol for {symbol}: {atm_call['greeks']['smv_vol']}")

        put_smv_val: Optional[float] = None
        if atm_put and isinstance(atm_put.get('greeks'), dict) and atm_put['greeks'].get('smv_vol') is not None:
            try: put_smv_val = float(atm_put['greeks']['smv_vol'])
            except (ValueError, TypeError): self.logger.warning(f"Could not parse put smv_vol for {symbol}: {atm_put['greeks']['smv_vol']}")
        
        avg_iv_approx: Optional[float] = None # Renamed for clarity in return dict
        if call_smv_val is not None and put_smv_val is not None:
            avg_iv_approx = (call_smv_val + put_smv_val) / 2.0
        elif call_smv_val is not None:
            avg_iv_approx = call_smv_val
        elif put_smv_val is not None:
            avg_iv_approx = put_smv_val
            
        if avg_iv_approx is None:
            self.logger.warning(f"Could not determine ATM SMV Vol for IV approximation for {symbol} on {closest_expiration_str}.")
            return None

        # Standardized key 'avg_5day_iv' for easier integration, even if target_dte is different
        # The actual DTE used is also provided for context.
        result_key_for_iv = f"avg_{target_dte}day_iv" if target_dte != 5 else "avg_5day_iv"

        result = {
            "symbol": symbol,
            "target_dte_for_iv_approx": target_dte,
            "source_expiration_for_iv_approx": closest_expiration_str,
            "actual_dte_of_source_options": actual_dte_of_closest_exp,
            "underlying_price_at_iv_approx_calc": current_price,
            "atm_call_strike_iv_approx": atm_call.get('strike') if atm_call else None,
            "atm_call_smv_vol_iv_approx": call_smv_val,
            "atm_put_strike_iv_approx": atm_put.get('strike') if atm_put else None,
            "atm_put_smv_vol_iv_approx": put_smv_val,
            result_key_for_iv: avg_iv_approx 
        }
        self.logger.info(f"IV approx for {symbol} (target DTE {target_dte}): {avg_iv_approx:.4f} (using {actual_dte_of_closest_exp}-DTE options)")
        return result

    def shutdown(self):
        """Placeholder for any cleanup actions if needed (e.g., closing persistent connections)."""
        self.logger.info("TradierDataFetcher shutdown. (No specific actions implemented in this version)")


# --- Main Test Block (Example Usage) ---
if __name__ == '__main__': # pragma: no cover
    # This block is for direct testing of TradierDataFetcher.
    # For this test to run, you MUST:
    # 1. Set the TRADIER_SANDBOX_ACCESS_TOKEN environment variable with a valid Tradier Developer Sandbox token.
    # OR
    # 2. Update the 'access_token_direct' in mock_tradier_config below with your sandbox token.
    
    module_test_logger_tradier = logging.getLogger(f"{__name__}_TradierTestMain")
    # Ensure the test logger also gets the handler if root doesn't have one
    if not module_test_logger_tradier.hasHandlers() and not logging.getLogger().hasHandlers():
        test_formatter = logging.Formatter('[%(levelname)s] (%(name)s:%(lineno)d) %(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
        test_handler = logging.StreamHandler(sys.stdout)
        test_handler.setFormatter(test_formatter)
        logging.getLogger().addHandler(test_handler) # Add to root to catch all module logs
        logging.getLogger().setLevel(logging.DEBUG)

    module_test_logger_tradier.setLevel(logging.DEBUG) 
    logger.setLevel(logging.DEBUG) # Set module logger to DEBUG for test output

    module_test_logger_tradier.info("--- Starting TradierDataFetcher Standalone Test (V2.4.1) ---")

    mock_tradier_config = {
        "tradier_api_settings": {
            "base_url": "https://sandbox.tradier.com/v1/", 
            "access_token_env_var": "TRADIER_SANDBOX_ACCESS_TOKEN", 
            "access_token_direct": "YOUR_TRADIER_SANDBOX_TOKEN_PLACEHOLDER", # Replace if not using ENV VAR
            "retry_config": {
                "max_retries": 2, 
                "base_delay_seconds": 0.5,
                "max_delay_seconds": 2.0,
                "jitter": True
            }
        },
        "system_settings": { 
             "log_level": "DEBUG" # For TradierFetcher's internal logger level setting
        }
    }
    
    # Check if placeholder token is still there and env var is not set
    token_env = os.getenv(mock_tradier_config["tradier_api_settings"]["access_token_env_var"])
    token_direct = mock_tradier_config["tradier_api_settings"]["access_token_direct"]
    if not token_env and token_direct == "YOUR_TRADIER_SANDBOX_TOKEN_PLACEHOLDER":
        module_test_logger_tradier.critical(
            f"TRADIER_SANDBOX_ACCESS_TOKEN env var not set AND 'access_token_direct' is placeholder. "
            f"Tradier tests WILL FAIL authentication. Please set the env var or update the placeholder in the test script."
        )
            
    tradier_fetcher = TradierDataFetcher(config=mock_tradier_config)

    if tradier_fetcher.initialization_failed:
        module_test_logger_tradier.critical("TradierDataFetcher failed to initialize properly in test. Aborting further tests.")
    else:
        test_sym = "AAPL" 
        
        module_test_logger_tradier.info(f"\n--- Testing get_underlying_quote for {test_sym} ---")
        quote_result = tradier_fetcher.get_underlying_quote(test_sym)
        if quote_result and not quote_result.get("error"):
            module_test_logger_tradier.info(f"Quote for {test_sym}: Last Price = {quote_result.get('last')}, Volume = {quote_result.get('volume')}")
        else: module_test_logger_tradier.error(f"Failed to get quote for {test_sym} or error in response: {quote_result}")

        module_test_logger_tradier.info(f"\n--- Testing get_ohlcv_data for {test_sym} (daily, last 10 trading days approx) ---")
        ohlcv_df = tradier_fetcher.get_ohlcv_data(test_sym, interval="daily", num_days_history=10)
        if not ohlcv_df.empty:
            module_test_logger_tradier.info(f"OHLCV DataFrame for {test_sym} (Shape: {ohlcv_df.shape}):\n{ohlcv_df.to_string()}")
        else: module_test_logger_tradier.error(f"Failed to get OHLCV data for {test_sym} or returned empty DataFrame.")

        module_test_logger_tradier.info(f"\n--- Testing get_option_expirations for {test_sym} ---")
        expirations_result = tradier_fetcher.get_option_expirations(test_sym)
        if expirations_result:
            module_test_logger_tradier.info(f"Expirations for {test_sym} (first 5): {expirations_result[:5]}")
            
            if expirations_result:
                future_expirations = [exp for exp in expirations_result if datetime.strptime(exp, '%Y-%m-%d').date() > (date.today() + timedelta(days=2))]
                test_expiry = future_expirations[0] if future_expirations else (expirations_result[0] if expirations_result else None)
                
                if test_expiry:
                    module_test_logger_tradier.info(f"\n--- Testing get_option_chain for {test_sym}, Expiry: {test_expiry} ---")
                    chain_result = tradier_fetcher.get_option_chain(test_sym, test_expiry)
                    if chain_result:
                        module_test_logger_tradier.info(f"Option chain for {test_sym} ({test_expiry}): Found {len(chain_result)} contracts. First contract (sample): {str(chain_result[0])[:250] if chain_result else 'N/A'}...")
                    else: module_test_logger_tradier.error(f"Failed to get option chain for {test_sym} ({test_expiry}) or error in response: {chain_result}")
                else:
                    module_test_logger_tradier.warning(f"No suitable future expiration found for {test_sym} to test option chain fetch.")
        else: module_test_logger_tradier.error(f"Failed to get expirations for {test_sym} or error in response: {expirations_result}")

        module_test_logger_tradier.info(f"\n--- Testing get_iv_approximation for {test_sym} (target DTE 5) ---")
        iv5_result_dict = tradier_fetcher.get_iv_approximation(test_sym, target_dte=5)
        if iv5_result_dict and iv5_result_dict.get("avg_5day_iv") is not None:
            module_test_logger_tradier.info(f"IV5 Approx for {test_sym}: {iv5_result_dict}")
        else: module_test_logger_tradier.error(f"Failed to get IV5 approximation for {test_sym}: {iv5_result_dict}")
        
        module_test_logger_tradier.info(f"\n--- Testing get_iv_approximation for {test_sym} (target DTE 0) ---")
        iv0_result_dict = tradier_fetcher.get_iv_approximation(test_sym, target_dte=0) # Test for 0 DTE
        if iv0_result_dict and iv0_result_dict.get("avg_0day_iv") is not None: # Note the key change
            module_test_logger_tradier.info(f"IV0 Approx for {test_sym}: {iv0_result_dict}")
        else: module_test_logger_tradier.error(f"Failed to get IV0 approximation for {test_sym}: {iv0_result_dict}")

        tradier_fetcher.shutdown()

    module_test_logger_tradier.info("--- TradierDataFetcher Standalone Test (V2.4.1) Finished ---")
