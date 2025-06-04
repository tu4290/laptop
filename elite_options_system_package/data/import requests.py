import requests
import json
from datetime import datetime, timedelta

# Tradier API base URL - Defaulting to Production
TRADIER_API_BASE_URL = "https://api.tradier.com/v1/"

# IMPORTANT: Replace 'YOUR_TRADIER_ACCESS_TOKEN' with your actual Tradier access token.
# This token should be your PRODUCTION access token.
# Keep your access token secure and do not share it publicly.
TRADIER_ACCESS_TOKEN = "J0bVBR60xoYQZw8tIEREcqVfcmfd" # Replace with your production token

# Standard headers for Tradier API requests
HEADERS = {
    "Authorization": f"Bearer {TRADIER_ACCESS_TOKEN}",
    "Accept": "application/json"
}

def get_stock_quote(symbol):
    """
    Fetches a stock quote from the Tradier API.
    Endpoint: /markets/quotes
    """
    endpoint = f"{TRADIER_API_BASE_URL}markets/quotes"
    params = {
        "symbols": symbol,
        "greeks": "false" # Set to "true" if you need greeks for options quotes
    }
    response_content = None # Initialize to ensure it's defined
    try:
        response = requests.get(endpoint, headers=HEADERS, params=params)
        response_content = response.text # Store response text for potential error logging
        response.raise_for_status() # Raises an HTTPError for bad responses (4XX or 5XX)
        
        data = response.json()
        # The 'quotes' key might contain a list or a single object depending on the API version/response
        if 'quotes' in data and 'quote' in data['quotes']:
            # Handle case where 'quote' is a single dictionary
            if isinstance(data['quotes']['quote'], list):
                 # If 'quote' is a list (e.g. when multiple symbols requested, but we only asked for one)
                if len(data['quotes']['quote']) > 0:
                    return data['quotes']['quote'][0]
                else:
                    print(f"Empty quote list received for {symbol}: {data}")
                    return None
            else: # 'quote' is a dictionary
                return data['quotes']['quote']
        elif 'quotes' in data and isinstance(data['quotes'], dict) and 'quote' in data['quotes']: # Some API versions might nest it this way
             return data['quotes']['quote']
        # Handle case where 'quotes' itself is the quote object (for single symbol request)
        elif 'symbol' in data.get('quotes', {}): # Check if 'quotes' is a dict and has 'symbol'
            return data['quotes']
        else:
            print(f"Unexpected quote data structure for {symbol}: {data}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching stock quote for {symbol}: {e}")
        if response_content: # Check if response_content was assigned
            print(f"Response content: {response_content}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON response for {symbol} quote.")
        if response_content:
            print(f"Response content: {response_content}")
        return None

def get_option_expirations(symbol):
    """
    Fetches option expiration dates for a given stock symbol.
    Endpoint: /markets/options/expirations
    """
    endpoint = f"{TRADIER_API_BASE_URL}markets/options/expirations"
    params = {
        "symbol": symbol,
        "includeAllRoots": "true", # To get all underlying roots
        "strikes": "false" # We only need dates here
    }
    response_content = None
    try:
        response = requests.get(endpoint, headers=HEADERS, params=params)
        response_content = response.text
        response.raise_for_status()
        data = response.json()
        if 'expirations' in data and 'date' in data['expirations']:
            # 'date' can be a single string if only one expiration, or a list
            if isinstance(data['expirations']['date'], list):
                return data['expirations']['date']
            else: # It's a single date string
                return [data['expirations']['date']]
        else:
            print(f"Unexpected expirations data structure for {symbol}: {data}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching option expirations for {symbol}: {e}")
        if response_content:
            print(f"Response content: {response_content}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON response for {symbol} expirations.")
        if response_content:
            print(f"Response content: {response_content}")
        return None

def get_option_chain(symbol, expiration_date):
    """
    Fetches the option chain for a given stock symbol and expiration date.
    Endpoint: /markets/options/chains
    """
    endpoint = f"{TRADIER_API_BASE_URL}markets/options/chains"
    params = {
        "symbol": symbol,
        "expiration": expiration_date,
        "greeks": "true" # Set to "true" to include option greeks (delta, gamma, theta, vega)
    }
    response_content = None
    try:
        response = requests.get(endpoint, headers=HEADERS, params=params)
        response_content = response.text
        response.raise_for_status()
        data = response.json()
        if 'options' in data and 'option' in data['options']:
            return data['options']['option'] # This is usually a list of option contracts
        else:
            print(f"Unexpected option chain data structure for {symbol} (exp: {expiration_date}): {data}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching option chain for {symbol} (exp: {expiration_date}): {e}")
        if response_content:
            print(f"Response content: {response_content}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON response for {symbol} option chain.")
        if response_content:
            print(f"Response content: {response_content}")
        return None

def get_historical_data(symbol, interval="daily", start_date=None, end_date=None):
    """
    Fetches historical OHLCV data for a given stock symbol.
    Endpoint: /markets/history
    """
    endpoint = f"{TRADIER_API_BASE_URL}markets/history"
    
    # Default to fetching the last 5 trading days if no dates are provided
    if start_date is None and end_date is None:
        end_date_dt = datetime.now()
        start_date_dt = end_date_dt - timedelta(days=7) # Fetch a bit more to ensure 5 trading days
        end_date = end_date_dt.strftime('%Y-%m-%d')
        start_date = start_date_dt.strftime('%Y-%m-%d')
    elif end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')


    params = {
        "symbol": symbol,
        "interval": interval, # e.g., 'daily', 'weekly', 'monthly', or '5min', '15min'
        "start": start_date,
        "end": end_date
        # Other params like 'session_filter' could be added if needed
    }
    response_content = None
    try:
        response = requests.get(endpoint, headers=HEADERS, params=params)
        response_content = response.text
        response.raise_for_status()
        data = response.json()
        # The history data is typically under 'history' -> 'day' (for daily)
        if 'history' in data and data['history'] and 'day' in data['history']:
            # 'day' can be a single object if only one day's data, or a list
            if isinstance(data['history']['day'], list):
                return data['history']['day']
            else: # It's a single day object
                return [data['history']['day']]
        elif data.get('history') is None: # Handles cases where history is null (e.g. no data for range)
            print(f"No historical data returned for {symbol} in the given range/interval. Response: {data}")
            return []
        else:
            print(f"Unexpected historical data structure for {symbol}: {data}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching historical data for {symbol}: {e}")
        if response_content:
            print(f"Response content: {response_content}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON response for {symbol} historical data.")
        if response_content:
            print(f"Response content: {response_content}")
        return None

if __name__ == "__main__":
    # --- IMPORTANT ---
    # 1. Make sure you have the 'requests' library installed: pip install requests
    # 2. Replace 'YOUR_TRADIER_ACCESS_TOKEN' with your actual PRODUCTION access token above.
    
    if TRADIER_ACCESS_TOKEN == "YOUR_TRADIER_ACCESS_TOKEN":
        print("Please replace 'YOUR_TRADIER_ACCESS_TOKEN' with your actual Tradier PRODUCTION access token in the script.")
    else:
        test_symbol = "AAPL" # Example stock symbol

        # --- Test 1: Get Stock Quote ---
        print(f"\n--- Fetching quote for {test_symbol} ---")
        quote = get_stock_quote(test_symbol)
        if quote:
            print(f"Symbol: {quote.get('symbol')}")
            print(f"Description: {quote.get('description')}")
            print(f"Last Price: {quote.get('last')}")
            print(f"Change: {quote.get('change')}")
            print(f"Volume: {quote.get('volume')}")
            # print(json.dumps(quote, indent=2)) # Uncomment to see full quote JSON
        else:
            print(f"Could not retrieve quote for {test_symbol}.")

        # --- Test 2: Get Option Expiration Dates ---
        print(f"\n--- Fetching option expirations for {test_symbol} ---")
        expirations = get_option_expirations(test_symbol)
        selected_expiration = None # Initialize
        if expirations and isinstance(expirations, list) and len(expirations) > 0 : # Check if expirations is a non-empty list
            print(f"Available expiration dates for {test_symbol}:")
            for exp_date in expirations[:5]: # Print first 5 for brevity
                print(exp_date)
            # print(json.dumps(expirations, indent=2)) # Uncomment to see full expirations JSON
            
            selected_expiration = expirations[0] # Use the first available expiration date
        elif expirations is None:
             print(f"Could not retrieve option expirations for {test_symbol}.")
        else: # Expirations might be an empty list or not a list
            print(f"No expiration dates found for {test_symbol} or unexpected format: {expirations}")

        # --- Test 3: Get Option Chain ---
        if selected_expiration: # Only proceed if we have an expiration date
            print(f"\n--- Fetching option chain for {test_symbol}, Expiration: {selected_expiration} ---")
            option_chain = get_option_chain(test_symbol, selected_expiration)
            if option_chain:
                print(f"Number of contracts in chain: {len(option_chain)}")
                # Print details for a few contracts (e.g., first 2 calls and first 2 puts)
                calls = [opt for opt in option_chain if opt and opt.get('option_type') == 'call']
                puts = [opt for opt in option_chain if opt and opt.get('option_type') == 'put']

                print("\nSample Calls:")
                for contract in calls[:2]: # Iterate safely
                    if contract: # Check if contract is not None
                        print(f"  Symbol: {contract.get('symbol')}, Strike: {contract.get('strike')}, Last: {contract.get('last')}, Bid: {contract.get('bid')}, Ask: {contract.get('ask')}, IV: {contract.get('greeks', {}).get('mid_iv')}")
                
                print("\nSample Puts:")
                for contract in puts[:2]: # Iterate safely
                     if contract: # Check if contract is not None
                         print(f"  Symbol: {contract.get('symbol')}, Strike: {contract.get('strike')}, Last: {contract.get('last')}, Bid: {contract.get('bid')}, Ask: {contract.get('ask')}, IV: {contract.get('greeks', {}).get('mid_iv')}")
                
                # print(json.dumps(option_chain[:2], indent=2)) # Uncomment to see first 2 contracts in full JSON
            else:
                print(f"Could not retrieve option chain for {test_symbol}, Expiration: {selected_expiration}.")
        else:
            print(f"\nSkipping option chain test as no expiration date was found for {test_symbol}.")

        # --- Test 4: Get Historical OHLCV Data ---
        print(f"\n--- Fetching historical (OHLCV) data for {test_symbol} (last 5 trading days) ---")
        # Example: Get daily data for the last 5 trading days (approx)
        end_date_hist = datetime.now().strftime('%Y-%m-%d')
        start_date_hist = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d') # Fetch a bit more to ensure we get enough data points

        historical_data = get_historical_data(test_symbol, interval="daily", start_date=start_date_hist, end_date=end_date_hist)
        if historical_data:
            print(f"Retrieved {len(historical_data)} data points.")
            # Print the last 5 available data points
            for day_data in historical_data[-5:]: # Print the most recent 5 data points from the fetched set
                if day_data: # Ensure day_data is not None
                    print(f"  Date: {day_data.get('date')}, Open: {day_data.get('open')}, High: {day_data.get('high')}, Low: {day_data.get('low')}, Close: {day_data.get('close')}, Volume: {day_data.get('volume')}")
            # print(json.dumps(historical_data, indent=2)) # Uncomment to see full historical data JSON
        else:
            print(f"Could not retrieve historical data for {test_symbol}.")


    # For unit testing these functions (similar to the yfinance example),
    # you would use `unittest.mock.patch` to mock `requests.get` and simulate
    # API responses without making actual network calls. This is crucial for
    # reliable and fast automated tests.
