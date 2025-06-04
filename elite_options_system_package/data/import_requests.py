import requests
import json
from datetime import datetime, timedelta

# Tradier API base URL - Defaulting to Production
TRADIER_API_BASE_URL = "https://api.tradier.com/v1/"

# IMPORTANT: Replace 'YOUR_TRADIER_ACCESS_TOKEN' with your actual Tradier access token.
# This token should be your PRODUCTION access token.
# Keep your access token secure and do not share it publicly.
TRADIER_ACCESS_TOKEN = "J0bVBR60xoYQZw8tIEREcqVfcmfd"  # Replace with your production token

# Standard headers for Tradier API requests
HEADERS = {
    "Authorization": f"Bearer {TRADIER_ACCESS_TOKEN}",
    "Accept": "application/json"
}

def get_underlying_quote_iv(symbol):
    """
    Fetches a stock quote from the Tradier API, requesting greeks to see if any IV is returned for the underlying.
    Note: Direct "underlying IV" from this endpoint might be limited or not what's typically used.
    smv_vol from an ATM option contract is often a better proxy.
    Endpoint: /markets/quotes
    """
    endpoint = f"{TRADIER_API_BASE_URL}markets/quotes"
    params = {
        "symbols": symbol,
        "greeks": "true"  # Request greeks
    }
    response_content = None
    try:
        response = requests.get(endpoint, headers=HEADERS, params=params)
        response_content = response.text
        response.raise_for_status()
        data = response.json()
        
        if 'quotes' in data and 'quote' in data['quotes']:
            quote_data = data['quotes']['quote']
            if isinstance(quote_data, list): # If multiple symbols were (accidentally) requested
                quote_data = quote_data[0] if quote_data else None
            
            if quote_data:
                # Tradier's quote endpoint with greeks=true for an equity/ETF might not
                # directly return an "underlying IV" in a standard field.
                # We're looking for any field that might represent it, e.g., 'iv30', 'iv90' if available
                # For this example, we'll just print the relevant parts of the quote.
                # A more common way to get "underlying IV" is to look at ATM option IVs or smv_vol.
                print(f"--- Underlying Quote Data for {symbol} (greeks requested) ---")
                print(f"  Symbol: {quote_data.get('symbol')}")
                print(f"  Last Price: {quote_data.get('last')}")
                # Check for common IV related fields, though they might not always be present for underlying quotes
                print(f"  Implied Volatility (30-day, if available): {quote_data.get('iv30')}") # Example field
                print(f"  Implied Volatility (90-day, if available): {quote_data.get('iv90')}") # Example field
                # You might need to inspect the full quote_data to see all available fields
                # print(json.dumps(quote_data, indent=2))
                return quote_data # Return the whole quote data for further inspection if needed
            else:
                print(f"No quote data found for {symbol} in response: {data}")
                return None
        else:
            print(f"Unexpected quote data structure for {symbol}: {data}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching underlying quote for {symbol}: {e}")
        if response_content:
            print(f"Response content: {response_content}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON response for {symbol} underlying quote.")
        if response_content:
            print(f"Response content: {response_content}")
        return None


def get_option_expirations(symbol):
    """
    Fetches option expiration dates for a given stock symbol.
    Endpoint: /markets/options/expirations
    """
    endpoint = f"{TRADIER_API_BASE_URL}markets/options/expirations"
    params = {"symbol": symbol, "includeAllRoots": "true", "strikes": "false"}
    response_content = None
    try:
        response = requests.get(endpoint, headers=HEADERS, params=params)
        response_content = response.text
        response.raise_for_status()
        data = response.json()
        if 'expirations' in data and 'date' in data['expirations']:
            dates = data['expirations']['date']
            return [dates] if isinstance(dates, str) else dates # Handle single date string or list
        else:
            print(f"Unexpected expirations data structure for {symbol}: {data}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching option expirations for {symbol}: {e}\nResponse: {response_content}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON for {symbol} expirations.\nResponse: {response_content}")
        return None

def get_option_chain_with_ivs(symbol, expiration_date):
    """
    Fetches the option chain and extracts relevant IV metrics.
    Endpoint: /markets/options/chains
    """
    endpoint = f"{TRADIER_API_BASE_URL}markets/options/chains"
    params = {"symbol": symbol, "expiration": expiration_date, "greeks": "true"}
    response_content = None
    try:
        response = requests.get(endpoint, headers=HEADERS, params=params)
        response_content = response.text
        response.raise_for_status()
        data = response.json()
        if 'options' in data and 'option' in data['options']:
            return data['options']['option']
        else:
            print(f"Unexpected option chain structure for {symbol} (exp: {expiration_date}): {data}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching option chain for {symbol} (exp: {expiration_date}): {e}\nResponse: {response_content}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON for {symbol} option chain.\nResponse: {response_content}")
        return None

def find_nearest_atm_options(options, underlying_price, num_options=3):
    """
    Finds a few options near the at-the-money price.
    Returns a dictionary with 'calls' and 'puts' lists.
    """
    if not options or underlying_price is None:
        return {'calls': [], 'puts': []}

    # Separate calls and puts
    calls = sorted([opt for opt in options if opt and opt.get('option_type') == 'call' and opt.get('strike') is not None], key=lambda x: x['strike'])
    puts = sorted([opt for opt in options if opt and opt.get('option_type') == 'put' and opt.get('strike') is not None], key=lambda x: x['strike'])

    # Find closest calls
    closest_calls = sorted(calls, key=lambda x: abs(x['strike'] - underlying_price))[:num_options * 2] # Get more to ensure we have enough on both sides if possible
    
    # Find closest puts
    closest_puts = sorted(puts, key=lambda x: abs(x['strike'] - underlying_price))[:num_options * 2]
    
    # Further refine to get some OTM and ITM if possible, or just closest
    # This is a simple selection, can be made more sophisticated
    
    return {'calls': sorted(closest_calls, key=lambda x: x['strike'])[:num_options], 
            'puts': sorted(closest_puts, key=lambda x: x['strike'])[:num_options]}


if __name__ == "__main__":
    if TRADIER_ACCESS_TOKEN == "YOUR_TRADIER_ACCESS_TOKEN":
        print("Please replace 'YOUR_TRADIER_ACCESS_TOKEN' with your actual Tradier PRODUCTION access token.")
    else:
        symbol_to_test = "SPY"
        today_str = datetime.now().strftime('%Y-%m-%d')
        print(f"--- IV Details for {symbol_to_test} for {today_str} ---")

        # 1. Get Underlying Quote (and attempt to see if any IV is present)
        underlying_quote = get_underlying_quote_iv(symbol_to_test)
        underlying_last_price = None
        if underlying_quote and underlying_quote.get('last') is not None:
            underlying_last_price = float(underlying_quote.get('last'))
        else:
            print(f"Could not get underlying price for {symbol_to_test} to find ATM options. Exiting.")
            exit()

        # 2. Get Option Expirations
        expirations = get_option_expirations(symbol_to_test)
        
        if not expirations:
            print(f"No expiration dates found for {symbol_to_test}. Exiting.")
            exit()

        # 3. Select the nearest expiration date (for "today" or nearest future)
        # For simplicity, we'll pick the very first expiration date >= today.
        # Tradier returns dates sorted, so the first one is usually the soonest.
        # A more robust selection might involve parsing dates and comparing.
        
        # Filter for expirations from today onwards
        valid_expirations = []
        for exp_str in expirations:
            try:
                exp_date_obj = datetime.strptime(exp_str, '%Y-%m-%d')
                if exp_date_obj.date() >= datetime.now().date():
                    valid_expirations.append(exp_str)
            except ValueError:
                print(f"Warning: Could not parse expiration date {exp_str}")
        
        if not valid_expirations:
            print(f"No future or current expiration dates found for {symbol_to_test}. Using first available from list: {expirations[0]}")
            # Fallback to just the first in the list if all are in the past (unlikely for SPY)
            # or if parsing failed for all.
            if expirations:
                 selected_expiration = expirations[0]
            else:
                print("No expirations available at all.")
                exit()
        else:
            selected_expiration = valid_expirations[0] # Takes the soonest valid one

        print(f"\nSelected Expiration Date: {selected_expiration}")

        # 4. Get Option Chain for the selected expiration
        option_chain = get_option_chain_with_ivs(symbol_to_test, selected_expiration)

        if not option_chain:
            print(f"Could not retrieve option chain for {symbol_to_test}, Expiration: {selected_expiration}. Exiting.")
            exit()

        print(f"\n--- Option IV Details for {len(option_chain)} contracts (Expiration: {selected_expiration}) ---")

        # 5. Find and print IVs for a few near-the-money options
        atm_options = find_nearest_atm_options(option_chain, underlying_last_price, num_options=3)

        print("\nNear-the-Money Calls:")
        if atm_options['calls']:
            for contract in atm_options['calls']:
                greeks = contract.get('greeks', {})
                print(f"  Symbol: {contract.get('symbol')}, Strike: {contract.get('strike')}")
                print(f"    Bid IV: {greeks.get('bid_iv'):.4f}, Ask IV: {greeks.get('ask_iv'):.4f}, Mid IV: {greeks.get('mid_iv'):.4f}, SMV Vol: {greeks.get('smv_vol'):.4f}")
        else:
            print("  No near-the-money calls found or data issue.")
            
        print("\nNear-the-Money Puts:")
        if atm_options['puts']:
            for contract in atm_options['puts']:
                greeks = contract.get('greeks', {})
                print(f"  Symbol: {contract.get('symbol')}, Strike: {contract.get('strike')}")
                print(f"    Bid IV: {greeks.get('bid_iv'):.4f}, Ask IV: {greeks.get('ask_iv'):.4f}, Mid IV: {greeks.get('mid_iv'):.4f}, SMV Vol: {greeks.get('smv_vol'):.4f}")
        else:
            print("  No near-the-money puts found or data issue.")
            
        # As a proxy for "underlying IV", you can also look at the smv_vol of an ATM option.
        # For example, the smv_vol of the first ATM call:
        if atm_options['calls'] and atm_options['calls'][0].get('greeks', {}).get('smv_vol') is not None:
            print(f"\nProxy for Underlying IV (SMV Vol from first ATM call): {atm_options['calls'][0]['greeks']['smv_vol']:.4f}")
        elif option_chain and option_chain[0].get('greeks', {}).get('smv_vol') is not None: # Fallback to first contract if no ATM calls
             print(f"\nProxy for Underlying IV (SMV Vol from first contract in chain): {option_chain[0]['greeks']['smv_vol']:.4f}")


        # To see the full data for a specific contract:
        # if atm_options['calls']:
        #     print("\nFull data for first ATM call:")
        #     print(json.dumps(atm_options['calls'][0], indent=2))
