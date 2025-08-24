#天風證劵-金融工程 量化选股模型 戴维斯双击 20170612
# ==============================================================================
# Step 0: Setup and Imports
# ==============================================================================
import eodhd
import pandas as pd
import numpy as np
import quantstats as qs
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings
import os

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# It's recommended to set the API key as an environment variable for security
# For demonstration, we set it here. Replace 'YOUR_EODHD_API_KEY' with your actual key.
# EODHD API Docs: https://eodhd.com/financial-apis/
API_KEY = os.environ.get('EODHD_API_KEY', 'YOUR_EODHD_API_KEY') 
if API_KEY == 'YOUR_EODHD_API_KEY':
    print("Please replace 'YOUR_EODHD_API_KEY' with your actual EODHD API key.")

client = eodhd.EodHdClient(API_KEY)
print(f"EODHD client initialized. User status: {client.get_user_data()['api_requests']} requests remaining.")


# ==============================================================================
# Step 1: Data Preparation
# ==============================================================================
# --- Helper functions to fetch and cache data to avoid redundant API calls ---

def get_cached_data(file_path):
    """Loads data from a cache file if it exists."""
    if os.path.exists(file_path):
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, index_col=0, parse_dates=True)
        elif file_path.endswith('.json'):
            return pd.read_json(file_path)
    return None

def save_cached_data(data, file_path):
    """Saves data to a cache file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if file_path.endswith('.csv'):
        data.to_csv(file_path)
    elif file_path.endswith('.json'):
        data.to_json(file_path)

def get_sp500_constituents(date):
    """
    Fetches historical S&P 500 constituents for a given date.
    Handles potential delistings by fetching constituents for each rebalancing period.
    """
    date_str = date.strftime('%Y-%m-%d')
    cache_path = f"cache/sp500_constituents/{date_str}.json"
    
    cached_data = get_cached_data(cache_path)
    if cached_data is not None:
        return cached_data['components'].tolist()

    print(f"Fetching S&P 500 constituents for {date_str}...")
    try:
        # Using GSPC.INDX for S&P 500 Index
        data = client.get_fundamental_equity("GSPC.INDX", filter_=f'General,Components&date={date_str}')
        if 'Components' in data and data['Components']:
            components = list(data['Components'].keys())
            save_cached_data(pd.DataFrame({'components': components}), cache_path)
            return components
        else:
            print(f"Warning: No components found for {date_str}. Using previous data if available.")
            return []
    except Exception as e:
        print(f"Error fetching constituents for {date_str}: {e}")
        return []

def get_ohlcv_data(tickers, start_date, end_date):
    """
    Fetches daily OHLCV data for a list of tickers.
    Uses caching to minimize API calls.
    """
    all_ohlcv = {}
    for ticker in tickers:
        cache_path = f"cache/ohlcv/{ticker}.csv"
        cached_data = get_cached_data(cache_path)
        
        # Check if cached data covers the required range
        if cached_data is not None and cached_data.index.min() <= start_date and cached_data.index.max() >= end_date:
            all_ohlcv[ticker] = cached_data.loc[start_date:end_date]
            continue
            
        print(f"Fetching OHLCV for {ticker}...")
        try:
            data = client.get_eod_historical_data(f"{ticker}.US", "d", start_date, end_date)
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                all_ohlcv[ticker] = df
                save_cached_data(df, cache_path)
        except Exception as e:
            print(f"Error fetching OHLCV for {ticker}: {e}")
            
    return all_ohlcv
    
def get_fundamental_data(tickers):
    """
    Fetches quarterly fundamental data for a list of tickers.
    Caches data to avoid repeated API calls.
    """
    all_fundamentals = {}
    for ticker in tickers:
        cache_path = f"cache/fundamentals/{ticker}.json"
        cached_data = get_cached_data(cache_path)
        if cached_data is not None:
            all_fundamentals[ticker] = cached_data
            continue
            
        print(f"Fetching fundamentals for {ticker}...")
        try:
            # We fetch quarterly income statements and valuation metrics
            data = client.get_fundamental_equity(f"{ticker}.US", filter_='Financials::Income_Statement::quarterly,Valuation')
            if 'Financials' in data:
                all_fundamentals[ticker] = data
                save_cached_data(pd.Series(data).to_json(), cache_path)
        except Exception as e:
            print(f"Error fetching fundamentals for {ticker}: {e}")

    return all_fundamentals

# ==============================================================================
# Step 2: Signal Generation Functions
# ==============================================================================

def calculate_factors(fundamentals, current_date):
    """
    Calculates the necessary factors (YoY growth, 2nd order growth) from fundamental data.
    This function carefully handles the lookahead bias by only using data with a filing_date
    before the current_date.
    """
    income_statements = fundamentals.get('Financials', {}).get('Income_Statement', {}).get('quarterly', {})
    
    if not income_statements:
        return None

    # Filter statements to avoid lookahead bias
    valid_statements = {
        date: data for date, data in income_statements.items() 
        if pd.to_datetime(data['filing_date']) < current_date
    }
    
    if len(valid_statements) < 6: # Need at least 6 quarters of data
        return None
        
    sorted_dates = sorted(valid_statements.keys(), reverse=True)
    
    # Identify the most recent quarters needed
    q0_date = sorted_dates[0]
    q1_date = sorted_dates[1]
    
    # Find the corresponding quarters from the previous year
    q4_date_target = (pd.to_datetime(q0_date) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    q5_date_target = (pd.to_datetime(q1_date) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    
    q4_date = min(valid_statements.keys(), key=lambda d: abs(pd.to_datetime(d) - pd.to_datetime(q4_date_target)))
    q5_date = min(valid_statements.keys(), key=lambda d: abs(pd.to_datetime(d) - pd.to_datetime(q5_date_target)))
    
    # Ensure we have unique quarters
    if len({q0_date, q1_date, q4_date, q5_date}) != 4:
        return None

    try:
        ni_q0 = float(valid_statements[q0_date]['netIncome'])
        ni_q1 = float(valid_statements[q1_date]['netIncome'])
        ni_q4 = float(valid_statements[q4_date]['netIncome'])
        ni_q5 = float(valid_statements[q5_date]['netIncome'])
        rev_q1 = float(valid_statements[q1_date]['totalRevenue'])

        # Basic health checks
        if ni_q1 <= 1e6 or ni_q4 <= 0 or ni_q5 <= 0 or rev_q1 <= 0: # Profit > 1M, positive base for growth calc
            return None

        yoy_growth_q0 = (ni_q0 / ni_q4) - 1
        yoy_growth_q1 = (ni_q1 / ni_q5) - 1
        
        if yoy_growth_q1 <= 0: # Need positive base for 2nd order growth
            return None
            
        second_order_growth = (yoy_growth_q0 / yoy_growth_q1) - 1

        return {
            'yoy_growth_q0': yoy_growth_q0,
            'yoy_growth_q1': yoy_growth_q1,
            'second_order_growth': second_order_growth
        }
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def get_trading_signals(all_fundamentals, current_date):
    """
    Generates trading signals based on the Davis Double Play strategy rules.
    This function embodies the logic from the report.
    """
    eligible_stocks = []
    
    for ticker, data in all_fundamentals.items():
        # --- Filter 1: Valuation ---
        # Rule: PE_ttm < 50 
        pe_ratio = data.get('Valuation', {}).get('TrailingPE', None)
        if pe_ratio is None or pe_ratio >= 50:
            continue
            
        # --- Filter 2: Listing Age (handled in main loop) ---
        # Rule: Listed for at least 1 year
        
        # --- Filter 3: Growth & Acceleration ---
        factors = calculate_factors(data, current_date)
        
        if factors is None:
            continue
            
        # Apply the growth condition checks
        if (factors['yoy_growth_q0'] > 0 and 
            factors['yoy_growth_q1'] > 0 and 
            factors['second_order_growth'] > 0):
            eligible_stocks.append({
                'ticker': ticker,
                'pe_ratio': pe_ratio,
                'yoy_growth_q0': factors['yoy_growth_q0']
            })
            
    if not eligible_stocks:
        return []
        
    # --- Final Selection & Ranking ---
    df = pd.DataFrame(eligible_stocks)
    
    group_a = df[(df['yoy_growth_q0'] > 0.20) & (df['yoy_growth_q0'] <= 1.00)]
    group_b = df[df['yoy_growth_q0'] > 1.00]
    
    # Sort Group A descending, Group B ascending
    group_a = group_a.sort_values(by='yoy_growth_q0', ascending=False)
    group_b = group_b.sort_values(by='yoy_growth_q0', ascending=True)
    
    # Combine and select top 25
    final_selection = pd.concat([group_a, group_b]).head(25)
    
    return final_selection['ticker'].tolist()


# ==============================================================================
# Step 3 & 4: Backtest Logic and Execution
# ==============================================================================

def run_backtest(start_date_str, end_date_str):
    """
    Main backtesting engine.
    """
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)
    initial_capital = 1_000_000
    
    # Define quarterly rebalancing dates
    # Adapted for US markets: Mid-month following quarter-end
    rebalance_dates = pd.date_range(start_date, end_date, freq='Q-NOV')[1:] # Start from first full quarter
    rebalance_dates = [d + pd.DateOffset(days=15) for d in rebalance_dates]
    
    print(f"Backtest from {start_date_str} to {end_date_str}")
    print(f"Rebalancing dates: {[d.strftime('%Y-%m-%d') for d in rebalance_dates]}")

    # --- Data Fetching for the entire period ---
    all_tickers = set()
    for date in rebalance_dates:
        constituents = get_sp500_constituents(date)
        all_tickers.update(constituents)
    
    print(f"Fetching data for {len(all_tickers)} unique tickers...")
    all_ohlcv = get_ohlcv_data(list(all_tickers), start_date, end_date)
    all_fundamentals = get_fundamental_data(list(all_tickers))
    
    # --- Backtest Loop ---
    portfolio_value = pd.Series(index=pd.date_range(start_date, end_date, freq='B'), dtype=float)
    cash = initial_capital
    positions = {} # {ticker: shares}
    current_date = start_date
    
    for rebal_date in rebalance_dates:
        # Calculate portfolio value up to the rebalance date
        while current_date < rebal_date and current_date <= end_date:
            today_val = cash
            for ticker, shares in positions.items():
                if ticker in all_ohlcv and current_date in all_ohlcv[ticker].index:
                    today_val += shares * all_ohlcv[ticker].loc[current_date, 'adjusted_close']
            portfolio_value[current_date] = today_val
            current_date += pd.DateOffset(days=1)
            
        if current_date > end_date:
            break

        print(f"\n--- Rebalancing on {rebal_date.strftime('%Y-%m-%d')} ---")
        
        # Liquidate existing portfolio
        liquidation_value = cash
        for ticker, shares in positions.items():
            if ticker in all_ohlcv and rebal_date in all_ohlcv[ticker].index:
                price = all_ohlcv[ticker].loc[rebal_date, 'adjusted_close']
                trade_value = shares * price
                liquidation_value += trade_value * (1 - 0.002) # 0.2% selling cost 
        
        cash = liquidation_value
        positions = {}
        
        # --- Generate new signals ---
        
        # Filter out stocks listed less than 1 year
        current_constituents = get_sp500_constituents(rebal_date)
        filtered_fundamentals = {}
        for ticker in current_constituents:
            if ticker in all_fundamentals:
                listing_date_str = all_fundamentals[ticker].get('General', {}).get('IPODate', '1900-01-01')
                listing_date = pd.to_datetime(listing_date_str)
                if (rebal_date - listing_date).days >= 365:
                    filtered_fundamentals[ticker] = all_fundamentals[ticker]

        target_tickers = get_trading_signals(filtered_fundamentals, rebal_date)
        print(f"Selected {len(target_tickers)} stocks: {target_tickers}")

        if not target_tickers:
            print("No stocks met the criteria. Holding cash.")
            continue
            
        # --- Execute new trades ---
        capital_per_stock = cash / len(target_tickers)
        
        for ticker in target_tickers:
            if ticker in all_ohlcv and rebal_date in all_ohlcv[ticker].index:
                price = all_ohlcv[ticker].loc[rebal_date, 'adjusted_close']
                invest_amount = capital_per_stock * (1 - 0.001) # 0.1% buying cost
                shares_to_buy = invest_amount / price
                positions[ticker] = shares_to_buy
                cash -= capital_per_stock

    # --- Final period valuation ---
    while current_date <= end_date:
        today_val = cash
        for ticker, shares in positions.items():
            if ticker in all_ohlcv and current_date in all_ohlcv[ticker].index:
                today_val += shares * all_ohlcv[ticker].loc[current_date, 'adjusted_close']
            elif ticker in all_ohlcv: # Handle cases where stock stops trading
                last_price = all_ohlcv[ticker]['adjusted_close'].iloc[-1]
                today_val += shares * last_price
        portfolio_value[current_date] = today_val
        current_date += pd.DateOffset(days=1)
        
    return portfolio_value.dropna()


# ==============================================================================
# Step 5: Performance Evaluation with QuantStats
# ==============================================================================

def evaluate_performance(strategy_values):
    """
    Generates a full performance report using QuantStats.
    """
    strategy_returns = qs.utils.to_returns(strategy_values)
    strategy_returns.index = strategy_returns.index.tz_localize('UTC')
    
    print("\n--- Strategy Performance Evaluation ---")
    
    # Fetch benchmark data (SPY)
    spy_ohlcv = client.get_eod_historical_data("SPY.US", "d", strategy_returns.index.min(), strategy_returns.index.max())
    spy_df = pd.DataFrame(spy_ohlcv)
    spy_df['date'] = pd.to_datetime(spy_df['date'])
    spy_df.set_index('date', inplace=True)
    spy_returns = qs.utils.to_returns(spy_df['adjusted_close']).tz_localize('UTC')
    
    # Generate and save the report
    qs.reports.html(strategy_returns, benchmark=spy_returns, output='davis_double_play_report.html', title='Davis Double Play Strategy')
    print("Performance report saved to 'davis_double_play_report.html'")
    
    return strategy_returns


# ==============================================================================
# Step 6: Deep Analysis of Returns
# ==============================================================================

def deep_analysis(strategy_returns):
    """
    Performs a deep analysis on returns by month, year, and market regime.
    """
    print("\n--- Deep Analysis of Returns ---")
    
    # --- i) Analysis by Month and Year ---
    print("\nMonthly Returns (%):")
    qs.stats.display_monthly_returns(strategy_returns)
    
    print("\nYearly Returns (%):")
    qs.plots.yearly_returns(strategy_returns, show=True)

    # --- ii) Analysis by Market Regime ---
    # Fetch benchmark data for regime classification
    spy_ohlcv = client.get_eod_historical_data("SPY.US", "d", strategy_returns.index.min(), strategy_returns.index.max())
    spy_df = pd.DataFrame(spy_ohlcv)
    spy_df['date'] = pd.to_datetime(spy_df['date'])
    spy_df.set_index('date', inplace=True)
    
    # Calculate moving averages for SPY
    spy_df['ma50'] = spy_df['adjusted_close'].rolling(50).mean()
    spy_df['ma200'] = spy_df['adjusted_close'].rolling(200).mean()
    spy_df.dropna(inplace=True)

    # Define regimes
    spy_df['regime'] = 'SIDEWAYS'
    spy_df.loc[(spy_df['adjusted_close'] > spy_df['ma50']) & (spy_df['ma50'] > spy_df['ma200']), 'regime'] = 'UPTREND'
    spy_df.loc[(spy_df['adjusted_close'] < spy_df['ma50']) & (spy_df['ma50'] < spy_df['ma200']), 'regime'] = 'DOWNTREND'

    # Align strategy returns with regimes
    analysis_df = pd.DataFrame({'strategy': strategy_returns.tz_localize(None)}) # remove tz for merge
    analysis_df = analysis_df.join(spy_df['regime']).dropna()

    print("\nPerformance by Market Regime:")
    regime_perf = analysis_df.groupby('regime')['strategy'].agg(
        sharpe=qs.stats.sharpe,
        cagr=qs.stats.cagr,
        max_drawdown=qs.stats.max_drawdown,
        volatility=qs.stats.volatility,
        total_return=lambda x: qs.stats.comp(x)
    ).reset_index()
    
    print(regime_perf)
    qs.plots.plot_yearly_returns(
        returns=analysis_df['strategy'],
        benchmark=None,
        yc=analysis_df['regime'],
        title='Yearly Returns by Market Regime'
    )


# ==============================================================================
# Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    if API_KEY != 'YOUR_EODHD_API_KEY':
        # 1. Run the backtest
        portfolio_history = run_backtest(start_date_str='2016-01-01', end_date_str='2025-06-30')
        
        if not portfolio_history.empty:
            # 2. Evaluate performance
            strategy_returns = evaluate_performance(portfolio_history)
            
            # 3. Perform deep analysis
            deep_analysis(strategy_returns)
        else:
            print("Backtest did not produce any results. Check criteria or data fetching.")
    else:
        print("Execution skipped. Please provide a valid EODHD API key.")

