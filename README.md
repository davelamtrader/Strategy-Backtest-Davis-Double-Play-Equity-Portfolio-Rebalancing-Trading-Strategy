# Strategy-Backtest-Davis-Double-Play-Equity-Portfolio-Rebalancing-Trading-Strategy

## Overview

This repository showcases a strategy inspired by the "Davis Double Play" concept, a value investing approach emphasizing the simultaneous growth of Earnings Per Share (EPS) and Price-to-Earnings (PE) multiple to generate superior returns . This strategy aims to identify and capitalize on companies exhibiting accelerating earnings growth while maintaining a reasonable valuation, a core tenet of the Davis Double Play strategy .

## Strategy Logic

The core principle is to identify stocks where both EPS and PE are expected to increase, creating a "multiplier effect" on returns . The strategy uses a combination of fundamental and quantitative analysis:

1.  **Fundamental Screening:**  Companies are screened based on earnings forecasts and growth potential. Emphasis is placed on identifying firms with accelerating earnings growth  .
2.  **Valuation Assessment:**  The strategy assesses valuation using the Price-to-Earnings (PE) ratio. It aims to identify stocks trading at a reasonable PE, with an eye towards future PEG ratio  .
3.  **"Double Play" Signal:**  The strategy looks for companies where accelerating earnings are combined with a relatively low valuation. This combination is expected to lead to simultaneous EPS and PE growth, the "double play"  .

## Data & Methodology

-   **Data Source:**  The attached Python code uses the EODHD API (replace 'YOUR_EODHD_API_KEY' with your actual API key) for fetching historical stock data, including price, earnings, and other fundamental data  .
-   **Key Metrics:**
    -   **PE Ratio (PE_ttm):**  Trailing twelve-month P/E ratio, collected on the day before the quarterly rebalance  .
    -   **Growth (Growth):**  Calculated using the quarterly year-over-year earnings growth  . The code calculates a second-order growth rate (acceleration) to identify accelerating earnings  .
-   **Screening & Portfolio Construction:**
    -   The strategy rebalances quarterly (April 20th, July 20th, October 20th, and January 20th) based on earnings announcements and forecasts  .
    -   The code screens for stocks meeting specific criteria, including positive earnings growth and a positive second-order growth rate (acceleration)  .
    -   The top-ranked stocks based on earnings acceleration are selected for the portfolio  .
-   **Backtesting:**  The included Python code provides a framework for backtesting the strategy using historical data. The code includes data fetching, data processing, and performance evaluation based on metrics like annualized returns and Sharpe ratio  .

## Code Structure

The Python code is organized modularly:

-   eodhd  library is used for data retrieval  .
-   Helper functions are used for data retrieval and data caching, reducing API calls  .
-   The  run_backtest  function executes the core backtesting logic  .
-   The  evaluate_performance  function calculates and reports key performance metrics using the  quantstats  library  .
-   The  deep_analysis  function provides a detailed analysis of the backtest  .
