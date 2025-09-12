**Until 23.06.2025:**

1. Data
- Setup Binance Client with functions to interact with broker
- Setup high market coins fetcher
- Setup Websocket Handler for OHCLV market data
- Setup Indicator Calculation of most used technical indicators using a highly-efficient library called ta-lib
- Setup Data Processor for preparing data for strategy
- Setup Data Engine to handle all data and finalize input for strategy

2. Strategy (Algorithm)
- Setup a trade signal class for easier use of signal properties
- Setup a base strategy class to be inherited e.g. "ma_crossover.py"
- Setup a algo engine overseeing strategy and finalizing trade signal input for execution

3. Execution
- Setup a portfolio manager responsible for allocating capital between defined assets
- Setup a basic risk manager to assess trade signal
- Setup a execution engine which uses the Binance Client to execute trade signal from strategy.

All engines are then managed inside main.py.


**From 30.06.2025 - 07.09.2025**

**Milestones achieved:**

1. Implement a portfolio allocationd and dynamic risk management system by summarising a book (this system is an improvement compared to previous system
   
- Learned about dynamic weight allocation
- Learned about volatitily targeting and how weights depend on that
- Learned about atr and volatility filtering
- Learned about Kelly Criterion and Fractionalized Kelly Criterion for more conservative approach (optimal bet sizing(
- Implemented the above

2. Implement a backtesting engine
   
- Problem: For my functionality I need a customized backtesting engine
- My backtesting engine design: customized data ingestion, strategy part stays same, customized exchange and order manager
- Further Improvement needed as its slow --> numpy and deques
- Current Plotting implementation: Use of quantstats_lumi --> needs to be changed to fully custom for speed and practicality
- Learned about important trading / performance metrics and how they're calculated: time under water, max drawdown, sharpe ratio, sortino ratio, winrate, CAGR
- Learned about the importance of sharpe: risk adjusted returns

3. Implement a Stress handler and Order tracker
   
- In flash crashed my bot should either turn of or rebalance portfolio more conservatively
- Order Tracker for stop losses and take profit market order tracking and closing them

4. Research more about strategies
   
- Read Ernest Chans book
- Currently reading Robert Carvers Advanced Futures Trading Strategies Book
- Researched about crypto inefficiencies which are capactiy constrained and which could be exploited
- Researched about arbitrage opportunities specifically funding arb, stat arb, pairs trading, basket trading, exchange arb
- Identified needs of change in infrastructure to make arbitrage work
- Read > 250 tweeter posts of reliable quants (cephalapod, quantymacro, vertox, hangukoquant, stat_arb, robot james, r_bit, etc.) to have a better idea of quantitative trading strategies --> arb was most prominent for a retail trader
- Listed to several episodes of "Flirting with Models" podcasts dedicated towards the quant space
- Watched Jim Simons' (Renaissance Technologies: Medaillon Fund) interview

5. Trading Workshop Millennium @ ETH (only open for Master students in a quantitative field)
   
- Understood Retail Trader Edges
- Talked to Millennium CEO for Switzerland and his advice

6. Identify Paper Structure

Intro:
- Wall street change
- Quant industry
- Trading algorithm and focus of this paper

What is a trading algorithm?
- Different types of trading algorithmn
- Structure of a trading algorithm
- How are they developed? --> Programming Languages

Portfolio Management and Risk Management - Quantitative Approach

General Strategy types and outline

(for now) Statisitical Abitrage
- The main idea
- Price exploitations
- Identify Correlation
- Machine Learning Model and Ridge Regression

My crypto futures trading algorithm outlined in detail

Performance Analysis (Backtest results)
- Comparing and analysing the different metrics used for assessing performance
- Comparing bot performance with other hedge funds (buy-side)

Outro

Appendix

Resource LIst


**Currently Working on:**

1. Sentiment Analysis
   
- Skimmed through few papers about Finbert sentiment analysis
- Watched several youtube videos about NLP and sentiment analysis
- Found github repo for sentiment analysis
- Identified sources to train model on huggingface (reddit, x, google trends)

2. Running algorithm on Server
   
- Talked to Marco Schmid about Server at school
- Researched about VPS to reduce latency as much as possible

3. Finishing Robert Carvers "Advanced Futures Trading Strategies Book"

4. Identifying Newsletters about futures trading strategies from Quantymacro, Vertox, Robot James (highly recommend), Stat_Arb


**Roughly Codebase size change: > +8000 loc**
