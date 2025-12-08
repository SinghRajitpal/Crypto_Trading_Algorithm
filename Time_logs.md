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

**Currently Working on (07.09.2025):**

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

**From 07.09.2025 - 05.12.2025 (submission)**

**Milestones achieved:**

1. Sentiment Analysis:
- Identified that historical data for crypto sentiment is costly
- For live/demo trading github repo exist but API's from Reddit, Twitter (X), Google Trends cost
--> Chose to skip sentiment analysis

2. Huge Fundamental Trading Shift and defining new trading setup:
- Read/skimmed the book "The Elements of Quantitative Investing" by Paleologo.
- Read/skimmed the book "Advanced Portfolio Management" by Paleologo
- Read/skimmed the book "Quantitatve Portfolio Management" by Paleologo
- Read a lot of twitter posts and threads of quantymacor, cephalpood, systematicls.

One key thing that I discovered which practicioners actually use is that they try to keep the optimal portfolio. This is done by choosing weights for assets and rebalancing (adjusting those weights) them.
This is a fundamental shift because this is proven to be a superior system. These books also discuss this in detail and also proper risk management.

The final system pseudocode which has to be implemented:
- Ingest Data (already implemented) and historical data and funding rate fetching
- Update risk model with EWMA
- Forecast return
- Run mean-variance optimization
- Add kelly overlay and check with constraints (caps)

And to validate the performance, a walk-forward system for backtesting is used as described in the books.

Hence, we need to do quite a bit of infrastructure change.

3. Data
- implemented fetching historical data and funding rates
- structure stayed similar

4. Forecasting System
- implemented a basic regression with ridge penalty mentioned in the books
- this serves more as a placeholder
- replaced the trade signal with a forecast signal object class
- The algorithm layer becomes the forecasting layer

5. New Execution system
- Moving from old to new portfolio and risk management system
- Implemented risk model with EWMA (and experimenting with GARCH)
- Implemented MVO
- implemented full kelly and fractional for position leveraging
- implemented calculating trades and executing them
- Integrated the forecasting layer with the new execution layer

6. Deeplearning models
- Revised the course machine learning specializaion on coursera and reading studies and papers on how to forecast more accurately and it seems that recurrent neural networks are best for sequential data. From them multiple studies have shown GRU has performed well.
- Implemented a GRU model based on those studies
- adapted the forecasting system to support this and changed the config file to support those parameter changes

7. Binance Testnet
- trades being set on Binance testnet but data on testnet is flawed
- looking for alternatives to replace this such as own custom live trading setup run but with simulated orders.
- too much effort for effective setup, hence keeping testnet but only for testing purposes and not performance testing.

8. Backtesting
- trying to identify a compatible structure:
The thing is we need to train the GRU Model, run complete portfolio backtest and maybe also reuse that trained model.
Solution:
- Layer A: Train model and make predictions in walk-forward fashion and this model is saved
- Layer B: Use predictions from Layer A and run full trading logic and do backtest
And when satisfied with results the saved model can be reused in live/testnet again

- implemented Layer A walk-forward model trainer and predictor, also calculated metrics which were often mentioned and are used to assess a models predictive power, also visualized training and validation loss
- implemented that results are shown in backtest results folder and named based on date and trial and also model is saved
- implemented Layer B to reuse predictions made in Layer A and run complete portfolio logic, also calculate benchmark metrics for BTCUSDT and trading setup metrics, also plots for equtiy curve for better comparison
- a simbroker is created to "simulate" the binance exchange. It tracks portfolio value and calculates metrics.
- implemented that results are shown in backtest results folder with all metrics and plots for the trading setup and benchmark
- designed to make them run separetely and together

To integrate this backtesting system with the live workflow, the data, forecasting, execution, main, config to use saved model for predictions. No retraining yet.

9. Testnet depreciation
Binance depreciated their testnet and introduced a demo mock trading exchange. It is essentially the same thing but the api has changed. The wrapper ccxt currently in use has not updated for the new update. Hence following problems observed:
- No connection to new demo mock exchange
- 


