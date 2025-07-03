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


**30.06 - 06.07: Week 1**


