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
