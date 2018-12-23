# Currency-Trading
A quantitative long-term currency trading strategy 

## Introduction to Currency Trading

### What is Currency Trading?

The exchange of currencies around the world in order to conduct foreign business and trade.

### What is the Forex Market?

Unlike the NYSE, the market for currency exchanges is open 5.5 days a week for 24 hours a day. Currencies are traded OTC entirely electornically via a network of computer. Due to the constant activity of exchanges located in various timezones, price quotes are always undergoing fluctuations. 

### How Is Currency Pair Trading Profitable? 

Because currencies are traded in pairs, whenever a trade is made, a trader is going long on one currency and shorting the other.

### How Can I Make Money?

This algorithm is designed to identify when the exchange rates are likeley to mean revert and thus executes trades at lows and sell at highs. 

## Investment Thesis

The AUD/JPY currency pair will be purchased when the price is within 1% of a retracement level price and when that price is at or below the lower Bollinger band. The AUD/JPY currency pair will be sold when the price is within 1% of a retracement level price and when that price is at or above the upper Bollinger band. 

## Why This Thesis?

Observing historical data of the exchange rate for the AUD/JPY currency pair, I noted that exchange rate movements occured as predicted by Fibonacci retracement often. That is, the levels at which one would expect the exchange rate to either break through a support or bounce at a level of resistance were very much in line with what actually occured in the historical data. 

One primary concern was that the Fibonacci retracement indicator did not offer any insight as to whether the retracement level would act as a point of support or resistance, which is where Bollinger Bands come in. Bollinger Bands utilize a moving N day standard deviation and SMA which serves to identify overbought and oversold levels. These levels correspond with indicating whether a price will bounce or not: a price above the upper band is likely to mean revert back towards the SMA and a price below the lower band is likely to mean revert up towards the SMA. 

We can thus use these two indicators in conjuction for resistance / bounce identifcation at fibonacci retracement level exchange rates. If the retracement level is not outside the bollinger band width, we conclude that the retracement level will act as a support zone and exchange rate will break through and continue in the direction it was heading. 
