# EC601 Project
This repository contains a semester project for a class in product design in electrical and computer engineering. 

## Background
The efficient market hypothesis (EMH) states that in an efficient market, for any given asset, the price of that asset reflects all available information that effects the price of that asset and therefore consistent excessive returns or “beating the market” is impossible. Investors are constantly seeking methods to make returns that outperform the market but often fail due to the EMH holding true for many publicly traded assets. Recent technical evaluations of various machine learning algorithms abilities to predict the prices of traditional assets, such as stocks, has shown to be very limited and certainly not been able to violate the EMH ([Fischer et al.](https://iranarze.ir/wp-content/uploads/2019/01/E10789-IranArze.pdf); [Gu et al.](https://academic.oup.com/rfs/article/33/5/2223/5758276)). Similar conclusions have been drawn when applying various machine learning algorithms to predict Bitcoin returns using technical, blockchain-based, sentiment-/interest-based, and asset-based data ([Jaquart et al.](https://www.sciencedirect.com/science/article/pii/S2405918821000027#bib3)). 

Market efficiency is a quantifiable metric that can change over time and varies across different assets ([Tran et al.](https://www.sciencedirect.com/science/article/pii/S1544612318305348)). Although there is mixed evidence pertaining to the market efficiency of Bitcoin, the general trend is that Bitcoin was not an efficient market in its early phases, pre-2017, and has become increasingly more efficient with time. An increasing degree of market efficiency for Bitcoin seems intuitive as the market has become increasingly competitive ([Jaquart et al.](https://www.sciencedirect.com/science/article/pii/S2405918821000027#bib3)) and considering that a competitive market is efficient because equilibrium is achieved where the demand price and supply price are equal.

## Problem
For investors, generating consistently excessive returns or “beating the market” is considered impossible in an efficient market according to the EMH.

## Approach
Find the value of adjusted market inefficiency magnitude (AMIM) for many different cryptocurrencies to identify inefficient markets that present opportunities for excessive returns. Apply long short-term memory (LSTM) neural networks to predict movements in price for the previously identified cryptocurrencies and develop a strategy that automatically executes trades to capitalize on the volatility of cryptocurrency. 

With major crypto trading platforms such a [Binance](https://www.binance.us/en/home) offering trading support for 50+ cryptocurrencies, it is possible that some of the lesser established cryptocurrencies are not very competitive leading to possibly inefficient markets. 

## Applicable Open Source Projects, APIs, and Research Papers of Interest

### 1. [A simple but powerful measure of market efficiency](https://www.sciencedirect.com/science/article/pii/S1544612318305348)
This paper details the derivation of a measure to quantify the level of market effeciencey for a given asset using its pricing history. This measure is named the adjusted market inefficiency magnitude (AMIM). In summary, Tran et al. find a method to normalize the autocorrelation found in the price function (price over time) for a given asset. If there is significant autocorrelation it can be determined that the underlying asset is not considered to be part of an efficient market. 

### 2. [Cryptocurrency Price Prediction Using Deep Learning](https://towardsdatascience.com/cryptocurrency-price-prediction-using-deep-learning-70cfca50dd3a)
This article summmarizes how to implement LTSM neural networks to predict cryptocurrencey prices and provides an open source git repo with the python scripts used to accomplish this.

### 3. [Short-term Bitcoin market prediction via machine learning](https://www.sciencedirect.com/science/article/pii/S2405918821000027#bib3)
This paper details the implementation of a LSTM neural network to predict Bitcoin prices using technical, blockchain-based, sentiment/interest-based, and asset-based data. The authors also quantify the performance of this model and provide details regarding the various data sources they used. 

### 4. [Binance API Documentation](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md)
This provides usefull information regarding APIs avialable for the crypto trading platform [Binance](https://www.binance.us/en/home). These APIs allow for the retrieval of relevant live data for various cryptocurrencies and placing market orders without user interaction.

### 5. [CryptoCompare API](https://min-api.cryptocompare.com)
This API provides a user-friendly way to obtain relevant technical cryptocurrency historical data that can be used for training/testing data sets. 

### 6. [Reddit APIs](https://www.reddit.com/dev/api/)
These APIs provide a way to query reddit for information relavent to a particular topic. This can be useful given there are a number of reddit pages dedicated to the dicussion of cryptocurrencey. 

### 7. [Twitter APIs](https://developer.twitter.com/en/docs/twitter-api)
These APIs provide a way to listen for tweets relating to specific topics and analyze their sentiment. This could be very usefull in obtaining real time public bullish/bearish sentiments relating to a specific cryptocurrencey. 


