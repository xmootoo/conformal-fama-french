# Conformal Time Series Forecasting with Fama French Factor Models
Project currently under development. The main idea is to test whether Fama-French factors (3 factor and 5 factor) are useful in forecasting of daily returns for S&P 500 with uncertainty quantification, in particular, using the conformal time series forecasting framework proposed in [Stankeviciute et al., 2021](https://proceedings.neurips.cc/paper/2021/hash/312f1ba2a72318edaaa995a67835fad5-Abstract.html). Will add more details on usage in the near future, however core components with docstrings are stored in `dataloading.py` which constructs the data pipeline, and `conformal.py` which
provides the evaluation scripts and statistical framework for conformal time series forecasting.

Data is obtained from:
* https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html. Select "Fama/French 5 Factors (2x3) [Daily]" and "Fama/French 3 Factors [Daily]".
* https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks.
