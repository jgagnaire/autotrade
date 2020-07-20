# autotrade

### Prepare working environment
```
conda create --name autotrade python=3
conda activate autotrade
pip install -r requirements.txt
```

### Use LinearRegression and KernelRidge to predict the price of a stock

```
python regressor.py
```

### Use K-Nearest Neighbors and Random Forests classifiers to predict an increase on the next day

```
python classifier.py
```

### External resources

* Yahoo! Finance API, providing end-of-day historical data in CSV format
* Technical Analysis indicators from the ta Python 3 package: https://github.com/bukosabino/ta/
* Python3 and the traditional Machine Learning frameworks and libraries: pandas, numpy, sklearn, matplotlib
* Amazon SageMaker notebook instance service for data exploration
* Amazon SageMaker training instance with GPU for DeepAR training
