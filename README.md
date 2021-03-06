[![codecov](https://codecov.io/gh/FriendlyUser/rbc-stock-trade-analyzer/branch/main/graph/badge.svg?token=3ZE6JY4C05)](https://codecov.io/gh/FriendlyUser/rbc-stock-trade-analyzer)
# rbc-stock-trade-analyzer
Simple rbc stock trade analyzer. For now only works with data from rbc

**Requirements**

* Python 3.7.1 and above
* Data from rbc (can just click export)

The data format should be as in `test.csv`.

| Date,Activity,Symbol,Quantity,Price,Value,Currency,Description,Symbol Description | 
|-----------------------------------------------------------------------------------| 
| "July 21, 2019",Sell,BB,81,4,324,CAD,lame,test                                    | 
| "May 21, 2019",Buy,AMC,8,10,-80,USD,lame,test                                     | 


To run

```sh
pip install -r requirements.txt
python make_report.py
```

To run tests
```sh
python -m unittest
```


Expected that all trades are recorded to get accurate results. 

Feel free to make a pr, typically since I trade a lot, I want estimations on how much profit I made and sometimes for specific stocks.

## Disclaimer

Project is not complete.

I started trading in 2019, and as a result I am lacking lots of experience and a lengthy trade history.

I am aware that rbc changed its data format arbitrary one time and added an extra field.

Works for the most part, can return misleading results if you purchase the same stock on CAD and USD.

For example CMC.CN on the CSE and CWSFF on the otc.

I am canadian and as a result the code is optimized for canadian tickers first.
