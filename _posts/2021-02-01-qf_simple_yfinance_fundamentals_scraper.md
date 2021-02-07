---
title: "[QF] Building a Simple Financial Statements Python WebScraper Using YahooFinance"
date: 2021-02-01
tags: [web-scraping, python, development]
header:
  image: "/images/financial-statements.jpg"
excerpt: "Using common webscraping python modules to build a simple scraper to scrape financial statements of public companies available on YahooFinance."
mathjax: "true"
layout: single
classes: wide
---


> ## [Detailed Explanation Pending]

## Final Product Class

Below is all the codes compiled together into a class called *YFinanceEquityFundamentalDataSource*:

```python

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import urllib.request as ur


class YFinanceEquityFundamentalDataSource(object):

    def __init__(self,
                ticker:str,
                *args,
                **kwargs   
    ):
        # URL link
        self._ticker = ticker
        self.url_is = 'https://finance.yahoo.com/quote/' + ticker + '/financials?p=' + ticker
        self.url_bs = 'https://finance.yahoo.com/quote/' + ticker + '/balance-sheet?p=' + ticker
        self.url_cf = 'https://finance.yahoo.com/quote/' + ticker + '/cash-flow?p='+ ticker

    #---| Properties
    @property
    def income_statement(self):
        return self._extract_and_clean(
                        self._read_div_ls(self.url_is)
        )
    @property
    def balance_sheet(self):
        return self._extract_and_clean(
                        self._read_div_ls(self.url_bs)
        )
    @property
    def cash_flow_statement(self):
        return self._extract_and_clean(
                        self._read_div_ls(self.url_cf)
        )


    #---| Back-end
    def _read_div_ls(self, url):
        # INSPIRATION:
        # https://towardsdatascience.com/web-scraping-for-accounting-analysis-using-python-part-1-b5fc016a1c9a
        
        read_data = ur.urlopen(url).read() 
        soup= BeautifulSoup(read_data,'lxml')        
        raw_ls= [] # Create empty list
        for l in soup.find_all('div'): 
            #Find all data structure that is ‘div’
            raw_ls.append(l.string) # add each element one by one to the list

            # Exclude certain columns if needed
            #raw_ls = [e for e in raw_ls if e not in ('Operating Expenses','Non-recurring Events')]

        ls = list(filter(None,raw_ls))
        return ls

    #---| Check if string can be converted to pd.Timestamp
    def _date_convertible(self, string:str):
        if "/" not in string:
            return False
        try:
            pd.Timestamp(string)
            return True
        except:
            return False

    #---| Check if string can be converted to numeric value
    def _is_numeric(self, string:str):
        if string == "-":
            return True
        try:
            float(string)
            return True
        except:
            try:
                float(string.replace(",",""))
                return True
            except:
                return False

    #----| Clean list of data (primarily for numeric checking)
    def _clean_data_piece(self, data_piece:list):
        cleaned = []
        for d in data_piece:
            if str(d) == "-":
                cleaned.append(np.nan)
            elif self._is_numeric(str(d)):
                cleaned.append(str(d).replace(",",""))
            else:
                cleaned.append(str(d))
        return cleaned

    #---| ETL
    def _extract_and_clean(self, ls):
        dates = [i for i in ls if self._date_convertible(str(i))]
        start_index = ls.index(dates[-1]) + 1

        dates = [pd.Timestamp(str(i)).date() for i in dates]
        if 'ttm' in ls:
            tuple_length = len(dates) + 1
            columns = ['Annual','TTM'] + dates
        else:
            tuple_length = len(dates)
            columns = ['Annual'] + dates
        data = []
        
        for i in range(start_index,len(ls)):
            data_piece = ls[i:i + tuple_length + 1]
            if not self._is_numeric(data_piece[0]):
                passed = False
                for d in data_piece[1:]:
                    if not self._is_numeric(d):
                        passed = False
                        break
                    else:
                        passed = True
                if passed:
                    data.append(tuple(self._clean_data_piece(data_piece)))
        df = pd.DataFrame(data, columns = columns).set_index("Annual")
        for c in df.columns:
            df[c] = pd.to_numeric(df[c])
        return df

```
