---
title: "[QF] Probing Volatility II: Portfolio Optimization & Random Matrix Theory"
date: 2020-06-03
tags: [research]
header:
  image: "/images/qf_vol_part1_banner.PNG"
excerpt: "Exploring portfolio management by finding minimum/maximum solutions of a constructed system of equations & constraints, specifically for optimization purposes, analytically & computationally, incorporating quantitative techniques though Random Matrix Theory to filter noises & tackling higher dimensions."
mathjax: "true"
layout: single
classes: wide
---

# Overview & Preparation
>*Optimization*, in general, is simply solving for minimum/maximum solution(s) of a *system of equation(s)*, satisfying given constraints.

Recalling our brief overview on the basics of Modern Portfolio Theory in my [first QF post](https://jp-quant.github.io/qf_intro/ "first QF post"), one of the metric to which used in evaluating performance of any investment, mirroring somewhat of a standardization technique, is the **Sharpe Ratio**, to which we will dub it as $$\boldsymbol{I_p}$$:

>$$\boldsymbol{I_p} = \frac{\bar{R_p} - R_f}{\sigma_p}$$, where
$$\bar{R_p}$$ = Average Returns
$$\sigma_p$$ = Volatility = Standard Deviation of Returns

In general, we want to **minimize** $$\sigma_p$$ & **maximize** $$\bar{R_p}$$, simply seeking to achieve not just as *highest* & as *consistent* of a returns rate as possible, but also having as little as possible in its maximum drawdown.
>In regards to selecting the optimal assets to pick in a portfolio, this is an extensive topic requiring much empirical & fundamental research with additional knowledge on top of our current topic at hand (we will save this for another section).

Let there be *M* amount of securities selected to invest a set amount of capital in, we seek for the optimal allocations, or weights, $$w = \begin{vmatrix} w_1\\ w_2\\ \vdots\\ w_M \end{vmatrix}$$  such that $$-1 \leq w_{i}\leq 1$$ and $$\sum_{1}^{M}w_i = 1$$, being 100% of our capital. 

>-  If you want **long only** positions, simply set $$0 \leq w_{i}\leq 1$$ instead.
- We are allowing short positions in all of our approaches since we want absolute returns & hedging opportunities, coming from a perspective of an institutional investor, hence each weight can be negative.

---

Before moving forward, we first need to address the context regarding *M securities* we seek to allocate our capital towards:
- At **this** moment in time $$t_N$$, as we are performing analysis to make decisions, being the latest timestamp (this is where back-testing will come in, as I will dedicate a series on building an event-driven one from scratch), we have *N* amount of data historically for *M* securities, hence the necessity for an *N x M* returns table.
- **Looking towards the future**  $$t_{N + dN}$$ , before we seek to find the optimal weights $$w$$, to compute $$\sigma_p$$ & $$\bar{R_p}$$ (again please refer to my [first QF post](https://jp-quant.github.io/qf_intro/ "first QF post") for the intricate details), as we will not yet be touching base on such extensive topic that having to deal with prediction (I will dedicate another series for this topic), we need to determine the answers for:

	- What is the returns $$\hat{r_i}$$ for each security $$i = 1,2...M$$?
		>In our demonstrative work, as being used as a factor in various predictive models, we will set $$\hat{r_i} = \bar{r_i}$$ being the **average returns "so far"** (subjective)

	- How much "uncertainty," or in other words, how much will $$\hat{r_i}$$ deviate? Or simply, what is $$\sigma_{\hat{r_i}}$$?
		>Given we have set $$\hat{r_i} = \bar{r_i}$$ , this will simply be the **standard deviations** of $$\bar{r_i}$$. Thus, we set $$\sigma_{\hat{r_i}} = \sigma_{\bar{r_i}}$$.

---
Unlike our [**previous post**](https://jp-quant.github.io/qf_volatility_p1/ "previous post") in this series on volatility, this time, we will start with a *N x M*  **daily close** $$RET$$ table of ~1,000 securities in the U.S. Equities market,  dating back **approximately 2 years worth of data** since May 1st 2020, or 600-700 data points available.

Before we proceed, it is important to note that we have over 1,000 securities available. This is dimensionally lacking in data as for our *N x M* returns table $$RET$$ (N < M). It is somewhat similar, although not completely accurate, to saying having 1-2 points when working with 3 assets. We will slightly touch base on this problem in later posts. Although, for now, when working with picking M assets,  we seek for **M < N** as much as we can, especially our purpose is to be able to work with **ANY** universe of investment targets, that being either:
- Randomizing selection of any given M < N
- Selection in any specific sector (luckily, with some having the entire sector < N)

First, load up our full $$RET$$ dataframe along with an information table of ALL the securities available (~1000 securities). We then proceed on checking the count of securities in each sector for all sectors, and index them for usage:

```python
#----| Import necessary modules
%matplotlib notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import minimize

#----| Preprocessed Daily Returns Table of ~1000 securities in the US Equities Market
_RET_ = pd.read_csv("ret.csv",index_col=0,header=0)

#----| Information of ALL available securities
universe_info = pd.read_csv("universeInfo.csv",index_col=0,header=0).astype("str")

#----| Identify & segment securities with all sectors
sectors = {s:universe_info[universe_info["sector"] == s].index for s in universe_info["sector"].unique()}
```


```python
_RET_.columns,_RET_.index
```

    (Index(['AMG', 'MSI', 'FBHS', 'AMD', 'EVR', 'NKE', 'NRG', 'EV', 'VRSN', 'SNPS',
            ...
            'FND', 'CVNA', 'AM', 'IR', 'JHG', 'ATUS', 'JBGS', 'BHF', 'ROKU',
            'SWCH'],
           dtype='object', length=984),
     Index(['10/9/2017', '10/10/2017', '10/11/2017', '10/12/2017', '10/13/2017',
            '10/16/2017', '10/17/2017', '10/18/2017', '10/19/2017', '10/20/2017',
            ...
            '4/20/2020', '4/21/2020', '4/22/2020', '4/23/2020', '4/24/2020',
            '4/27/2020', '4/28/2020', '4/29/2020', '4/30/2020', '5/1/2020'],
           dtype='object', name='date', length=635))


```python
universe_info
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>sector</th>
    </tr>
    <tr>
      <th>symbol</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>AMG</th>
      <td>AFFILIATED MANAGERS GROUP</td>
      <td>Financials</td>
    </tr>
    <tr>
      <th>MSI</th>
      <td>MOTOROLA SOLUTIONS INC</td>
      <td>Information Technology</td>
    </tr>
    <tr>
      <th>FBHS</th>
      <td>FORTUNE BRANDS HOME &amp; SECURI</td>
      <td>Industrials</td>
    </tr>
    <tr>
      <th>AMD</th>
      <td>ADVANCED MICRO DEVICES</td>
      <td>Information Technology</td>
    </tr>
    <tr>
      <th>EVR</th>
      <td>EVERCORE INC - A</td>
      <td>Financials</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>ATUS</th>
      <td>ALTICE USA INC- A</td>
      <td>Telecommunication Services</td>
    </tr>
    <tr>
      <th>JBGS</th>
      <td>JBG SMITH PROPERTIES</td>
      <td>Real Estate</td>
    </tr>
    <tr>
      <th>BHF</th>
      <td>BRIGHTHOUSE FINANCIAL INC</td>
      <td>Financials</td>
    </tr>
    <tr>
      <th>ROKU</th>
      <td>ROKU INC</td>
      <td>Telecommunication Services</td>
    </tr>
    <tr>
      <th>SWCH</th>
      <td>SWITCH INC - A</td>
      <td>Information Technology</td>
    </tr>
  </tbody>
</table>
</div>




```python
for s in sectors:
    print(s+":",len(sectors[s]))
```

    Financials: 147
    Information Technology: 143
    Industrials: 125
    Consumer Discretionary: 116
    Utilities: 37
    Health Care: 98
    Telecommunication Services: 54
    Materials: 57
    Real Estate: 78
    Energy: 44
    Consumer Staples: 50
    Others: 35
    

Before we proceed on finding the "optimal" $$w$$ solutions, it's useful to have a function that could take in any given allocation weight & the respective returns table to compute the some evaluation metrics & returns (as well as cummulative returns) data:

```python
def evaluate_w(RET,_w_):
    portfolio_ret = RET.dot(_w_.reindex(RET.columns).values)
    _cov_ = RET.cov() #---| to calculate std
    _mean_ = portfolio_ret.mean()
    _std_ = np.sqrt(np.dot(_w_.T,np.dot(_cov_,_w_)))
    metrics = pd.Series(data=[_mean_,_std_,(_mean_/_std_),portfolio_ret.min()],
                        index=["avg_ret","std","sharpe","max_drawdown"])
    return {"returns":portfolio_ret,"cummulative_returns":(1+portfolio_ret).cumprod(),"metrics":metrics}
```

Since we are going to perform evaluations on multiple weights and compare them to each other, we also wrote a bulk evaluation function that takes in an *M x W* matrix, as *W* different weights of *M* securities, and an *N x M* returns table with N returns of such M securities:

```python
def evaluate_bulk_w(_RET_,all_w):
    RESULTS = {w:evaluate_w(_RET_,all_w[w]) for w in all_w.columns}
    all_metrics = pd.DataFrame({w:RESULTS[w]["metrics"] for w in RESULTS})
    all_cumret = pd.DataFrame({w:RESULTS[w]["cummulative_returns"] for w in RESULTS})
    return {"metrics":all_metrics.T,"cum_ret":all_cumret}
```


In addition, to further evaluate the performances of our extract allocations weights, we will need to split our $$RET$$ data into in-sample & out-sample. Performing calculations to find such weights on the in-sample, then using the weights to apply on the out-sample. This is equivalent to saying:
> If we use the "optimal" weights, calculated from in-sample data, being the latest possible date and invest at that date, *without any predictive features*, how well will such weight perform in the out-sample, or the future?

```python
def ioSampleSplit(df,inSample_pct=0.75):
    inSample = df.head(int(len(df)*inSample_pct))
    return (inSample,df.reindex([i for i in df.index if i not in inSample.index]))
```


Lastly, for daring purposes, as we will obtain results that are very much interesting & sensible, we are going to work with the **Others** sector, representing **all different funds & indexes**, comprising the entire "market" as much as we can, with components as independent they can. Observe the description of such sector of securities:



```python
universe_info.reindex(sectors["Others"]).sort_index()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>sector</th>
    </tr>
    <tr>
      <th>symbol</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DIA</th>
      <td>SPDR DJIA TRUST</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>EEM</th>
      <td>ISHARES MSCI EMERGING MARKET</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>EFA</th>
      <td>ISHARES MSCI EAFE ETF</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>EWJ</th>
      <td>ISHARES MSCI JAPAN ETF</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>FEZ</th>
      <td>SPDR EURO STOXX 50 ETF</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>FXI</th>
      <td>ISHARES CHINA LARGE-CAP ETF</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>GDX</th>
      <td>VANECK GOLD MINERS</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>GLD</th>
      <td>SPDR GOLD SHARES</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>HYG</th>
      <td>ISHARES IBOXX HIGH YLD CORP</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>IBB</th>
      <td>ISHARES NASDAQ BIOTECHNOLOGY</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>IEF</th>
      <td>ISHARES 7-10 YEAR TREASURY B</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>IWM</th>
      <td>ISHARES RUSSELL 2000 ETF</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>IYR</th>
      <td>ISHARES US REAL ESTATE ETF</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>MDY</th>
      <td>SPDR S&amp;P MIDCAP 400 ETF TRST</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>PFF</th>
      <td>ISHARES PREFERRED &amp; INCOME S</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>QQQ</th>
      <td>INVESCO QQQ TRUST SERIES 1</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>SHY</th>
      <td>ISHARES 1-3 YEAR TREASURY BO</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>SMH</th>
      <td>VANECK SEMICONDUCTOR</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>SPY</th>
      <td>SPDR S&amp;P 500 ETF TRUST</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>TLT</th>
      <td>ISHARES 20+ YEAR TREASURY BO</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>VNQ</th>
      <td>VANGUARD REAL ESTATE ETF</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>XBI</th>
      <td>SPDR S&amp;P BIOTECH ETF</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>XHB</th>
      <td>SPDR S&amp;P HOMEBUILDERS ETF</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>XLB</th>
      <td>MATERIALS SELECT SECTOR SPDR</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>XLE</th>
      <td>ENERGY SELECT SECTOR SPDR</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>XLF</th>
      <td>FINANCIAL SELECT SECTOR SPDR</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>XLI</th>
      <td>INDUSTRIAL SELECT SECT SPDR</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>XLK</th>
      <td>TECHNOLOGY SELECT SECT SPDR</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>XLP</th>
      <td>CONSUMER STAPLES SPDR</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>XLU</th>
      <td>UTILITIES SELECT SECTOR SPDR</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>XLV</th>
      <td>HEALTH CARE SELECT SECTOR</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>XLY</th>
      <td>CONSUMER DISCRETIONARY SELT</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>XME</th>
      <td>SPDR S&amp;P METALS &amp; MINING ETF</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>XOP</th>
      <td>SPDR S&amp;P OIL &amp; GAS EXP &amp; PR</td>
      <td>Others</td>
    </tr>
    <tr>
      <th>XRT</th>
      <td>SPDR S&amp;P RETAIL ETF</td>
      <td>Others</td>
    </tr>
  </tbody>
</table>
</div>


In addition, we split our full $$RET$$ data into 75% in-sample & 25% out-sample, letting the default variable **RET** as in-sample & **RET_outSample** as, of course, the out-sample data. Subsequent optimization steps in solving for the desired allocations weight $$w$$ will be performed on in-sample data:

```python
RET,RET_outSample = ioSampleSplit(_RET_[sectors["Others"]])
```

```python
RET.shape, RET_outSample.shape
```




    ((476, 35), (159, 35))


Like before, we will use pandas covariance function to calculated its sample covariance matrix $$\boldsymbol{C_{\sigma}}$$:


```python
RET.cov()
```


---
# Mean-Variance Optimization
Given a selected M amount of securities, we obtain our Symmetric *M x M* Covariance Matrix ($$\boldsymbol{C_{\sigma}}$$), calculated from an *N x M* $$RET$$ returns table of *N* returns data of such M securities (again, details in [first QF post](https://jp-quant.github.io/qf_intro/ "first QF post") ), a portfolio's volatility ($$\sigma_p$$) is calculated as:

>$$\sigma_p = \sqrt{w^\top \cdot \boldsymbol{C_{\sigma}} \cdot w}$$

Thus, our objective is to **find an allocation weight** $$w = \begin{vmatrix} w_1\\ w_2\\ \vdots\\ w_M \end{vmatrix}$$ that would *minimize* $$\sigma_p$$, or simply put will result in the **smallest possible** $$\sigma_p$$, such that $$-1 \leq w_{i}\leq 1$$ and $$\sum_{1}^{M}w_i = 1$$

---
## 1. Risk Minimization
>Commonly known as the **Minimum Variance Portfolio**, minimizing portfolio's risk serves as the first step for the overaching topic of Portfolio Optimization.

### 1a. Analytical Solutions
Going back to my earlier remark above on solving optimization problems, we can simply construct this problem with constraints utilizing the [Lagrangian method](https://scholar.harvard.edu/files/david-morin/files/cmchap6.pdf "Lagrangian method") ($$\mathcal{L}$$). My personal exposure to such mathematical technique stemmed from my Physics background, specifically on the topic of [Lagrangian Mechanics](https://en.wikipedia.org/wiki/Lagrangian_mechanics); such modeling technique can be applied to any other system optimization problems in other fields, which in this case being finance.
>**IMPORTANT**: The subsequent analytical steps has to follow the fact that the Covariance Matrix $$\boldsymbol{C_{\sigma}}$$ has to be **invertible**, meaning it has to be positive definite, or that all our **real eigen values decomposed has to be all positive (>0)**


In finding the optimal allocations weight $$w$$ of a *Minimum Variance Portfolio*, we seek for the solution of:

>$$\underset{w}{min} (w^\top \cdot \boldsymbol{C_{\sigma}} \cdot w)$$

Subjected to the constraint:
>$$w^T \cdot \boldsymbol{\vec{1}} = \begin{vmatrix} w_1 & w_2 & \cdots & w_M \end{vmatrix} \cdot \begin{vmatrix} 1\\ 1\\ \vdots\\ 1 \end{vmatrix} = \sum_{i=1}^{M}w_i = 1$$

Constructing a Lagrangian, we obtain:

>$$\mathcal{L}(w,\lambda_{w}) = (w^\top \cdot \boldsymbol{C_{\sigma}} \cdot w) + \lambda_w[(w^T \cdot \boldsymbol{\vec{1}}) - 1]$$


Recall we established that $$\boldsymbol{C_{\sigma}}$$ is an **invertible Hermitian Matrix**, following the property of their **eigen values being real**. Under the assumption of invertibility, the quadratic part is positive definite and there exists a unique minimizer (read more on Hermitian Matrices [here](https://mathworld.wolfram.com/HermitianMatrix.html "here")). Thus, in solving for $$w$$,the gradient of such Lagrangian follows:
> $$\nabla \mathcal{L}(w,\lambda_{w}) = \begin{vmatrix} 0_1 &0_2 &\cdots&0_M & 0_{\lambda_w} \end{vmatrix}^T$$

Representing the FOCs, with (M+1) dimensions (with the additional dimension being the constraint represented by the Langrage Multiplier $$\lambda_w$$)

After solving the gradient equation through some algebra (I am dumping this down as much as possible, though for the specifics, check out this [**lecture notes**](https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf "lecture notes")), in addition to the initial constraint, we obtain the equation:

>$$\boldsymbol{C_\sigma}\cdot w = -\lambda_w \boldsymbol{\vec{1}}$$

Proceeding with the assumption that $$\boldsymbol{C_{\sigma}}$$ is invertible, we have a unique solution:

> $$w = -\lambda_w\cdot \boldsymbol{C_{\sigma}^{-1}} \cdot \boldsymbol{\vec{1}} = \frac{-\lambda_w\cdot \boldsymbol{C_{\sigma}^{-1}} \cdot \boldsymbol{\vec{1}}}{1}$$

Substituting the denominator **1** with our initial constraint equation $$w^\top \cdot \boldsymbol{\vec{1}} = 1$$, we obtain:

> $$w = \frac{-\lambda_w\cdot \boldsymbol{C_{\sigma}^{-1}} \cdot \boldsymbol{\vec{1}}}{w^\top \cdot \boldsymbol{\vec{1}}} = \frac{-\lambda_w\cdot \boldsymbol{C_{\sigma}^{-1}} \cdot \boldsymbol{\vec{1}}}{(-\lambda_w\cdot \boldsymbol{C_{\sigma}^{-1}} \cdot \boldsymbol{\vec{1}})^\top \cdot \boldsymbol{\vec{1}}}$$

Thus, canceling the Lagrange Multiplier $$\lambda_w$$, we finally have our solution:
> $$w = \frac{\boldsymbol{C_{\sigma}^{-1}} \cdot \boldsymbol{\vec{1}}}{(\boldsymbol{C_{\sigma}^{-1}} \cdot \boldsymbol{\vec{1}})^\top \cdot \boldsymbol{\vec{1}}}$$

The Lagrangian Method can be further complexified, adding more constraints into the system, with each constraint associated with a Lagrange Multiplier, increasing the dimensionality of the FOCs, then proceed on solving it backwards:
> We can establish an equality constraint(=) on the portfolio's return, then solve for the global minimum variance solution of such return. Iterating over a range of all possible returns and perform the same step will give you the [efficient frontier](https://en.wikipedia.org/wiki/Efficient_frontier "efficient frontier")
For the additional mathematics on adding the returns equality constraint, again, check out this [lecture notes](https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf "lecture notes").

When establishing *inequality constraints(<,>,<=,>=)*, the mathematics can get very arduous to go over, let alone an intricate list of equality constraints (=). This is where computational power comes in handy.

### 1b. Computational Approach
> As there already exist many posts online from different individuals showing methods of optimization utilizing CS algorithms, iterating over different potential solutions *smartly*, given an initial guess, to arrive at the solution needed. Therefore, I will not touch base on this approach in much details. [Here](https://kyle-stahl-mn.com/stock-portfolio-optimization "Here") is an example of such work, specifically on our current topic, if you need more ABC on each step.

Simply put, we will utilize scipy's minimization function in seeking for the solution of our optimization problem, establishing constraints & boundaries for the input $$w$$ until we reach the global minimum. Since the symmetric covariance matrix, assumed invertible, contains quadratic components with a unique minimizer, we opt for the Sequential Least Square Programming (SLSQP) Algorithm for our computational approach.
I encourage learning more on the details of different algorithms for optimization, specifically in knowing the context of the problem constructed (linearity, convexity, etc), as well as, again, the mathematics behind them as much as you can. Scipy offers alot of different optimization functions, as well as algorithms to be implemented when solving; learn about the mathematical optimization [here](http://scipy-lectures.org/advanced/mathematical_optimization/ "here")

---
### 1c. Implementations

Below is our written function that both analytically & computationally, depends on input argument (analytical on default), solves for the global minimum variance weight $$w$$, a unique solution that minimizes portfolio's risk $$\sigma_p$$, requiring only an input of any *M x M* covariance matrix of M securities:

```python
def minVar(covariance_matrix,analytical_method=True):
    if analytical_method:
        inv_cov = np.linalg.pinv(covariance_matrix)
        ones = np.ones(len(inv_cov))
        inv_dot_ones = np.dot(inv_cov, ones)
        return pd.Series(data=(inv_dot_ones/np.dot(inv_dot_ones,ones)),
                         index=covariance_matrix.index)
    #----| Constraint of sum(w_i) = 1
    constraints = ({'type':'eq','fun': lambda allocations:(np.sum(allocations) - 1)})
    targets = list(covariance_matrix.columns)
    #----| Boundary is subjective & yields interesting results, having no boundary ~ all possible leverages?
    bounds = [(None,None) for s in targets]
    init_guess = [1/len(targets) for s in targets]
    assert(len(bounds) == len(targets) == len(init_guess))
    def volatility(allocations):
        return np.sqrt(np.dot(allocations.T,np.dot(covariance_matrix,allocations)))
    def optimize(allocations):
        return (volatility(allocations))
    result = minimize(optimize,
                      x0=init_guess,
                      method='SLSQP',
                      bounds = bounds,
                      constraints = constraints)
    return pd.Series(data=result.x,index=targets)
```

Calculate the Minimum Variance allocations weights for both methods & construct table of weights:

```python
allWeights = pd.DataFrame({
    "Analytical":minVar(RET.cov()),
    "Computational":minVar(RET.cov(),analytical_method = False)})
```

Observe the the weights comparisons between the two:

```python
f = plt.figure(figsize=(16,8))
ax_a = f.add_subplot(121,title="Minimum Variance Weights (Analytical)")
ax_c = f.add_subplot(122,title="Minimum Variance Weights (Computational)")
allWeights["Analytical"].plot(kind="bar",ax=ax_a)
allWeights["Computational"].plot(kind="bar",ax=ax_c)
```

<img src="https://jp-quant.github.io/images/vol_2/a_vs_c_w.png">

The **Minimum Variance Portfolio** tells us to allocate the majority of our capital in to *IEF, TLT & SHY*, all of which are **T-Bills**! We will observe this result more in-depths below in the next section on Eigen Portfolios.


### *Conclusions & Remarks*
- The discrepancies between our analytical & computational approaches are very small. I theorize that this is due to either when approximating the inverse covariance matrix when solving analytically, or computational errors. I will add an explanation once I have empirically identified the problem.
- It is **much faster to use the analytical method** to compute our Minimum Variance Portfolio, especially as M increases to which requires more combinations for iterations when utilizing computational approach.
- We tend to opt for computation when we don't have the concrete mathematics to solve for the optimization problem, or that the constraint is neither linear or quadratic, or even convex (harder to find global minimum/maximum solution).
- The mathematical rigour can get intensive the more we complexify our Lagrangian, so I will add them more if we have the time. For now, it's much easier to use the computational approach for **adding more constraints or changing optimization target**
	> For example, instead of $$\sigma_p$$, maybe we can change it to $$\boldsymbol{I_p}$$, and perform maximization instead of minimization. Perhaps I will demonstrate this as well later, although it is fairly easy to change up the code (check out [**Scipy Optimization Documentation**](https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html "Scipy Optimization Documentation"))

---
## 2. Eigen Portfolios
Recall our brief introduction in the first post, eigen portfolios are simply the eigen vectors "scaled" by the summation of its components. We need to address the following **important** fundamental concepts & empirical findings:

- The idea of an eigen portfolio is simply an **allocation vector** that **aligns itself in the exact direction of the eigen vector**.

- For the eigen vectors $$e_{i} = \begin{vmatrix} e_{i_1}\\ e_{i_2}\\ \vdots\\ e_{i_M} \end{vmatrix}$$, each being a unit vector such that $$\left \| e_i \right \| = 1$$,  when multiplied (scaled) by the summation of its components, we obtain the **eigen portfolios** $$w_{e_i} = \frac{e_i}{e_i \cdot \vec{\boldsymbol{1}}} = \frac{e_i}{\sum_{n=1}^{M}e_{i_n}}$$

- The eigen portfolios $$w_{e_i}$$ **no longer are unit vectors**, meaning that $$\left \| w_{e_i} \right \| \neq 1$$.

- The eigen values $$\lambda_i$$ are interpreted as an **exposure factor** associated with its respective eigen portfolio $$w_{e_i}$$.

- It is *empirically* proven, through other research works, that $$w_{e_1}$$, being the eigen portfolio associated with the eigen vector with the **largest eigen value**, is the **market portfolio**, given the universe of M securities comprising the market.

Modifying our function written to extract the eigen pairs in the [**previous post**](https://jp-quant.github.io/qf_volatility_p1/ "previous post"), below function scaled the eigen vectors with their weights summation into eigen portfolios and return the portfolios' weights along with everything else:

```python
def EIGEN(_ret_):
    _pca = PCA().fit(_ret_)
    eVecs = _pca.components_
    eVals = _pca.explained_variance_
    _eigenNames = ["eigen_"+str(e+1) for e in range(len(eVals))]
    _eigenValues = pd.Series(eVals,index=_eigenNames,name="eigenValues")
    _eigenVectors = pd.DataFrame(eVecs.T,index=_ret_.columns,columns=_eigenNames)
    _eigenPortfolios = pd.DataFrame({e:(_eigenVectors[e]/_eigenVectors[e].sum()) for e in _eigenVectors})
    return {"lambdas":_eigenValues,"vectors":_eigenVectors,"portfolios":_eigenPortfolios}
```

In constructing our *allWeights* table of different allocations weights, to which in this case being all the eigen portfolios, we will go ahead and **add the Minimum Variance Portfolio** into it as well:

```python
_eigen_ = EIGEN(RET)
eigenWeights = _eigen_["portfolios"]
allWeights = pd.concat([eigenWeights,minVar(RET.cov(),True).reindex(eigenWeights.index)],1)
```

Briefly observe the first top 10 eigen values' **explained variances**:


```python
(_eigen_["lambdas"].sort_values(ascending=False)/_eigen_["lambdas"].sum()).head(10)
```




    eigen_1     0.599924
    eigen_2     0.076345
    eigen_3     0.065515
    eigen_4     0.044836
    eigen_5     0.043632
    eigen_6     0.028974
    eigen_7     0.022040
    eigen_8     0.018703
    eigen_9     0.014721
    eigen_10    0.012076
    Name: eigenValues, dtype: float64


Now that we have extracted the weights from **in-sample** data, we proceed on performing evaluations for **BOTH in-sample & out-sample**. Observe plot results for the evaluations of the **Top 5 Eigen Portfolios + Minimum Variance Portfolio**:

```python
to_plot = ["eigen_1","eigen_2","eigen_3","eigen_4","eigen_5","MinVar"]

#----| In-Sample: RET
bulk_evals = evaluate_bulk_w(RET,allWeights)
f = plt.figure(figsize=(9,6))
ax_m = f.add_subplot(121,title="In-Sample Evaluation")
ax_cr = f.add_subplot(122,title="In-Sample Cummulative Returns")
bulk_evals["metrics"].T[to_plot].plot(kind="bar",ax=ax_m)
bulk_evals["cum_ret"][to_plot].plot(ax=ax_cr)

#---| Out-Sample: RET_outSample
bulk_evals = evaluate_bulk_w(RET_outSample,allWeights)
f = plt.figure(figsize=(9,6))
ax_m = f.add_subplot(121,title="Out-Sample Evaluation")
ax_cr = f.add_subplot(122,title="Out-Sample Cummulative Returns")
bulk_evals["metrics"].T[to_plot].plot(kind="bar",ax=ax_m)
bulk_evals["cum_ret"][to_plot].plot(ax=ax_cr)
```

<img src="https://jp-quant.github.io/images/vol_2/mv_vs_e_1.png">
<img src="https://jp-quant.github.io/images/vol_2/mv_vs_e_2.png">

Recall that we are working with the sector **Others**, containing **35 securities** representing **a diverse pool of different funds & indexes**, comprising the entire "market" as much as we can, with components as independent they can, ranging from different asset classes (stocks, commodities, T-bills, etc) as well as domestic & foreign securities (refer to the table above in our Preparation section for information).

The interesting thing to point out is the **out-sample result**, the timeframe to which when the latest *COVID-19 market crash* occured whereas only the **eigen_3** and **MinVar** portfolios exhibit positive performance, with **eigen_3** also having the highest sharpe ratio for **both** samples.

If we plot out the weights for **eigen_1, eigen_2 and MinVar**, we obtain:
<img src="https://jp-quant.github.io/images/vol_2/e_123.png">

Notice how the **eigen_1** weights **have the majority of positive allocations in indexes affiliated with stocks!**. This confirms our statement above on it being the **market portfolio**. Notice how for *GLD & GDX*, being commodities, specifically gold, hedged out in weights, while indexes of foreign securities like *EWJ* have very little weights in comparison to the others; lastly, all our negative allocations, or short positions, are in Tresury Bills (IEF, SHY, TLT), all statistically exhibit negative correlation with the market!

[MORE ANALYSIS CONTENT IN PROGRESS]


---
# Random Matrix Theory
[CONTENT IN-PROGRESS]

http://web.eecs.umich.edu/~rajnrao/Acta05rmt.pdf


```python
def _RMT_(RET,Q=None,sigma=None,include_plots=False,
                          exclude_plotting_outliers=True,auto_scale=True):
    #----| Part 1: Marchenko-Pastur Theoretical eVals
    T,N = RET.shape
    Q = Q or (T/N) #---| optimizable
    sigma = sigma or 1 #---| optimizable
    min_theoretical_eval = np.power(sigma*(1 - np.sqrt(1/Q)),2)
    max_theoretical_eval = np.power(sigma*(1 + np.sqrt(1/Q)),2)
    theoretical_eval_linspace = np.linspace(min_theoretical_eval,max_theoretical_eval,500)
    def marchenko_pastur_pdf(x,sigma,Q):
        y=1/Q
        b=np.power(sigma*(1 + np.sqrt(1/Q)),2) # Largest eigenvalue
        a=np.power(sigma*(1 - np.sqrt(1/Q)),2) # Smallest eigenvalue
        return (1/(2*np.pi*sigma*sigma*x*y))*np.sqrt((b-x)*(x-a))*(0 if (x > b or x <a ) else 1)
    pdf = np.vectorize(lambda x:marchenko_pastur_pdf(x,sigma,Q))
    eVal_density = pdf(theoretical_eval_linspace)
    
    #-----| Part 2a: Calculate Actual eVals from Correlation Matrix & Construct Filtered eVals
    corr = RET.corr()
    eVals,eVecs = np.linalg.eigh(corr.values)
    noise_eVals = eVals[eVals <= max_theoretical_eval]
    outlier_eVals = eVals[eVals > max_theoretical_eval]
    filtered_eVals = eVals.copy()
    filtered_eVals[filtered_eVals <= max_theoretical_eval] = 0
    #-----| Part 2b: Construct Filtered Correlation Matrix from Filtered eVals
    filtered_corr = np.dot(eVecs,np.dot(
                        np.diag(filtered_eVals),np.transpose(eVecs)
                                        ))
    np.fill_diagonal(filtered_corr,1)
    #----| Part 2c: Construct Filtered Covariance Matrix from Filtered Correlation Matrix
    cov = RET.cov()
    standard_deviations = np.sqrt(np.diag(cov.values))
    filtered_cov = (np.dot(np.diag(standard_deviations), 
                        np.dot(filtered_corr,np.diag(standard_deviations))))
    result = {
        "raw_cov":cov,"raw_corr":corr,
        "filtered_cov":pd.DataFrame(data=filtered_cov,index=cov.index,columns=cov.columns),
        "filtered_corr":pd.DataFrame(data=filtered_corr,index=corr.index,columns=corr.columns),
        "Q":Q,"sigma":sigma,
        "min_theoretical_eval":min_theoretical_eval,
        "max_theoretical_eval":max_theoretical_eval,
        "noise_eVals":noise_eVals,
        "outlier_eVals":outlier_eVals,
        "eVals":eVals,"eVecs":eVecs
    }
    if include_plots:
        #----| Plot A = Eigen Densities Comparison of Actual vs Theoretical
        MP_ax = plt.figure().add_subplot(111)
        eVals_toPlot = eVals.copy() if (not exclude_plotting_outliers) else eVals[eVals <= max_theoretical_eval+1]
        MP_ax.hist(eVals_toPlot, density = True, bins=100)
        MP_ax.set_autoscale_on(True)
        MP_ax.plot(theoretical_eval_linspace,eVal_density, linewidth=2, color = 'r')
        MP_ax.set_title(("Q = "+ str(Q) + " | sigma = " + str(sigma)))
        result["MP_ax"] = MP_ax
        
        #----| Plot B = Original vs Filtered Correlation Matrix (from filtered eVals)
        f = plt.figure()
        FILTERED_ax = f.add_subplot(121,title="Original")
        FILTERED_ax.imshow(corr)
        FILTERED_ax = f.add_subplot(122,title="Filtered")
        a = FILTERED_ax.imshow(filtered_corr)
        cbar = f.colorbar(a, ticks=[-1, 0, 1])
        result["FILTERED_ax"] = FILTERED_ax
    return result
```
<img src="https://jp-quant.github.io/images/vol_2/rmt_0.png">
<img src="https://jp-quant.github.io/images/vol_2/rmt_1.png">
<img src="https://jp-quant.github.io/images/vol_2/rmt_2.png">