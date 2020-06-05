---
title: "[QF] Probing Volatility II: Portfolio Optimization & Random Matrix Theory"
date: 2020-06-03
tags: [research]
header:
  image: "/images/qf_vol_part1_banner.PNG"
excerpt: "Exploring portfolio management by finding minimum/maximum solutions of a constructed system of equations & constraints, specifically for optimization purposes, analytically & computationally, incorporating quantitative techniques though Random Matrix Theory to filter noises & tackling higher dimensions."
mathjax: "true"
---

# Overview & Preparation
>*Optimization*, in general, is simply solving for minimum/maximum solution(s) of a *system of equation(s)*, satisfying given constraints.

Recalling our brief overview on the basics of Modern Portfolio Theory in my [first QF post](https://jp-quant.github.io/qf_intro/ "first QF post"), one of the metric to which used in evaluating performance of any investment, mirroring somewhat of a standardization technique, is the **Sharpe Ratio**, to which we will dub it as <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{I}" title="\boldsymbol{I}" />:

><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\boldsymbol{I}&space;=&space;\frac{\bar{R_p}&space;-&space;R_f}{\sigma_p}" title="\boldsymbol{I} = \frac{\bar{R_p} - R_f}{\sigma_p}" />, where in the perspective of evaluating a portfolio,
<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bar{R_p}" title="\bar{R_p}" /> = Average Returns
<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\sigma_p" title="\sigma_p" /> = Volatility = Standard Deviation of Returns

In general, we want to **minimize** $$\sigma_p$$ & **maximize** $$\bar{R_p}$$, simply seeking to achieve not just as *highest* & as *consistent* of a returns rate as possible, but also having as little as possible in its maximum drawdown.
>In regards to selecting the optimal assets to pick in a portfolio, this is an extensive topic requiring much empirical & fundamental research with additional knowledge on top of our current topic at hand (we will save this for another section).

Let there be *M* amount of securities selected to invest a set amount of capital in, we seek for the optimal allocations, or weights, <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;w&space;=&space;\begin{vmatrix}&space;w_1\\&space;w_2\\&space;\vdots\\&space;w_M&space;\end{vmatrix}" title="w = \begin{vmatrix} w_1\\ w_2\\ \vdots\\ w_M \end{vmatrix}" />  such that <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;-1&space;\leq&space;w_{i}\leq&space;1" title="-1 \leq w_{i}\leq 1" /> and <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\sum_{1}^{M}w_i&space;=&space;1" title="\sum_{1}^{M}w_i = 1" />, being 100% of our capital. 

>-  If you want **long only** positions, simply set -- <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;0&space;\leq&space;w_{i}\leq&space;1" title="0 \leq w_{i}\leq 1" /> instead.
- We are allowing short positions in all of our approaches since we want absolute returns & hedging opportunities, coming from a perspective of an institutional investor, hence each weight can be negative.

---

Before moving forward, we first need to address the context regarding *M securities* we seek to allocate our capital towards:
- At **this** moment in time $$t_N$$, as we are performing analysis to make decisions, being the latest timestamp (this is where back-testing will come in, as I will dedicate a series on building an event-driven one from scratch), we have *N* amount of data historically for *M* securities, hence the necessity for an *N x M* returns table.
- **Looking towards the future**  $$t_{N + dN}$$ , before we seek to find the optimal weights $$w$$, to compute $$\sigma_p$$ & $$\bar{R_p}$$ (again please refer to my [first QF post](https://jp-quant.github.io/qf_intro/ "first QF post") for the intricate details), as we will not yet be touching base on such extensive topic that having to deal with prediction (I will dedicate another series for this topic), we need to determine the answers for:

	- What is the returns <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\hat{r_i}" title="" /> for each security <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;i&space;=&space;1,2...M" title="i = 1,2...M" />?
		>In our demonstrative work, as being used as a factor in various predictive models, we will set <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\hat{r_i}&space;=&space;\bar{r_i}" title="\hat{r_i} = \bar{r_i}" /> being the **average returns "so far"** (subjective)

	- How much "uncertainty," or in other words, how much will  <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\hat{r_i}" title="\hat{r_i}" />  deviate, or simply <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\sigma_{\hat{r_i}}" title="\sigma_{\hat{r_i}}" />?
		>Given we have set <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\hat{r_i}&space;=&space;\bar{r_i}" title="\hat{r_i} = \bar{r_i}" /> , this will simply be the **standard deviations** of  <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bar{r_i}" title="\bar{r_i}" />. Thus, we set <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\sigma_{\hat{r_i}}&space;=&space;\sigma_{\bar{r_i}}" title="\sigma_{\hat{r_i}} = \sigma_{\bar{r_i}}" />

---
Unlike our [**previous post**](https://jp-quant.github.io/qf_volatility_p1/ " previous post") in this series on volatility, this time, we will start with a *N x M*  **daily close** $$RET$$ table of ~1,000 securities in the U.S. Equities market,  dating back approximately **2 years worth of data** since May 1st 2020, or 500-600 data points available.

Before we proceed, it is important to note that we have over 1,000 securities available. This is dimensionally lacking in data as for our *N x M* returns table $$RET$$ (N < M). It is somewhat similar, although not completely accurate, to saying having 1-2 points when working with 3 assets. We will slightly touch base on this problem in later posts. Although, for now, when working with picking M assets,  we seek for **M < N** as much as we can, that being either:
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
_RET_
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
      <th>AMG</th>
      <th>MSI</th>
      <th>FBHS</th>
      <th>AMD</th>
      <th>EVR</th>
      <th>NKE</th>
      <th>NRG</th>
      <th>EV</th>
      <th>VRSN</th>
      <th>SNPS</th>
      <th>...</th>
      <th>FND</th>
      <th>CVNA</th>
      <th>AM</th>
      <th>IR</th>
      <th>JHG</th>
      <th>ATUS</th>
      <th>JBGS</th>
      <th>BHF</th>
      <th>ROKU</th>
      <th>SWCH</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10/9/2017</th>
      <td>-0.002572</td>
      <td>-0.003809</td>
      <td>0.000606</td>
      <td>0.017978</td>
      <td>-0.015949</td>
      <td>-0.017318</td>
      <td>0.009760</td>
      <td>-0.003002</td>
      <td>-0.001366</td>
      <td>0.002785</td>
      <td>...</td>
      <td>0.007816</td>
      <td>-0.029932</td>
      <td>0.002488</td>
      <td>0.006744</td>
      <td>0.000287</td>
      <td>-0.025749</td>
      <td>-0.007929</td>
      <td>-0.014859</td>
      <td>0.056152</td>
      <td>-0.091909</td>
    </tr>
    <tr>
      <th>10/10/2017</th>
      <td>0.005651</td>
      <td>-0.000224</td>
      <td>-0.005774</td>
      <td>0.016931</td>
      <td>-0.003221</td>
      <td>0.000194</td>
      <td>-0.001944</td>
      <td>0.006392</td>
      <td>-0.015146</td>
      <td>-0.000363</td>
      <td>...</td>
      <td>-0.002688</td>
      <td>-0.063414</td>
      <td>0.016757</td>
      <td>0.004471</td>
      <td>0.006016</td>
      <td>-0.017418</td>
      <td>0.006713</td>
      <td>-0.000333</td>
      <td>-0.067858</td>
      <td>0.048765</td>
    </tr>
    <tr>
      <th>10/11/2017</th>
      <td>-0.003387</td>
      <td>0.002019</td>
      <td>0.006683</td>
      <td>0.013053</td>
      <td>-0.012987</td>
      <td>-0.009750</td>
      <td>-0.008601</td>
      <td>0.002387</td>
      <td>0.001571</td>
      <td>0.002175</td>
      <td>...</td>
      <td>0.004030</td>
      <td>0.024868</td>
      <td>0.000000</td>
      <td>-0.001488</td>
      <td>-0.002001</td>
      <td>-0.033837</td>
      <td>0.000912</td>
      <td>0.009439</td>
      <td>0.031340</td>
      <td>-0.045090</td>
    </tr>
    <tr>
      <th>10/12/2017</th>
      <td>-0.004120</td>
      <td>0.006032</td>
      <td>0.002872</td>
      <td>0.022793</td>
      <td>-0.005242</td>
      <td>-0.003927</td>
      <td>0.008211</td>
      <td>0.004559</td>
      <td>-0.002682</td>
      <td>0.022435</td>
      <td>...</td>
      <td>-0.001341</td>
      <td>-0.016986</td>
      <td>-0.009330</td>
      <td>0.001116</td>
      <td>0.007130</td>
      <td>-0.027839</td>
      <td>0.000304</td>
      <td>-0.003964</td>
      <td>-0.001269</td>
      <td>0.032485</td>
    </tr>
    <tr>
      <th>10/13/2017</th>
      <td>-0.001084</td>
      <td>0.001447</td>
      <td>-0.000151</td>
      <td>0.001407</td>
      <td>-0.001315</td>
      <td>0.002947</td>
      <td>0.014689</td>
      <td>0.009839</td>
      <td>0.002035</td>
      <td>-0.001535</td>
      <td>...</td>
      <td>-0.010525</td>
      <td>0.000714</td>
      <td>-0.008920</td>
      <td>0.002600</td>
      <td>0.003121</td>
      <td>-0.022925</td>
      <td>-0.005177</td>
      <td>-0.000497</td>
      <td>-0.026154</td>
      <td>0.000507</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4/27/2020</th>
      <td>0.101339</td>
      <td>0.010676</td>
      <td>0.068505</td>
      <td>0.005503</td>
      <td>0.065444</td>
      <td>0.011253</td>
      <td>0.032592</td>
      <td>0.041703</td>
      <td>0.009851</td>
      <td>0.007227</td>
      <td>...</td>
      <td>0.065038</td>
      <td>-0.021146</td>
      <td>0.020834</td>
      <td>0.033578</td>
      <td>0.061943</td>
      <td>0.012889</td>
      <td>0.031936</td>
      <td>0.055195</td>
      <td>0.029987</td>
      <td>0.043699</td>
    </tr>
    <tr>
      <th>4/28/2020</th>
      <td>0.020766</td>
      <td>-0.060111</td>
      <td>0.058889</td>
      <td>-0.017500</td>
      <td>0.012124</td>
      <td>-0.006398</td>
      <td>0.000890</td>
      <td>0.012772</td>
      <td>-0.014860</td>
      <td>-0.015100</td>
      <td>...</td>
      <td>0.024220</td>
      <td>-0.084890</td>
      <td>-0.046422</td>
      <td>0.017230</td>
      <td>0.020241</td>
      <td>-0.015946</td>
      <td>0.018743</td>
      <td>0.068626</td>
      <td>-0.074796</td>
      <td>-0.053236</td>
    </tr>
    <tr>
      <th>4/29/2020</th>
      <td>0.031928</td>
      <td>0.023714</td>
      <td>0.036964</td>
      <td>-0.033895</td>
      <td>0.044477</td>
      <td>-0.008255</td>
      <td>0.027215</td>
      <td>0.046558</td>
      <td>0.001515</td>
      <td>0.039897</td>
      <td>...</td>
      <td>0.074486</td>
      <td>0.022998</td>
      <td>-0.028479</td>
      <td>0.035244</td>
      <td>0.005620</td>
      <td>0.008764</td>
      <td>0.034730</td>
      <td>0.072330</td>
      <td>0.001340</td>
      <td>-0.010768</td>
    </tr>
    <tr>
      <th>4/30/2020</th>
      <td>-0.005275</td>
      <td>-0.034916</td>
      <td>-0.066805</td>
      <td>-0.023952</td>
      <td>-0.041566</td>
      <td>-0.010157</td>
      <td>-0.032569</td>
      <td>-0.012187</td>
      <td>-0.009029</td>
      <td>-0.005648</td>
      <td>...</td>
      <td>-0.008689</td>
      <td>-0.057956</td>
      <td>0.054067</td>
      <td>-0.041757</td>
      <td>0.108469</td>
      <td>-0.014906</td>
      <td>-0.018097</td>
      <td>-0.057439</td>
      <td>0.014206</td>
      <td>-0.021890</td>
    </tr>
    <tr>
      <th>5/1/2020</th>
      <td>-0.073998</td>
      <td>-0.035243</td>
      <td>0.066223</td>
      <td>-0.049096</td>
      <td>-0.032900</td>
      <td>-0.018991</td>
      <td>-0.016843</td>
      <td>-0.070259</td>
      <td>-0.021568</td>
      <td>-0.048777</td>
      <td>...</td>
      <td>-0.011147</td>
      <td>-0.052667</td>
      <td>0.002103</td>
      <td>-0.033214</td>
      <td>-0.020317</td>
      <td>-0.038466</td>
      <td>-0.028078</td>
      <td>-0.039672</td>
      <td>-0.061316</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>635 rows × 984 columns</p>
</div>




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
<p>984 rows × 2 columns</p>
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

Since we are going to perform evaluations on multiple weights and compare them to each other, we also wrote a bulk evaluation function that takes in an *M x W* matrix, as *W* different weights of *M* securities, and an *N x M* returns table with N returns of  such M securities:

```python
def evaluate_bulk_w(_RET_,all_w):
    RESULTS = {w:evaluate_w(_RET_,all_w[w]) for w in all_w.columns}
    all_metrics = pd.DataFrame({w:RESULTS[w]["metrics"] for w in RESULTS})
    all_cumret = pd.DataFrame({w:RESULTS[w]["cummulative_returns"] for w in RESULTS})
    return {"metrics":all_metrics.T,"cum_ret":all_cumret}
```


In addition, to further evaluate the performances of our extract allocations weights, we will need to split our $$RET$$ data into in-sample & out-sample. Performing calculations to find such weights on the in-sample, then using the weights to apply on the out-sample. This is equivalent to saying:
> If we use the "optimal" weights, calculated from in-sample data, being the latest possible date and invest at that date, **without any predictive features**, how well will such weight perform?

```python
def ioSampleSplit(df,inSample_pct=0.75):
    inSample = df.head(int(len(df)*inSample_pct))
    return (inSample,df.reindex([i for i in df.index if i not in inSample.index]))
```

---
# Mean-Variance Optimization
Given a selected M amount of securities, we obtain our Symmetric *M x M* Covariance Matrix (<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\boldsymbol{C_{\sigma}}" title="\boldsymbol{C_{\sigma}}" />), calculated from an *N x M* $$RET$$ returns table of *N* returns data of such M securities (again, details in [first QF post](https://jp-quant.github.io/qf_intro/ "first QF post") ), a portfolio's volatility (risk) is calculated as:

><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\sigma_p&space;=&space;\sqrt{w^\top&space;\cdot&space;\boldsymbol{C_{\sigma}}&space;\cdot&space;w}" title="\sigma_p = \sqrt{w^\top \cdot \boldsymbol{C_{\sigma}} \cdot w}" />

Thus, our objective is to **find an allocation weight** <img src="https://latex.codecogs.com/gif.latex?w&space;=&space;\begin{vmatrix}&space;w_1\\&space;w_2\\&space;\vdots\\&space;w_M&space;\end{vmatrix}" title="w = \begin{vmatrix} w_1\\ w_2\\ \vdots\\ w_M \end{vmatrix}" />  that would *minimize* <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\sigma_p" title="\sigma_p" />, or simply put will result in the **smallest possible** <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\sigma_p" title="\sigma_p" />, such that <img src="https://latex.codecogs.com/gif.latex?-1&space;\leq&space;w_{i}\leq&space;1" title="-1 \leq w_{i}\leq 1" /> and <img src="https://latex.codecogs.com/gif.latex?\sum_{1}^{M}w_i&space;=&space;1" title="\sum_{1}^{M}w_i = 1" /> 

We're just going to select our M securities as the *Utilities* sector for our demonstrative work, since it has the least amount to encompasses such sector. In addition, we split our full $$RET$$ data into 75% in-sample & 25% out-sample, letting the default variable **RET** as in-sample & **RET_outSample** as, of course, the out-sample data. Subsequent optimization steps in solving for the desired allocations weight $$w$$ will be performed on in-sample data:

```python
RET,RET_outSample = ioSampleSplit(_RET_[sectors["Utilities"]])
```

```python
RET.shape, RET_outSample.shape
```




    ((476, 37), (159, 37))


Like before, we will use pandas covariance function to calculated its sample covariance matrix:


```python
RET.cov()
```

---
## 1. Minimizing Risk
>Commonly known as the **Minimum Variance Portfolio**, minimizing portfolio's risk serves as the first step for the overaching topic of Portfolio Optimization.

### 1a. Analytical Solutions
>**Personal Commentary**: It is important to understand the mathematics behind methods of solving optimization problems, especially aiding in knowing the context of the problem and how solutions exist within a certain boundary, the convexity or linearity, as well as grasping the analytical abstraction drawn to describe any system. There might exist patterns within our analytical steps, to which could allow us to find out some invariances that are even more powerful than solving for such solution.

Going back to my earlier remark above on solving optimization problems, we can simply construct this problem with constraints utilizing the [Lagrangian method](https://scholar.harvard.edu/files/david-morin/files/cmchap6.pdf "Lagrangian method"). My personal exposure to such mathematical technique stemmed from my Physics background, specifically on the topic of [Lagrangian Mechanics](https://en.wikipedia.org/wiki/Lagrangian_mechanics); such modeling technique can be applied to any other system optimization problems in other fields, which in this case being finance.
>**IMPORTANT**: The subsequent analytical steps has to follow the fact that the Covariance Matrix <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\boldsymbol{C_{\sigma}}" title="\boldsymbol{C_{\sigma}}" /> has to be **invertible**, meaning it has to be positive definite, or that all our **real eigen values decomposed has to be all positive (>0)**


In finding the optimal allocations weight $$w$$ of a *Minimum Variance Portfolio*, we seek for the solution of:

><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\underset{w}{min}&space;(w^\top&space;\cdot&space;\boldsymbol{C_{\sigma}}&space;\cdot&space;w)" title="\underset{w}{min} (w^\top \cdot \boldsymbol{C_{\sigma}} \cdot w)" />

Subjected to the constraint:
><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;w^T&space;\cdot&space;\boldsymbol{\vec{1}}&space;=&space;\begin{vmatrix}&space;w_1&space;&&space;w_2&space;&&space;\cdots&space;&&space;w_M&space;\end{vmatrix}&space;\cdot&space;\begin{vmatrix}&space;1\\&space;1\\&space;\vdots\\&space;1&space;\end{vmatrix}&space;=&space;\sum_{i=1}^{M}w_i&space;=&space;1" title="w^T \cdot \boldsymbol{\vec{1}} = \begin{vmatrix} w_1 & w_2 & \cdots & w_M \end{vmatrix} \cdot \begin{vmatrix} 1\\ 1\\ \vdots\\ 1 \end{vmatrix} = \sum_{i=1}^{M}w_i = 1" />

Constructing a Lagrangian, we obtain:

><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\mathcal{L}(w,\lambda_{w})&space;=&space;(w^\top&space;\cdot&space;\boldsymbol{C_{\sigma}}&space;\cdot&space;w)&space;&plus;&space;\lambda_w[(w^T&space;\cdot&space;\boldsymbol{\vec{1}})&space;-&space;1]" title="\mathcal{L}(w,\lambda_{w}) = (w^\top \cdot \boldsymbol{C_{\sigma}} \cdot w) + \lambda_w[(w^T \cdot \boldsymbol{\vec{1}}) - 1]" />


Recall we established that <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\boldsymbol{C_{\sigma}}" title="\boldsymbol{C_{\sigma}}" /> is an **invertible Hermitian Matrix**, following the property of their **eigen values being real**. Under the assumption of invertibility, the quadratic part is positive definite and there exists a unique minimizer (read more on Hermitian Matrices [here](https://mathworld.wolfram.com/HermitianMatrix.html "here")). Thus, in solving for $$w$$,the gradient of such Lagrangian follows:
> <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\nabla&space;\mathcal{L}(w,\lambda_{w})&space;=&space;\begin{vmatrix}&space;0_1&space;&0_2&space;&\cdots&0_M&space;&&space;0_{\lambda_w}&space;\end{vmatrix}^T" title="\nabla \mathcal{L}(w,\lambda_{w}) = \begin{vmatrix} 0_1 &0_2 &\cdots&0_M & 0_{\lambda_w} \end{vmatrix}^T" />

Representing the FOCs, with (M+1) dimensions (with the additional dimension being the constraint represented by the Langrage Multiplier $$\lambda_w$$)

After solving the gradient equation through some algebra (I am dumping this down as much as possible, though for the specifics, check out this [**lecture notes**](https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf "lecture notes")), in addition to the initial constraint, we obtain the equation:

><img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\boldsymbol{C_\sigma}\cdot&space;w&space;=&space;-\lambda_w&space;\boldsymbol{\vec{1}}" title="\boldsymbol{C_\sigma}\cdot w = -\lambda_w \boldsymbol{\vec{1}}" />

Proceeding with the assumption that <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\boldsymbol{C_{\sigma}}" title="\boldsymbol{C_{\sigma}}" /> is invertible, we have a unique solution:

> <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;w&space;=&space;-\lambda_w\cdot&space;\boldsymbol{C_{\sigma}^{-1}}&space;\cdot&space;\boldsymbol{\vec{1}}&space;=&space;\frac{-\lambda_w\cdot&space;\boldsymbol{C_{\sigma}^{-1}}&space;\cdot&space;\boldsymbol{\vec{1}}}{1}" title="w = -\lambda_w\cdot \boldsymbol{C_{\sigma}^{-1}} \cdot \boldsymbol{\vec{1}} = \frac{-\lambda_w\cdot \boldsymbol{C_{\sigma}^{-1}} \cdot \boldsymbol{\vec{1}}}{1}" />

Substituting the denominator **1** with our initial constraint equation <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;w^\top&space;\cdot&space;\boldsymbol{\vec{1}}&space;=&space;1" title="w^\top \cdot \boldsymbol{\vec{1}} = 1" />, we obtain:

> <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;w&space;=&space;\frac{-\lambda_w\cdot&space;\boldsymbol{C_{\sigma}^{-1}}&space;\cdot&space;\boldsymbol{\vec{1}}}{w^\top&space;\cdot&space;\boldsymbol{\vec{1}}}&space;=&space;\frac{-\lambda_w\cdot&space;\boldsymbol{C_{\sigma}^{-1}}&space;\cdot&space;\boldsymbol{\vec{1}}}{(-\lambda_w\cdot&space;\boldsymbol{C_{\sigma}^{-1}}&space;\cdot&space;\boldsymbol{\vec{1}})^\top&space;\cdot&space;\boldsymbol{\vec{1}}}" title="w = \frac{-\lambda_w\cdot \boldsymbol{C_{\sigma}^{-1}} \cdot \boldsymbol{\vec{1}}}{w^\top \cdot \boldsymbol{\vec{1}}} = \frac{-\lambda_w\cdot \boldsymbol{C_{\sigma}^{-1}} \cdot \boldsymbol{\vec{1}}}{(-\lambda_w\cdot \boldsymbol{C_{\sigma}^{-1}} \cdot \boldsymbol{\vec{1}})^\top \cdot \boldsymbol{\vec{1}}}" />

Thus, canceling the Lagrange Multiplier $$\lambda_w$$, we finally have our solution:
> <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;w&space;=&space;\frac{\boldsymbol{C_{\sigma}^{-1}}&space;\cdot&space;\boldsymbol{\vec{1}}}{(\boldsymbol{C_{\sigma}^{-1}}&space;\cdot&space;\boldsymbol{\vec{1}})^\top&space;\cdot&space;\boldsymbol{\vec{1}}}" title="w = \frac{\boldsymbol{C_{\sigma}^{-1}} \cdot \boldsymbol{\vec{1}}}{(\boldsymbol{C_{\sigma}^{-1}} \cdot \boldsymbol{\vec{1}})^\top \cdot \boldsymbol{\vec{1}}}" />

The Lagrangian Method can be further complexified, adding more constraints into the system, with each constraint associated with a Lagrange Multiplier, increasing the dimensionality of the FOCs, then proceed on solving it backwards:
> We can establish an equality constraint(=) on the portfolio's return, then solve for the global minimum variance solution of such return. Iterating over a range of all possible returns and perform the same step will give you the [efficient frontier](https://en.wikipedia.org/wiki/Efficient_frontier "efficient frontier")
For the additional mathematics on adding the returns equality constraint, again, check out this [lecture notes](https://faculty.washington.edu/ezivot/econ424/portfolioTheoryMatrix.pdf "lecture notes").

When establishing *inequality constraints(<,>,<=,>=)*, the mathematics can get very arduous to go over, let alone an intricate list of equality constraints (=). This is where computational power comes in.

### 1b. Computational Approach
> As there already exist many posts online from different individuals showing methods of optimization utilizing CS algorithms, iterating over different guesses to arrive at the solution needed. Therefore, I will not touch base on this approach in much details. [Here](https://kyle-stahl-mn.com/stock-portfolio-optimization "Here") is an example of such work, specifically on our current topic, if you need more details on each step.

Simply put, we will utilize scipy's minimization function in seeking for the solution of our optimization problem, establishing constraints & boundaries for the input $$w$$ until we reach the global minimum. Since the symmetric covariance matrix contains quadratic components with a unique minimizer, we opt for the Sequential Least Square Programming (SLSQP) Algorithm for our computational approach.
I encourage learning more on the details of different algorithms for optimization, specifically in knowing the context of the problem constructed (linearity, convexity, etc), as well as, again, the mathematics behind them as much as you can. Scipy offers alot of different optimization functions, as well as algorithms to be implemented when solving; learn about the mathematical optimization [here](http://scipy-lectures.org/advanced/mathematical_optimization/ "here")

---
### 1c. Implementations & Comparing Two Approahces

Below is our written function that both analytically & computationally, depends on input argument, solves for the global minimum variance weight $$w$$, a unique solution that minimizes portfolio's risk (<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\sigma_p" title="\sigma_p" />), requiring only an input of any *M x M* covariance matrix of M securities:

```python
def minVar(covariance_matrix,analytical_method=False):
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
    "Analytical":minVar(RET.cov(),True),
    "Computational":minVar(RET.cov())
})
```

Observe the the weights comparisons between the two:

```python
f = plt.figure(figsize=(16,4))
ax_a = f.add_subplot(121,title="Minimum Variance Weights (Analytical)")
ax_c = f.add_subplot(122,title="Minimum Variance Weights (Computational)")
allWeights["Analytical"].plot(kind="bar",ax=ax_a)
allWeights["Computational"].plot(kind="bar",ax=ax_c)
```

<img src="https://jp-quant.github.io/images/vol_2/a_vs_c_w.png" alt="1" border="0">

We proceed on first observing the **in-sample** evaluation results between Analytical & Computational solutions, seeing how much they do line up with each other:

```python
bulk_evals = evaluate_bulk_w(RET,allWeights)
f = plt.figure(figsize=(16,12))
ax_m = f.add_subplot(121,title="In-Sample Evaluation")
ax_cr = f.add_subplot(122,title="In-Sample Cummulative Returns")
bulk_evals["metrics"].T.plot(kind="bar",ax=ax_m)
bulk_evals["cum_ret"].plot(ax=ax_cr)
```
<img src="https://jp-quant.github.io/images/vol_2/a_vs_c_1.png" alt="1" border="0">

With the weights extracted from our in-sample, do the same thing for **out-sample**:

```python
bulk_evals = evaluate_bulk_w(RET_outSample,allWeights)
f = plt.figure(figsize=(16,12))
ax_m = f.add_subplot(121,title="Out-Sample Evaluation")
ax_cr = f.add_subplot(122,title="Out-Sample Cummulative Returns")
bulk_evals["metrics"].T.plot(kind="bar",ax=ax_m)
bulk_evals["cum_ret"].plot(ax=ax_cr)
```

<img src="https://jp-quant.github.io/images/vol_2/a_vs_c_2.png" alt="1" border="0">

> **REMARK**: There are some discrepancies between our analytical & computational approaches, if anything a very small one. I theorize that this is due to either when approximating the inverse covariance matrix when solving analytically,or computational iteration approximations. I will add an explanation once I have empirically identified the problem.



---
### 2. Eigen Portfolios
Recall our brief mention in the first post, eigen portfolios are simply the eigen vectors normalized such that the weights follow capital allocation summation <img src="https://latex.codecogs.com/gif.latex?\sum_{1}^{M}w_i&space;=&space;1" title="\sum_{1}^{M}w_i = 1" />.
>Since each eigen value associated with each eigen vector represents the independent directional variance, that together describes the covariance of M securities, it is **empirically proven** that:
- The **largest eigen portfolio** associating with the **largest eigen value**, representing a portfolio with the **highest variance**, is referred to as the **market portfolio** when M securities are picked as securities that comprises the market.

>From a **historical standpoint** (this is important to keep in mind in regards to Random Matrix Theory), we **pose the question**:
- Would the **last eigen portfolio**, being one associated with the **least eigen value**, represent a portfolio with the **lowest variance**?

Modifying our function written to extract the eigen pairs in the [**previous post**](https://jp-quant.github.io/qf_volatility_p1/ " previous post"), below function normalizes the eigen vectors with their weights summation into eigen portfolios and return the portfolios' weights:
```python
def eigen_w(_ret_):
    _pca = PCA().fit(_ret_)
    eVecs = _pca.components_
    eVals = _pca.explained_variance_
    _eigenNames = ["eigen_"+str(e+1) for e in range(len(eVals))]
    _eigenValues = pd.Series(eVals,index=_eigenNames,name="eigenValues")
    _eigenVectors = pd.DataFrame(eVecs.T,index=ret.columns,columns=_eigenNames)
    return pd.DataFrame({e:(_eigenVectors[e]/_eigenVectors[e].sum()) for e in _eigenVectors})
```

In constructing our *allWeights* table of different allocations, to which in this case being all the eigen portfolios' weights, we will go ahead and **add the Minimum Variance Portfolio Weights** into it as well:

```python
eigenWeights = eigen_w(RET)
allWeights = pd.concat([eigenWeights,minVar(RET.cov()).reindex(eigenWeights.index)],1)
```

Perform evaluations of the weights on **in-sample** data & plot the results of the **Top 5 with the LEAST Volatility**, we obtain:

```python
bulk_evals = evaluate_bulk_w(RET,allWeights)
to_plot = list(bulk_evals["metrics"].sort_values("std").index[:5])

f = plt.figure(figsize=(16,12))
ax_m = f.add_subplot(121,title="In-Sample Evaluation")
ax_cr = f.add_subplot(122,title="In-Sample Cummulative Returns")
bulk_evals["metrics"].T[to_plot].plot(kind="bar",ax=ax_m)
bulk_evals["cum_ret"][to_plot].plot(ax=ax_cr)
```

<img src="https://jp-quant.github.io/images/vol_2/mv_vs_e_1.png" alt="1" border="0">


Now, if we evaluate the weights on **out-sample** data, again plotting the **Top 5 with the LEAST Volatility**:

```python
bulk_evals = evaluate_bulk_w(RET_outSample,allWeights)
to_plot = list(bulk_evals["metrics"].sort_values("std").index[:5])

f = plt.figure(figsize=(16,12))
ax_m = f.add_subplot(121,title="Out-Sample Evaluation")
ax_cr = f.add_subplot(122,title="Out-Sample Cummulative Returns")
bulk_evals["metrics"].T[to_plot].plot(kind="bar",ax=ax_m)
bulk_evals["cum_ret"][to_plot].plot(ax=ax_cr)
```

<img src="https://jp-quant.github.io/images/vol_2/mv_vs_e_2.png" alt="1" border="0">


[ANALYSIS IN-PROGRESS]

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
    filtered_eVals[filtered_eVals <= max_theoretical_eval] = 0 #---| if u dont filter nothing changes...
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
<img src="https://jp-quant.github.io/images/vol_2/rmt_0.png" alt="1" border="0">
<img src="https://jp-quant.github.io/images/vol_2/rmt_1.png" alt="1" border="0">
<img src="https://jp-quant.github.io/images/vol_2/rmt_2.png" alt="1" border="0">