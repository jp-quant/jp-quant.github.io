---
title: "[QF] Probing Volatility I: Eigen Decomposition of Covariance Matrices"
date: 2020-05-08
tags: [research]
header:
  image: "/images/qf_intro_banner.jpg"
excerpt: "Exploring volatility through a linear algebraic approach, topics commonly known as Principal Component Analysis (PCA), Clustering Algorithms, etc... in various other non-finance purposes."
mathjax: "true"
---

# The Covariance Matrix
>*Exploring the mathematical applications when tackling volatility & correlational properties of securities in the market, as well as the market itself, by dissecting covariance matrices*


From basic statistics, for <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;n=1,2,...,N" title="n=1,2,...,N" /> time indexes of <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;r_{i_n}" title="r_{i_n}" /> and  <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;r_{j_n}" title="r_{j_n}" /> as returns data of asset i and j, their covariance <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\sigma_{ij}" title="\sigma_{ij}" /> is calculated as:
- <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\sigma_{ij}&space;=&space;\frac{\sum_{1}^{N}(r_{i_n}&space;-&space;\bar{r_i})(r_{j_n}&space;-&space;\bar{r_j})}{N-1}" title="\sigma_{ij} = \frac{\sum_{1}^{N}(r_{i_n} - \bar{r_i})(r_{j_n} - \bar{r_j})}{N-1}" /> for **sample size**
- <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\sigma_{ij}&space;=&space;\frac{\sum_{1}^{N}(r_{i_n}&space;-&space;\bar{r_i})(r_{j_n}&space;-&space;\bar{r_j})}{N}" title="\sigma_{ij} = \frac{\sum_{1}^{N}(r_{i_n} - \bar{r_i})(r_{j_n} - \bar{r_j})}{N}" /> for **population size**
 
Recall our brief overview in my [initial post](https://jp-quant.github.io/qf_intro/ "initial post") on the basic essential topics of this research series, given a constructed returns table $$RET$$, we obtain the de-meaned version:
<img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\overline{RET}&space;=&space;\begin{vmatrix}&space;(r_{1_1}&space;-&space;\bar{r_{1}})&(r_{2_1}&space;-&space;\bar{r_{2}})&space;&\hdots&space;&(r_{M_1}&space;-&space;\bar{r_{M}})&space;\\&space;(r_{1_2}&space;-&space;\bar{r_{1}})&space;&(r_{2_2}&space;-&space;\bar{r_{2}})&space;&\hdots&space;&(r_{M_2}&space;-&space;\bar{r_{M}})&space;\\&space;\vdots&\vdots&space;&\ddots&space;&\vdots&space;\\&space;(r_{1_N}&space;-&space;\bar{r_{1}})&(r_{2_N}&space;-&space;\bar{r_{2}})&space;&\hdots&space;&(r_{M_N}&space;-&space;\bar{r_{M}})&space;\end{vmatrix}" title="\overline{RET} = \begin{vmatrix} (r_{1_1} - \bar{r_{1}})&(r_{2_1} - \bar{r_{2}}) &\hdots &(r_{M_1} - \bar{r_{M}}) \\ (r_{1_2} - \bar{r_{1}}) &(r_{2_2} - \bar{r_{2}}) &\hdots &(r_{M_2} - \bar{r_{M}}) \\ \vdots&\vdots &\ddots &\vdots \\ (r_{1_N} - \bar{r_{1}})&(r_{2_N} - \bar{r_{2}}) &\hdots &(r_{M_N} - \bar{r_{M}}) \end{vmatrix}" />

A *N x M* matrix representing N interval (hourly/daily/weekly/etc) returns of M securities, we can construct a **sample size** Covariance Matrix (<img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\boldsymbol{C}" title="\boldsymbol{C}" />):

<img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\boldsymbol{C}&space;=&space;\frac{\overline{RET}^\top&space;\cdot&space;\overline{RET}}{N-1}&space;=&space;\begin{vmatrix}&space;\sigma_1^2&space;&&space;\sigma_{12}&space;&&space;\hdots&space;&&space;\sigma_{1M}&space;\\&space;\sigma_{21}&space;&&space;\sigma_2^2&space;&&space;\hdots&space;&&space;\sigma_{2M}\\&space;\vdots&space;&\vdots&space;&\ddots&space;&\vdots&space;\\&space;\sigma_{M1}&space;&&space;\sigma_{M2}&space;&\hdots&space;&\sigma_M^2&space;\end{vmatrix}" title="\boldsymbol{C} = \frac{\overline{RET}^\top \cdot \overline{RET}}{N-1} = \begin{vmatrix} \sigma_1^2 & \sigma_{12} & \hdots & \sigma_{1M} \\ \sigma_{21} & \sigma_2^2 & \hdots & \sigma_{2M}\\ \vdots &\vdots &\ddots &\vdots \\ \sigma_{M1} & \sigma_{M2} &\hdots &\sigma_M^2 \end{vmatrix}" />

where <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\boldsymbol{C}" title="\boldsymbol{C}" /> is a *M x M*   **square matrix** such that:
- The diagonal values <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\sigma_i^2" title="\sigma_i^2" /> = **variances** of individual securities <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;i&space;=&space;1,2,...,M" title="i = 1,2,...,M" />
- <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\sigma_{ij}" title="\sigma_{ij}" /> where <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;i\neq&space;j" title="i\neq j" /> = **covariances** between two different assets i & j

Important points to highlight:
- Although we can find the standard deviation of individual securities <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;i&space;=&space;1,2,...,M" title="i = 1,2,...,M" /> as <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\sigma_i&space;=&space;\sqrt{\sigma_i^2}" title="\sigma_i = \sqrt{\sigma_i^2}" /> , there is no "standard deviation" of two securities, where the covariance is written as <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\sigma_{ij}" title="\sigma_{ij}" />.
- The covariance value is an **unbounded** measurement value that describes a different perspective, a perspective of looking at the "movement relationship" between two different entities.
- The covariance value <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\sigma_{ij}" title="\sigma_{ij}" /> encompasses/derives a relevant metric **correlation** <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\rho_{ij}&space;=&space;\frac{\sigma_{ij}}{\sigma_i&space;\sigma_j}" title="\rho_{ij} = \frac{\sigma_{ij}}{\sigma_i \sigma_j}" />, where such value, unlike covariance, is **bounded** between (-1,1)

---
For demonstration purposes moving forward, I have preprocessed & cleaned a returns table $$RET$$, logarithmically calculated from **daily close prices from Apil 2019 - April 2020** (encompassing the latest COVID-19 market crash), of 950-1000 securities on the U.S. Equities Market, including individual stocks in all industries/sectors as well as ETFs of domestic (*SPY, XLK, XLV, etc...*) & foreign securities (*FXI, EWJ, etc...* ). Data obtained with IEX Cloud API

```python
#---| Initialize all modules needed |---#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#----| Load preprocess RET Table
RET = pd.read_csv("ret.csv",index_col=0,header=0)
RET
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
      <th>UNH</th>
      <th>EVR</th>
      <th>AMD</th>
      <th>NKE</th>
      <th>NRG</th>
      <th>EV</th>
      <th>VRSN</th>
      <th>SNPS</th>
      <th>PKI</th>
      <th>...</th>
      <th>Z</th>
      <th>PLNT</th>
      <th>PEN</th>
      <th>MSGS</th>
      <th>PSTG</th>
      <th>HPE</th>
      <th>MTCH</th>
      <th>SQ</th>
      <th>TEAM</th>
      <th>UA</th>
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
      <th>4/1/2019</th>
      <td>0.039181</td>
      <td>-0.006981</td>
      <td>0.026032</td>
      <td>0.032385</td>
      <td>0.012040</td>
      <td>-0.016616</td>
      <td>0.023294</td>
      <td>0.023945</td>
      <td>0.020798</td>
      <td>0.012684</td>
      <td>...</td>
      <td>0.016274</td>
      <td>0.014590</td>
      <td>-0.023260</td>
      <td>-0.000205</td>
      <td>0.031175</td>
      <td>0.030005</td>
      <td>-0.006202</td>
      <td>0.018514</td>
      <td>0.013082</td>
      <td>-0.001591</td>
    </tr>
    <tr>
      <th>4/2/2019</th>
      <td>-0.000180</td>
      <td>-0.004613</td>
      <td>-0.001178</td>
      <td>0.014687</td>
      <td>-0.010142</td>
      <td>0.013787</td>
      <td>0.006523</td>
      <td>0.004935</td>
      <td>-0.000510</td>
      <td>-0.001846</td>
      <td>...</td>
      <td>0.024340</td>
      <td>0.012116</td>
      <td>0.005624</td>
      <td>0.014667</td>
      <td>0.013696</td>
      <td>-0.005044</td>
      <td>0.007438</td>
      <td>-0.009479</td>
      <td>0.015079</td>
      <td>0.010034</td>
    </tr>
    <tr>
      <th>4/3/2019</th>
      <td>0.018240</td>
      <td>0.005834</td>
      <td>-0.002468</td>
      <td>0.081451</td>
      <td>0.001185</td>
      <td>-0.000472</td>
      <td>0.006959</td>
      <td>0.006454</td>
      <td>-0.000766</td>
      <td>0.012041</td>
      <td>...</td>
      <td>0.027533</td>
      <td>0.009448</td>
      <td>-0.012190</td>
      <td>0.004362</td>
      <td>0.025559</td>
      <td>0.001895</td>
      <td>0.011228</td>
      <td>0.018998</td>
      <td>-0.002078</td>
      <td>0.005764</td>
    </tr>
    <tr>
      <th>4/4/2019</th>
      <td>0.006065</td>
      <td>0.006285</td>
      <td>0.007814</td>
      <td>0.002409</td>
      <td>0.009544</td>
      <td>-0.002365</td>
      <td>0.010466</td>
      <td>-0.010475</td>
      <td>-0.018567</td>
      <td>-0.007841</td>
      <td>...</td>
      <td>-0.001615</td>
      <td>-0.007042</td>
      <td>-0.016749</td>
      <td>-0.011583</td>
      <td>-0.026876</td>
      <td>0.013785</td>
      <td>-0.037143</td>
      <td>-0.033387</td>
      <td>-0.045579</td>
      <td>0.032385</td>
    </tr>
    <tr>
      <th>4/5/2019</th>
      <td>0.012367</td>
      <td>0.005603</td>
      <td>0.001917</td>
      <td>-0.003789</td>
      <td>0.001406</td>
      <td>-0.004033</td>
      <td>0.006134</td>
      <td>0.016571</td>
      <td>0.011473</td>
      <td>0.011991</td>
      <td>...</td>
      <td>0.003763</td>
      <td>0.010125</td>
      <td>0.004551</td>
      <td>0.006717</td>
      <td>0.004384</td>
      <td>0.004966</td>
      <td>-0.005629</td>
      <td>0.006820</td>
      <td>0.010020</td>
      <td>-0.006597</td>
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
      <th>3/26/2020</th>
      <td>0.101575</td>
      <td>0.085379</td>
      <td>0.053125</td>
      <td>0.062323</td>
      <td>0.064807</td>
      <td>0.071551</td>
      <td>0.073436</td>
      <td>0.052101</td>
      <td>0.053364</td>
      <td>0.052086</td>
      <td>...</td>
      <td>0.022066</td>
      <td>0.039924</td>
      <td>0.079247</td>
      <td>0.010383</td>
      <td>0.109337</td>
      <td>0.114583</td>
      <td>-0.031277</td>
      <td>0.067172</td>
      <td>0.062938</td>
      <td>0.061196</td>
    </tr>
    <tr>
      <th>3/27/2020</th>
      <td>-0.008481</td>
      <td>-0.051996</td>
      <td>-0.001504</td>
      <td>-0.019558</td>
      <td>-0.012774</td>
      <td>0.045344</td>
      <td>-0.022240</td>
      <td>-0.011378</td>
      <td>-0.028454</td>
      <td>-0.054757</td>
      <td>...</td>
      <td>-0.072186</td>
      <td>-0.038137</td>
      <td>-0.023598</td>
      <td>0.009727</td>
      <td>-0.098243</td>
      <td>-0.069807</td>
      <td>-0.017452</td>
      <td>-0.049201</td>
      <td>-0.040977</td>
      <td>-0.051534</td>
    </tr>
    <tr>
      <th>3/30/2020</th>
      <td>0.069726</td>
      <td>0.035772</td>
      <td>0.003863</td>
      <td>0.027109</td>
      <td>0.025504</td>
      <td>-0.037936</td>
      <td>0.039860</td>
      <td>0.080825</td>
      <td>0.038871</td>
      <td>0.022604</td>
      <td>...</td>
      <td>-0.023468</td>
      <td>-0.036764</td>
      <td>0.049167</td>
      <td>-0.048365</td>
      <td>-0.023118</td>
      <td>0.023152</td>
      <td>0.064909</td>
      <td>0.030647</td>
      <td>-0.007692</td>
      <td>0.010759</td>
    </tr>
    <tr>
      <th>3/31/2020</th>
      <td>-0.021578</td>
      <td>-0.007590</td>
      <td>-0.013585</td>
      <td>-0.051007</td>
      <td>-0.031409</td>
      <td>-0.042728</td>
      <td>-0.046351</td>
      <td>-0.045906</td>
      <td>-0.004029</td>
      <td>-0.016207</td>
      <td>...</td>
      <td>-0.028464</td>
      <td>0.002055</td>
      <td>-0.011524</td>
      <td>-0.064122</td>
      <td>-0.008097</td>
      <td>-0.034416</td>
      <td>-0.009494</td>
      <td>-0.048808</td>
      <td>-0.028017</td>
      <td>-0.042508</td>
    </tr>
    <tr>
      <th>4/1/2020</th>
      <td>-0.051881</td>
      <td>-0.049568</td>
      <td>-0.009599</td>
      <td>-0.040840</td>
      <td>-0.043348</td>
      <td>-0.015527</td>
      <td>-0.062040</td>
      <td>-0.026046</td>
      <td>-0.022219</td>
      <td>-0.060932</td>
      <td>...</td>
      <td>-0.135993</td>
      <td>-0.139947</td>
      <td>-0.049035</td>
      <td>-0.014964</td>
      <td>-0.089231</td>
      <td>-0.013479</td>
      <td>-0.069436</td>
      <td>-0.112428</td>
      <td>-0.023664</td>
      <td>-0.078700</td>
    </tr>
  </tbody>
</table>
</div>

We will use *Pandas* to perform calculation for our covariance matrix, as it is automatically [normalized by (N-1)](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.cov.html "normalized by (N-1)") for sample population. We confirm this by performing our own calculation to check for sanity

```python
def sample_covariance(ret):
    demeaned_RET = pd.DataFrame({
        s:(ret[s]-ret[s].mean()) for s in ret.columns})
    return ((demeaned_RET.T.dot(demeaned_RET))/(len(ret)-1))

raw_cov = sample_covariance(RET)
pd_cov = RET.cov()
```


```python
False not in [np.allclose(raw_cov[i],pd_cov[i]) for i in raw_cov.columns]
```




    True



They indeed do match. We proceed onto the next step.

## Eigen Decomposition
>*This topic is a part of Linear Algebra, an extremely powerful spectrum within mathematics, that elevated my perspective from reality to abstraction, specifically in modeling nature using infinite-dimensional abstract vector space. For further academic readings for ground-up studies, I recommend watching this MIT lecture [series](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/ "series").*

From a linear algebraic perspective, specifically in our approach, a matrix acts as a vector space, defined by a system of equations, containing subspaces such as row space, column space, etc... as well as an operator that perform transformation, or mapping, of vectors from one subspace to another.
>**Example**: Casting a shadow of an object = mapping a 3D entity to its 2D form

In this work, we are exploring a mapping from N-dimensional space to another N-dimensional space, specifically working with a square matrix. The main point of Eigen Decomposition is simply to find **[invariant subspaces](https://en.wikipedia.org/wiki/Invariant_subspace "invariant subspaces") under transformation by a square matrix**. The analysis of such extracted eigen pairs is otherwise known as Principal Component Analysis (PCA), commonly used in other non-finance topics for dimensionality reduction, clustering data features, etc...

> **NOTE**: Unlike the common usages such as compression of images' pixels, or classification of different plant types, we are utilizing PCA to extract eigen components from a covariance matrix calculated from time series datasets of financial instruments. Always keep this context in mind when approaching mathematically, abstract or applied.

Given our *M x M* covariance matrix <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\boldsymbol{C}" title="\boldsymbol{C}" /> of M securities, as square matrix calculated from returns of *N* time indexes, we seek to find <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;e_i" title="e_i" /> and <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\lambda_i" title="\lambda_i" /> , respectively as the **eigen vectors** and **eigen values**, where for <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;i&space;=&space;1,2,...,M" title="i = 1,2,...,M" />, comes with such **eigen pair** that satisfy the condition:
<img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\dpi{150}&space;\boldsymbol{C}&space;\cdot&space;e_i&space;=&space;\lambda_i&space;e_i" title="\boldsymbol{C} \cdot e_i = \lambda_i e_i" /> such that:
- <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;e_{i}&space;=&space;\begin{vmatrix}&space;e_{i_1}\\&space;e_{i_2}\\&space;\vdots\\&space;e_{i_M}&space;\end{vmatrix}" title="e_{i} = \begin{vmatrix} e_{i_1}\\ e_{i_2}\\ \vdots\\ e_{i_M} \end{vmatrix}" /> , where we can construct an *M x M* matrix of the linearly independent eigen vectors as <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\boldsymbol{E}&space;=&space;\begin{vmatrix}&space;e_1&space;&e_2&space;&\hdots&space;&e_M&space;\end{vmatrix}&space;=\begin{vmatrix}&space;e_{1_1}&space;&&space;e_{2_1}&space;&\hdots&space;&e_{M_1}&space;\\&space;e_{1_2}&space;&&space;e_{2_2}&space;&\hdots&space;&e_{M_2}&space;\\&space;\vdots&space;&\vdots&space;&\ddots&space;&\vdots&space;\\&space;e_{1_M}&space;&e_{2_M}&space;&\hdots&space;&e_{M_M}&space;\end{vmatrix}" title="\boldsymbol{E} = \begin{vmatrix} e_1 &e_2 &\hdots &e_M \end{vmatrix} =\begin{vmatrix} e_{1_1} & e_{2_1} &\hdots &e_{M_1} \\ e_{1_2} & e_{2_2} &\hdots &e_{M_2} \\ \vdots &\vdots &\ddots &\vdots \\ e_{1_M} &e_{2_M} &\hdots &e_{M_M} \end{vmatrix}" />

- <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\lambda_i" title="\lambda_i" /> is a **real** value, each associated with <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;e_i" title="e_i" />,  to which we can diagonalize it into another *M x M* square matrix:
<img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\boldsymbol{V}&space;=&space;\begin{vmatrix}&space;\lambda_1&space;&0&space;&\hdots&space;&0&space;\\&space;0&space;&\lambda_2&space;&\hdots&space;&0&space;\\&space;\vdots&space;&\vdots&space;&\ddots&space;&\vdots&space;\\&space;0&space;&0&space;&\hdots&space;&\lambda_M&space;\end{vmatrix}" title="\boldsymbol{V} = \begin{vmatrix} \lambda_1 &0 &\hdots &0 \\ 0 &\lambda_2 &\hdots &0 \\ \vdots &\vdots &\ddots &\vdots \\ 0 &0 &\hdots &\lambda_M \end{vmatrix}" />


Due to the nature of <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\boldsymbol{C}" title="\boldsymbol{C}" /> being not just as a square matrix, but also a **Hermitian Matrix**, our values will be **real** & the matrix is diagonalizable such that:
<img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\boldsymbol{C}&space;=&space;\boldsymbol{E^\top}&space;\cdot&space;\boldsymbol{V}&space;\cdot&space;\boldsymbol{E}&space;=&space;\begin{vmatrix}&space;\sigma_1^2&space;&&space;\sigma_{12}&space;&&space;\hdots&space;&&space;\sigma_{1M}&space;\\&space;\sigma_{21}&space;&&space;\sigma_2^2&space;&&space;\hdots&space;&&space;\sigma_{2M}\\&space;\vdots&space;&\vdots&space;&\ddots&space;&\vdots&space;\\&space;\sigma_{M1}&space;&&space;\sigma_{M2}&space;&\hdots&space;&\sigma_M^2&space;\end{vmatrix}" title="\boldsymbol{C} = \boldsymbol{E^\top} \cdot \boldsymbol{V} \cdot \boldsymbol{E} = \begin{vmatrix} \sigma_1^2 & \sigma_{12} & \hdots & \sigma_{1M} \\ \sigma_{21} & \sigma_2^2 & \hdots & \sigma_{2M}\\ \vdots &\vdots &\ddots &\vdots \\ \sigma_{M1} & \sigma_{M2} &\hdots &\sigma_M^2 \end{vmatrix}" />

To observe the significance of to why this is powerful when it comes to applying it on the covariance matrix, we will first highlight the significance of the eigen pairs (<img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;e_i" title="e_i" /> & <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\lambda_i" title="\lambda_i" />)
- Eigen vectors <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;e_i" title="e_i" /> are **normalized independent vectors** from each other, meaning that they are **orthogonal** to each other with <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;||e_i||&space;=&space;1" title="||e_i|| = 1" />
- Since there are *M* eigen vectors <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;e_i" title="e_i" /> extracted from an *M x M* matrix, their orthogonality means that they represent the **basis** of M-dimensional vector space.
- Therefore, in other words, we perceive the eigen vectors as the directional vectors basis, the axis that builds the entire subspace that described by such matrix, or in this case the covariance matrix <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\boldsymbol{C}" title="\boldsymbol{C}" />
- Each eigen value <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\lambda_i" title="\lambda_i" /> simply represents the **directional variance** of its associated eigen vector <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;e_i" title="e_i" /> , independent from one another, such that together they **explain** the total volatility of *M* securities, without any specified allocations.

---
We will first visualize the meaning of such extract eigen pairs by giving an example of two securities -  **SPY**(*S&P500 ETF*) and **IEF**(*7-10 Year Treasury ETF*).
Plotting their returns against each other in **2-Dimensional Space** and we have:

```python
ax = RET.plot.scatter("SPY","IEF",alpha=0.2)
ax.set_aspect("equal")
```
<img src="https://i.ibb.co/pRnprzW/1.png" alt="1" border="0">

Observe the relationship between two assets, exhibiting negative correlation, reflecting behaviorially through the financial market (i.e.: As the market goes down, demand falls for equities and rises in treasury, especially during COVID-19 market crash ). 

```python
RET[["SPY","IEF"]].corr()
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
      <th>SPY</th>
      <th>IEF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SPY</th>
      <td>1.000000</td>
      <td>-0.516561</td>
    </tr>
    <tr>
      <th>IEF</th>
      <td>-0.516561</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>

If we plot the cummulative returns between the two assets, we can distinctly observe their correlational properties as exhibited through the scatter plot:

```python
(1+RET[["SPY","IEF"]]).cumprod().plot()
```

<img src="https://i.ibb.co/Zd9h67Z/2.png" alt="2" border="0">


Using numpy's linalg module to extract our eigen pairs. Below is our pre-coded function to perform PCA, working on dataframe being inputted & return the eigen vectors & eigen values:

```python
#----| Perform Eigen Decomposition on Covariance Matrix of any given returns table, standardized/normalized or not
#----- dynamic for any NxM size of returns matrix
def EIGEN(ret,_simple=False):
    _pca = PCA().fit(ret)
    eVecs = _pca.components_
    eVals = _pca.explained_variance_
    _eigenNames = ["eigen_"+str(e+1) for e in range(len(eVals))]
    _eigenValues = pd.Series(eVals,index=_eigenNames,name="eigenValues")
    _eigenVectors = pd.DataFrame(eVecs.T,index=ret.columns,columns=_eigenNames)
    return _eigenVectors,_eigenValues
```
With each eigen vector magnified by 3 times the square root of its associated eigen value (since each eigen value represents the directional variance of such eigen vector, as stated above), graphing such vectors on top of the scatter plot, each in both directions, using *matplpotlib* we obtain:

```python
ax = plt.figure().gca()
ret = RET[["SPY","IEF"]]
x,y = list(ret.columns)
vectors,lambdas = EIGEN(ret)
ax.scatter(ret[x], ret[y], alpha=0.1)
ax.set_xlabel(x)
ax.set_ylabel(y)
_start = ret.mean(axis=0)
_colors = pd.Series(["red","green"],index=lambdas.index)
for e in lambdas.index:
    v = vectors[e] * (3*np.sqrt(lambdas[e]))
    ax.plot([_start[x]-v[x],_start[x]+v[x]],
              [_start[y]-v[y],_start[y]+v[y]],color=_colors[e],linewidth=2,label=e)
ax.set_aspect("equal")
ax.legend()
```

<img src="https://i.ibb.co/8B6N4M3/4.png" alt="4" border="0">

> **REMARK**: The eigen components are orthogonal to each other, encapsulating the magnitudes of standard deviation in both dimensions. We interpret M eigen values as **independently explained variances** that, together, describe the **total variance** of our M selected securities.Such mathematical approach can take us further to any M-dimensional space such that we can describe the entire variance of M securities, in such abstract vector space that we can't visualize, down to the independent components. 

### Plotly 3D Interactive Visualization
For the sake of illustration & fancy observation, below are functions written to actually plot such results for 3 Dimensional Space (M=3) using [Plotly](https://plotly.com/python/ "Plotly"), a powerful interactive data visualization module:

```python
import plotly.graph_objects as go
#----| Plot (with labels) & return result of 3 selected assets with each other with given table
def EIGEN_3D(ret):
    assert(ret.shape[1] == 3),"3 assets only"
    vectors,lambdas = EIGEN(ret)
    _start = ret.mean(axis=0)
    x,y,z = list(ret.columns)
    explained_variance = lambdas/lambdas.sum()
    _names = pd.Series([(i+" - "+str(
                    (explained_variance[i]*100).round(2))+"%") for i in explained_variance.index],
                           index=explained_variance.index)
    result = {"vectors":vectors,"lambdas":lambdas,
              "explained_variance":explained_variance}

    _DATA_ = []
    #----| SCATTER PLOT OF RETURNS DATA |----#
    _DATA_.append(go.Scatter3d(x=RET[x],y=RET[y],z=RET[z],mode="markers",marker=dict(
                                size=5,opacity=0.2),name=" - ".join([ret.index[0],ret.index[-1]])))

    #----| LINE PLOT OF EIGEN COMPONENTS |----#
    for e in lambdas.index:
        v = vectors[e] * (3*np.sqrt(lambdas[e]))
        _DATA_.append(go.Scatter3d(x=[_start[x]-v[x],_start[x]+v[x]],
                                   y=[_start[y]-v[y],_start[y]+v[y]],
                                   z=[_start[z]-v[z],_start[z]+v[z]],
                                   mode="lines",line=dict(width=6),name=_names[e]))

    fig = go.Figure(_DATA_)
    fig.update_layout(
        title=" -vs- ".join(list(ret.columns)),
        scene=dict(
        xaxis_title=x,
        yaxis_title=y,
        zaxis_title=z,aspectmode="auto"))
    result["figure"] = fig
    return result
```
Observe the results for 3 assets picked as **SPY**(*S&P500 ETF*), **IEF**(*7-10 Year Treasury ETF*) & **GLD** (SPDR Gold Shares), fundamentally representing different asset classes (stocks vs treasury vs gold), thus exihibiting minimal correlations to each other dimensionally.

```python
result = EIGEN_3D(RET[["SPY","IEF","GLD"]])
result["figure"].show()
```

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<div id="b91eaac0-18ff-4d09-bca5-ce6c8bae7116" class="plotly-graph-div" style="height:100%; width:100%;"></div>
<script type="text/javascript">
    
        window.PLOTLYENV=window.PLOTLYENV || {};
        
    if (document.getElementById("b91eaac0-18ff-4d09-bca5-ce6c8bae7116")) {
        Plotly.newPlot(
            'b91eaac0-18ff-4d09-bca5-ce6c8bae7116',
            [{"marker": {"opacity": 0.2, "size": 5}, "mode": "markers", "name": "5/1/2019 - 5/1/2020", "type": "scatter3d", "x": [-0.007544886999999999, -0.002161273, 0.00974017, -0.004123717, -0.016840691, -0.0013901920000000002, -0.003030358, 0.0050107979999999995, -0.025451317999999997, 0.009003003, 0.005840356999999999, 0.009218585999999999, -0.006486057, -0.006634047, 0.008975273, -0.0030761729999999998, -0.012293865, 0.002265809, -0.009344036, -0.006733309, 0.002727438, -0.013566868999999999, -0.002546197, 0.021474434, 0.008624874, 0.006481635, 0.009957284, 0.004578413, -0.000242269, -0.001766877, 0.004117867, -0.00110566, 0.00038020800000000004, 0.010416581999999999, 0.0022546379999999998, 0.009509003, -0.006306602, -0.00122524, -0.009856343, -0.0009978839999999999, 0.003539705, 0.0051326029999999995, 0.009037536, 0.002600957, 0.00796335, -0.001138533, -0.0055100259999999995, 0.00124577, 0.004766709, 0.002341451, 0.0044669720000000005, 0.000332557, -0.003464014, -0.006594718, 0.003654227, -0.0055704840000000005, 0.002453494, 0.00712461, 0.0046885220000000005, -0.004788517, 0.006677655, -0.0018227920000000002, -0.0024577379999999997, -0.011000696, -0.008746067, -0.0075579969999999995, -0.030534605, 0.01392556, 0.000590514, 0.019430106000000003, -0.006834829000000001, -0.012248078999999999, 0.015432086000000001, -0.030013529, 0.002638292, 0.014647166999999999, 0.011975778999999999, -0.007692081, 0.008102492, -0.000307792, -0.026023194, 0.010997753999999998, -0.003931329, 0.007016840999999999, 0.012692141, -0.000444422, -0.005864315, 0.011286416, 0.012773465, 0.000771981, 0.000503145, -0.00023476900000000001, 0.007085828000000001, 0.003457795, -0.0006640330000000001, -0.0030935579999999997, 0.0025287829999999997, 0.000597987, -6.64e-05, -0.009343367, -0.000234706, -0.007877768, 0.0058973359999999996, -0.002085366, -0.005401769, 0.004627058, -0.011966042, -0.017822595, 0.008159359, 0.013441383999999999, -0.004323926, -0.015646543, 0.009451605, 0.006740714, 0.010313567, -0.001114432, 0.009851634, -0.0016072870000000002, 0.0029447220000000003, -0.00438678, 0.00675633, -0.003272123, 0.0029053770000000002, 0.001632653, 0.0040865879999999995, 0.005620779, -0.00029678, 0.003062487, -0.0026668000000000004, 0.009221192, 0.00400972, -0.001106771, 0.000227965, 0.0035106, 0.002463055, -0.0019115820000000002, 0.002105775, 0.000323572, 0.001454781, 0.007210254, 0.000737404, -0.000288485, -0.003725715, -0.001610203, 0.0022214, 0.007720314, 0.00226313, 0.004447558, -0.003715528, -0.008531085, -0.006729045, 0.006151289, 0.001796369, 0.009092567, -0.003149108, -0.001115698, 0.002834622, 0.008582113, 0.0074454890000000004, 0.000219068, 6.26e-05, 0.004090624, -0.0005298999999999999, 0.001526599, 3.11e-05, 0.00530917, -0.000247755, -0.005528479, 0.002426355, 0.009308434, -0.007601075, 0.0038077590000000004, -0.002815727, 0.005315381, 0.00675782, -0.002881846, 0.006853743000000001, -0.0015257860000000001, 0.0022573370000000003, 0.008283945, 0.0031077009999999996, -0.001960046, 0.000120729, 0.0011462010000000001, -0.008932821, -0.016159304, 0.010424609, -0.000826307, 0.0032401070000000003, -0.018324838, 0.007401133000000001, 0.015126286999999999, 0.011481876, 0.00335913, -0.005343914, 0.007437654, 0.0017314989999999998, 0.006422095, -0.001067489, 0.001600807, -0.00258034, 0.004769885, -0.004116756, -0.010351658, -0.03372785, -0.030770689, -0.0036850159999999997, -0.0459515, -0.004210391, 0.042395049000000004, -0.029050342000000003, 0.041173651, -0.033806769, -0.016669275, -0.081312592, 0.050450580999999994, -0.049976639, -0.100568916, 0.082028067, -0.115886536, 0.05258493400000001, -0.051959739000000005, 0.002122745, -0.049913409000000006, -0.025900727999999998, 0.086731006, 0.014859236000000001, 0.056748625, -0.030238206, 0.031959541, -0.01501761, -0.046049017000000005, 0.02281315, -0.014559675, 0.065006803, 0.001018887, 0.033017318, 0.01510269, -0.009172056999999999, 0.029066306, -0.021477097, 0.004812709, 0.026656932, -0.017774961000000002, -0.030833807999999997, 0.021951836000000002, -7.170000000000001e-05, 0.013842406000000002, 0.014315531999999999, -0.004609108, 0.025841767, -0.009354349, -0.026830154], "y": [-0.002458165, -0.002843873, 0.001328147, 0.002556456, 0.0033984740000000003, -0.001886615, 0.0026402659999999996, -0.0006594130000000001, 0.004981915999999999, -0.0009380860000000001, 0.00318591, -0.001966569, 0.000468582, -0.0016879220000000001, -0.001126866, 0.00300216, 0.00607053, -0.000279369, 0.004367834, -0.000370989, 0.003611282, 0.0066329119999999995, 0.002842344, -0.003669056, 0.0, -0.000183807, 0.003944416, -0.004955957, 0.0, 0.002572821, 0.002474682, 0.0007320639999999999, -0.00018296599999999998, 0.001919474, 0.00273573, 0.0016378529999999999, -0.004099673, 0.003463049, 0.001545385, -0.004095942, 0.00300533, 0.000454566, -0.003003962, 0.003549169, 0.001905886, -0.006367131, -0.001004245, -0.000639591, 0.000548246, -0.004945065, 0.001101019, 0.002106903, -0.002198608, 0.004666279000000001, 0.002461596, -0.0018228220000000002, 0.00045601699999999997, -0.001642636, 0.001460254, -0.001734289, -0.000182732, 0.000913325, 0.00045635, 0.002551719, 0.009511371999999999, 0.00207142, 0.008599892, 0.0016042779999999998, 0.0, 8.909999999999998e-05, -0.001514949, 0.006311974000000001, -0.00355114, 0.006648077, 0.00634084, -0.002637132, -0.004764435, 0.004412291, -0.0026450379999999997, -0.002209555, 0.006701965, -0.0009672460000000001, 0.0043014590000000005, 0.000350324, -0.001928303, 0.000263169, 0.000526131, 0.001576735, -0.008085113, 0.000617475, -0.00619197, -0.007302553000000001, -0.00098368, -0.002239241, -0.009188426, 0.004334484, 0.002609675, 0.00035942099999999996, 0.000449095, 0.005730675, 0.0014275520000000001, 0.0049804440000000005, -0.006586584, 0.002140946, 0.001424755, 0.000800534, 0.001244002, 0.0034572970000000002, 0.005207656, 0.0019349170000000002, -0.003344485, 0.002113607, -0.002554505, -0.0062818169999999994, -0.006857576999999999, 0.0027665, -0.004824022, 0.001968152, -0.000357558, 0.000893655, -0.0036690720000000002, 0.00232829, 0.00035771800000000003, -0.00017884299999999998, -0.0021486129999999997, -0.003771218, 0.0008992000000000001, 0.004841756, 0.006330545999999999, -0.0033831940000000004, -0.00509635, -0.005392787, 0.002879772, -0.008029998, -0.0013597429999999998, 0.001087942, 0.00045296, 0.00271346, 0.00504551, -0.000809171, 0.0017074820000000002, 0.001883831, 0.003667759, -0.0025033539999999997, -0.00017905099999999998, 0.00044756699999999995, 0.001877599, -0.002593803, -0.0008063430000000001, -0.005031909, 0.008878584, -0.003846675, -0.002333095, -0.003059207, 0.000810701, -0.001171224, 0.004138185, -0.007931542, 0.0011756730000000001, 9.04e-05, -0.0027149340000000004, -0.000634547, 0.0, -0.000907194, 0.001722966, 0.00144862, 0.001446524, -0.0006325969999999999, -0.0036225359999999996, 0.004616437, 0.006660691, -0.001077103, -0.001437944, -0.002341078, 0.000720916, 0.002159634, -0.001169223, 0.001977884, 0.002511663, -0.002062689, -0.000808299, 0.003945486, 0.000178971, 0.002502235, 0.0035637959999999997, 0.00673642, -0.003362537, 0.004774967, 0.0009698889999999999, 0.005010779, -0.00219462, -0.005817044, -0.0043407079999999995, 0.000709975, 0.004779193, 0.001940549, -0.002382109, -0.002564898, 0.000885191, 0.002474374, 0.001587442, -0.000176258, 0.002904292, 0.004034385, 0.007759749, 0.00277537, -0.000519796, 0.00484095, 0.011148387, -0.001877935, 0.011552966000000001, -0.0021983610000000002, 0.008848497, 0.010100674, 0.008930107, -0.01852751, -0.01002916, 0.000508087, -0.006540135, 0.026073807, -0.025392317999999997, -0.013977846, 0.003276147, 0.02515856, 0.011765332, -0.0068253969999999995, 0.000584478, 0.002417776, 0.007383781, 0.0023117579999999997, 0.00181264, 0.00287699, 0.00016415, 0.00073834, -0.004520618, -0.004458396, -0.001241979, 0.002151784, -0.002151784, 0.0013247229999999999, 0.00889627, 0.00106553, -0.002624889, 0.002624889, 0.002781415, -0.00310915, 0.0006553619999999999, 0.00024565, -0.005335973000000001, 0.0043530109999999995, -0.000491844, -0.00238066, 0.00032870400000000003], "z": [-0.006622541, -0.003827915, 0.0059021740000000005, 0.001325271, 0.003305515, -0.002478111, 0.002395606, 0.001895891, 0.010159858, -0.001713377, -0.000571779, -0.007216103000000001, -0.007102771999999999, -8.290000000000001e-05, -0.0023236520000000003, -0.000997506, 0.007704773000000001, 0.001072298, -0.0038822170000000003, 0.000330989, 0.0067612390000000005, 0.013386862, 0.014329661, 0.000319668, 0.002553668, 0.003262386, 0.005624445, -0.010003259, -0.001197557, 0.005179088, 0.00610072, -0.00031600599999999997, -0.000632311, 0.005047329000000001, 0.006038997, 0.024866146000000002, 0.00661373, 0.014741543000000001, 0.0019392860000000001, -0.009207686, -0.000978142, 0.002706362, -0.019559414, 0.021134748999999998, 0.001647694, -0.01121282, -0.00645335, 0.00349757, 0.01566415, -0.008479396, 0.00623523, 0.0, -0.008498528, 0.016331174, 0.014019266, -0.014836958, -0.00014874299999999998, -0.005668696, 0.005073502, -0.007095142, 0.00164757, 0.006637605, 0.003339397, -0.013199788999999998, 0.023738226, -0.003819322, 0.013885052, 0.008095454, 0.015075483, 0.005022115, -0.003321675, 0.009651701, -0.005977304, 0.0068182880000000005, 0.006632945, -0.006422809, -0.011765256000000002, 0.00776511, -0.0031693520000000003, -0.0025427329999999997, 0.019400405, 0.000138715, 0.009525196, -0.002820488, -0.007190296, -0.002570606, 0.013817146, 0.006224156999999999, -0.02429381, -0.008559654, -0.003741489, -0.008594719, 0.006045323000000001, 0.002054189, -0.008313545, 0.008313545, 0.001979359, -0.006305146, 0.0040427020000000004, 0.011751181999999999, 0.005580761, 0.00527303, -0.018719552, -0.000282068, -0.005161758, -0.015647088, 0.0054578230000000005, 0.011606097, 0.00452042, 0.0, -0.008563695999999999, 0.00870463, 0.00091559, -0.008767644, -0.005554779, 0.003991168, -0.006995031999999999, 0.005713893, 0.001423386, -0.00106735, -0.004781454, 0.002928678, 0.002351015, 0.00709072, 0.002328947, -0.008637222, -0.00277689, 0.00547518, 0.009948927, 0.000912313, -0.0028801259999999998, -0.016312418000000002, 0.004281134, -0.015643258, -0.006384698, -0.002404811, 0.00269591, 0.003994051, 0.004194698, -0.002529177, 0.002962109, 0.00050485, 0.000504595, -0.005492136999999999, -0.001885835, -0.004803153, 0.004803153, -0.005313934, 0.006184762, -0.00050789, 0.009534200000000001, -0.001366759, 0.000575705, -0.009977669, -0.000290698, 0.0028307040000000003, 0.0068619580000000005, -0.003533445, 0.004396879, -0.000215789, 7.190000000000001e-05, 0.002586208, 0.0010039439999999999, 0.0030772559999999996, 0.009387737, 0.007826587, -0.000351235, 0.002105559, 0.00189122, 0.007320932, 0.01318124, 0.010434877, 0.0039274159999999995, -0.007529798000000001, -0.0056676719999999995, 0.006008074, -0.00744717, -0.0008919080000000001, 0.005817352, -0.00157077, 0.0018436960000000001, 0.0010909589999999999, 0.00034068099999999996, 0.002245586, 0.005828549000000001, 0.00680206, -0.008966856, 0.005403228, 6.74e-05, 0.005775704, -0.006516869000000001, -0.013094254, 0.0012285010000000001, 0.00537398, 0.002642367, 0.002567916, -0.00344793, -0.000813008, 0.005677225, 0.004169755, 0.012737327, 0.005814354, 0.004076270999999999, 0.0149135, 0.008945006, -0.018035978, 0.004360992, 0.00019482400000000002, -0.037176052, 0.005511136999999999, 0.030950374, 0.001752963, 0.02137094, 0.000380904, 0.00164891, -0.02132714, -0.003566684, -0.040705606, -0.030991589, -0.011512131, 0.013464439, -0.020123102, -0.019086466, 0.01488433, 0.04323148, 0.04738958099999999, -0.013784268, 0.012740699, -0.006481413, 0.004391002, -0.032364854, 0.009411834, 0.016260521, 0.00492531, 0.027333463, -0.005368797, -0.008947886, 0.025788113999999997, 0.016995098, 0.007837369, -0.0051151009999999995, -0.000865373, -0.019608471000000002, 0.007100919, -0.006848696, 0.019479919, 0.00990564, -0.0042947490000000005, -0.006662579, -0.0044665090000000005, 0.0055181959999999995, -0.018282728999999998, 0.00615232]}, {"line": {"width": 6}, "mode": "lines", "name": "eigen_1 - 75.54%", "type": "scatter3d", "x": [0.059828716703644674, -0.060136568529731635], "y": [-0.007250993838332416, 0.00835096201224546], "z": [0.004017009839781896, -0.0018323790966988931]}, {"line": {"width": 6}, "mode": "lines", "name": "eigen_2 - 21.93%", "type": "scatter3d", "x": [0.0005635591511639279, -0.0008714109772508839], "y": [-0.0059210758660686825, 0.007021044039981726], "z": [-0.030882732638242143, 0.03306736338132515]}, {"line": {"width": 6}, "mode": "lines", "name": "eigen_3 - 2.53%", "type": "scatter3d", "x": [0.0013513948466569094, -0.0016592466727438657], "y": [0.0113200632733247, -0.010220095099411656], "z": [-0.0010535385746635755, 0.0032381693177465786]}],
            {"scene": {"aspectmode": "auto", "xaxis": {"title": {"text": "SPY"}}, "yaxis": {"title": {"text": "IEF"}}, "zaxis": {"title": {"text": "GLD"}}}, "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "SPY -vs- IEF -vs- GLD"}},
            {"responsive": true}
        )
    };
    
</script>

---
# Applications
*Now that we understand & visualize PCA implementation on time-series, we will proceed on discussing what we can use it for in the context of quant finance & further the horizons of our approach on dissecting volatility as a whole.*

---
[IN PROGRESS]







