---
title: "[QF] Introduction & Essential Foundations"
date: 2020-05-04
tags: [qf]
header:
  image: "/images/qf_intro_banner.jpg"
excerpt: "Preface, Inspiration, Brief Coverage on Foundational Topics & Personal Remarks Before Moving Forward"
mathjax: "true"
---


# Preface
---
Since my independent exploratory journey into this world of Quantitative Finance, as a young mathematicians finishing off my final year of undergraduate career, I have spent hundreds to thousands of hours studying he fundamentals of finance as a whole, cultivating experiences & skills in data science through academic projects, independently and collaboratively with fellow peers, as well as watching/reading graduate-level lectures on Financial Engineering, Mathematics, Computer Science, etc...My garnered knowledge & work have all lead me to develop an intricate customizable Quant Engine, containing tens of thousands lines of Python codes, designed with an event-driven architecture for backtesting & live-trading purposes, to which allowed me to not just coming up with trading strategies & performing research work, but also being able to backtest the strategies to validate the developed models and continuously improving my many developed quantitative trading algorithms.

This series of posts under Quantitative Finance (QF) contain compositions of notes & coding contents from independent research, primarily aims toward the overarching goal of capturing alphas & developing alpha signals, whereas "alpha" plays a major part in almost all existing quantitative trading strategies out there (almost all leading firms have divisions focusing purely on capturing alphas). In the process of presenting my work & approach, I will try my best to shed light on relevant topics to which constitute the overall development of alpha design, though not in details, as I will attach links for further readings on those relevant topics (academic lectures, other posts, etc...)

# Inspiration
When it comes to systematic investing utilizing quantitative approaches, one thing to always keep in mind is that models (or rules) developed is only reliable at a "certain time," to which such time variable "t" in terms of its period/frequency, can be as dynamic as the reliability of the model developed itself.

In another words, as said by Igor Tulchinsky, the founder of WorldQuant:

> "***An infinite number of possible rules describe reality, and we are always struggling to discover and refine them. Yet, paradoxically, there is only one rule that governs them all. That rule is: no rule ever works perfectly.***"

The decision to compose publication(s) of my research notes & implementation results was propelled after reading Tulchinsky's book on [*Finding Alphas*](https://www.amazon.com/Finding-Alphas-Quantitative-Approach-Strategies/dp/1119057868), especially after realizing that my mathematical approach drew significant parallel to the content of his book, though Tulchinsky has provided me the terminologies & credible academic foundations needed to not just affirm my mathematical interpretations in tackling such topic, as well as my doubt on the reliability of my developed models & strategies, but also articulate my understandings & personal approaches that could potentially help other passionate participants in the world of Quantitative Finance.

# Foundations & Basic Essentials
*Brief coverage on basic foundational topics & key-points on personal approach & understanding towards alpha designs.*

---

### Alpha & Beta

In general finance & economics, a commonly known model called the [Capital Asset Pricing Model](https://corporatefinanceinstitute.com/resources/knowledge/finance/what-is-capm-formula/ " Capital Asset Pricing Model") (CAPM) can be perceived to serve as the propelling root of cause for the concept of alpha:

$$R_a = R_f + \beta (R_m - R_f)$$

where:

$$R_a = $$  Expected Return of Security
$$R_m = $$ Expected Return of The Market
$$R_f = $$ Risk-free Rate
$$\beta = $$ Beta of Security


**Beta**, by definition, is the security's "sensitivity" to market's risk/volatility movement. Although, it is better to understand it from a quantitative approach, when it comes to alpha, as we proceed on understanding Beta as the **"regressed slope of the security's returns with respect to the returns of the selected market/benchmark** (e.g.: 0.5 beta = *on average*, for every 1% returns on the market, our security returns at 0.5%). We will demonstrate this terminology further as we move forward.

In addition, notice we are using "expected returns" as this is important to proceed on understanding alpha design. Recall that since CAPM aims to model a security/portfolio, say AAPL, to a certain market as a benchmark to our choosing, say the SPY index, or even the XLK (Technology Index from SPDR) to compare AAPL with its competitors in the sector its operating in, to obtain its expected rate of returns in able to price equity investments.
>**REMARK**: This concept of modeling performance (return rate in this case) of a certain investment target with respect to another is my first exposure to understanding alpha design, as the ultimate goal of investing & trading in finance is about performing better than the market (personally as absolute as possible)


First, we will start with the textbook definition:
> **Alpha** (or [Jensen's Alpha](https://financetrain.com/jensens-alpha/ "Jensen's Alpha"))  is the risk-adjusted/excess returns of a portfolio, used to evaluate its returns performance with respect to its expected returns against certain selected benchmark (using CAPM model).

Jensen's $$\alpha = R_p - (R_f + \beta (R_m - R_f))$$
where $$R_p = $$ Portfolio's Return

As CAPM aims to evaluate the expected return of a certain target, Alpha is used to evaluate actual performance of such target, usually a portfolio, strategy, or individual security (notice that $$R_p$$ is not an expected value but a realized/actual value).

It is important to highlight the definition of alpha being **risk-adjusted/excess returns** as this is the foundation to which we will use to proceed on understanding & designing alphas. On a mathematical standpoint, **whereas beta is the slope, alpha is the y-intercept** when regressing returns against a selected benchmark, or that alpha is the returns of a certain *investment target*  when the expected *benchmark*  returns 0% on average. In other words,

>**Alpha** is the **independent returns** of a portfolio with respect to a selected benchmark.

It is valuable to have high alpha as best as we can, such that statistically, given the classic definition of alpha, regardless of market's movement, bullish or bearish, even if it crashes, the alpha component of a given portfolio return will not be affected. This is the power of finding alphas in the investing world.

---
### Modern Portfolio Theory 
From a fundamental perspective of portfolio management, controlling risk tolerance & maximizing returns, the topic of Modern Portfolio Theory (MPT), or mean-variance analysis, is essential in almost all aspects of investment.
As such topic can start from a very simplistic implementation to a much more complex & layered one, we will be using the mathematics behind the concept in a much wider range of purposes than just maximizing returns & minimizing risks, approaching them through a quantitative perspective by utilizing abstract vector space & matrix theory.

Assume we have constructed a portfolio of selected securities (this is an intricate process that we will further explore), we seek to generate maximal returns with as minimal risk as possible, both in-sample & out-sample.
If we have *M* amount of the selected securities in such portfolio, and given a selected *N* time indexes (minute/hour/day/week/month/year/etc), we have their N **historical prices** of M securities, each defined as:
<img src="https://latex.codecogs.com/gif.latex?\vec{p_i}&space;=&space;\begin{vmatrix}&space;p_{i_1}\\&space;p_{i_2}\\&space;\vdots\\&space;p_{i_N}&space;\end{vmatrix}" title="\vec{p_i} = \begin{vmatrix} p_{i_1}\\ p_{i_2}\\ \vdots\\ p_{i_N} \end{vmatrix}" /> where <img src="https://latex.codecogs.com/gif.latex?i&space;=&space;1,2,...,M" title="i = 1,2,...,M" /> indexed as each security in such portfolio

> **NOTE**: Historical prices data are subjected to readjustment for any purpose. We can define them as simple as the open or close prices for each time index, or to other customizable approaches, like setting it as the average between the two, etc...it is your choice to model the prices for whatever purposes that might follow your theoretical way on tackling the objective at hand, whether that be assessing assets' relationships or building predictive models.

We then proceed on defining the securities *logarithmic* returns vectors  as:

<img src="https://latex.codecogs.com/gif.latex?\vec{r_i}&space;=&space;\begin{vmatrix}&space;r_{i_1}\\&space;r_{i_2}\\&space;\vdots\\&space;r_{i_N}&space;\end{vmatrix}&space;=&space;\begin{vmatrix}&space;log(\frac{p_{i_2}}{p_{i_1}})\\&space;\\&space;log(\frac{p_{i_3}}{p_{i_2}})\\&space;\vdots\\&space;log(\frac{p_{i_N}}{p_{i_{N-1}}})&space;\end{vmatrix}" title="\vec{r_i} = \begin{vmatrix} r_{i_1}\\ r_{i_2}\\ \vdots\\ r_{i_N} \end{vmatrix} = \begin{vmatrix} log(\frac{p_{i_2}}{p_{i_1}})\\ \\ log(\frac{p_{i_3}}{p_{i_2}})\\ \vdots\\ log(\frac{p_{i_N}}{p_{i_{N-1}}}) \end{vmatrix}" /> 

>- Another way to calculate the interval returns can as straight forward as percentage change of one point to the next.
- We use log to smoothen the returns shape & minimize outliers. This can be good & bad, depending on the situation, as we will further explore.

We can construct a table of such defined prices & returns of *M* securities as an *N x M* matrix, where M columns as securities & N rows as time indexes, such that:
<img src="https://latex.codecogs.com/gif.latex?PRICE&space;=&space;\begin{vmatrix}&space;p_{1_1}&p_{2_1}&space;&\hdots&space;&p_{M_1}&space;\\&space;p_{1_2}&p_{2_2}&space;&\hdots&space;&p_{M_2}&space;\\&space;\vdots&\vdots&space;&\ddots&space;&\vdots&space;\\&space;p_{1_N}&p_{2_N}&space;&\hdots&space;&p_{M_N}&space;\end{vmatrix}" title="PRICE = \begin{vmatrix} p_{1_1}&p_{2_1} &\hdots &p_{M_1} \\ p_{1_2}&p_{2_2} &\hdots &p_{M_2} \\ \vdots&\vdots &\ddots &\vdots \\ p_{1_N}&p_{2_N} &\hdots &p_{M_N} \end{vmatrix}" /> , <img src="https://latex.codecogs.com/gif.latex?RET&space;=&space;\begin{vmatrix}&space;r_{1_1}&r_{2_1}&space;&\hdots&space;&r_{M_1}&space;\\&space;r_{1_2}&r_{2_2}&space;&\hdots&space;&r_{M_2}&space;\\&space;\vdots&\vdots&space;&\ddots&space;&\vdots&space;\\&space;r_{1_N}&r_{2_N}&space;&\hdots&space;&r_{M_N}&space;\end{vmatrix}" title="RET = \begin{vmatrix} r_{1_1}&r_{2_1} &\hdots &r_{M_1} \\ r_{1_2}&r_{2_2} &\hdots &r_{M_2} \\ \vdots&\vdots &\ddots &\vdots \\ r_{1_N}&r_{2_N} &\hdots &r_{M_N} \end{vmatrix}" />


Next, we ask the question on what are the most **optimal allocations**, which we define optimal differently depending on the situation, though primarily, we often regard to a portfolio with the optimal allocations to which maximizes a specific ratio that's widely being used in the investment world to evaluate performance of any portfolio/strategy/security:

>**Sharpe Ratio** (or Information Ratio as dubbed by Tulchinsky) = <img src="https://latex.codecogs.com/gif.latex?\frac{\bar{R_p}&space;-&space;R_f}{\sigma_p}" title="\frac{\bar{R_p} - R_f}{\sigma_p}" /> where, simplistically speaking,
<img src="https://latex.codecogs.com/gif.latex?\sigma_p" title="\sigma_p" /> = Portfolio's Volatility = Standard Deviation of Portfolio's Returns,
<img src="https://latex.codecogs.com/gif.latex?\bar{R_p}" title="\bar{R_p}" /> = Portfolio's Returns

We perceive Portfolio Allocations ~ Direction (long/short) + Magnitude (0-100% of available capital), where for the given *M* securities to allocate out capital towards, With 100% = 1 capital available , we define our allocations/weights as:
<img src="https://latex.codecogs.com/gif.latex?w&space;=&space;\begin{vmatrix}&space;w_1\\&space;w_2\\&space;\vdots\\&space;w_M&space;\end{vmatrix}" title="w = \begin{vmatrix} w_1\\ w_2\\ \vdots\\ w_M \end{vmatrix}" />  such that <img src="https://latex.codecogs.com/gif.latex?-1&space;\leq&space;w_{i}\leq&space;1" title="-1 \leq w_{i}\leq 1" /> and <img src="https://latex.codecogs.com/gif.latex?\sum_{1}^{M}w_i&space;=&space;1" title="\sum_{1}^{M}w_i = 1" /> 

> **IMPORTANT:** Both  <img src="https://latex.codecogs.com/gif.latex?\sigma_p" title="\sigma_p" /> & <img src="https://latex.codecogs.com/gif.latex?\bar{R_p}" title="\bar{R_p}" /> can be tweaked & incorporated with much more complexity, such as predictive data added, etc..There are a multitude of ways to compute & determine such value, to which we will further explore as many as we can in our pending research work.

For the sake of simplicity at the moment, performing analysis from the historical standpoint, we opt to first, calculate the individual portfolio's returns  ($$\vec{R_p}$$) for *N* time indexes, given allocations $$w$$, satisfying the condition defined above, for *M* amount of selected securities in such portfolio:

<img src="https://latex.codecogs.com/gif.latex?\vec{R_p}&space;=&space;RET\cdot&space;w&space;=&space;\begin{vmatrix}&space;r_{1_1}&r_{2_1}&space;&\hdots&space;&r_{M_1}&space;\\&space;r_{1_2}&r_{2_2}&space;&\hdots&space;&r_{M_2}&space;\\&space;\vdots&\vdots&space;&\ddots&space;&\vdots&space;\\&space;r_{1_N}&r_{2_N}&space;&\hdots&space;&r_{M_N}&space;\end{vmatrix}&space;\cdot&space;\begin{vmatrix}&space;w_1\\&space;w_2\\&space;\vdots\\&space;w_M&space;\end{vmatrix}&space;=&space;\begin{vmatrix}&space;R_{p_1}\\&space;R_{p_2}\\&space;\vdots\\&space;R_{p_N}&space;\end{vmatrix}" title="\vec{R_p} = RET\cdot w = \begin{vmatrix} r_{1_1}&r_{2_1} &\hdots &r_{M_1} \\ r_{1_2}&r_{2_2} &\hdots &r_{M_2} \\ \vdots&\vdots &\ddots &\vdots \\ r_{1_N}&r_{2_N} &\hdots &r_{M_N} \end{vmatrix} \cdot \begin{vmatrix} w_1\\ w_2\\ \vdots\\ w_M \end{vmatrix} = \begin{vmatrix} R_{p_1}\\ R_{p_2}\\ \vdots\\ R_{p_N} \end{vmatrix}" />

Then, we can compute <img src="https://latex.codecogs.com/gif.latex?\bar{R_p}" title="\bar{R_p}" /> as the average returns for N time indexes, being the most commonly used & basic way of calculating it from a historical analysis standpoint, such that <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\bar{R_p}&space;=&space;\frac{\sum_{1}^{N}R_{p_i}}{N}" title="\bar{R_p} = \frac{\sum_{1}^{N}R_{p_i}}{N}" />


For portfolio's volatility $$\sigma_p$$, being the standard deviation of $$R_p$$, we seek to assess this topic in a much more intricate way, as assets exhibit correlational properties, as well as their volatility relationships between each other, and to the market they are in. For any allocations to which we seek to determine to be optimal in our process of optimization, we first calculate the covariance matrix (<img src="https://latex.codecogs.com/gif.latex?\boldsymbol{C}" title="\boldsymbol{C}" />) for *M* selected securities, using the constructed returns table:

<img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;RET&space;=&space;\begin{vmatrix}&space;r_{1_1}&r_{2_1}&space;&\hdots&space;&r_{M_1}&space;\\&space;r_{1_2}&r_{2_2}&space;&\hdots&space;&r_{M_2}&space;\\&space;\vdots&\vdots&space;&\ddots&space;&\vdots&space;\\&space;r_{1_N}&r_{2_N}&space;&\hdots&space;&r_{M_N}&space;\end{vmatrix}&space;=&space;\begin{vmatrix}&space;r_{1}&space;&r_{2}&space;&\hdots&space;&r_{M}&space;\end{vmatrix}" title="RET = \begin{vmatrix} r_{1_1}&r_{2_1} &\hdots &r_{M_1} \\ r_{1_2}&r_{2_2} &\hdots &r_{M_2} \\ \vdots&\vdots &\ddots &\vdots \\ r_{1_N}&r_{2_N} &\hdots &r_{M_N} \end{vmatrix} = \begin{vmatrix} r_{1} &r_{2} &\hdots &r_{M} \end{vmatrix}" />

where <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;r_{i}&space;=&space;\begin{vmatrix}&space;r_{i_1}\\&space;r_{i2}\\&space;\vdots\\&space;r_{i_N}&space;\end{vmatrix}" title="r_{i} = \begin{vmatrix} r_{i_1}\\ r_{i2}\\ \vdots\\ r_{i_N} \end{vmatrix}" /> represents *N* returns of individual security i in M selected securities, such that we can compute the **average returns** <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\bar{r_{i}}" title="\bar{r_{i}}" /> which, for simplicity, defined as <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\bar{r_{i}}&space;=&space;\frac{\sum_{n=1}^{N}&space;r_{i_n}}{N}" title="\bar{r_{i}} = \frac{\sum_{n=1}^{N} r_{i_n}}{N}" /> (subjected to variation - weighted, rolling, etc...).

We can proceed on de-meaning columns of $$RET$$, obtaining::

<img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\overline{RET}&space;=&space;\begin{vmatrix}&space;(r_{1_1}&space;-&space;\bar{r_{1}})&(r_{2_1}&space;-&space;\bar{r_{2}})&space;&\hdots&space;&(r_{M_1}&space;-&space;\bar{r_{M}})&space;\\&space;(r_{1_2}&space;-&space;\bar{r_{1}})&space;&(r_{2_2}&space;-&space;\bar{r_{2}})&space;&\hdots&space;&(r_{M_2}&space;-&space;\bar{r_{M}})&space;\\&space;\vdots&\vdots&space;&\ddots&space;&\vdots&space;\\&space;(r_{1_N}&space;-&space;\bar{r_{1}})&(r_{2_N}&space;-&space;\bar{r_{2}})&space;&\hdots&space;&(r_{M_N}&space;-&space;\bar{r_{M}})&space;\end{vmatrix}" title="\overline{RET} = \begin{vmatrix} (r_{1_1} - \bar{r_{1}})&(r_{2_1} - \bar{r_{2}}) &\hdots &(r_{M_1} - \bar{r_{M}}) \\ (r_{1_2} - \bar{r_{1}}) &(r_{2_2} - \bar{r_{2}}) &\hdots &(r_{M_2} - \bar{r_{M}}) \\ \vdots&\vdots &\ddots &\vdots \\ (r_{1_N} - \bar{r_{1}})&(r_{2_N} - \bar{r_{2}}) &\hdots &(r_{M_N} - \bar{r_{M}}) \end{vmatrix}" />

We can now calculate our **sample** covariance matrix:

<img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\boldsymbol{C}&space;=&space;\frac{\overline{RET}^\top&space;\cdot&space;\overline{RET}}{N-1}&space;=&space;\begin{vmatrix}&space;\sigma_1^2&space;&&space;\sigma_{12}&space;&&space;\hdots&space;&&space;\sigma_{1M}&space;\\&space;\sigma_{21}&space;&&space;\sigma_2^2&space;&&space;\hdots&space;&&space;\sigma_{2M}\\&space;\vdots&space;&\vdots&space;&\ddots&space;&\vdots&space;\\&space;\sigma_{M1}&space;&&space;\sigma_{M2}&space;&\hdots&space;&\sigma_M^2&space;\end{vmatrix}" title="\boldsymbol{C} = \frac{\overline{RET}^\top \cdot \overline{RET}}{N-1} = \begin{vmatrix} \sigma_1^2 & \sigma_{12} & \hdots & \sigma_{1M} \\ \sigma_{21} & \sigma_2^2 & \hdots & \sigma_{2M}\\ \vdots &\vdots &\ddots &\vdots \\ \sigma_{M1} & \sigma_{M2} &\hdots &\sigma_M^2 \end{vmatrix}" />

where <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{C}" title="\boldsymbol{C}" /> is a *M x M* **square matrix**, such that:
- The diagonal values <img src="https://latex.codecogs.com/gif.latex?\sigma_i^2" title="\sigma_i^2" /> = **variances** of individual securities <img src="https://latex.codecogs.com/gif.latex?i&space;=&space;1,2,...,M" title="i = 1,2,...,M" />
- <img src="https://latex.codecogs.com/gif.latex?\sigma_{ij}" title="\sigma_{ij}" /> = <img src="https://latex.codecogs.com/gif.latex?\sigma_{ji}" title="\sigma_{ji}" /> = **covariances** between two different assets i & j, where <img src="https://latex.codecogs.com/gif.latex?i\neq&space;j" title="i\neq j" />

><img src="https://latex.codecogs.com/gif.latex?\sigma_{ij}&space;=&space;\sigma_{ji}" title="\sigma_{ij} = \sigma_{ji}" /> implies that  <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{C}" title="\boldsymbol{C}" /> is a *Symmetric*, or **Hermitian Matrix**. This is a very mathematically important fact that opens us to various options of exploration & analysis to which we will further explore in this series.

Thus, we proceed on calculating $$\sigma_p$$ using the covariance matrix <img src="https://latex.codecogs.com/gif.latex?\boldsymbol{C}" title="\boldsymbol{C}" />, given any allocation $$w$$ that satisfy the conditions above:

<img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\sigma_p&space;=&space;\sqrt{w^\top&space;\cdot&space;\boldsymbol{C}&space;\cdot&space;w}" title="\sigma_p = \sqrt{w^\top \cdot \boldsymbol{C} \cdot w}" />

---
### Summary & Conclusion
Putting them together, again, from the historical analysis standpoint, we want a portfolio with allocations $$w$$ for M securities, with their given (hourly/daily/weekly/etc) returns table $$RET$$, an *N x M* matrix, such that:
- <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\vec{R_p}&space;=&space;RET\cdot&space;w" title="\vec{R_p} = RET\cdot w" /> = Portfolio's individual returns through given N time indexes

- <img src="https://latex.codecogs.com/gif.latex?\dpi{110}&space;\boldsymbol{C}&space;=&space;\frac{RET^\top&space;\cdot&space;RET}{N}" title="\boldsymbol{C} = \frac{RET^\top \cdot RET}{N}" /> = Covariance Matrix (Hermitian) of M selected securities, calculated from N time indexes (different N's = different values)

Thus, with:
- <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\bar{R_p}&space;=&space;\frac{\sum_{1}^{N}R_{p_i}}{N}" title="\bar{R_p} = \frac{\sum_{1}^{N}R_{p_i}}{N}" /> = Portfolio's Average Returns (for simplicity)

- <img src="https://latex.codecogs.com/gif.latex?\dpi{120}&space;\sigma_p&space;=&space;\sqrt{w^\top&space;\cdot&space;\boldsymbol{C}&space;\cdot&space;w}" title="\sigma_p = \sqrt{w^\top \cdot \boldsymbol{C} \cdot w}" /> = Portfolio's Volatility/Standard Deviations of N Returns

- $$R_f$$ = Risk-Free Rate (usually set as returns rate of T-Bills, or any relatively risk-free assets, or 0 for simplicity)

Portfolio's Sharpe/Information Ratio = <img src="https://latex.codecogs.com/gif.latex?\dpi{150}&space;\frac{\frac{\sum_{1}^{N}R_{p_i}}{N}&space;-&space;R_f}{\sqrt{w^\top&space;\cdot&space;\boldsymbol{C}&space;\cdot&space;w}}" title="\frac{\frac{\sum_{1}^{N}R_{p_i}}{N} - R_f}{\sqrt{w^\top \cdot \boldsymbol{C} \cdot w}}" />

There are many potential applications we can have from understanding mean-variance analysis, especially portfolio **risk management**, as well as integrating predictive models & establishing **boundaries for optimization** exploited by computational power. Thus, this can help us figure out the optimal allocations to be holding at any given time, or even serving as a powerful overarching tool to **perform analysis on assets** using historical data, such as *industry analysis, correlational relationships, etc*. We can even utilize the individual terms/components of Portfolio's Sharpe Ratio to **design a custom system of steps & equations for optimization** to further seek mathematically unsolvable solutions, whereas such solutions can help in finding or evaluating reliable alpha signals (we will further explore this in later posts).

---
# Personal Remarks on Moving Forward
It is important to recognize that for the majority of cases, or all of them at a relative level, higher alpha is desired as alpha represents the independent (relative to the market) excess returns of a portfolio.  The important concepts we will keep as reminders are:
- The concept of alpha is a statistical measure, sacrificially constricted with some potential trade-offs in beta by the overarching market's risk, especially with lesser dependency on predictive features due to the further compounded risk of prediction/speculation in itself.
- There is no rewards without risks, as risk = volatility = movements in values to capitalize upon. No risk = $$R_f$$ as for any individual retail investors should have in their portfolio. We are seeking to understand such risk & placing ourselves such that we can generate superior returns.
- All potential securities in the market, each with their own risk measure comprises such market as a whole, the market to which we are setting as a benchmark.
- Constructing a portfolio means picking a selected N amount of securities WITHIN THE MARKET for your specific returns objective
- Generating returns from such portfolio by strategically allocating capital sizing & positioning of the selected securities, as well as adding & removing securities within such portfolio.
- Beta, as defined above, measures the sensitivity to market's movement of a certain security or constructed portfolio.

With that being said, as we will further dive deeper into subsequent topics & ways to tackle our objective of generate alphas, through data-driven research results, the following key-points are essential to consider in my personal approach on generating optimal returns when it comes to systematic investing & trading:
- A portfolio with high beta and low alpha (even negative alpha) can still be preferable compared to one with low beta & high alpha ([smart beta strategies](https://www.quantilia.com/smart-beta-strategies/ "smart beta strategies")), depending on market condition as well as such preference is subjected to the specific time for the cyclic market's nature.
- The "best performing" portfolio is the one that, on a fundamental level, generate, on average, the highest positive returns as consistent as it can, knowing WHEN and HOW to align itself with market's volatility/movement (increase beta magnitude) & when not to (decrease beta magnitude), such that such portfolio's measured alpha reaches its global maximum given stated condition(s).
- Prediction is an unavoidable component needed to adjust positions through time, such that we could satisfy the ladder objective, in able to maximize alpha in an absolute sense.
-  Quantifying reality to establish definitions of market's conditions is a gray area needed to be carefully treaded. In other words, we seek to approach the topic not just in a purely quantitative way, but rather in a quantitatively rigorous way that is built from & reflects the fundamental foundations of finance as a whole.
