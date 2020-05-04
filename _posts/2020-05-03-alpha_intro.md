---
title: "QF Alpha-Research: Introduction"
date: 2020-05-03
tags: [alpha]
header:
  image: "/images/alpha_universe/alpha_banner.jpg"
excerpt: "Research works on methods of finding & picking individual assets to formulate an alpha universe for investment portfolios"
mathjax: "true"
---

# Preface
---
In this series of research on capturing alphas & developing alpha signals, whereas "alpha" plays a major part in almost all existing quantitative trading strategies out there (almost all leading firms have divisions focusing purely on capturing alphas), it is important to keep in mind that we are touching base on a very intricate and in-depth topic that drives the frontier of quant finance, pioneered and led by the smartest people in the world.

In the process of presenting my work & approach, I will try my best to shed light on relevant topics to which constitute the overarching development of alpha design, though not in details, as I will attach links for further readings on those relevant topics.

Please keep in mind that these works will be opened for continuous improvement and potential revisions in the upcoming future as I am just a sprouting young mathematician who has much more to learn during this era of pioneers & adventurers, following the footstep of the frontier explorers in such industry.

# Inspiration
When it comes to systematic investing utilizing quantitative approaches, one thing to always keep in mind is that models (or rules) developed is only reliable at a "certain time," to which such time variable "t" in terms of its period/frequency, can be as dynamic as the reliability of the model developed itself.

In another words, as said by Igor Tulchinsky, the founder of WorldQuant:

> "***An infinite number of possible rules describe reality, and we are always struggling to discover and refine them. Yet, paradoxically, there is only one rule that governs them all. That rule is: no rule ever works perfectly.***"

The decision to compose publication(s) of my personal findings through independent research was propelled after reading Tulchinsky's book on [*Finding Alphas*](https://www.amazon.com/Finding-Alphas-Quantitative-Approach-Strategies/dp/1119057868), especially after realizing that my mathematical approach drew significant parallel to the content of his book, though Tulchinsky has provided me the terminologies & credible academic foundations needed to not just affirmed my mathematical interpretations in tackling such topic, but also articulate my findings that could potentially contribute to the on-going research of capturing alphas in the world of quantitative finance.

# Foundation & Approach
*Brief coverage on basic foundational topics & key-points on personal approach & understanding towards alpha designs.*

---
### CAPM & Beta

In general finance & economics, there exists a commonly known model called the[Capital Asset Pricing Model](https://corporatefinanceinstitute.com/resources/knowledge/finance/what-is-capm-formula/ " Capital Asset Pricing Model") (CAPM) to which can be perceived to serve as the propelling root of cause for the concept of alpha:

$$R_a = R_f + \beta (R_m - R_f)$$

where:

$$R_a = $$  Expected Return of Security
$$R_m = $$ Expected Return of The Market
$$R_f = $$ Risk-free Rate
$$\beta = $$ Beta of Security

Note that from a CAPM perspective, we need to recognize that the "security" is our investment target, and that the "market" is the benchmark to which we try to evaluate our security expected returns in comparison 

**Beta**, by definition, is the security's "sensitivity" to market's risk/volatility movement. Although, it is better to understand it from a quantitative approach, when it comes to alpha, as we proceed on understanding Beta as the **"regressed slope of the security's returns with respect to the returns of the selected market/benchmark** (e.g.: 0.5 beta = *on average*, for every 1% returns on the market, our security returns at 0.5%). We will demonstrate this terminology further as we move forward.

In addition, notice we are using "expected returns" as this is important to proceed on understanding alpha design. Recall that since CAPM aims to model a security/portfolio, say AAPL, to a certain market as a benchmark to our choosing, say the SPY index, or even the XLK (Technology Index from SPDR) to compare AAPL with its competitors in the sector its operating in, to obtain its expected rate of returns in able to price equity investments.
>**REMARK**: This concept of modeling performance (return rate in this case) of a certain investment target with respect to another is the foundation to understand alpha design, as the ultimate goal of investing & trading in finance is about performing better than the market, absolute or not.

---
### Alpha
First, we will start with the textbook definition:
> **Alpha** (or [Jensen's Alpha](https://financetrain.com/jensens-alpha/ "Jensen's Alpha"))  is the risk-adjusted/excess returns of a portfolio, used to evaluate its returns performance with respect to its expected returns against certain selected benchmark (using CAPM model).

Jensen's $$\alpha = R_p - (R_f + \beta (R_m - R_f))$$
where $$R_p = $$ Portfolio's Return

This looks very similar to CAPM because it is, as it incorporates CAPM into the equation. As CAPM aims to evaluate the expected return of a certain target, Alpha is used to evaluate actual performance of such target, usually a portfolio, managed by certain individual, or financial firm (notice that $$R_p$$ is not an expected value but a realized/actual value)

It is important to highlight the definition of alpha being **risk-adjusted/excess returns** as this is the foundation to which we will use to proceed on understanding & designing alphas. On a mathematical standpoint, whereas beta is the slope, alpha is the **y-intercept** when regressing returns against a selected benchmark, or that alpha is the returns of a certain *investment target*  when the expected *benchmark*  returns 0% on average. In other words,

>**Alpha** is the **independent returns** of a portfolio with respect to a selected benchmark.

It is valuable to have high alpha as best as we can, such that statistically, given the definition of alpha, regardless of market's movement, bullish or bearish, even if it crashes, the alpha component of a given portfolio return will not be affected. This is the power of finding alphas in the investing world. 

---
# Personal Remarks & Important Key Points on Moving Forward
It is important to recognize that although for the majority of cases, or all of them at a relative level, higher alpha is desired as alpha represents the independent (relative to the market) excess returns of a portfolio. However, **That does NOT mean higher alpha means better performance on an absolute sense.** The important concepts we will keep as reminders are:
- There is no rewards without risks, as risk = volatility = movements in values to capitalize (if you want no risks, you should just keep your capital in T-bills, or even cash, as they are classified as the Risk-free rate - $$R_f$$)
- All potential securities in the market, each with their own risk measure comprises such market as a whole, the market to which we are setting as a benchmark.
- Constructing a portfolio means picking a selected N amount of securities WITHIN THE MARKET for your specific returns objective
- Generating returns from such portfolio by strategically allocating capital sizing & positioning of the selected securities, as well as adding & removing securities within such portfolio.
- Beta, as defined above, measures the sensitivity to market's movement of a certain security or constructed portfolio.

With that being said, as we will further explore the meanings behind beta & alpha through data-driven research results, the following key-points are essential to consider in my personal approach on generating optimal returns when it comes to systematic investing & trading:
- A portfolio with high beta and low alpha (even negative alpha) can still be preferable compared to one with low beta & high alpha ([smart beta strategies](https://www.quantilia.com/smart-beta-strategies/ "smart beta strategies"))
- The "best performing" portfolio is the one that, on a fundamental level, generate positive returns by knowing WHEN and HOW to align itself with market's volatility/movement (increase beta) & when not to (increase alpha).
-  Quantifying reality to establish definitions of market's condition is a gray are need to be carefully treaded. In other words, we seek to approach the topic not just in a purely quantitative way, but rather in a quantitatively rigorous way that reflect the fundamental foundation of finance as a whole.

