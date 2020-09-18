# Introduction & Inspiration
In the recent years, Facebook released an open-source tool for Python & R, called [fbprophet](https://facebook.github.io/prophet/), allowing scientists & developers to not just tackle the complexity & non-linearity in time-series analysis, but also allow for a robust model-building process to forecast any time-series data while accounting for *uncertainty* in every step of the way.

 
After spending a couple weeks on my spare time reading Facebook's published [research paper](https://peerj.com/preprints/3190/) on the released forecaster, conducted by PhD mathematicians, as well as breaking down the codes from their [open-source repository](https://github.com/facebook/prophet/tree/master/python) (Python), I was able to understand, in details, the mathematics & computational executions, thus built my own time-series forecaster with additional complexities added, utilizing their GLM mathematical foundations.

In this post, we will:

1. Mathematically understand the difference between **Ordinary v.s. Generalized** Linear Models.
1. Explore the concept of "going Bayesian" (Probabilistic Modelling) & the benefits from doing so.
2. Break down the mathematics of FBProphet's. 
3. Build our own version in Python with additional flexibility & creative concepts added, utilizing *PyMC3* instead of *Stan* (like fbprophet does) as backend sampling.


---
## 1. Bayes' Theorem & Going Bayesian
Linearity is the foundational language that explains the totality of composition for any reality we are trying to model, as linear regression becomes the basis for the overarching factor-modeling technique in the modern data-driven world, especially neural network models in Machine Learning.

  

However, the main difference between an ordinary linear regression model and a generalized one falls under the concept of **symmetry** and **normality**, a theoretical zero-sum framework towards reality.

  

The intuition, as I summarize below, is best explained through this Wiki [page](https://en.wikipedia.org/wiki/Generalized_linear_model#Maximum_likelihood  "Wikipedia page"):

  

---

**Ordinary linear** regression predicts Y, the expected value of a given unknown quantity (the response variable, a random variable), as a linear combination of a set of observed values X's (predictors):

  

$$Y = \sum \alpha_{i} X_{i}$$

  

- This is **appropriate when the response variable can vary indefinitely in either direction**, thus when asked about the distribution of such predicted values, we only need $\mu$ and $\sigma$ to describe the symmetric property of its deviation.

  

However this is **not generally true** for the observables when tackling real-world problems. Most real-world data are *not* normally distributed on an absolute sense. For example (by Wiki):

  

>  -  *Suppose a linear prediction model learns from some data (perhaps primarily drawn from large beaches) that a 10 degree temperature decrease would lead to 1,000 fewer people visiting the beach. This model is unlikely to generalize well over different sized beaches.*

  

**Generalized linear** models cover all these situations by allowing for response variables that have arbitrary distributions (rather than simply normal distributions), and for an arbitrary function of the response variable (the link function) to vary linearly with the predicted values (rather than assuming that the response itself must vary linearly). For example (by Wiki):

> The case above of predicted number of beach attendees would typically be modeled with a Poisson distribution and a log link, while the case of predicted probability of beach attendance would typically be modeled with a Bernoulli distribution (or binomial distribution, depending on exactly how the problem is phrased) and a log-odds (or logit) link function.

  

---

In short, a generalized linear model covers all possible ways of how different distributions of difference factors can "hierarchically" impact the defined distribution of the observed, hence the Bayesian approach, built from the amazing yet simple **Baye's Theorem**:

  

$$ P(A \mid B) = \frac{P(A) P(B \mid A)}{P(B)}$$

  

This allows us to tackle complex time-series problems, especially ones that exhibit non-linearity, while not having to worry about data stationarity that classical models, such as ARIMA, heavily rely on.

<!--stackedit_data:
eyJoaXN0b3J5IjpbMzc0MTQ3MzU5XX0=
-->