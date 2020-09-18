In the recent years, Facebook released an open-source tool for Python & R, called [fbprophet](https://facebook.github.io/prophet/), allowing scientists & developers to not just tackle the complexity & non-linearity in time-series analysis, but also allow for a robust regression model-building process (linear & logistic) to forecast any time-series data while accounting for *uncertainty* in every defined variables (priors & posteriors).

 
After spending a couple weeks on my spare time reading Facebook's published [research paper](https://peerj.com/preprints/3190/) on the released forecaster, conducted by PhD mathematicians, as well as breaking down the codes from their [open-source repository](https://github.com/facebook/prophet/tree/master/python) (Python), I was able to understand, in details, the mathematics & computational executions, thus built my own time-series forecaster with additional complexities added, utilizing their mathematical foundations.

In this post, we will:

1. Mathematically understand the difference between **Ordinary v.s. Generalized** Linear Models (GLM) & why we are using GLM.
2. Explore the concept of "going Bayesian" (Bayes' Theorem), the benefits from doing so & the extended philosophy behind it.
3. Break down the mathematics of FBProphet. 
4. Build our own version in Python with additional flexibility & creative concepts added, utilizing **PyMC3** instead of *Stan* (like fbprophet does) as backend.
 
---
## 1. The Superiority of Generalized Linear Models
Linearity is the foundational language that explains the totality of composition for any reality we are trying to model, as regression becomes the basis for the overarching factor-modeling technique in the modern data-driven world, especially neural network models in Machine Learning.

  

However, the main difference between a regular ordinary linear regression model and a generalized one falls under the concept of **symmetry & normality**, a *theoretical* zero-sum framework towards reality.

  

The intuition, as I summarize below, is best explained from [Wikipedia](https://en.wikipedia.org/wiki/Generalized_linear_model#Maximum_likelihood  "Wikipedia page"):

  

---

**Ordinary linear** regression predicts Y, the expected value of a given unknown quantity (the response variable, a random variable), as a linear combination of a set of observed values X's (predictors):

  

$$Y = \sum \alpha_{i} X_{i}$$

  

- This is **appropriate when the response variable can vary indefinitely in either direction**.
- When asked about the distribution of such predicted values, we only need $\mu$ and $\sigma$ to describe the symmetric property of its deviation.

  

However this is **not generally true when tackling problems from real-world data**, as such data typically are *not* normally distributed, as many exhibit certain properties aside from just skewness & kurtosis, but rather fatter tails or positively bounded, etc. For example (by Wiki):

  

>  -  *Suppose a linear prediction model learns from some data (perhaps primarily drawn from large beaches) that a 10 degree temperature decrease would lead to 1,000 fewer people visiting the beach. This model is unlikely to generalize well over different sized beaches.*

  

**Generalized linear** models cover all these situations by allowing for response variables that have arbitrary distributions (rather than simply normal distributions), and for an arbitrary function of the response variable (the link function) to vary linearly with the predicted values (rather than assuming that the response itself must vary linearly). For example (by Wiki):

> The case above of predicted number of beach attendees would typically be modeled with a Poisson distribution and a log link, while the case of predicted probability of beach attendance would typically be modeled with a Bernoulli distribution (or binomial distribution, depending on exactly how the problem is phrased) and a log-odds (or logit) link function.

  

---

In short, a generalized linear model covers all possible ways of how different distributions of difference factors can "hierarchically" impact the defined distribution of the observed. This allows us to tackle complex time-series problems, especially ones that exhibit non-linearity, while not having to worry about data stationarity that classical time-series models, such as ARIMA, heavily rely on.


---
## 2. Going Bayesian

All Bayesian techniques & implementations in modern days, even in machine learning neural networks, are built from the beautiful statistical foundation pioneered by Thomas Bayes himself, back in the late 1700s, called **Baye's Theorem**:
$$ P(A \mid B) = \frac{P(A) P(B \mid A)}{P(B)}$$

This approach of modeling variables, both the priors & posteriors, as distributions have not been heavily explored & implemented back then due to high computational demands. However, our accelerating technological advancement has allowed Bayesians to pave themselves a vital role for data modelling approaches in modern days, especially in building neural network models.

Without having to shed light much on the applications of such simple, yet powerful, concept, below are video links explaining the meaning & power of Bayesian Statistics. I **highly encourage** checking them out even if you already know the mathematics, since it is not about just knowing what and how to use it, but when & why we are using it: 

- [Bayes Theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM&t=528s&ab_channel=3Blue1Brown)  *by 3Blue1Brown*
- [The Bayesian Trap](https://www.youtube.com/watch?v=R13BD8qKeTg&ab_channel=Veritasium) *by Veritasium*




  



<!--stackedit_data:
eyJoaXN0b3J5IjpbLTk0OTU5MDg1MywtOTk1MzU0NzQ4LC0xMT
M1NzIwNTQwXX0=
-->