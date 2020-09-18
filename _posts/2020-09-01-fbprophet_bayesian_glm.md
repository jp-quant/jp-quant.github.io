In the recent years, Facebook released an open-source tool for Python & R, called [fbprophet](https://facebook.github.io/prophet/), allowing scientists & developers to not just tackle the complexity & non-linearity in time-series analysis, but also allow for a robust regression model-building process (linear & logistic) to forecast any time-series data while accounting for *uncertainty* in every defined variables (priors & posteriors).

 
After spending a couple weeks on my spare time reading Facebook's published [research paper](https://peerj.com/preprints/3190/) on the released forecaster, conducted by PhD mathematicians, as well as breaking down the codes from their [open-source repository](https://github.com/facebook/prophet/tree/master/python) (Python), I was able to understand, in details, the mathematics & computational executions, thus built my own time-series forecaster with additional complexities added, utilizing their mathematical foundations.

---
In this post, we will:

1. Mathematically understand the difference between **Ordinary v.s. Generalized** Linear Models (GLM) & why we are using GLM to build time-series forecasters.
2. Explore the concept of "going Bayesian" (Bayes' Theorem), the benefits from doing so & the extended philosophy behind it.
3. Break down the mathematics of FBProphet. 
4. Build our own version in Python with additional flexibility & creative concepts added, utilizing **PyMC3** instead of *Stan* (like fbprophet does) as backend sampler (since I am not yet super fluent in Stan by the time I write this post).
 
---
---

## 1. The Superiority of Generalized Linear Models
Linearity is the foundational language that explains the totality of composition for any reality we are trying to model, as regression becomes the basis for the overarching factor-modeling technique in the modern data-driven world, especially neural network models in Machine Learning.

  

However, the main difference between a regular ordinary linear regression model and a generalized one falls under the concept of **symmetry & normality**, a *theoretical* zero-sum framework towards reality as a whole (not as what we observed incrementally through time).

  

The intuition, as I summarize below, is best explained by this [Wikipedia page](https://en.wikipedia.org/wiki/Generalized_linear_model#Maximum_likelihood  "Wikipedia page"):

  

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

In short, a generalized linear model covers all possible ways of how different distributions of difference factors, abstract or real, can "hierarchically" impact the defined distribution of the observed. This allows us to tackle complex time-series problems, especially ones that exhibit non-linearity, while retain uncertainty in our models, as well as  not having to worry about data stationarity that classical time-series models, such as ARIMA, heavily rely on


---
## 2. Going Bayesian

All Bayesian techniques & implementations in modern days, even in machine learning neural networks, are built from the beautiful statistical foundation pioneered by Thomas Bayes himself, back in the late 1700s, called **Bayes Theorem**:
$$ P(A \mid B) = \frac{P(A) P(B \mid A)}{P(B)}$$
> **Interesting Fact**: Bayes never officially publish such theorem but rather assumed it was intuitive. It was Laplace who later on wrote about it when, after stumbling on Bayes' publication, utilizing the theorem in his own mathematical work


This approach of modeling variables, both the priors & posteriors, as distributions have not been heavily explored & implemented back then due to high computational demands. However, our accelerating technological advancement has allowed Bayesians to find themselves a vital role for data modelling approaches in modern days, especially in building neural network models.

Though I can spend time writing in details on the applications of such simple, yet powerful, concept of Bayesian Statistics, there exist many informational & captivating explanations already conducted by much more credible individuals than me. I **highly encourage** checking them out even if you already know the mathematics, since *it is not about just knowing what and how to use it, but also when & why we are using it*: 

> - [**Bayes Theorem**](https://www.youtube.com/watch?v=HZGCoVF3YvM&t=528s&ab_channel=3Blue1Brown) *by 3Blue1Brown*
> - [**The Bayesian Trap**](https://www.youtube.com/watch?v=R13BD8qKeTg&ab_channel=Veritasium) *by Veritasium*
> - [**Demystifying Bayesian Deep Learning**](https://www.youtube.com/watch?v=s0S6HFdPtlA&t=906s&ab_channel=PyData) by Eric J. Ma (PyData)*

---
Recalling from above:
> **Generalized Linear Models** allows for response variables that have arbitrary distributions, and for an arbitrary function of the response variable to vary linearly with the predicted values (rather than assuming that the response itself must vary linearly)

Shortly put in details, implementation of Bayesian Statistics on time-series GLM is powerful due to:

1. The ability for us to define **priors** (initial beliefs) as *any distributions* $P(A), P(B),...$, whereas such priors $A, B, ...$ are variables/features that can be as abstract or realistic as we want.

2. We can formulate complex functions (link functions), as $f(A,B,...)$ with defined priors variables $A,B,...$ to the **posteriors**, being the observed, to which such observed values modeled as either a factor of predicted values, or predicted values themselves, *are also defined as a distribution.*

3. We can then update such priors with the arrival of new observed data, using Bayes' Theorem, specifically the concept of *conditional probability* $P(Observed \mid X)$ *(where $X$ being a conditioned variable, either as a prior or a function of a group of priors)*

We will demonstrate these implementations in the upcoming sections.
To summarize:
- The **observable is the unknown** posterior, to which **dependent conditionally on the defined priors** beliefs, to which such **priors are updated to "fit" the observable** when new observable data arrive

The main **philosophy** behind Bayesian Statistics is that:
> Our understanding of the **reality** we are trying to model through time, to predict its future values, **is never static & objective, but rather dynamic & conditional**. Going Bayesian means we are accepting that **we will never know the full picture of reality completely, and that we can only infer from the data of such reality we collected so far to forecast its future values with compounded uncertainty**, as per defining all of the parameters & functions of our entire model as distributions.

---
## 3. The Mathematics behind FBProphet
The overarching formula is:
### $$\boldsymbol{Y}(t) \sim [\boldsymbol{G}(t) \cdot(1 + \boldsymbol{S}_{m}(t)) +  \boldsymbol{S}_{a}(t)] \pm \boldsymbol{\epsilon}_{t}$$

where, all as **tensors**  :
>$\boldsymbol{Y}$ = Observed to fit & predict
> $\boldsymbol{G}$ = Trend/Growth
> $\boldsymbol{S}_{m}$ = Multiplicative Seasonal Components
> $\boldsymbol{S}_{a}$ = Additive Seasonal Components
> $\boldsymbol{\epsilon}$ = Unknown Errors (e.g.:  $\sigma$ of the observed)

---
Modeling Trend/Growth $\boldsymbol{G}$
---
Without worrying about their meanings at the moment, We first define **3 priors**:
$$ k \sim \mathcal{N}(0,\theta)$$

$$ m \sim \mathcal{N}(0,\theta) $$

$$\delta \sim Laplace(0,\tau)$$

where **$\theta$ and $\tau$ being the scales** of the priors' distributions (or simply $\sigma_G$ measuring the deviation of such priors). This is viewed as hyper-parameters for tuning with cross-validation (employed by fbprophet) or any other custom tuning methods.

As default, set by fbprophet, we will opt with:
$$\theta = 5$$

$$\tau = 0.05$$

The effect of priors' scaling values will be demonstrated in our work later on, as well as an extended creative idea on defining scales as priors themselves, although for now, we stick with them being as default constants.

We now explore their relative meanings & dimensions in our trend model:
- $k$ =  1-Dimensional Growth Rate
- $m$ = 1-Dimensional Growth Offset
- $\delta$ = N-Dimensional Growth Rate Changepoints Adjustments

With the priors, fbprophet's model use them to calculate **3 types of trends**:



<!--stackedit_data:
eyJoaXN0b3J5IjpbMTgzNTk1NDU1MSwtNTYyMjQ2NjE0LC0xMj
k4Nzg5MTI0LDQ4MjQxNDkyNCwxMTA2NDYwNDE1LDE3MzA0Mjc3
OTEsMTA5ODk5MzQ0MCwtMTY0MDU1NzgyNSwtOTk1MzU0NzQ4LC
0xMTM1NzIwNTQwXX0=
-->