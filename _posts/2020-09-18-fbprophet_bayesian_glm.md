---
title: "Time-Series Forecasting: FBProphet & Going Bayesian with Generalized Linear Models (GLM)"
date: 2020-09-18
tags: [research, development, time-series, predictive-modeling]
header:
  image: "/images/qf_glm_banner.PNG"
excerpt: "Building time-series forecaster as hierarchichal GLM bayesian models with PyMC3 as backend sampler, inspired by Facebook's open-source powerful tool for R & Python, called fbprophet"
mathjax: "true"
layout: single
classes: wide
---


In the recent years, Facebook released an open-source tool for Python & R, called [fbprophet](https://facebook.github.io/prophet/), allowing scientists & developers to not just tackle the complexity & non-linearity in time-series analysis, but also allow for a robust regression model-building process, to forecast any time-series data while accounting for *uncertainty* in every defined variables (priors & posteriors) of any built models.

 
After spending a couple weeks on my spare time reading Facebook's published [research paper](https://peerj.com/preprints/3190/) on fbprophet, conducted by PhD mathematicians, as well as breaking down the codes from their [open-source repository](https://github.com/facebook/prophet/tree/master/python) (Python), I was able to understand, in details, the mathematics & computational executions, thus built my own time-series forecaster with additional complexities added, utilizing their mathematical foundations.

---
As an attempt of mine to explain the model in more applicable details, and alternatively recreate it with further implementations added, in this post, we will:

1. Mathematically understand the difference between **Ordinary v.s. Generalized** Linear Models (GLM) & why we are using GLM to build time-series forecasters.
2. Explore the concept of "going Bayesian" (Bayes' Theorem), the benefits from doing so & the extended philosophy behind it.
3. Break down the mathematics of FBProphet. 
4. Build our own version in Python with additional flexibility & creative concepts added, utilizing *PyMC3* instead of *Stan* (like fbprophet does) as backend sampler (since I am not yet super fluent in Stan by the time I write this post).
 

---

## 1. The Superiority of Generalized Linear Models
Linearity is the foundational language that explains the totality of composition for any reality we are trying to model, as regression becomes the basis for the overarching factor-modeling technique in the modern data-driven world, especially neural network models in Machine Learning.

  

However, the main difference between a regular ordinary linear regression model and a generalized one falls under the concept of **symmetry & normality**, a *theoretical* zero-sum framework towards reality as a whole (not as what we observed incrementally through time).

  

The intuition, as I summarize below, is best explained by this [Wikipedia page](https://en.wikipedia.org/wiki/Generalized_linear_model#Maximum_likelihood  "Wikipedia page"):

  

---

**Ordinary linear** regression predicts Y, the expected value of a given unknown quantity (the response variable, a random variable), as a linear combination of a set of observed values X's (predictors):

  

$$Y = \sum \alpha_{i} X_{i}$$

  

- This is **appropriate when the response variable can vary indefinitely in either direction**.
- When asked about the distribution of such predicted values, we only need $$\mu$$ and $$\sigma$$ to describe the symmetric property of its deviation.

  

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


This approach of **modeling variables**, both the priors & posteriors, **as distributions** have not been heavily explored & implemented back then due to high computational demands. However, our accelerating technological advancement has allowed Bayesians to find themselves a vital role for data modelling approaches in modern days, especially in building neural network models.

Though I can spend time writing in details on the applications of such simple, yet powerful, concept of Bayesian Statistics, there exist many informational & captivating explanations already conducted by much more credible individuals than me. I **highly encourage** checking them out even if you already know the mathematics, since *it is not about just knowing what and how to use it, but also when & why we are using it*: 

> - [**Bayes Theorem**](https://www.youtube.com/watch?v=HZGCoVF3YvM&t=528s&ab_channel=3Blue1Brown) *by 3Blue1Brown*
> - [**The Bayesian Trap**](https://www.youtube.com/watch?v=R13BD8qKeTg&ab_channel=Veritasium) *by Veritasium*
> - [**Demystifying Bayesian Deep Learning**](https://www.youtube.com/watch?v=s0S6HFdPtlA&t=906s&ab_channel=PyData) [Extra] by Eric J. Ma (PyData)*

---
Let's start by recalling the brief explanation about GLM from above:
> **Generalized Linear Models** allows for response variables that have arbitrary distributions, and for an arbitrary function of the response variable to vary linearly with the predicted values (rather than assuming that the response itself must vary linearly)

Shortly put in details, implementation of Bayesian Statistics on time-series GLM is powerful due to:

1. The ability for us to define **priors** (initial beliefs) as *any distributions* $$P(A), P(B),...$$, whereas such priors $$A, B, ...$$ are variables/features that can be as abstract or realistic as we want.

2. We can formulate functions (link functions), as $$f(A,B,...)$$ with defined priors variables $$A,B,...$$, as transformed variables, to define the **posteriors**, being the observed, to which such observed values modeled as either a factor of predicted values, or predicted values themselves, *are also defined as a distribution.*

3. We can then update such priors with the arrival of new observed data, using Bayes' Theorem, specifically the concept of *conditional probability* $$P(Observed \mid X)$$ *(where $$X$$ being a conditioned variable, either as a prior or a function of a group of priors)*

We will demonstrate these implementations in the upcoming sections.
To summarize:
- The **observable is the unknown** posterior, to which **dependent conditionally on the defined priors** beliefs, to which such **priors are updated to "fit" the observable** when new observable data arrive throughout time (hence variable t).

The main **philosophy** behind Bayesian Statistics is that:
> Our understanding of the **reality** we are trying to model through time, to predict its future values, **is never static & objective, but rather dynamic & conditional**. Going Bayesian means we are accepting that **we will never know the full picture of reality completely, and that we can only infer from the data of such reality we collected so far to forecast its future values with compounded uncertainty**, as per defining all of the parameters & functions of our entire model hierarchically as distributions.

---
## 3. The Mathematics behind FBProphet

> **Disclosure**: The content below is somewhat a detailed summary, or rather a concise alternative explanation based on my personal understanding of fbprophet's GLM. If you want to check out the original published paper, click [here](https://peerj.com/preprints/3190/).

---
Starting with the model's overarching formula:

### $$\boldsymbol{Y}(t) \sim [\boldsymbol{G}(t) \cdot(1 + \boldsymbol{S}_{m}(t)) +  \boldsymbol{S}_{a}(t)] \pm \boldsymbol{\epsilon}_{t}$$

where, as **tensors** (except $$\boldsymbol{\epsilon}$$):

- $$\boldsymbol{Y}$$ = Observable to fit & predict (data of prediction target)

- $$\boldsymbol{G}$$ = Trend/Growth

- $$\boldsymbol{S}_{m}$$ = Multiplicative Seasonal Components

- $$\boldsymbol{S}_{a}$$ = Additive Seasonal Components

- $$\boldsymbol{\epsilon}$$ = Unknown Errors (set as $$\sigma$$ of the observed by fbprophet)


----
### Scaling timestamps to $$\boldsymbol{t}$$ (for "time-series"?)
---
As we are obviously trying to build a predictive model on time-series data, which under our assumptions moving forward, being all real numbers, or, simply put, such data are numeric data (integers & floats). As time is basically the essence of our model-building, before we touch base on any components of our model, **we need to define a numeric transformation on a given array of timestamp instances (they are not numbers), that, the aftermath result from such transformation, sortedly retains the periodicity & frequency of the original sorted array of timestamps given**.

There are many ways of approaching this while avoiding look-ahead bias. Some can define it as the integers field. I personally opt for the same method fbprophet employs, scaling it directly through min-max standardization, into a "Gaussian-like" bound (as I like to call it):

Given such array $$D$$ containing $$N$$ amount of timestamp instances,

$$D = \begin{bmatrix} d_1 & d_2  &\cdots  & d_n \end{bmatrix}$$

Algorithmically speaking (summarized):
- When fitting, we perform standardization on such array $$D$$, view as $$D_fit$$, to obtain $$\boldsymbol{t}$$ as a numeric array of values between (0,1), such that $$\boldsymbol{t} = \frac{D - min(D)}{max(D) - min(D)}$$. Notice how such transformation cancels out our units and left us with purely numeric values, while capturing information of the timeframe we are working with.
- When predicting, we use the fitted $$min(D)$$ & $$max(D)$$ values, aka the $$min(D_fit)$$ & $$max(D_fit)$$ above, to perform the exact same scaling procedure on any given array $$D$$, to which in predictive context viewed as $$D_pred$$. The resulted $$\boldsymbol{t}$$ values that are out of bound (0,1) represents stamps before (<0) or after (>1) the timeframe of data we fitted (the priors of our model on). 

---
### Modeling Trend [$$\boldsymbol{G}(t)$$]
---
Without worrying about their meanings at the moment, we first define **3 essential priors** for our trend model:

$$ k \sim \mathcal{N}(0,\theta)$$

$$ m \sim \mathcal{N}(0,\theta) $$

$$\delta \sim Laplace(0,\tau)$$

where **$$\theta$$ and $$\tau$$ being the scales** of the priors' distributions (or simply $$\sigma_G$$ measuring the deviation of such priors). This is viewed as hyper-parameters for tuning with cross-validation (employed by fbprophet) or any other custom tuning methods.

As default, set by fbprophet:

$$\theta = 5$$

$$\tau = 0.05$$

The effect of priors' scaling values will be demonstrated in our work later on, as well as an extended creative idea on defining scales as priors themselves, although for now, we stick with them being as default constants.

We now explore their relative meanings & dimensions in our trend model:

- $$\boldsymbol{k}$$ [*1-Dimensional*] = Growth Rate
- $$\boldsymbol{m}$$ [*1-Dimensional*] = Growth Offset
- $$\boldsymbol{\delta}$$ [*N-Dimensional*] = Growth Rate Changepoints Adjustments

Notice how while $$k$$ and $$m$$ are 1 dimensional, or simply as constants, $$\delta$$ is an $$N$$ dimensional variable, where such *integer* value $$N$$ is also a hyper-parameter, though not as important as scales (as advised by Facebook), for tuning.
> Our $$\delta$$ here is somewhat similar to the commonly known concept in mathematics called *Dirac Delta* in differential equation, used to tackle problems with piece-wise regressions & step-functions. 

Before finalizing our trend model, we also need to define a couple last components, although **these will NOT be as priors with distributions needed to be sampled for fit** but rather **most of which are transformed variables**, being **calculation results using the defined priors & hyper-parameters** above. These are:

$$\boldsymbol{s}, A,\gamma$$

> For every given $$\boldsymbol{t}$$ as the **numeric timesteps** of $$K$$ dimensional length, meaning there being K amount of timestamps given to fit or predict,

> $$\boldsymbol{t} =\begin{bmatrix} t_1 & t_2  &\cdots  & t_k \end{bmatrix}$$

> and $$\delta$$ of $$N$$ dimensional length, representing $$N$$ amount of changepoints occurring at values in $$\boldsymbol{t}$$, we subsequently compute those N changepoint values from $$\boldsymbol{t}$$, defined as $$\boldsymbol{s}$$, being N-dimensional as well, such that:

> For $$i = 1,2,...N$$, where $$s_i \in \boldsymbol{t}$$ and $$N \leq K$$, we define

> $$\boldsymbol{s} =\begin{bmatrix} s_1 & s_2  &\cdots  & s_n \end{bmatrix}$$

> We then compute the $$K$$ x $$N$$ matrix $$A$$, called the **Determining Matrix**, with **boolean entries as binaries** (1 = True, 0 = False):

> $$A_t = \begin{bmatrix} t_{1} \geq s_1  & t_{1} \geq s_2  & \dots  & t_{1} \geq s_n \\ t_{2} \geq s_1  & t_{2} \geq s_2  & \dots  & t_{2} \geq s_n \\ \vdots & \vdots & \ddots & \vdots \\ t_{k} \geq s_1  & t_{k} \geq s_2 & \dots & t_{k} \geq s_n \end{bmatrix}$$

> Lastly, from $$\delta$$ being the **changepoints adjustment for the growth rate**, we define a transformed variable:

> $$\gamma = -s \delta$$

> Where $$\gamma$$ being the **changepoints adjustment for the growth offset**.

---
Now, with all the defined components to model our $$\boldsymbol{G}(t)$$, we proceed on using them to calculate **3 types of trends**:

**Linear Trend** (mainly used)
> **$$ G(\boldsymbol{t}) = (k + A_{\boldsymbol{t}} \delta) \boldsymbol{t} + (m + A_{\boldsymbol{t}} \gamma) $$**

**Logistic Trend** (Non-Linear & Saturating Growth Applications)
> $$G(\boldsymbol{t}) = \frac{C(\boldsymbol{t})}{1 + exp[-(k + A_{\boldsymbol{t}} \delta)(\boldsymbol{t} - (m + A_{\boldsymbol{t}} \gamma))]}$$

> Where $$C =$$ cap/maximum for logistic convergences/saturating point(s), can be given or include in our model to fit.

**Flat Trend** (for simplicity)
> $$G(\boldsymbol{t}) = m \vec{\boldsymbol{1}_{\boldsymbol{t}}}$$

> No changepoints incorporated, defined purely with a constant linear trend value as prior $$m$$ (or $$k$$) following the distribution $$\mathcal{N}(0,\theta)$$, or $$\mathcal{N}(0,5)$$ by default. 

---

### Trend Model (with changepoints $$\delta$$) Demonstration & Example

> **Remarks**: Following with our technical demonstrative work below, we will only showcase the effect of *Linear Trend*, as it's being the default & conventional approach in most cases. Though in short, we primarily seek to elucidate the importance of our defined priors & their transformed variables, the roles they play thus how, together, they construct our trend model as a component in the overarching GLM.

---

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
```

```python
# GIVEN & ASSIGNED
t = np.arange(1000) #---| timesteps (though as integers for demonstration, this is scaled between (0,1) while fitting in fbprophet)
n_changepoints = 25 #---| establish the amount of changepoints in our timesteps

# PRIORS
k = 1 #---| fixed as constant for demonstration
m = 5 #---| fixed as constant for demonstration
delta = np.random.normal(size=n_changepoints) #----| Changepoints Adjustment for Growth Rate (k-term)


# TRANSFORMED
s = np.sort(np.random.choice(t, n_changepoints, replace=False)) #---| n_changepoints from such timesteps
A = (t[:, None] > s) * 1 #---| Determining Matrix (*1 turns booleans into 1 & 0)
gamma = -s * delta #----| Changepoints Adjustment for Growth Offset (m-term)


# FINALIZE
growth = (k + np.dot(A,delta)) * t #---| Growth Rate (k-term)
offset = m + np.dot(A,gamma) #---| Growth Offset (m-term)
trend = growth + offset

# PLOT
plt.figure(figsize=(16, 9))
#----| 3 subplots indexing
n = 310
i = 0
for t, f in zip(['Trend = Growth Rate + Growth Offset','Growth Rate', 'Growth Offset'],
                [trend, growth, offset]):
    i += 1
    plt.subplot(n + i)
    plt.title(t)
    plt.yticks([])
    plt.vlines(s, min(f), max(f), lw=0.75, linestyles='--')
    plt.plot(f)
```

<img src="https://jp-quant.github.io/images/glm_bayesian/demo_1.png">

Notice how the full Trend as an addition of Growth Rate & Offsets is basically a piece-wise regression of such pair of components, where the defined $$N$$ amount of changepoints (n_changepoints) resulted in our $$\delta$$ dictating the multiplicative magnitude of our two terms at those $$N$$ specific changepoints in numeric timesteps $$\boldsymbol{t}$$.

[MORE IN PROGRESS AS OF 9-18-2020]