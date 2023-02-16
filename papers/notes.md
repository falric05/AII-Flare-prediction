# __Notes for project__

## __Stochastic differential equation__

A __stochastic differential equation__ (__SDE__) is a [differential equation](https://en.wikipedia.org/wiki/Differential_equation) in which one or more of the terms is a [stochastic process](https://en.wikipedia.org/wiki/Stochastic_process), resulting in a solution which is also a stochastic process. Typically, SDEs contain a variable which represents random white noise calculated as the derivative of Brownian motion or the [Wiener process](https://en.wikipedia.org/wiki/Wiener_process) (the one used by this project). However, other types of random behaviour are possible, such as jump processes. Random differential equations are conjugate to stochastic differential equations.

## __Autonomous Systems__

In mathematics, an __autonomous system__ or autonomous differential equation is a system of ordinary differential equations which does not explicitly depend on the independent variable. When the variable is time, they are also called time-invariant systems.

Many laws in physics, where the independent variable is usually assumed to be time, are expressed as autonomous systems because it is assumed the laws of nature which hold now are identical to those for any point in the past or future.

## __Wiener method__

In mathematics, the Wiener process is a real-valued continuous-time stochastic process named in honor of American mathematician Norbert Wiener and it has applications throughout the mathematical sciences.

The Wiener process $W_{t}$ is characterised by the following properties:

1. $W_0=0$
2. $W$ has independent increments: for every $t>0$, the future increments $W_{t+u}-W_{t}$, $u\geq 0$, are independent of the past values $W_s$,  $s<t$.
3. $W$ has Gaussian increments: $W_{t+u}-W_{t}$ is normally distributed with mean $0$ and variance $u$, $W_{t+u}-W_{t}\sim {\mathcal {N}}(0,u)$.
4. $W$ has continuous paths: $W_{t}$ is continuous in $t$.

## __Milstein method__

In mathematics, the __Milstein method__ is a technique for the approximate [numerical solution](https://en.wikipedia.org/wiki/Numerical_analysis) of a [stochastic differential equation](https://en.wikipedia.org/wiki/Stochastic_differential_equation) (SDE).

A __stochastic differential equation__ (__SDE__) is a differential equation in which one or more of the terms is a stochastic process, resulting in a solution which is also a stochastic process.

### __Description__

Consider the autonomous Itō stochastic differential equation:
$$dX_t=a(X_t)d_t+b(X_t)dW_t$$
with:

* initial condition $X_0=x_0$
* $W_t$ stands for [Wiener process](https://en.wikipedia.org/wiki/Wiener_process), and suppose that we wish to solve this SDE on some interval of time $[0, T]$.

Then the __Milstein approximation__ to the true solution $X$ is the _Markov chain_ $Y$ defined as follows:

* partition the interval $[0,T]$ into $N$ equal subintervals of width $\Delta t >0$:

$$0=\tau_0<\tau_1<...<\tau_N=T \hspace{0.5cm}\text{with } \tau_n:=n\Delta t \text{ and } \Delta t = \frac{T}{N}$$

* set $Y_0 = x_0$
* recursively define $Y_n$ for $1\leq n \leq N$, with $b'$ derivative of $b(x)$ with respect to $x$ and $\Delta W_n = W_{\tau_{n+1} - W_{\tau_n}}$ are independent and identically distributed normal random variables with expected value 0 and variance $\Delta t$, by:
$$Y_{n+1}=Y_n+a(Y_n)\Delta t + b (Y_n) \Delta W_n + \frac{1}{2} b (Y_n) b' (Y_n) \Big((\Delta W_n)^2 - \Delta t \Big)$$
Then $Y_n$ will approximate $X_{\tau_n}$ for $0 \leq n \leq N$, and increasing $N$ will yield a better approximation.
