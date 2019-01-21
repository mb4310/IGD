## Implicit Gradient Descent

In this repository we implement implicit forms of common gradient descent algorithms available in pyTorch 1.0 and compare performance to traditional (explicit) SGD methods. For a detailed discussion of the theoretical advantages of implicit gradient descent, see [this paper](http://faculty.chicagobooth.edu/workshops/econometrics/PDF%202016/ptoulis_ISGD.pdf). To paraphrase, a key finding is that whereas in explicit methods the learning rate schedule needs to be carefully adjusted to balance statistical efficiency and numerical stability, with implicit methods the stability constraint effectively vanishes. Effectively, any learning rate (or sequence thereof) yields a stable procedure allowing for higher rates which yields faster convergence. On the other hand as we shall see computing the update is computationally expensive and in many cases intractable, so in this repo we experiment to find situations in which the trade-off is beneficial. 

A little background: an update rule for gradient descent typically looks something like:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;$x_{n&plus;1}&space;=&space;x_n&space;-&space;\delta_n&space;\nabla&space;F(x_{n})$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{100}&space;$x_{n&plus;1}&space;=&space;x_n&space;-&space;\delta_n&space;\nabla&space;F(x_{n})$" title="$x_{n+1} = x_n - \delta_n \nabla F(x_{n})$" /></a>

where  <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;$x_0$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{100}&space;$x_0$" title="$x_0$" /></a>  is given. 

The implicit update is instead:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;$x_{n&plus;1}&space;=&space;x_n&space;-&space;\delta_n&space;\nabla&space;F(x_{n&plus;1})$" target="_blank"><img src="https://latex.codecogs.com/png.latex?\dpi{100}&space;$x_{n&plus;1}&space;=&space;x_n&space;-&space;\delta_n&space;\nabla&space;F(x_{n&plus;1})$" title="$x_{n+1} = x_n - \delta_n \nabla F(x_{n+1})$" /></a>

where here we choose use a first-order expansion to approximate <a href="https://www.codecogs.com/eqnedit.php?latex=$\nabla&space;F(x_{n&plus;1})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\nabla&space;F(x_{n&plus;1})" title="$\nabla F(x_{n+1})" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=$\nabla&space;F(x_{n&plus;1})&space;\approx&space;\nabla&space;F(x_n)&space;-&space;H_F(x_{n})(x_{n&plus;1}&space;-&space;x_n)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$\nabla&space;F(x_{n&plus;1})&space;\approx&space;\nabla&space;F(x_n)&space;-&space;H_F(x_{n})(x_{n&plus;1}&space;-&space;x_n)" title="$\nabla F(x_{n+1}) \approx \nabla F(x_n) - H_F(x_{n})(x_{n+1} - x_n)" /></a>

where <a href="https://www.codecogs.com/eqnedit.php?latex=$H_F(\cdot)$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$H_F(\cdot)$" title="$H_F(\cdot)$" /></a> denotes the Hessian of F. Now we can simply rearrange to find our update rule:

<a href="https://www.codecogs.com/eqnedit.php?latex=$x_{n&plus;1}&space;=&space;x_n&space;-&space;\delta_n&space;(I&space;&plus;&space;\delta_n&space;H_F(x_n))^{-1}&space;\nabla&space;F(x_n)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$x_{n&plus;1}&space;=&space;x_n&space;-&space;\delta_n&space;(I&space;&plus;&space;\delta_n&space;H_F(x_n))^{-1}&space;\nabla&space;F(x_n)" title="$x_{n+1} = x_n - \delta_n (I + \delta_n H_F(x_n))^{-1} \nabla F(x_n)" /></a>

In our investigation we restrict our attention to compositions of affine functions, bounded non-linearities and ReLU (MLP, CNN and RNN) and so we can approximate well the Hessian without computing second-order partials of loss w.r.t parameters (see background.pdf for details). This enables us to utilize pyTorch auto-differentiation tools, and the increase in workload from explicit method is due exclusively to needed to invert a square matrix of size MxM where M is the number of parameters. In addition, rather than straight-forward SGD update we also implement alternative update-rules (such as implicit ADAM). 

Early experiments suggests the procedure pays dividends on certain tasks when we deal with number of parameters on the order of tens of thousands but no more. While it is true most modern CNN/RNN architectures have order of millions (or more) parameters, so certainly these methods cannot train such architecture from scratch, it is common today (popularized for example by Jeremy Howard and his [courses @ fast.ai](https://course.fast.ai/) to take pre-trained models, attach custom (e.g problem-specific) heads (with order of tens of thousand parameters), freeze earlier parameteres and train the head separately. We feel these approaches may prove useful to this sort of paradigm. 

