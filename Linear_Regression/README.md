# Linear Regression and Gradient-Based Supervised Learning

This repository presents supervised learning projects developed to study and implement neural network training from first principles, with emphasis on gradient-based optimization methods.

---

# Supervised Learning via Gradient Descent

A classical problem in applied mathematics and engineering is the curve-fitting problem, in which a model must be adjusted to approximate experimental observations obtained from a physical system.

Consider a dataset composed of input-output pairs:

$$
(x_i, y_i), \quad i = 1, \dots, N
$$

The objective is to determine a model $f$ such that

$$
f(x_i) \approx y_i
$$

To quantify the discrepancy between predictions and observed data, we define a cost function. A widely used choice is the Mean Squared Error (MSE), given by

$$
J = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - f(x_i) \right)^2
$$

The learning problem consists of minimizing $J$ with respect to the model parameters. If the model depends on parameters $z_1, z_2, \dots, z_\ell$, we write:

$$
J(z_1, z_2, \dots, z_\ell) = 
\frac{1}{N} \sum_{i=1}^{N} 
\left( y_i - f(x_i; z_1, z_2, \dots, z_\ell) \right)^2
$$

The minimization is performed through gradient descent, which updates parameters iteratively according to

$$
z_i^{(k+1)} = z_i^{(k)} - \mu \nabla_{z_i} J
$$

where:

- $k$ denotes the iteration (or epoch),
- $\mu$ is the learning rate,
- $\nabla_{z_i} J$ is the partial derivative of the cost with respect to parameter $z_i$.

The negative sign reflects the minimization objective, ensuring movement in the direction of steepest descent.

---

# Linear Regression

In the case of simple linear regression, the model is defined as

$$
f(x_i) = a x_i + b
$$

The goal is to determine parameters $a$ and $b$ that minimize

$$
J(a,b) = \frac{1}{N} \sum_{i=1}^{N} 
\left( y_i - (a x_i + b) \right)^2
$$

## Gradient Computation

The partial derivative with respect to $a$ is:

$$
\nabla_a J =
\frac{\partial}{\partial a}
\left(
\frac{1}{N}
\sum_{i=1}^{N}
(y_i - (a x_i + b))^2
\right)
$$

Applying linearity of differentiation and the chain rule:

$$
\nabla_a J =
-\frac{2}{N}
\sum_{i=1}^{N}
x_i \left( y_i - (a x_i + b) \right)
$$

Similarly, for $b$:

$$
\nabla_b J =
-\frac{2}{N}
\sum_{i=1}^{N}
\left( y_i - (a x_i + b) \right)
$$

## Parameter Update Rule

The gradient descent updates are therefore:

$$
\begin{cases}
a^{(k+1)} = a^{(k)} - \mu \nabla_a J \\
b^{(k+1)} = b^{(k)} - \mu \nabla_b J
\end{cases}
$$

These equations define the iterative learning dynamics of the linear model.

---

# Multiple Linear Regression

The linear model can be extended to multiple input variables. Suppose each observation is represented by a vector:

$$
\mathbf{x}_i = (x_{i1}, x_{i2}, \dots, x_{iM})
$$

The model becomes:

$$
f(\mathbf{x}_i) =
\sum_{j=1}^{M} w_j x_{ij} + \theta
$$

where:

- $w_j$ are the weights,
- $\theta$ is the bias term.

The cost function is:

$$
J(\mathbf{w}, \theta) =
\frac{1}{N}
\sum_{i=1}^{N}
\left(
y_i -
\left(
\sum_{j=1}^{M} w_j x_{ij} + \theta
\right)
\right)^2
$$

---

## Gradient with Respect to the Weights

For a given weight $w_k$:

$$
\nabla_{w_k} J =
-\frac{2}{N}
\sum_{i=1}^{N}
x_{ik}
\left(
y_i -
\left(
\sum_{j=1}^{M} w_j x_{ij} + \theta
\right)
\right)
$$

## Gradient with Respect to the Bias

$$
\nabla_{\theta} J =
-\frac{2}{N}
\sum_{i=1}^{N}
\left(
y_i -
\left(
\sum_{j=1}^{M} w_j x_{ij} + \theta
\right)
\right)
$$

---

## Iterative Learning Rule

The parameter updates are given by:

$$
\begin{cases}
w_k^{(n+1)} = w_k^{(n)} - \mu \nabla_{w_k} J \\
\theta^{(n+1)} = \theta^{(n)} - \mu \nabla_{\theta} J
\end{cases}
$$

where:

- $n$ denotes the iteration index,
- $k$ identifies the corresponding weight.

These equations characterize the supervised learning process for the multivariable linear regression model under gradient descent optimization.
