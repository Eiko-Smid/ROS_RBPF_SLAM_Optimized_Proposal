# RBPF SLAM System with Optimized Proposal Distribution

## RBPF SLAM Algorithm

The core of this project is a **Rao-Blackwellized Particle Filter (RBPF) SLAM** algorithm with an optimized
proposal distribution.

In SLAM, the goal is to estimate the robot trajectory $x_{1:t}$ and the map $m$ from the control inputs
$u_{1:t}$ and sensor measurements $z_{1:t}$:

```math
p(x_{1:t}, m \mid z_{1:t}, u_{1:t})
```

The key idea of RBPF SLAM is to **factorize the posterior** into two separate parts. The first part is the
**trajectory estimation**, and the second part is the **map estimation** based on the estimated trajectory:

```math
p(x_{1:t}, m \mid z_{1:t}, u_{1:t})
=
p(m \mid x_{1:t}, z_{1:t})
\cdot
p(x_{1:t} \mid z_{1:t}, u_{1:t})
```

In this project, trajectory estimation is performed by a **particle filter**, while **Occupancy Grid Mapping**
is used to estimate the map. Each particle in the algorithm contains its own pose, map, and weight.

Conceptually, a particle is therefore a set:

```math
X_t^{[i]}
=
\left(
x_{1:t}^{[i]}, m_t^{[i]}, w_t^{[i]}
\right)
```

### Optimized Proposal Distribution

A standard particle filter can use the **motion model** directly as its **proposal distribution**:

```math
x_t^{[i]}
\sim
p(x_t \mid x_{t-1}^{[i]}, u_t)
```

This is simple, but it has an important disadvantage: the current laser measurement $z_t$ is **not considered
when proposing the new particle pose**. Only the odometry source is used. Since the odometry source is
**quite noisy**, the difference between the proposal and the actual **target distribution** can be large.

In this case, a **large number of samples** is needed to capture the meaningful area of the target distribution.
Many particles also end up in regions where the target distribution has a low value and therefore receive a
**low weight**. This wastes particles. Since every particle needs to be processed, it is also a **waste of
computational resources**.

A better proposal incorporates the current **measurement together with the odometry** information to find the
**meaningful area of the target distribution**:

```math
q(x_t)
\approx
p(x_t \mid x_{t-1}, u_t, z_t, m_{t-1})
```

The exact distribution is difficult to compute for the nonlinear scan-matching and occupancy-grid measurement
models used in this project. Instead, the proposal is **approximated locally by a Gaussian**:

```math
q(x_t)
\approx
\mathcal{N}(\mu_t, \Sigma_t)
```

For each particle, the proposal is estimated approximately as follows:

1. The **kinematic motion model** predicts the next pose from the wheel odometry.
2. The **ICP scan matcher** corrects this prediction using the current laser scan and the particle map from $t-1$.
3. A deterministic set of candidate poses is generated around the scan-matched pose.
4. The candidates are evaluated according to the product of the **measurement and motion models**.
5. Their weighted distribution is **approximated by a Gaussian** with mean.

For all $x_j \in \{x_1, \ldots, x_k\}$:

```math
\mu_t^{(i)}
=
\mu_t^{(i)}
+
p(z_t \mid m_{t-1}^{(i)}, x_j)
p(x_t \mid x_{t-1}^{(i)}, u_t)
```

```math
\eta^{(i)}
=
\eta^{(i)}
+
p(z_t \mid m_{t-1}^{(i)}, x_j)
p(x_t \mid x_{t-1}^{(i)}, u_t)
```

Normalize the mean:

```math
\mu_t^{(i)}
=
\frac{\mu_t^{(i)}}{\eta^{(i)}}
```

Finally, compute the covariance matrix.

For all $x_j \in \{x_1, \ldots, x_k\}$:

```math
\Sigma_t^{(i)}
=
\Sigma_t^{(i)}
+
(x_j - \mu_t)
\cdot
(x_j - \mu_t)^T
\cdot
p(z_t \mid m_{t-1}^{(i)}, x_j)
\cdot
p(x_t \mid x_{t-1}^{(i)}, u_t)
```

Normalize the covariance matrix:

```math
\Sigma_t^{(i)}
=
\frac{\Sigma_t^{(i)}}{\eta^{(i)}}
```

Finally, the new particle pose is sampled from the resulting Gaussian:

```math
x_t^{[i]}
\sim
\mathcal{N}(\mu_t, \Sigma_t)
```
