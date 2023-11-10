from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from torch.utils import data
import pdb

def alpha_fn(pi, lambda_):
    return (pi * lambda_) ** -1 + 1.0 - lambda_**-1


def beta_fn(pi, lambda_):
    return lambda_ * (pi) ** -1 + 1.0 - lambda_


def f_mu_linear(x, t, u):
    mu = (
        (2 * t - 1) * x
        + (2.0 * t - 1)
        # - 2 * np.sin((4 * t - 2) * x)
        # - (theta * u - 2) # * (1 + 0.5 * x)
        + u
    )
    return mu


def f_mu(x, t, u, theta=4.0):
    mu = (
        (2 * t - 1) * x
        + (2.0 * t - 1)
        - 2 * np.sin((4 * t - 2) * x)
        - (theta * u - 2) * (1 + 0.5 * x)
        + u
    )
    return mu


def complete_propensity(x, u, gamma, beta=0.75, effective_conf=1.0):
    logit = beta * x + 0.5
    nominal = (1 + np.exp(-logit)) ** -1
    alpha = alpha_fn(nominal, gamma)
    beta = beta_fn(nominal, gamma)
    a = 1 / beta
    b = 1 / alpha
    t_x = (nominal - a) / (b - a)
    ind = u > t_x
    putative = a * ind + b * (1 - ind)
    p = np.random.binomial(1, p=effective_conf, size=nominal.size)
    return (1 - p) * nominal.reshape(-1) + p * putative.reshape(-1)


class Synthetic(data.Dataset):
    def __init__(
        self,
        num_examples_obs,
        num_examples_rct,
        gamma_star,
        effective_conf,
        num_groups=1,
        mode="pi",
        p_u="uniform",
        theta=4.0,
        beta=0.75,
        sigma_y=0.1,
        domain=2.0,
        domain_rct=1.0,
        seed=1331,
        linear=True
        # split=None,
    ):
        super(Synthetic, self).__init__()
        rng = np.random.default_rng(seed=seed)
        self.num_examples = num_examples_obs
        self.num_examples_rct = num_examples_rct
        self.dim_input = 1
        self.dim_treatment = 1
        self.dim_output = 1
        self.x = rng.uniform(-domain, domain, size=(num_examples_obs, 1)).astype("float32")
        self.linear = linear
        self.gamma_star = gamma_star
        self.effective_conf = effective_conf
        self.num_groups = num_groups

        if p_u == "uniform":
            self.u = rng.uniform(size=(num_examples_obs, 1)).astype("float32")
        else:
            raise NotImplementedError(f"{p_u} is not a supported distribution")

        self.pi = (
            complete_propensity(
                x=self.x, u=self.u, gamma=gamma_star, beta=beta, effective_conf=self.effective_conf
            )
            .astype("float32")
            .ravel()
        )

        self.t = rng.binomial(1, self.pi).astype("float32")
        eps = (sigma_y * rng.normal(size=self.t.shape)).astype("float32")
        if self.linear:
            self.mu0 = f_mu_linear(x=self.x, t=0.0, u=self.u).astype("float32").ravel()
            self.mu1 = f_mu_linear(x=self.x, t=1.0, u=self.u).astype("float32").ravel()
        else:
            self.mu0 = f_mu(x=self.x, t=0.0, u=self.u, theta=theta).astype("float32").ravel()
            self.mu1 = f_mu(x=self.x, t=1.0, u=self.u, theta=theta).astype("float32").ravel()

        self.y0 = self.mu0 + eps
        #self.y0 = np.random.binomial(n=1,p=(self.y0 - self.y0.min())/(self.y0.max()-self.y0.min()))
        self.y1 = self.mu1 + eps
        #self.y1 = np.random.binomial(n=1,p=(self.y1 - self.y1.min())/(self.y1.max()-self.y1.min()))
        self.y = self.t * self.y1 + (1 - self.t) * self.y0
        self.tau = self.mu1 - self.mu0
        if mode == "pi":
            self.inputs = self.x
            self.targets = self.t
        elif mode == "mu":
            self.inputs = np.hstack([self.x, np.expand_dims(self.t, -1)])
            self.targets = self.y
        else:
            raise NotImplementedError(
                f"{mode} not supported. Choose from 'pi'  for propensity models or 'mu' for expected outcome models"
            )
        self.y_mean = np.array([0.0], dtype="float32")
        self.y_std = np.array([1.0], dtype="float32")

        group_assigner = assign_groups(num_groups, domain)
        self.s = group_assigner(self.x)

        if domain_rct > domain:
            raise ValueError("The RCT domain must be smaller than the observational domain")
        self.rct = RCT(
            num_examples=num_examples_rct,
            domain=domain_rct,
            sigma_y=sigma_y,
            theta=theta,
            linear=linear,
            seed=seed,
            group_assigner=group_assigner,
        )

        if self.rct.s is not None:
            self.s_rct = np.unique(self.rct.s)
        self.s_obs = np.unique(self.s)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index : index + 1]

    def tau_fn(self, x):
        return f_mu(x=x, t=1.0, u=1.0, theta=0.0) - f_mu(x=x, t=0.0, u=1.0, theta=0.0)


def assign_groups(num_groups: int, domain: float) -> Callable[..., np.ndarray]:
    """
    Assigns elements to groups based on their value.

    Args:
        num_groups (int): The number of groups to divide the elements into.
        domain (float): The range of values to consider for grouping.

    Returns:
        Callable[..., np.ndarray]: A function that assigns elements to groups based on their value.
    """

    def inner(X: np.ndarray) -> np.ndarray:
        """
        Assigns elements in `X` to groups based on their value.

        Args:
            X (np.ndarray): The list of elements to assign to groups.

        Returns:
            np.ndarray: A list of group indices indicating the group each element belongs to.

        Raises:
            ValueError: If any value in `X` is larger than `inner.domain` or smaller than `-inner.domain`.
        """

        # Check if any value in X is out of range
        if any(num > inner.domain or num < -inner.domain for num in X):
            raise ValueError("Value out of range")

        # Calculate the group size
        range_value = inner.domain - (-inner.domain)
        group_size = range_value / inner.num_groups

        # Assign elements to groups based on their value
        group_indices = [int((num - (-inner.domain)) // group_size) for num in X]

        return np.array(group_indices)

    inner.num_groups = num_groups
    inner.domain = domain

    return inner


@dataclass
class RCT:
    """
    Represents a Randomized Controlled Trial (RCT) dataset.
    """

    num_examples: int
    domain: float = 1.0
    sigma_y: float = 0.1
    theta: float = 4.0
    linear: bool = True
    seed: int = 1331
    group_assigner: Optional[Callable] = None
    x: np.ndarray = field(init=False)
    t: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    pi: np.ndarray = field(init=False)
    mu0: np.ndarray = field(init=False)
    mu1: np.ndarray = field(init=False)
    y0: np.ndarray = field(init=False)
    y1: np.ndarray = field(init=False)
    tau: np.ndarray = field(init=False)
    s: Optional[np.ndarray] = field(init=False, default=None)

    def __post_init__(self):
        self.create_data(self.seed)

    def create_data(self, seed) -> None:
        """
        Generates the data for the RCT dataset.
        """
        rng = np.random.default_rng(seed=seed)
        self.u = rng.uniform(size=(self.num_examples, 1)).astype(np.float32)
        self.x = rng.uniform(-self.domain, self.domain, size=(self.num_examples, 1)).astype(
            np.float32
        )
        self.pi = np.full((self.num_examples,), 0.5).astype("float32").ravel()
        self.t = rng.binomial(1, self.pi).astype(np.float32)
        eps = (self.sigma_y * rng.normal(size=self.t.shape)).astype(np.float32)

        if self.linear:
            self.mu0 = f_mu_linear(x=self.x, t=0.0, u=self.u).astype(np.float32).ravel()
            self.mu1 = f_mu_linear(x=self.x, t=1.0, u=self.u).astype(np.float32).ravel()
        else:
            self.mu0 = f_mu(x=self.x, t=0.0, u=self.u, theta=self.theta).astype(np.float32).ravel()
            self.mu1 = f_mu(x=self.x, t=1.0, u=self.u, theta=self.theta).astype(np.float32).ravel()

        self.y0 = self.mu0 + eps
        self.y1 = self.mu1 + eps
        self.y = self.t * self.y1 + (1 - self.t) * self.y0
        self.tau = self.mu1 - self.mu0

        if self.group_assigner is not None:
            self.s = self.group_assigner(self.x)
