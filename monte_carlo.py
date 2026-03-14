"""
Monte Carlo Option Pricing Simulator
=====================================
Simulates 1000+ stock price paths using Geometric Brownian Motion (GBM)
to price European call and put options via probability.

A Black-Scholes analytical solution is also provided for result validation.
"""

import math
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Black-Scholes analytical pricing (used for comparison / validation)
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Cumulative distribution function for the standard normal distribution."""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def black_scholes_price(
    spot: float,
    strike: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiry: float,
    option_type: str = "call",
) -> float:
    """
    Compute the Black-Scholes analytical price for a European option.

    Parameters
    ----------
    spot            : Current stock price (S0).
    strike          : Strike price (K).
    risk_free_rate  : Annualised risk-free interest rate (r), e.g. 0.05 for 5 %.
    volatility      : Annualised volatility (sigma), e.g. 0.20 for 20 %.
    time_to_expiry  : Time to expiry in years (T).
    option_type     : ``"call"`` or ``"put"``.

    Returns
    -------
    float
        Analytical Black-Scholes option price.
    """
    if time_to_expiry <= 0:
        if option_type == "call":
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)

    d1 = (
        math.log(spot / strike)
        + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry
    ) / (volatility * math.sqrt(time_to_expiry))
    d2 = d1 - volatility * math.sqrt(time_to_expiry)

    if option_type == "call":
        return spot * _norm_cdf(d1) - strike * math.exp(
            -risk_free_rate * time_to_expiry
        ) * _norm_cdf(d2)
    if option_type == "put":
        return strike * math.exp(
            -risk_free_rate * time_to_expiry
        ) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")


# ---------------------------------------------------------------------------
# Stock path simulation
# ---------------------------------------------------------------------------

def simulate_stock_paths(
    spot: float,
    risk_free_rate: float,
    volatility: float,
    time_to_expiry: float,
    num_paths: int = 10_000,
    num_steps: int = 252,
    random_seed: int | None = None,
) -> np.ndarray:
    """
    Simulate stock price paths using Geometric Brownian Motion (GBM).

    The discrete GBM update rule is::

        S(t+dt) = S(t) * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)

    where Z ~ N(0, 1) is a standard normal random variable.

    Parameters
    ----------
    spot            : Current stock price (S0).
    risk_free_rate  : Annualised risk-free interest rate (r).
    volatility      : Annualised volatility (sigma).
    time_to_expiry  : Time to expiry in years (T).
    num_paths       : Number of simulated paths (>= 1 000 recommended).
    num_steps       : Number of discrete time steps per path (default 252 ≈ trading days/year).
    random_seed     : Optional seed for reproducibility.

    Returns
    -------
    np.ndarray of shape ``(num_paths, num_steps + 1)``
        Each row is one simulated price path, starting at ``spot``.
    """
    if num_paths < 1:
        raise ValueError("num_paths must be >= 1")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if time_to_expiry <= 0:
        raise ValueError("time_to_expiry must be > 0")

    rng = np.random.default_rng(random_seed)

    dt = time_to_expiry / num_steps
    drift = (risk_free_rate - 0.5 * volatility ** 2) * dt
    diffusion = volatility * math.sqrt(dt)

    # Standard normal random shocks: shape (num_paths, num_steps)
    z = rng.standard_normal((num_paths, num_steps))
    log_returns = drift + diffusion * z  # shape (num_paths, num_steps)

    # Cumulative log returns, prepend 0 so path starts at spot
    cumulative = np.concatenate(
        [np.zeros((num_paths, 1)), np.cumsum(log_returns, axis=1)], axis=1
    )
    paths = spot * np.exp(cumulative)  # shape (num_paths, num_steps + 1)
    return paths


# ---------------------------------------------------------------------------
# Monte Carlo option pricer
# ---------------------------------------------------------------------------

class MonteCarloOptionPricer:
    """
    Price European options using Monte Carlo simulation.

    Parameters
    ----------
    spot            : Current stock price (S0).
    strike          : Strike price (K).
    risk_free_rate  : Annualised risk-free interest rate (r).
    volatility      : Annualised volatility (sigma).
    time_to_expiry  : Time to expiry in years (T).
    num_paths       : Number of simulated price paths (default 10 000).
    num_steps       : Time steps per path (default 252).
    random_seed     : Optional seed for reproducibility.
    """

    def __init__(
        self,
        spot: float,
        strike: float,
        risk_free_rate: float,
        volatility: float,
        time_to_expiry: float,
        num_paths: int = 10_000,
        num_steps: int = 252,
        random_seed: int | None = None,
    ) -> None:
        if spot <= 0:
            raise ValueError("spot must be > 0")
        if strike <= 0:
            raise ValueError("strike must be > 0")
        if volatility <= 0:
            raise ValueError("volatility must be > 0")
        if time_to_expiry <= 0:
            raise ValueError("time_to_expiry must be > 0")
        if num_paths < 1000:
            raise ValueError("num_paths must be >= 1000 for reliable MC estimates")

        self.spot = spot
        self.strike = strike
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.time_to_expiry = time_to_expiry
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.random_seed = random_seed

        # Simulate once; cache the paths for reuse.
        self._paths: np.ndarray = simulate_stock_paths(
            spot=self.spot,
            risk_free_rate=self.risk_free_rate,
            volatility=self.volatility,
            time_to_expiry=self.time_to_expiry,
            num_paths=self.num_paths,
            num_steps=self.num_steps,
            random_seed=self.random_seed,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def paths(self) -> np.ndarray:
        """Simulated stock price paths, shape ``(num_paths, num_steps + 1)``."""
        return self._paths

    @property
    def terminal_prices(self) -> np.ndarray:
        """Terminal (expiry) stock price for each path."""
        return self._paths[:, -1]

    def price(self, option_type: str = "call") -> dict:
        """
        Estimate the option price and associated statistics.

        The price is computed as the discounted expected payoff across all
        simulated paths::

            price = exp(-r * T) * mean(payoffs)

        Parameters
        ----------
        option_type : ``"call"`` or ``"put"``.

        Returns
        -------
        dict with keys:

        ``"price"``
            Monte Carlo option price estimate.
        ``"std_error"``
            Standard error of the estimate.
        ``"confidence_interval_95"``
            Tuple (lower, upper) 95 % confidence interval.
        ``"black_scholes_price"``
            Analytical Black-Scholes price for comparison.
        ``"num_paths"``
            Number of simulated paths.
        ``"option_type"``
            Option type string.
        ``"probability_in_the_money"``
            Fraction of paths that finished in the money.
        """
        option_type = option_type.lower()
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

        S_T = self.terminal_prices
        K = self.strike

        if option_type == "call":
            payoffs = np.maximum(S_T - K, 0.0)
            in_the_money = S_T > K
        else:
            payoffs = np.maximum(K - S_T, 0.0)
            in_the_money = S_T < K

        discount = math.exp(-self.risk_free_rate * self.time_to_expiry)
        discounted_payoffs = discount * payoffs

        mc_price = float(np.mean(discounted_payoffs))
        std_error = float(np.std(discounted_payoffs, ddof=1) / math.sqrt(self.num_paths))

        ci_lower = mc_price - 1.96 * std_error
        ci_upper = mc_price + 1.96 * std_error

        bs_price = black_scholes_price(
            spot=self.spot,
            strike=self.strike,
            risk_free_rate=self.risk_free_rate,
            volatility=self.volatility,
            time_to_expiry=self.time_to_expiry,
            option_type=option_type,
        )

        return {
            "price": mc_price,
            "std_error": std_error,
            "confidence_interval_95": (ci_lower, ci_upper),
            "black_scholes_price": bs_price,
            "num_paths": self.num_paths,
            "option_type": option_type,
            "probability_in_the_money": float(np.mean(in_the_money)),
        }

    def plot_simulation(self, num_paths_to_plot: int = 100, show: bool = True):
        """
        Plot a graphical representation of simulated stock-price paths.

        Parameters
        ----------
        num_paths_to_plot : Number of paths to draw on the chart.
        show              : If True, display the plot immediately.

        Returns
        -------
        tuple
            (figure, axis) for further customization if needed.
        """
        if num_paths_to_plot < 1:
            raise ValueError("num_paths_to_plot must be >= 1")

        selected = min(num_paths_to_plot, self.num_paths)
        x_axis = np.linspace(0.0, self.time_to_expiry, self.num_steps + 1)

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#111111")
        ax.set_facecolor("#111111")

        for i in range(selected):
            ax.plot(x_axis, self._paths[i], linewidth=1, alpha=0.55)

        ax.axhline(self.strike, color="#ff6666", linestyle="--", linewidth=1, alpha=0.8)
        ax.set_title("Monte Carlo Stock Price Simulation", color="white")
        ax.set_xlabel("Time (years)", color="white")
        ax.set_ylabel("Stock Price", color="white")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2)

        param_text = (
            f"Strike: {self.strike:.0f}\n"
            f"Volatility: {self.volatility:.1f}\n"
            f"Risk-free rate: {self.risk_free_rate:.2f}\n"
            f"T: {self.time_to_expiry:.1f}"
        )
        ax.text(
            0.02,
            0.95,
            param_text,
            transform=ax.transAxes,
            fontsize=9,
            color="white",
            verticalalignment="top",
            bbox=dict(facecolor="#222222", edgecolor="white", alpha=0.8),
        )

        fig.tight_layout()
        if show:
            plt.show()
        return fig, ax
