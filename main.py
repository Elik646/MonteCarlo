"""
Monte Carlo Option Pricing Simulator – Demo
============================================
Run this script to see the simulator in action.

Usage
-----
    python main.py

Example output (values will differ due to randomness unless a seed is fixed):

    ============================================================
     Monte Carlo Option Pricing Simulator
    ============================================================
    Parameters
    ----------
    Spot price          : $100.00
    Strike price        : $105.00
    Risk-free rate      :   5.00 %
    Volatility          :  20.00 %
    Time to expiry      :   1.00 years
    Simulated paths     : 10,000
    Time steps / path   : 252
    ------------------------------------------------------------
    CALL option
    MC price            :   $8.02   (± 0.09 std-err)
    95 % CI             : [ $7.85 ,  $8.19 ]
    Black-Scholes price :   $8.02
    P(in-the-money)     :  45.23 %
    ------------------------------------------------------------
    PUT option
    MC price            :   $8.11   (± 0.05 std-err)
    95 % CI             : [ $8.01 ,  $8.20 ]
    Black-Scholes price :   $8.13
    P(in-the-money)     :  54.77 %
    ============================================================
"""

from monte_carlo import MonteCarloOptionPricer


def _print_result(result: dict) -> None:
    label = result["option_type"].upper()
    ci_lo, ci_hi = result["confidence_interval_95"]
    print(f"\n  {label} option")
    print(f"  MC price            : ${result['price']:>8.2f}   (± {result['std_error']:.2f} std-err)")
    print(f"  95 % CI             : [${ci_lo:>7.2f} , ${ci_hi:>7.2f} ]")
    print(f"  Black-Scholes price : ${result['black_scholes_price']:>8.2f}")
    print(f"  P(in-the-money)     : {result['probability_in_the_money'] * 100:>6.2f} %")
    print("  " + "-" * 58)


def main() -> None:
    # ------------------------------------------------------------------ #
    # Example parameters – feel free to adjust these.                     #
    # ------------------------------------------------------------------ #
    spot = 100.0           # Current stock price ($)
    strike = 105.0         # Strike price ($)
    risk_free_rate = 0.05  # Risk-free rate (5 %)
    volatility = 0.20      # Volatility (20 %)
    time_to_expiry = 1.0   # Time to expiry (1 year)
    num_paths = 10_000    # Number of simulated paths
    num_steps = 252       # Daily steps over one trading year
    random_seed = 42      # Fixed seed for reproducibility

    print("\n" + "=" * 62)
    print("  Monte Carlo Option Pricing Simulator")
    print("=" * 62)
    print("  Parameters")
    print("  ----------")
    print(f"  Spot price          : ${spot:.2f}")
    print(f"  Strike price        : ${strike:.2f}")
    print(f"  Risk-free rate      : {risk_free_rate * 100:>6.2f} %")
    print(f"  Volatility          : {volatility * 100:>6.2f} %")
    print(f"  Time to expiry      : {time_to_expiry:>6.2f} years")
    print(f"  Simulated paths     : {num_paths:,}")
    print(f"  Time steps / path   : {num_steps:,}")
    print("  " + "-" * 58)

    pricer = MonteCarloOptionPricer(
        spot=spot,
        strike=strike,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        time_to_expiry=time_to_expiry,
        num_paths=num_paths,
        num_steps=num_steps,
        random_seed=random_seed,
    )

    _print_result(pricer.price("call"))
    _print_result(pricer.price("put"))

    print("\n" + "=" * 62)
    pricer.plot_simulation()


if __name__ == "__main__":
    main()
