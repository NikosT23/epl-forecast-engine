
import numpy as np
from scipy.stats import poisson


GOALS = np.arange(11)


def score_probabilities(home_xg, away_xg, rho, ensemble_probs):
    # Calculate Dixon-Coles scoreline probabilities
    home = poisson.pmf(GOALS, home_xg)
    away = poisson.pmf(GOALS, away_xg)

    scores = np.outer(home, away)

    # Dixon-Coles correction for low-scoring matches
    scores[0, 0] *= 1 - home_xg * away_xg * rho
    scores[0, 1] *= 1 + home_xg * rho
    scores[1, 0] *= 1 + away_xg * rho
    scores[1, 1] *= 1 - rho

    if (scores < 0).any():
        raise ValueError("Invalid score probabilities")

    scores /= scores.sum()

    # Separate home wins, draws and away wins
    home_mask = np.tril(np.ones_like(scores), -1).astype(bool)
    draw_mask = np.eye(len(GOALS), dtype=bool)
    away_mask = np.triu(np.ones_like(scores), 1).astype(bool)

    # Adjust each outcome to match our ensemble probabilities
    for mask, target in zip(
        [home_mask, draw_mask, away_mask],
        ensemble_probs,
    ):
        current = scores[mask].sum()

        if current <= 0:
            raise ValueError("Cannot adjust score probabilities")

        scores[mask] *= target / current

    return scores / scores.sum()


def simulate_match(scores, simulations=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # Randomly select scorelines based on their probabilities
    probabilities = scores.flatten()

    outcomes = rng.choice(
        len(probabilities),
        size=simulations,
        p=probabilities,
    )

    home_goals, away_goals = np.unravel_index(outcomes, scores.shape)

    return home_goals, away_goals


def main():
    # Manchester City vs Fulham from our current predictions
    ensemble = np.array([0.697, 0.173, 0.130])

    # Example expected goals for testing (not fitted model values)
    home_xg = 2.0
    away_xg = 0.9
    rho = -0.058

    scores = score_probabilities(
        home_xg, away_xg, rho, ensemble
    )

    home_goals, away_goals = simulate_match(
        scores, simulations=10000,
        rng=np.random.default_rng(42),
    )

    print("Manchester City vs Fulham — 10,000 simulations")

    print(f"Home wins: {(home_goals > away_goals).mean():.2%}")
    print(f"Draws:     {(home_goals == away_goals).mean():.2%}")
    print(f"Away wins: {(home_goals < away_goals).mean():.2%}")

    print(f"\nAverage home goals: {home_goals.mean():.2f}")
    print(f"Average away goals: {away_goals.mean():.2f}")

    assert np.isclose(scores.sum(), 1)


if __name__ == "__main__":
    main()
