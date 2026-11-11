
import numpy as np
from scipy.stats import poisson


GOALS = np.arange(11) # possible goals bounded from 0-10 including edges


def score_probabilities(home_xg, away_xg, rho, ensemble_probs):
    # Calculate Dixon-Coles scoreline probabilities using Poisson
    home = poisson.pmf(GOALS, home_xg)
    away = poisson.pmf(GOALS, away_xg)

    scores = np.outer(home, away) # Combine probs into 11x11 matrix

    # Dixon-Coles correction for low-scoring matches
    scores[0, 0] *= 1 - home_xg * away_xg * rho
    scores[0, 1] *= 1 + home_xg * rho
    scores[1, 0] *= 1 + away_xg * rho
    scores[1, 1] *= 1 - rho

    if (scores < 0).any():
        raise ValueError("Invalid score probabilities")

    scores /= scores.sum() # Normalize matrix to total 100%

    # Separate home wins, draws and away wins w/ masks (matrix containing T or F vals)
    home_mask = np.tril(np.ones_like(scores), -1).astype(bool) # Below diagonal-> HG>AG
    draw_mask = np.eye(len(GOALS), dtype=bool) # Diagonal-> Draws (0-0, 1-1,...)
    away_mask = np.triu(np.ones_like(scores), 1).astype(bool) # Above diagonal-> AG>HG

    # Adjust matrix w/ our complete ensemble probabilities
    for mask, target in zip([home_mask, draw_mask, away_mask],ensemble_probs):
        current = scores[mask].sum() # add up prob of outcome

        if current <= 0:
            raise ValueError("Cannot adjust score probabilities")

        # Adjust scoreline: eg. DC homeWin=60% and Ensemble homeWin=70%-> factor=0.7/0.6
        scores[mask] *= target / current 

    return scores / scores.sum() # Normalize matrix


def simulate_match(scores, simulations=10000, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # Convert 11x11 matrix to 1 dimensional array
    probabilities = scores.flatten()
    # Randomly select scorelines based on probabilities, sim for 10000 and more likely scores appear more often
    outcomes = rng.choice(len(probabilities),size=simulations,p=probabilities)
    # Convert selected to scores: eg. if select [2,1] then home=2 and away=1
    home_goals, away_goals = np.unravel_index(outcomes, scores.shape)

    return home_goals, away_goals # arrays containing simulated goals


def main():
    # Manchester City vs Fulham from our current predictions
    ensemble = np.array([0.697, 0.173, 0.130])

    # Example expected goals for testing (not fitted model values)
    home_xg = 2.0
    away_xg = 0.9
    rho = -0.058

    scores = score_probabilities(home_xg, away_xg, rho, ensemble)

    home_goals, away_goals = simulate_match(scores, simulations=10000,rng=np.random.default_rng(42))

    print("Manchester City vs Fulham — 10,000 simulations")

    print(f"Home wins: {(home_goals > away_goals).mean():.2%}")
    print(f"Draws:     {(home_goals == away_goals).mean():.2%}")
    print(f"Away wins: {(home_goals < away_goals).mean():.2%}")

    print(f"\nAverage home goals: {home_goals.mean():.2f}")
    print(f"Average away goals: {away_goals.mean():.2f}")

    assert np.isclose(scores.sum(), 1)


if __name__ == "__main__":
    main()
