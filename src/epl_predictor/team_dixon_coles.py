
import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import poisson


GOALS = np.arange(11)


def fit_team_model(train):
    """Train team attack, defense and home advantage."""

    teams = sorted(set(train["HomeTeam"]) | set(train["AwayTeam"]))
    team_ids = {team: i for i, team in enumerate(teams)}
    n = len(teams)

    home = train["HomeTeam"].map(team_ids).to_numpy()
    away = train["AwayTeam"].map(team_ids).to_numpy()

    home_goals = train["FTHG"].to_numpy()
    away_goals = train["FTAG"].to_numpy()

    def loss(params):
        attack = params[:n]
        defense = params[n:2 * n]
        home_advantage = params[-1]

        home_xg = np.exp(home_advantage + attack[home] - defense[away])
        away_xg = np.exp(attack[away] - defense[home])

        # Poisson negative log likelihood (constant terms omitted)
        likelihood = np.sum(
            home_xg - home_goals * np.log(home_xg)
            + away_xg - away_goals * np.log(away_xg)
        )

        # Prevent extreme team ratings
        penalty = 10 * np.sum(attack**2 + defense**2)

        return likelihood + penalty

    initial = np.zeros(2 * n + 1)
    initial[-1] = np.log(1.2)

    bounds = [(-1.5, 1.5)] * (2 * n) + [(-0.5, 1.0)]

    result = minimize(
        loss,
        initial,
        method="L-BFGS-B",
        bounds=bounds,
    )

    if not result.success:
        raise RuntimeError(f"Training failed: {result.message}")

    attack = result.x[:n]
    defense = result.x[n:2 * n]
    home_advantage = result.x[-1]

    # Expected goals for training matches
    home_xg = np.exp(home_advantage + attack[home] - defense[away])
    away_xg = np.exp(attack[away] - defense[home])

    # Find Dixon-Coles rho using training matches only
    best_rho = 0.0
    best_loss = float("inf")

    for rho in np.linspace(-0.08, 0.08, 81):
        correction = np.ones(len(train))

        mask = (home_goals == 0) & (away_goals == 0)
        correction[mask] = 1 - home_xg[mask] * away_xg[mask] * rho

        mask = (home_goals == 0) & (away_goals == 1)
        correction[mask] = 1 + home_xg[mask] * rho

        mask = (home_goals == 1) & (away_goals == 0)
        correction[mask] = 1 + away_xg[mask] * rho

        mask = (home_goals == 1) & (away_goals == 1)
        correction[mask] = 1 - rho

        if (correction <= 0).any():
            continue

        current_loss = -np.log(correction).sum()

        if current_loss < best_loss:
            best_loss = current_loss
            best_rho = float(rho)

    return {
        "team_ids": team_ids,
        "attack": attack,
        "defense": defense,
        "home_advantage": home_advantage,
        "rho": best_rho,
    }


def get_expected_goals(model, home, away):
    """Calculate expected goals from fitted team strengths."""

    home_id = model["team_ids"].get(home)
    away_id = model["team_ids"].get(away)

    # Teams absent from training receive neutral ratings
    home_attack = model["attack"][home_id] if home_id is not None else 0.0
    home_defense = model["defense"][home_id] if home_id is not None else 0.0

    away_attack = model["attack"][away_id] if away_id is not None else 0.0
    away_defense = model["defense"][away_id] if away_id is not None else 0.0

    home_xg = np.exp(
        model["home_advantage"] + home_attack - away_defense
    )

    away_xg = np.exp(
        away_attack - home_defense
    )

    return float(home_xg), float(away_xg)


def predict_probabilities(home_xg, away_xg, rho):
    """Calculate home-win, draw and away-win probabilities."""

    home_probs = poisson.pmf(GOALS, home_xg)
    away_probs = poisson.pmf(GOALS, away_xg)

    # Score probability matrix (rows = home goals, columns = away goals)
    scores = np.outer(home_probs, away_probs)

    # Dixon-Coles correction for low-scoring results
    scores[0, 0] *= 1 - home_xg * away_xg * rho
    scores[0, 1] *= 1 + home_xg * rho
    scores[1, 0] *= 1 + away_xg * rho
    scores[1, 1] *= 1 - rho

    if (scores < 0).any():
        raise ValueError("Invalid Dixon-Coles correction")

    scores /= scores.sum()

    home_win = np.tril(scores, -1).sum()
    draw = np.trace(scores)
    away_win = np.triu(scores, 1).sum()

    return np.array([home_win, draw, away_win])


def main():
    from epl_predictor.config import PROCESSED_DATA_DIR

    # Train on the last three completed seasons
    data = pd.read_csv(
        PROCESSED_DATA_DIR / "epl_matches_with_xg.csv"
    )

    train = data[data["Season"].isin([
        "2023-24", "2024-25", "2025-26"
    ])]

    model = fit_team_model(train)

    # Example prediction
    home = "Arsenal"
    away = "Chelsea"

    home_xg, away_xg = get_expected_goals(model, home, away)

    probs = predict_probabilities(
        home_xg, away_xg, model["rho"]
    )

    print(f"\n{home} vs {away}")
    print(f"Expected goals: {home_xg:.2f} - {away_xg:.2f}")
    print(f"Home win: {probs[0]:.2%}")
    print(f"Draw:     {probs[1]:.2%}")
    print(f"Away win: {probs[2]:.2%}")
    print(f"Rho: {model['rho']:.4f}")


if __name__ == "__main__":
    main()
