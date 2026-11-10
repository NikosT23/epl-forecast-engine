
import numpy as np

from epl_predictor.elo_model import HOME_ADVANTAGE


def predict_match(home_elo, away_elo):
    # Account for home advantage
    difference = home_elo + HOME_ADVANTAGE - away_elo

    # Expected home result based on Elo
    expected_home = 1 / (1 + 10 ** (-difference / 400))

    # Draw probability decreases as the Elo gap increases
    draw = 0.28 * np.exp(-abs(difference) / 400)

    # Prevent impossible negative win probabilities
    draw = min(draw, 2 * expected_home, 2 * (1 - expected_home))

    # Convert expected result into three probabilities
    home_win = expected_home - draw / 2
    away_win = 1 - expected_home - draw / 2

    return np.array([home_win, draw, away_win])


def main():
    # Test three different matchups
    matches = [
        ("Equal teams", 1500, 1500),
        ("Strong home team", 1800, 1500),
        ("Strong away team", 1500, 1800),
    ]

    for name, home_elo, away_elo in matches:
        home, draw, away = predict_match(home_elo, away_elo)

        print(f"\n{name}")
        print(f"Home win: {home:.2%}")
        print(f"Draw:     {draw:.2%}")
        print(f"Away win: {away:.2%}")

        assert np.isclose(home + draw + away, 1)
        assert min(home, draw, away) >= 0


if __name__ == "__main__":
    main()
