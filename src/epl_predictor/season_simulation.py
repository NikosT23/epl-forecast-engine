
import numpy as np
import pandas as pd
from epl_predictor.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from epl_predictor.monte_carlo import score_probabilities, simulate_match
from epl_predictor.team_dixon_coles import fit_team_model, get_expected_goals


SIMULATIONS = 10000
PROBS = ["HomeWin", "Draw", "AwayWin"]
KEYS = ["Date", "HomeTeam", "AwayTeam"]


def simulate_season():
    # Load current-season matches and ensemble predictions
    matches = pd.read_csv(RAW_DATA_DIR / "epl_2026_27.csv")
    predictions = pd.read_csv(PROCESSED_DATA_DIR / "current_predictions.csv")

    # Separate completed from unfinished
    completed = matches[matches["Status"] == "FINISHED"].copy()
    remaining = matches[matches["Status"] != "FINISHED"].copy()

    if len(matches) != 380:
        raise ValueError("Expected 380 season fixtures")

    # Match predictions to unfinished fixtures
    remaining = remaining.merge(predictions[KEYS + PROBS],on=KEYS,how="left",validate="one_to_one")

    if remaining[PROBS].isna().any().any():
        raise ValueError("Some fixtures have no predictions")

    # Prepare Dixon-Coles training data
    historical = pd.read_csv(PROCESSED_DATA_DIR / "epl_matches_with_xg.csv")

    recent = historical[historical["Season"].isin(["2023-24", "2024-25", "2025-26"])]

    # Train DC to est atk,def,home_adv and rho
    train = pd.concat([recent, completed], ignore_index=True)
    model = fit_team_model(train)

    # Initialize league standings
    teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    # Give every team numeric ids
    team_ids = {team: i for i, team in enumerate(teams)}
    n = len(teams)

    # Create 3 arrays w/ 10000 rows+20 cols. Row=sim season, Col=Team
    points = np.zeros((SIMULATIONS, n), dtype=int)
    goals_for = np.zeros((SIMULATIONS, n), dtype=int)
    goals_against = np.zeros((SIMULATIONS, n), dtype=int)

    # Add actual results from completed matches
    for _, match in completed.iterrows():
        # Numeric ids of teams
        h = team_ids[match["HomeTeam"]]
        a = team_ids[match["AwayTeam"]]

        #Actual final score
        hg = int(match["FTHG"])
        ag = int(match["FTAG"])

        # Add actual scored and conceded, [:,h] means 10000 sims for home team
        goals_for[:, h] += hg
        goals_against[:, h] += ag

        goals_for[:, a] += ag
        goals_against[:, a] += hg

        # Actual league points 
        if hg > ag:
            points[:, h] += 3
        elif hg < ag:
            points[:, a] += 3
        else:
            points[:, h] += 1
            points[:, a] += 1

    # Simulate all remaining fixtures
    rng = np.random.default_rng(42)
    for _, match in remaining.iterrows():
        home = match["HomeTeam"]
        away = match["AwayTeam"]

        # Team ids
        h = team_ids[home]
        a = team_ids[away]

        # Expected calculated from DC
        home_xg, away_xg = get_expected_goals(model, home, away)

        # Ensemble probs for match
        ensemble = match[PROBS].to_numpy(dtype=float)

        # Probs for every possible scoreline
        scores = score_probabilities(home_xg, away_xg, model["rho"], ensemble)

        # Sim 10000 times and hg,ag are arrays containing simulated scores
        hg, ag = simulate_match(scores, simulations=SIMULATIONS, rng=rng)

        # Update simulated goals
        goals_for[:, h] += hg
        goals_against[:, h] += ag

        goals_for[:, a] += ag
        goals_against[:, a] += hg

        # Boolean arrays indicating result of each of 10000 sim matches
        home_wins = hg > ag
        away_wins = hg < ag
        draws = hg == ag

        points[:, h] += 3 * home_wins + draws
        points[:, a] += 3 * away_wins + draws

    # Calculate final standings for every simulation
    goal_difference = goals_for - goals_against

    # Each team's finishing pos in every sim
    positions = np.zeros((SIMULATIONS, n), dtype=int)

    # Final table calc
    for i in range(SIMULATIONS):
        # Sort by points, goal difference and goals scored
        order = np.lexsort((
            np.arange(n),
            -goals_for[i],
            -goal_difference[i],
            -points[i]
        ))

        # Assign pos 1-20 to teams in sorted order
        positions[i, order] = np.arange(1, n + 1)

    # Calculate season outcome probabilities
    results = pd.DataFrame({
        "Team": teams,
        "ExpectedPoints": points.mean(axis=0),
        "ExpectedPosition": positions.mean(axis=0),
        "Title": (positions == 1).mean(axis=0),
        "Top4": (positions <= 4).mean(axis=0),
        "Relegation": (positions >= 18).mean(axis=0)
    })

    results = results.sort_values("ExpectedPosition").reset_index(drop=True)

    # Verify league outcome probabilities
    assert np.isclose(results["Title"].sum(), 1) # All title probs must add up to 1
    assert np.isclose(results["Top4"].sum(), 4) # Combined top 4 probs must add up to 4
    assert np.isclose(results["Relegation"].sum(), 3) # Combined last 3 probs must add up to 3

    output = PROCESSED_DATA_DIR / "season_simulation.csv"
    results.to_csv(output, index=False)

    print(f"Completed matches: {len(completed)}")
    print(f"Remaining fixtures: {len(remaining)}")
    print(f"Simulations: {SIMULATIONS}")
    print(f"Saved to: {output}")

    print("\nProjected final standings:")
    print(
        results.to_string(
            index=False,
            formatters={
                "ExpectedPoints": "{:.1f}".format,
                "ExpectedPosition": "{:.1f}".format,
                "Title": "{:.1%}".format,
                "Top4": "{:.1%}".format,
                "Relegation": "{:.1%}".format
            },
        )
    )

    return results


if __name__ == "__main__":
    simulate_season()
