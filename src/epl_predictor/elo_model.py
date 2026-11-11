
import pandas as pd
from epl_predictor.config import PROCESSED_DATA_DIR


START_ELO = 1500
K = 20
HOME_ADVANTAGE = 100
SEASON_REGRESSION = 0.10


def expected_home_score(home_elo, away_elo):
    difference = home_elo + HOME_ADVANTAGE - away_elo
    return 1 / (1 + 10 ** (-difference / 400))


def update_ratings(home_elo, away_elo, home_goals, away_goals):
    # Calculate the expected result
    expected = expected_home_score(home_elo, away_elo)

    # Actual result: win = 1, draw = 0.5, loss = 0
    if home_goals > away_goals:
        actual = 1
    elif home_goals < away_goals:
        actual = 0
    else:
        actual = 0.5

    # Larger victories produce slightly larger Elo changes
    margin = abs(home_goals - away_goals)

    if margin <= 1:
        multiplier = 1
    else:
        multiplier = min(2, 1 + 0.25 * (margin - 1)) # increase by 0.25 for each goal BUT never exceed 2

    change = K * multiplier * (actual - expected) # if actual>expected then home gains elo, otherwise home loses elo

    return home_elo + change, away_elo - change


def calculate_elo(matches):
    matches = matches.copy()
    matches["Date"] = pd.to_datetime(matches["Date"])

    matches = matches.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    ratings = {} # stores current elo
    home_ratings = [] # elo before match
    away_ratings = [] # elo before match

    previous_season = None

    # Group by date so same-day results cannot affect each other
    for date, games in matches.groupby("Date", sort=True):
        season = games.iloc[0]["Season"]

        # Move Elo ratings 10% towards 1500 between seasons
        if previous_season is not None and season != previous_season:
            for team in ratings:
                ratings[team] = (ratings[team] * (1 - SEASON_REGRESSION)+ START_ELO * SEASON_REGRESSION)

        previous_season = season
        changes = {}

        # Store pre-match ratings before applying results
        for _, match in games.iterrows():
            home = match["HomeTeam"]
            away = match["AwayTeam"]

            # if team is new assign starting 1500
            home_elo = ratings.get(home, START_ELO)
            away_elo = ratings.get(away, START_ELO)

            home_ratings.append(home_elo)
            away_ratings.append(away_elo)

            new_home, new_away = update_ratings(home_elo,away_elo,match["FTHG"],match["FTAG"])

            changes[home] = changes.get(home, 0) + new_home - home_elo
            changes[away] = changes.get(away, 0) + new_away - away_elo

        # Update ratings only after the entire day's matches
        for team, change in changes.items():
            ratings[team] = ratings.get(team, START_ELO) + change

    matches["HomeElo"] = home_ratings
    matches["AwayElo"] = away_ratings
    matches["EloDifference"] = matches["HomeElo"] - matches["AwayElo"] # Lightgbm feature

    return matches, ratings


def main():
    matches = pd.read_csv(PROCESSED_DATA_DIR / "epl_features.csv")

    matches, ratings = calculate_elo(matches)

    output = PROCESSED_DATA_DIR / "epl_features_with_elo.csv"
    matches.to_csv(output, index=False)

    print(f"Matches: {len(matches)}")
    print("Elo features: 3")
    print(f"Saved to: {output}")

    print("\nTop 10 final Elo ratings:")
    top_teams = sorted(ratings, key=ratings.get, reverse=True)

    for team in top_teams[:10]:
        print(f"{team}: {ratings[team]:.1f}")


if __name__ == "__main__":
    main()
