
import pandas as pd
from epl_predictor.config import PROCESSED_DATA_DIR


def build_features():
    df = pd.read_csv(PROCESSED_DATA_DIR / "epl_matches_with_xg.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Create home and away team statistics
    home = df[["Date", "HomeTeam", "FTHG", "FTAG", "HomeXG", "AwayXG"]].copy()
    home.columns = ["Date", "Team", "Goals", "Conceded", "XG", "XGA"]

    away = df[["Date", "AwayTeam", "FTAG", "FTHG", "AwayXG", "HomeXG"]].copy()
    away.columns = ["Date", "Team", "Goals", "Conceded", "XG", "XGA"]

    # Calculate match points
    for team_df in [home, away]:
        team_df["Points"] = 0
        team_df.loc[team_df["Goals"] == team_df["Conceded"], "Points"] = 1
        team_df.loc[team_df["Goals"] > team_df["Conceded"], "Points"] = 3

    teams = pd.concat([home, away], ignore_index=True)
    teams = teams.sort_values(["Team", "Date"])

    # Date + Team must identify each team's match uniquely
    if teams.duplicated(["Date", "Team"]).any():
        raise ValueError("A team has multiple matches on the same date")

    # Calculate rolling averages from the previous 5 matches
    stats = ["Points", "Goals", "Conceded", "XG", "XGA"]

    for stat in stats:
        previous = teams.groupby("Team")[stat].shift(1)
        teams[f"{stat}5"] = (previous.groupby(teams["Team"]).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True))

    # Prepare home and away features
    features = stats

    # Create separate tables with dates and team names
    home_features = teams[["Date", "Team"]].copy()
    away_features = teams[["Date", "Team"]].copy()

    # Rename team columns for merging
    home_features = home_features.rename(columns={"Team": "HomeTeam"})
    away_features = away_features.rename(columns={"Team": "AwayTeam"})

    # Add the rolling statistics with home/away prefixes
    for stat in stats:
        home_features[f"Home{stat}5"] = teams[f"{stat}5"]
        away_features[f"Away{stat}5"] = teams[f"{stat}5"]
    # Merge features with original matches
    
    df = df.merge(home_features, on=['Date', 'HomeTeam'], how='left')
    df = df.merge(away_features, on=['Date', 'AwayTeam'], how='left')

    if len(df) != 4560:
        raise ValueError(f"Expected 4560 matches, found {len(df)}")

    output = PROCESSED_DATA_DIR / "epl_features.csv"
    df.to_csv(output, index=False)

    print(f"Matches: {len(df)}")
    print(f"Rolling features: {len(stats) * 2}")
    print(f"Saved to: {output}")

    return df


if __name__ == "__main__":
    build_features()
