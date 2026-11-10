
import numpy as np
import pandas as pd

from epl_predictor.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from epl_predictor.elo_model import calculate_elo
from epl_predictor.lightgbm_model import FEATURES


def build_current_features():
    # Load historical data, current fixtures and frozen xG baseline
    historical = pd.read_csv(
        PROCESSED_DATA_DIR / "epl_matches_with_xg.csv"
    )

    current = pd.read_csv(
        RAW_DATA_DIR / "epl_2026_27.csv"
    )

    baseline = pd.read_csv(
        RAW_DATA_DIR / "current_xg_baseline.csv"
    ).set_index("Team")

    current["Kickoff"] = pd.to_datetime(current["Kickoff"], utc=True)

    completed = current[current["Status"] == "FINISHED"].copy()
    upcoming = current[current["Status"] != "FINISHED"].copy()

    if completed[["FTHG", "FTAG"]].isna().any().any():
        raise ValueError("Completed matches have missing scores")

    # Update Elo using all historical and current completed matches
    elo_columns = [
        "Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"
    ]

    elo_matches = pd.concat([
        historical[elo_columns],
        completed[elo_columns],
    ], ignore_index=True)

    _, ratings = calculate_elo(elo_matches)

    # Build team match history for points and goals
    home = completed[
        ["Kickoff", "HomeTeam", "FTHG", "FTAG"]
    ].copy()
    home.columns = ["Kickoff", "Team", "Goals", "Conceded"]

    away = completed[
        ["Kickoff", "AwayTeam", "FTAG", "FTHG"]
    ].copy()
    away.columns = ["Kickoff", "Team", "Goals", "Conceded"]

    teams = pd.concat([home, away], ignore_index=True)

    teams["Points"] = 0
    teams.loc[teams["Goals"] == teams["Conceded"], "Points"] = 1
    teams.loc[teams["Goals"] > teams["Conceded"], "Points"] = 3

    teams = teams.sort_values("Kickoff")

    team_form = {}

    for team, games in teams.groupby("Team"):
        last_five = games.tail(5)

        team_form[team] = {
            "Points5": last_five["Points"].mean(),
            "Goals5": last_five["Goals"].mean(),
            "Conceded5": last_five["Conceded"].mean(),
        }

    # Load actual match-level xG collected after the baseline
    xg_path = RAW_DATA_DIR / "current_match_xg.csv"

    team_xg = {}
    actual_xg_teams = 0

    if xg_path.exists():
        xg_matches = pd.read_csv(xg_path)

        if not xg_matches.empty:
            keys = ["Date", "HomeTeam", "AwayTeam"]

            if xg_matches.duplicated(keys).any():
                raise ValueError("Duplicate match-level xG records")

            # Only use xG for verified completed fixtures
            checked = xg_matches.merge(
                completed[keys],
                on=keys,
                how="left",
                indicator=True,
                validate="one_to_one",
            )

            if (checked["_merge"] != "both").any():
                raise ValueError("xG contains an unmatched fixture")

            if checked[["HomeXG", "AwayXG"]].isna().any().any():
                raise ValueError("Missing match-level xG values")

            if (checked[["HomeXG", "AwayXG"]] < 0).any().any():
                raise ValueError("Negative xG values found")

            # Convert match xG into individual team performances
            xg_home = xg_matches[
                ["Date", "HomeTeam", "HomeXG", "AwayXG"]
            ].copy()
            xg_home.columns = ["Date", "Team", "XG", "XGA"]

            xg_away = xg_matches[
                ["Date", "AwayTeam", "AwayXG", "HomeXG"]
            ].copy()
            xg_away.columns = ["Date", "Team", "XG", "XGA"]

            xg_teams = pd.concat(
                [xg_home, xg_away],
                ignore_index=True,
            )

            xg_teams = xg_teams.sort_values("Date")

            for team, games in xg_teams.groupby("Team"):
                # Switch only when five actual xG matches are available
                if len(games) >= 5:
                    last_five = games.tail(5)

                    team_xg[team] = {
                        "XG5": last_five["XG"].mean(),
                        "XGA5": last_five["XGA"].mean(),
                    }

                    actual_xg_teams += 1

    # Build features for all unfinished fixtures
    rows = []

    for _, match in upcoming.iterrows():
        home_team = match["HomeTeam"]
        away_team = match["AwayTeam"]

        if home_team not in team_form or away_team not in team_form:
            raise ValueError("Missing match history for a team")

        if home_team not in baseline.index or away_team not in baseline.index:
            raise ValueError("Missing xG baseline for a team")

        home_elo = ratings[home_team]
        away_elo = ratings[away_team]

        row = match.to_dict()

        # Actual rolling form
        for stat in ["Points5", "Goals5", "Conceded5"]:
            row[f"Home{stat}"] = team_form[home_team][stat]
            row[f"Away{stat}"] = team_form[away_team][stat]

        # Frozen baseline or genuine rolling-five xG
        row["HomeXG5"] = (
            team_xg[home_team]["XG5"]
            if home_team in team_xg
            else baseline.loc[home_team, "XG"]
        )

        row["HomeXGA5"] = (
            team_xg[home_team]["XGA5"]
            if home_team in team_xg
            else baseline.loc[home_team, "XGA"]
        )

        row["AwayXG5"] = (
            team_xg[away_team]["XG5"]
            if away_team in team_xg
            else baseline.loc[away_team, "XG"]
        )

        row["AwayXGA5"] = (
            team_xg[away_team]["XGA5"]
            if away_team in team_xg
            else baseline.loc[away_team, "XGA"]
        )

        # Updated Elo ratings
        row["HomeElo"] = home_elo
        row["AwayElo"] = away_elo
        row["EloDifference"] = home_elo - away_elo

        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError("No unfinished fixtures found")

    if df[FEATURES].isna().any().any():
        raise ValueError("Missing prediction features")

    if not np.isfinite(df[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Invalid prediction features")

    output = PROCESSED_DATA_DIR / "current_features.csv"
    df.to_csv(output, index=False)

    print(f"Completed matches: {len(completed)}")
    print(f"Unfinished fixtures: {len(df)}")
    print(f"Features: {len(FEATURES)}")
    print(f"Teams using actual rolling xG: {actual_xg_teams}/20")
    print(f"Teams using baseline xG: {len(baseline) - actual_xg_teams}/20")
    print(f"Saved to: {output}")

    return df


if __name__ == "__main__":
    build_current_features()
