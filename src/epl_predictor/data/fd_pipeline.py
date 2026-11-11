
import pandas as pd
from epl_predictor.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from epl_predictor.data.fd_processing import clean_matches


def validate_season(df, season):
    # Each EPL season has 380 matches and 20 teams
    if len(df) != 380:
        raise ValueError(f"{season}: Expected 380 matches, found {len(df)}")

    if not (df['Season'] == season).all():
        raise ValueError(f"{season}: Incorrect season label")

    teams = set(df["HomeTeam"]) | set(df["AwayTeam"]) # Remove duplicates and combine sets

    if len(teams) != 20:
        raise ValueError(f"{season}: Expected 20 teams, found {len(teams)}")

    # Each team plays 19 home and 19 away matches
    if not df["HomeTeam"].value_counts().eq(19).all():
        raise ValueError(f"{season}: Invalid home match counts")

    if not df["AwayTeam"].value_counts().eq(19).all():
        raise ValueError(f"{season}: Invalid away match counts")


def build_historical_dataset():
    

    all_matches = []

    for year in range(2014, 2026):
        season = f"{year}-{str(year + 1)[-2:]}" #2014-15, 2015-16, ..., 2025-26
        file_path = RAW_DATA_DIR / f"EPL_{season}.csv"

        # Load season, assign label and clean
        df = pd.read_csv(file_path)
        df["Season"] = season
        df = clean_matches(df)

        # Check that the season is complete
        validate_season(df, season)

        all_matches.append(df)

        print(f"{season}: {len(df)} matches")

    # Combine all 12 season dfs into a dataset
    combined = pd.concat(all_matches, ignore_index=True)
    combined = combined.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)
    output_path = PROCESSED_DATA_DIR / "epl_matches_2014_2026.csv"
    combined.to_csv(output_path,index=False,date_format="%Y-%m-%d")

    print("\nHistorical dataset successfully created!")
    print(f"Seasons: {combined['Season'].nunique()}")
    print(f"Matches: {len(combined)}")
    print(f"Saved: {output_path}")

    return combined


if __name__ == "__main__":
    build_historical_dataset()
