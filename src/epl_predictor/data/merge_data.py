# Merging Understat and Football-Data

import pandas as pd
from epl_predictor.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

TEAM_MAP = { # for understat to match the fd dataset
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Queens Park Rangers": "QPR",
    "West Bromwich Albion": "West Brom",
    "Wolverhampton Wanderers": "Wolves",
}

def merge_data():
    football = pd.read_csv(PROCESSED_DATA_DIR / "epl_matches_2014_2026.csv")
    understat = pd.read_csv(RAW_DATA_DIR / "understat_xg_2014_2026.csv")

    understat["HomeTeam"] = understat["HomeTeam"].replace(TEAM_MAP)
    understat["AwayTeam"] = understat["AwayTeam"].replace(TEAM_MAP)

    # Column renaming to avoid confusion due to similarity
    understat = understat.rename(columns={
        "Date":'UnderstatDate',
        "HomeGoals":'UnderstatHomeGoals',
        "AwayGoals":'UnderstatAwayGoals',
    })

    # Merge using Seas, Home, Away and add _merge to show whether each match is in both sets
    merged = football.merge(understat,on = ["Season", "HomeTeam", "AwayTeam"],how = "left",validate = "1:1",indicator=True)
    unmatched = merged[merged["_merge"] != "both"]

    print(f"Football-Data matches: {len(football)}")
    print(f"Understat matches: {len(understat)}")
    print(f"Matched matches: {len(merged) - len(unmatched)}")
    print(f"Unmatched matches: {len(unmatched)}")

    if not unmatched.empty:
        print("\nUnmatched fixtures:")
        print(unmatched[["Season", "HomeTeam", "AwayTeam"]].to_string(index=False))
        raise ValueError("Some fixtures could not be matched.")

   
    # Keep original Football-Data columns plus Understat xG + ID
    merged = merged[list(football.columns) + ["HomeXG", "AwayXG", "UnderstatID"]]
    merged = merged.sort_values(["Date", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    output_path = PROCESSED_DATA_DIR / "epl_matches_with_xg.csv"
    merged.to_csv(output_path, index=False)

    print(f"\nSuccessfully merged {len(merged)} matches!")
    print(f"Missing xG values: {merged[['HomeXG', 'AwayXG']].isna().sum().sum()}")
    print(f"Saved to: {output_path}")

    return merged


if __name__ == "__main__":
    merge_data()
















