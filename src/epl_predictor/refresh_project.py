
import pandas as pd

from epl_predictor.config import RAW_DATA_DIR
from epl_predictor.data.download_current import download_current_season
from epl_predictor.data.current_match_xg import download_match_xg
from epl_predictor.current_features import build_current_features
from epl_predictor.current_predictions import generate_predictions
from epl_predictor.season_simulation import simulate_season

BASELINE_DATE = "2026-11-09"


def refresh_project():
    print("\n1. Updating current-season results and fixtures")
    download_current_season()

    print("\n2. Downloading missing match-level xG")
    download_match_xg()

    # Don't generate new predictions with missing weekly xG
    fixtures = pd.read_csv(RAW_DATA_DIR / "epl_2026_27.csv")

    completed = fixtures[
        (fixtures["Status"] == "FINISHED")
        & (fixtures["Date"] > BASELINE_DATE)
    ]

    xg_path = RAW_DATA_DIR / "current_match_xg.csv"

    if not completed.empty:
        if not xg_path.exists():
            raise ValueError("Missing current match-level xG data")

        xg = pd.read_csv(xg_path)

        missing = completed.merge(
            xg[["Date", "HomeTeam", "AwayTeam"]],
            on=["Date", "HomeTeam", "AwayTeam"],
            how="left",
            indicator=True,
            validate="one_to_one",
        )

        if (missing["_merge"] != "both").any():
            raise ValueError("Some completed matches are missing xG")

    print("\n3. Updating rolling features and Elo ratings")
    build_current_features()

    print("\n4. Training models and generating predictions")
    generate_predictions()

    print("\n5. Simulating the remaining Premier League season")
    simulate_season()
    
    print("\nWeekly refresh completed successfully!")


if __name__ == "__main__":
    refresh_project()
