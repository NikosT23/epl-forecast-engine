
import os
import pandas as pd
import requests

from dotenv import load_dotenv
from epl_predictor.config import RAW_DATA_DIR


TEAM_MAP = {
    "Brighton & Hove Albion": "Brighton",
    "AFC Bournemouth": "Bournemouth",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham",
    "Hull City": "Hull",
    "Coventry City": "Coventry",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
}


def download_xg_baseline():
    output = RAW_DATA_DIR / "current_xg_baseline.csv"

    # Only download the initial baseline once
    if output.exists():
        print(f"Baseline already exists: {output}")
        return pd.read_csv(output)

    load_dotenv()

    api_key = os.getenv("STRIKERAPI_API_KEY")

    if not api_key:
        raise ValueError("STRIKERAPI_API_KEY is missing from .env")

    url = "https://mcp.strikerapi.com/v1/get_team_stats"

    response = requests.post(url,headers={"x-api-key": api_key.strip()},json={"season": 2026,"league": "soccer",},timeout=30)

    response.raise_for_status()

    # Extract team statistics
    data = response.json()["data"]
    teams = pd.DataFrame(data["teams"])

    print("API method:", data.get("method"))

    # Convert team names to our project's naming convention
    teams["Team"] = teams["display_name"].replace(TEAM_MAP)

    # Keep only the data required for rolling xG features
    teams = teams[["Team", "games", "xg_for", "xg_against"]].copy()

    teams.columns = ["Team", "Matches", "XG", "XGA"]

    # Validate the response
    if len(teams) != 20:
        raise ValueError(f"Expected 20 teams, found {len(teams)}")

    if teams["Team"].duplicated().any():
        raise ValueError("Duplicate team names found")

    if teams[["XG", "XGA"]].isna().any().any():
        raise ValueError("Missing xG values")

    # Compare teams against our current-season fixtures
    fixtures = pd.read_csv(RAW_DATA_DIR / "epl_2026_27.csv")
    fixture_teams = set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"])
    baseline_teams = set(teams["Team"])

    if baseline_teams != fixture_teams:
        print("Missing teams:", fixture_teams - baseline_teams)
        print("Unexpected teams:", baseline_teams - fixture_teams)
        raise ValueError("Team names do not match current fixtures")

    # Baseline should represent the 100 completed EPL matches
    completed = fixtures[fixtures["Status"] == "FINISHED"]
    appearances = pd.concat([completed["HomeTeam"],completed["AwayTeam"]]).value_counts()

    for _, team in teams.iterrows():
        expected_matches = appearances.get(team["Team"], 0)

        if team["Matches"] != expected_matches:
            raise ValueError(f"{team['Team']}: API reports {team['Matches']} matches, "f"but EPL fixtures contain {expected_matches}")

    # Save the initial baseline permanently
    teams = teams.sort_values("Team").reset_index(drop=True)
    teams.to_csv(output, index=False)

    print("\nCurrent-season xG baseline:")
    print(teams.to_string(index=False))

    print(f"\nTeams: {len(teams)}")
    print(f"Saved to: {output}")

    print("\nCredits used:", response.headers.get("X-Credits-Used"))
    print("Credits remaining:", response.headers.get("X-Credits-Remaining"))

    return teams


if __name__ == "__main__":
    download_xg_baseline()
