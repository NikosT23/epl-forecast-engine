
import os
import pandas as pd
import requests
from dotenv import load_dotenv
from epl_predictor.config import RAW_DATA_DIR


URL = "https://mcp.strikerapi.com/v1"
BASELINE_DATE = "2026-11-09" # Earlier xG info is covered 

TEAM_MAP = {
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion": "Brighton",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham"
}

COLUMNS = ["Date", "HomeTeam", "AwayTeam", "HomeXG", "AwayXG"]


def download_match_xg():
    output = RAW_DATA_DIR / "current_match_xg.csv"
    fixtures = pd.read_csv(RAW_DATA_DIR / "epl_2026_27.csv")

    if output.exists():
        saved = pd.read_csv(output)
    else:
        saved = pd.DataFrame(columns=COLUMNS)

    # Only completed matches after our initial xG baseline
    missing = fixtures[(fixtures["Status"] == "FINISHED")& (fixtures["Date"] > BASELINE_DATE)].copy()

    keys = ["Date", "HomeTeam", "AwayTeam"]

    if saved.duplicated(keys).any():
        raise ValueError("Duplicate saved xG matches")

    # Match completed to downloaded xg
    missing = missing.merge(saved[keys],on=keys,how="left",indicator=True)
    # Keep only fixt not appearing on our saved data
    missing = missing[missing["_merge"] == "left_only"]

    print(f"Matches needing xG: {len(missing)}")

    # If match already has xG then return our saved data / No API req
    if missing.empty:
        return saved

    load_dotenv()
    api_key = os.getenv("STRIKERAPI_API_KEY") # Retrieve key

    if not api_key:
        raise ValueError("Missing StrikerAPI API key")

    headers = {"x-api-key": api_key.strip()}

    # Find the StrikerAPI IDs for the missing matches
    games = []
    offset = 0

    while True:
        response = requests.post(
            f"{URL}/list_games",
            headers=headers,
            json={
                "league": "soccer",
                "season": 2026,
                "stage": "league",
                "status": "final",
                "from": missing["Date"].min(),
                "to": missing["Date"].max(),
                "limit": 100,
                "offset": offset,
            },
            timeout=30,
        )

        response.raise_for_status() # error for unsuccessful HTTP response
        data = response.json()["data"] # JSON to dictionary

        games.extend(data["games"]) # add matches to list

        # Stop if no more pages 
        if data.get("next_offset") is None:
            break

        offset = data["next_offset"] # continue from next page

    print(f"API matches found: {len(games)}")

    # Match fixtures by date, home team and away team
    for _, match in missing.iterrows():
        found = []

        # Standardize names
        for game in games:
            home = TEAM_MAP.get(game["home"]["name"], game["home"]["name"])
            away = TEAM_MAP.get(game["away"]["name"], game["away"]["name"])

            # Check all info is correct for fixture
            if (
                game["date"] == match["Date"]
                and home == match["HomeTeam"]
                and away == match["AwayTeam"]
                and game["home"]["score"] == match["FTHG"]
                and game["away"]["score"] == match["FTAG"]
            ):
                found.append(game)

        if len(found) != 1:
            print(f"Skipping unmatched fixture: {match['HomeTeam']} vs {match['AwayTeam']}")
            continue

        game = found[0]

        # Download shot events for this match
        response = requests.post(
            f"{URL}/get_events",
            headers=headers,
            json={
                "game_id": game["game_id"],
                "kind": "shots",
                "limit": 300,
            },
            timeout=30,
        )

        response.raise_for_status() # Error if req failed
        data = response.json()["data"] # extract response

        if data.get("next_offset") is not None:
            raise ValueError("Shot event pagination required")

        # Init both teams w/ zero xg
        home_xg = 0.0
        away_xg = 0.0
        shots = data["events"] # List of shot events

        if not shots:
            raise ValueError(f"No shots returned for {game['game_id']}")

        # Add up xg vals for each team
        for event in shots:
            shot = event.get("shot") # extract shot info

            if shot is None or shot.get("xg") is None:
                raise ValueError("Shot is missing xG")

            # Shot belongs to home team?
            if str(event["team_id"]) == str(game["home"]["team_id"]):
                home_xg += float(shot["xg"])

            # Shot belongs to away team?
            elif str(event["team_id"]) == str(game["away"]["team_id"]):
                away_xg += float(shot["xg"])

            # Unexpected team
            else:
                raise ValueError("Unknown shot team")

        row = {
            "Date": match["Date"],
            "HomeTeam": match["HomeTeam"],
            "AwayTeam": match["AwayTeam"],
            "HomeXG": round(home_xg, 3),
            "AwayXG": round(away_xg, 3)
        }

        # Save to our existing xg dtset
        saved = pd.concat([saved, pd.DataFrame([row])],ignore_index=True)
        saved.to_csv(output, index=False)

        print(
            f"{row['HomeTeam']} vs {row['AwayTeam']}: "
            f"{home_xg:.2f} - {away_xg:.2f} xG"
        )

    print(f"\nSaved xG matches: {len(saved)}")
    print(f"Saved to: {output}")

    return saved


if __name__ == "__main__":
    download_match_xg()
