
import os
import pandas as pd
import requests
from dotenv import load_dotenv
from epl_predictor.config import RAW_DATA_DIR


TEAM_MAP = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton",
    "Burnley FC": "Burnley",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Coventry City FC": "Coventry",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Hull City AFC": "Hull",
    "Ipswich Town FC": "Ipswich",
    "Leeds United FC": "Leeds",
    "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest",
    "Southampton FC": "Southampton",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
}


def download_current_season():
    load_dotenv()

    api_key = os.getenv("FOOTBALL_DATA_API_KEY")

    if not api_key:
        raise ValueError("FOOTBALL_DATA_API_KEY is missing from .env")

    url = "https://api.football-data.org/v4/competitions/PL/matches"

    response = requests.get(url,headers={"X-Auth-Token": api_key},params={"season": 2026},timeout=30)

    response.raise_for_status()
    matches = response.json()["matches"]

    rows = []

    for match in matches:
        home_name = match["homeTeam"]["name"]
        away_name = match["awayTeam"]["name"]

        if home_name not in TEAM_MAP or away_name not in TEAM_MAP:
            raise ValueError(f"Unknown team: {home_name} / {away_name}")

        home = TEAM_MAP[home_name]
        away = TEAM_MAP[away_name]

        finished = match["status"] == "FINISHED"
        score = match["score"]["fullTime"]

        home_goals = score["home"] if finished else None
        away_goals = score["away"] if finished else None

        result = None

        if finished:
            if home_goals > away_goals:
                result = "H"
            elif home_goals < away_goals:
                result = "A"
            else:
                result = "D"

        rows.append({
            "MatchID": match["id"],
            "Season": "2026-27",
            "Date": match["utcDate"][:10],
            "Kickoff": match["utcDate"],
            "Matchday": match["matchday"],
            "HomeTeam": home,
            "AwayTeam": away,
            "FTHG": home_goals,
            "FTAG": away_goals,
            "FTR": result,
            "Status": match["status"],
            "HomeCrest": match["homeTeam"]["crest"],
            "AwayCrest": match["awayTeam"]["crest"],
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Kickoff").reset_index(drop=True)

    if df.empty:
        raise ValueError("No matches returned by the API")

    if df["MatchID"].duplicated().any():
        raise ValueError("Duplicate match IDs found")

    completed = df["Status"] == "FINISHED"

    if df.loc[completed, ["FTHG", "FTAG", "FTR"]].isna().any().any():
        raise ValueError("Completed matches have missing results")

    if len(df) != 380:
        print(f"Warning: Expected 380 fixtures, received {len(df)}")

    output = RAW_DATA_DIR / "epl_2026_27.csv"
    df.to_csv(output, index=False)

    print(f"Total fixtures: {len(df)}")
    print(f"Completed: {completed.sum()}")
    print(f"Remaining / unfinished: {(~completed).sum()}")
    print(f"Teams: {len(set(df['HomeTeam']) | set(df['AwayTeam']))}")
    print(f"Saved to: {output}")

    return df


if __name__ == "__main__":
    download_current_season()
