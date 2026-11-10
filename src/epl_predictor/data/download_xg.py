# Understat xg downloader
import pandas as pd
from understatapi import UnderstatClient
from epl_predictor.config import RAW_DATA_DIR



def download_xg():
    all_matches = []

    with UnderstatClient() as understat:
        for year in range(2014,2026):
            season = f"{year}-{str(year + 1)[-2:]}"
            print(f"Downloading Understat season {season}...") 
            matches = understat.league(league="EPL").get_match_data(season=str(year))

            completed = [match for match in matches if match.get("isResult")]

            if len(completed) != 380:
                print(f"WARNING: Expected 380 matches for season {year}, but got {len(completed)}")

            for match in completed:
                all_matches.append({
                    "UnderstatID": match["id"],
                    "Season": season,
                    "DateTime": match["datetime"],
                    "HomeTeam": match["h"]["title"],
                    "AwayTeam": match["a"]["title"],
                    "HomeGoals": int(match["goals"]["h"]),
                    "AwayGoals": int(match["goals"]["a"]),
                    "HomeXG": float(match["xG"]["h"]),
                    "AwayXG": float(match["xG"]["a"]),
                })

            print(f"Downloaded {len(completed)} completed matches for season {year}/{str(year + 1)[-2:]}")

    df = pd.DataFrame(all_matches)

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["Date"] = df["DateTime"].dt.strftime("%Y-%m-%d")

    if df['UnderstatID'].duplicated().any():
        print("WARNING: Duplicate UnderstatIDs found.")

    df = df.sort_values("DateTime").reset_index(drop=True)

    output_path = RAW_DATA_DIR / "understat_xg_2014_2026.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} unique Understat matches to {output_path}")
    return df


if __name__ == "__main__":
    download_xg()






