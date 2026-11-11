
import pandas as pd
from epl_predictor.config import RAW_DATA_DIR

BASE_URL = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"


SEASONS = {
    "2014-15": "1415",
    "2015-16": "1516",
    "2016-17": "1617",
    "2017-18": "1718",
    "2018-19": "1819",
    "2019-20": "1920",
    "2020-21": "2021",
    "2021-22": "2122",
    "2022-23": "2223",
    "2023-24": "2324",
    "2024-25": "2425",
    "2025-26": "2526",
}

def download_season(season_name, season_code):
    url = BASE_URL.format(season_code) 
    print(f"Downloading {season_name}...")

    df = pd.read_csv(url) # streams the CSV into a DF over HTTP
    df = df.dropna(how='all')  # Drop rows where all elements are NaN

    df["Season"] = season_name # Column to id season
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors='coerce').dt.strftime('%Y-%m-%d')

    output_path = RAW_DATA_DIR / f"EPL_{season_name}.csv"
    df.to_csv(output_path, index=False)

for season_name, season_code in SEASONS.items():
    try:
        download_season(season_name, season_code)
    except Exception as e:
        print(f"ERROR downloading {season_name}: {e}")


















