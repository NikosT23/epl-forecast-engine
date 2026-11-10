# Cleanup and Validation of data

import pandas as pd

REQUIRED_COLUMNS = [
    "Season",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTHG", # Full Time Home Goals
    "FTAG", # Full Time Away Goals
    "FTR",  # Full Time Result
]

OPTIONAL_COLUMNS = [
    "HS",   # Home Shots
    "AS",   # Away Shots
    "HST",  # Home Shots on Target
    "AST",  # Away Shots on Target
    "B365H", # Bet365 Home Odds
    "B365D", # Bet365 Draw Odds
    "B365A", # Bet365 Away Odds
]


def clean_matches(df:pd.DataFrame):
    df = df.copy()

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns] 
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(how = "all")

    df['Date'] = pd.to_datetime(df['Date'], format = "%Y-%m-%d" ,errors='raise')

    for column in ['Season', 'HomeTeam', 'AwayTeam', 'FTR']:
        df[column] = df[column].astype(str).str.strip() 

    for column in ['FTHG', 'FTAG']:
        df[column] = pd.to_numeric(df[column], errors='raise')

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Missing values found in required columns")

    if (df['HomeTeam'] == df['AwayTeam']).any():
        raise ValueError("HomeTeam and AwayTeam cannot be the same")

    if (df['HomeTeam'] == '').any() or (df['AwayTeam'] == '').any():
        raise ValueError("HomeTeam and AwayTeam cannot be empty strings")

    for column in ['FTHG', 'FTAG']:
        if (df[column] < 0).any():
            raise ValueError(f"{column} cannot be negative")
        df[column] = df[column].astype(int)

    expected_result = pd.Series('D', index=df.index)

    expected_result[df['FTHG'] > df['FTAG']] = 'H'
    expected_result[df['FTHG'] < df['FTAG']] = 'A'

    if not expected_result.equals(df['FTR']):
        raise ValueError("FTR does not match FTHG and FTAG")

    fixture_columns = ['Season', 'HomeTeam', 'AwayTeam']
    if df.duplicated(subset=fixture_columns).any():
        raise ValueError("Duplicate fixtures found in the data")

    available_optional = [col for col in OPTIONAL_COLUMNS if col in df.columns]
    for column in available_optional:
        df[column] = pd.to_numeric(df[column], errors='coerce')

    df = df[REQUIRED_COLUMNS + available_optional]

    df = df.sort_values(['Date', 'HomeTeam', 'AwayTeam']).reset_index(drop=True)

    return df
