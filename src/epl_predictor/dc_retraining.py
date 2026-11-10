
import numpy as np
import pandas as pd

from epl_predictor.config import PROCESSED_DATA_DIR
from epl_predictor.team_dixon_coles import (
    fit_team_model,
    get_expected_goals,
    predict_probabilities,
)


RETRAIN_EVERY = 50


def evaluate(predictions):
    labels = np.array(["H", "D", "A"])
    probabilities = predictions[["HomeWin", "Draw", "AwayWin"]].to_numpy()

    predicted = labels[np.argmax(probabilities, axis=1)]
    actual = predictions["FTR"].to_numpy()

    accuracy = np.mean(predicted == actual)

    actual_ids = pd.Series(actual).map({
        "H": 0,
        "D": 1,
        "A": 2,
    }).to_numpy()

    correct_probs = probabilities[np.arange(len(actual)), actual_ids]
    log_loss = -np.log(np.clip(correct_probs, 1e-15, 1)).mean()

    return accuracy, log_loss


def main():
    data = pd.read_csv(PROCESSED_DATA_DIR / "epl_features_with_elo.csv")
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    seasons = sorted(data["Season"].unique())
    results = []

    # Evaluate 2021-22 through 2025-26
    for season in seasons:
        if season < "2021-22":
            continue

        print(f"\nEvaluating {season}")

        season_index = seasons.index(season)
        previous_seasons = seasons[max(0, season_index - 3):season_index]

        test = data[data["Season"] == season].copy()
        test = test.sort_values("Date").reset_index(drop=True)

        start = 0

        while start < len(test):
            end = min(start + RETRAIN_EVERY, len(test))

            # Keep matches played on the same date in one block
            while end < len(test) and test.loc[end, "Date"] == test.loc[end - 1, "Date"]:
                end += 1

            block = test.iloc[start:end]
            cutoff = block.iloc[0]["Date"]

            # Previous 3 seasons plus current-season matches before this block
            train = data[
                data["Season"].isin(previous_seasons)
                | (
                    (data["Season"] == season)
                    & (data["Date"] < cutoff)
                )
            ]

            model = fit_team_model(train)

            for _, match in block.iterrows():
                home_xg, away_xg = get_expected_goals(
                    model,
                    match["HomeTeam"],
                    match["AwayTeam"],
                )

                probs = predict_probabilities(
                    home_xg, away_xg, model["rho"]
                )

                results.append({
                    "Season": season,
                    "Date": match["Date"],
                    "HomeTeam": match["HomeTeam"],
                    "AwayTeam": match["AwayTeam"],
                    "FTR": match["FTR"],
                    "HomeWin": probs[0],
                    "Draw": probs[1],
                    "AwayWin": probs[2],
                })

            print(f"  Predicted {end}/{len(test)} matches")
            start = end

        season_results = pd.DataFrame(results)
        season_results = season_results[season_results["Season"] == season]

        accuracy, log_loss = evaluate(season_results)

        print(f"Accuracy: {accuracy:.2%}")
        print(f"Log Loss: {log_loss:.4f}")

    predictions = pd.DataFrame(results)

    output = PROCESSED_DATA_DIR / "dixon_coles_predictions.csv"
    predictions.to_csv(output, index=False)

    print("\nOverall results:")

    accuracy, log_loss = evaluate(predictions)
    print(f"Matches: {len(predictions)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Log Loss: {log_loss:.4f}")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
