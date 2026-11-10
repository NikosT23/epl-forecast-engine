
import numpy as np
import pandas as pd

from epl_predictor.config import PROCESSED_DATA_DIR
from epl_predictor.elo_probs import predict_match


KEYS = ["Season", "Date", "HomeTeam", "AwayTeam"]
PROBS = ["HomeWin", "Draw", "AwayWin"]


def evaluate(data):
    probs = data[PROBS].to_numpy()
    actual = data["FTR"].to_numpy()

    labels = np.array(["H", "D", "A"])
    predicted = labels[np.argmax(probs, axis=1)]
    accuracy = np.mean(predicted == actual)

    actual_ids = pd.Series(actual).map({
        "H": 0, "D": 1, "A": 2
    }).to_numpy()

    correct_probs = probs[np.arange(len(data)), actual_ids]
    log_loss = -np.log(np.clip(correct_probs, 1e-15, 1)).mean()

    return accuracy, log_loss


def main():
    # Load historical predictions
    elo = pd.read_csv(PROCESSED_DATA_DIR / "epl_features_with_elo.csv")
    dc = pd.read_csv(PROCESSED_DATA_DIR / "dixon_coles_predictions.csv")
    lgbm = pd.read_csv(PROCESSED_DATA_DIR / "lightgbm_predictions.csv")

    # Calculate Elo probabilities using pre-match ratings
    elo_probs = np.array([
        predict_match(home, away)
        for home, away in zip(elo["HomeElo"], elo["AwayElo"])
    ])

    elo[PROBS] = elo_probs

    # Merge predictions for the same matches
    data = lgbm[KEYS + ["FTR"]].copy()

    for name, model in [
        ("Elo", elo),
        ("DC", dc),
        ("LightGBM", lgbm),
    ]:
        model_probs = model[KEYS + PROBS].copy()
        model_probs = model_probs.rename(
            columns={col: f"{name}_{col}" for col in PROBS}
        )

        data = data.merge(
            model_probs,
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )

    if len(data) != len(lgbm):
        raise ValueError("Some model predictions could not be matched")

    # Average probabilities from all three models
    for col in PROBS:
        data[col] = (
            data[f"Elo_{col}"]
            + data[f"DC_{col}"]
            + data[f"LightGBM_{col}"]
        ) / 3

    # Evaluate individual seasons
    for season, games in data.groupby("Season"):
        accuracy, log_loss = evaluate(games)

        print(f"\n{season}")
        print(f"Matches: {len(games)}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Log Loss: {log_loss:.4f}")

    # Evaluate validation and 2025-26 separately
    validation = data[data["Season"] <= "2024-25"]
    evaluation = data[data["Season"] == "2025-26"]

    for name, games in [
        ("VALIDATION 2021-25", validation),
        ("EVALUATION 2025-26", evaluation),
        ("OVERALL", data),
    ]:
        accuracy, log_loss = evaluate(games)

        print(f"\n{name}")
        print(f"Matches: {len(games)}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Log Loss: {log_loss:.4f}")

    output = PROCESSED_DATA_DIR / "ensemble_predictions.csv"
    data.to_csv(output, index=False)

    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
