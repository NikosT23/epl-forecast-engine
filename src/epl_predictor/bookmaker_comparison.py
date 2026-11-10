
import numpy as np
import pandas as pd

from epl_predictor.config import PROCESSED_DATA_DIR


PROBS = ["HomeWin", "Draw", "AwayWin"]


def evaluate(probs, actual):
    labels = np.array(["H", "D", "A"])

    predicted = labels[np.argmax(probs, axis=1)]
    accuracy = np.mean(predicted == actual)

    actual_ids = pd.Series(actual).map({
        "H": 0,
        "D": 1,
        "A": 2,
    }).to_numpy()

    correct_probs = probs[np.arange(len(actual)), actual_ids]
    log_loss = -np.log(np.clip(correct_probs, 1e-15, 1)).mean()

    return accuracy, log_loss


def main():
    matches = pd.read_csv(
        PROCESSED_DATA_DIR / "epl_features_with_elo.csv"
    )

    ensemble = pd.read_csv(
        PROCESSED_DATA_DIR / "ensemble_predictions.csv"
    )

    keys = ["Season", "Date", "HomeTeam", "AwayTeam"]

    # Merge bookmaker odds with our historical predictions
    data = ensemble.merge(
        matches[keys + ["B365H", "B365D", "B365A"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )

    # Remove matches with unavailable or invalid odds
    odds = ["B365H", "B365D", "B365A"]

    for col in odds:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    valid = data[odds].notna().all(axis=1)
    valid &= (data[odds] > 1).all(axis=1)

    data = data[valid].copy()

    if data.empty:
        raise ValueError("No matches have valid Bet365 odds")

    # Decimal odds -> implied probabilities
    bookmaker_probs = 1 / data[odds].to_numpy()

    # Normalize to remove the bookmaker's margin
    bookmaker_probs /= bookmaker_probs.sum(axis=1, keepdims=True)

    data["BookmakerHome"] = bookmaker_probs[:, 0]
    data["BookmakerDraw"] = bookmaker_probs[:, 1]
    data["BookmakerAway"] = bookmaker_probs[:, 2]

    print(f"Matches with valid odds: {len(data)}")

    # Evaluate both models on exactly the same matches
    for name, subset in [
        ("VALIDATION 2021-25", data[data["Season"] <= "2024-25"]),
        ("EVALUATION 2025-26", data[data["Season"] == "2025-26"]),
        ("OVERALL", data),
    ]:
        print(f"\n{name} ({len(subset)} matches)")

        if subset.empty:
            continue

        actual = subset["FTR"].to_numpy()

        ensemble_probs = subset[PROBS].to_numpy()
        market_probs = subset[
            ["BookmakerHome", "BookmakerDraw", "BookmakerAway"]
        ].to_numpy()

        for model_name, probs in [
            ("Ensemble", ensemble_probs),
            ("Bet365", market_probs),
        ]:
            accuracy, log_loss = evaluate(probs, actual)

            print(
                f"{model_name}: "
                f"Accuracy {accuracy:.2%} | "
                f"Log Loss {log_loss:.4f}"
            )

    output = PROCESSED_DATA_DIR / "bookmaker_comparison.csv"
    data.to_csv(output, index=False)

    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
