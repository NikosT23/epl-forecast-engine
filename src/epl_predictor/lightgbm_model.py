
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from epl_predictor.config import PROCESSED_DATA_DIR


FEATURES = [
    "HomePoints5", "AwayPoints5",
    "HomeGoals5", "AwayGoals5",
    "HomeConceded5", "AwayConceded5",
    "HomeXG5", "AwayXG5",
    "HomeXGA5", "AwayXGA5",
    "HomeElo", "AwayElo", "EloDifference",
]


def train_model(train):
    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=100,
        learning_rate=0.03,
        num_leaves=7,
        max_depth=3,
        min_child_samples=50,
        reg_lambda=2,
        random_state=42,
        verbosity=-1,
        n_jobs=1,
    )

    model.fit(train[FEATURES], train["FTR"])
    return model


def evaluate_model():
    data = pd.read_csv(
        PROCESSED_DATA_DIR / "epl_features_with_elo.csv"
    )

    # Remove matches without complete rolling statistics
    data = data.dropna(subset=FEATURES).copy()

    seasons = sorted(data["Season"].unique())
    results = []

    for season in seasons:
        if season < "2021-22":
            continue

        # Train on all earlier seasons
        train = data[data["Season"] < season]
        test = data[data["Season"] == season]

        model = train_model(train)

        # LightGBM returns probabilities in alphabetical class order
        raw_probs = model.predict_proba(test[FEATURES])

        class_ids = [
            list(model.classes_).index(label)
            for label in ["H", "D", "A"]
        ]

        probs = raw_probs[:, class_ids]

        for i, (_, match) in enumerate(test.iterrows()):
            results.append({
                "Season": season,
                "Date": match["Date"],
                "HomeTeam": match["HomeTeam"],
                "AwayTeam": match["AwayTeam"],
                "FTR": match["FTR"],
                "HomeWin": probs[i, 0],
                "Draw": probs[i, 1],
                "AwayWin": probs[i, 2],
            })

        # Evaluate this season
        predicted = np.array(["H", "D", "A"])[np.argmax(probs, axis=1)]
        actual = test["FTR"].to_numpy()

        accuracy = np.mean(predicted == actual)

        actual_ids = pd.Series(actual).map({
            "H": 0, "D": 1, "A": 2
        }).to_numpy()

        correct_probs = probs[np.arange(len(test)), actual_ids]
        log_loss = -np.log(np.clip(correct_probs, 1e-15, 1)).mean()

        print(f"\n{season}")
        print(f"Training matches: {len(train)}")
        print(f"Test matches: {len(test)}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Log Loss: {log_loss:.4f}")

    predictions = pd.DataFrame(results)

    output = PROCESSED_DATA_DIR / "lightgbm_predictions.csv"
    predictions.to_csv(output, index=False)

    # Overall performance
    probs = predictions[["HomeWin", "Draw", "AwayWin"]].to_numpy()
    actual = predictions["FTR"].to_numpy()

    predicted = np.array(["H", "D", "A"])[np.argmax(probs, axis=1)]
    accuracy = np.mean(predicted == actual)

    actual_ids = pd.Series(actual).map({
        "H": 0, "D": 1, "A": 2
    }).to_numpy()

    log_loss = -np.log(
        np.clip(probs[np.arange(len(actual)), actual_ids], 1e-15, 1)
    ).mean()

    print("\nOverall results:")
    print(f"Matches: {len(predictions)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Log Loss: {log_loss:.4f}")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    evaluate_model()
