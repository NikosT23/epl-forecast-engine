
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from epl_predictor.config import PROCESSED_DATA_DIR

# Our 13 LGBM features
FEATURES = [
    "HomePoints5", "AwayPoints5",
    "HomeGoals5", "AwayGoals5",
    "HomeConceded5", "AwayConceded5",
    "HomeXG5", "AwayXG5",
    "HomeXGA5", "AwayXGA5",
    "HomeElo", "AwayElo", "EloDifference",
]


def train_model(train):
    #Build LGBM model that predicts 3 outcomes
    model = LGBMClassifier(
        objective="multiclass",
        n_estimators=100, # 100 decision trees
        learning_rate=0.03, # Each new tree contribution 
        num_leaves=7, # max leaves per tree
        max_depth=3, # depth of each tree
        min_child_samples=50, # min num of train samples in a leaf to prevet overfitting
        reg_lambda=2, # L2 regularization
        random_state=42, #Fixed seed
        verbosity=-1, # Hide train messages
        n_jobs=1, # CPU threads
    )

    # Train using 13 features as X and FTR as Y
    model.fit(train[FEATURES], train["FTR"])
    return model


def evaluate_model():
    data = pd.read_csv( PROCESSED_DATA_DIR / "epl_features_with_elo.csv")

    # Remove matches without complete rolling statistics
    data = data.dropna(subset=FEATURES).copy()

    seasons = sorted(data["Season"].unique())
    results = []

    # Walk forward eval
    for season in seasons:
        if season < "2021-22":
            continue

        # Train on all earlier seasons than current test season
        train = data[data["Season"] < season]
        test = data[data["Season"] == season]

        model = train_model(train) # Train new model for this eval season

        # Returns one prob per class
        raw_probs = model.predict_proba(test[FEATURES])

        # LightGBM returns probabilities in alphabetical class order but we need in H-D-A
        class_ids = [list(model.classes_).index(label)for label in ["H", "D", "A"]]

        probs = raw_probs[:, class_ids] # Rearrange cols into H-D-A

        # Save individual predictions for each test match
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

        # Evaluate this season, find most likely result
        predicted = np.array(["H", "D", "A"])[np.argmax(probs, axis=1)]
        actual = test["FTR"].to_numpy() # actual outcome

        accuracy = np.mean(predicted == actual)

        actual_ids = pd.Series(actual).map({"H": 0, "D": 1, "A": 2}).to_numpy()

        correct_probs = probs[np.arange(len(test)), actual_ids] # prob of actual result
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

    predicted = np.array(["H", "D", "A"])[np.argmax(probs, axis=1)] # Most likely outcome for each match
    accuracy = np.mean(predicted == actual)

    actual_ids = pd.Series(actual).map({"H": 0, "D": 1, "A": 2}).to_numpy()

    log_loss = -np.log(np.clip(probs[np.arange(len(actual)), actual_ids], 1e-15, 1)).mean()

    print("\nOverall results:")
    print(f"Matches: {len(predictions)}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Log Loss: {log_loss:.4f}")
    print(f"Saved to: {output}")


if __name__ == "__main__":
    evaluate_model()
