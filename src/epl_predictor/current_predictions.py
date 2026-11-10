
import joblib
import numpy as np
import pandas as pd

from epl_predictor.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
)
from epl_predictor.elo_probs import predict_match
from epl_predictor.lightgbm_model import train_model, FEATURES
from epl_predictor.team_dixon_coles import (
    fit_team_model,
    get_expected_goals,
    predict_probabilities,
)


def generate_predictions():
    # Load historical training data and current fixtures
    historical = pd.read_csv(
        PROCESSED_DATA_DIR / "epl_features_with_elo.csv"
    )

    current = pd.read_csv(
        RAW_DATA_DIR / "epl_2026_27.csv"
    )

    fixtures = pd.read_csv(
        PROCESSED_DATA_DIR / "current_features.csv"
    )

    # 1. Train final LightGBM model
    train = historical.dropna(subset=FEATURES).copy()

    lightgbm = train_model(train)

    lgbm_probs = lightgbm.predict_proba(fixtures[FEATURES])

    # Reorder probabilities to Home, Draw, Away
    class_ids = [
        list(lightgbm.classes_).index(label)
        for label in ["H", "D", "A"]
    ]

    lgbm_probs = lgbm_probs[:, class_ids]

    # Save final LightGBM model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lightgbm, MODELS_DIR / "lightgbm_model.joblib")

    # 2. Train Dixon-Coles using recent completed matches
    previous_seasons = ["2023-24", "2024-25", "2025-26"]

    dc_train = historical[
        historical["Season"].isin(previous_seasons)
    ].copy()

    completed = current[current["Status"] == "FINISHED"]

    dc_train = pd.concat([
        dc_train,
        completed,
    ], ignore_index=True)

    dc_model = fit_team_model(dc_train)

    # 3. Generate individual model probabilities
    elo_probs = []
    dc_probs = []

    for _, match in fixtures.iterrows():
        home = match["HomeTeam"]
        away = match["AwayTeam"]

        # Elo prediction
        elo_probs.append(
            predict_match(match["HomeElo"], match["AwayElo"])
        )

        # Dixon-Coles prediction
        home_xg, away_xg = get_expected_goals(
            dc_model, home, away
        )

        dc_probs.append(
            predict_probabilities(
                home_xg, away_xg, dc_model["rho"]
            )
        )

    elo_probs = np.array(elo_probs)
    dc_probs = np.array(dc_probs)

    # 4. Equal-weight ensemble
    ensemble_probs = (elo_probs + dc_probs + lgbm_probs) / 3

    if not np.allclose(ensemble_probs.sum(axis=1), 1):
        raise ValueError("Probabilities do not sum to 1")

    if (ensemble_probs < 0).any():
        raise ValueError("Negative probabilities found")

    # 5. Save predictions
    results = fixtures[
        ["Date", "Kickoff", "HomeTeam", "AwayTeam"]
    ].copy()

    results["HomeWin"] = ensemble_probs[:, 0]
    results["Draw"] = ensemble_probs[:, 1]
    results["AwayWin"] = ensemble_probs[:, 2]

    labels = np.array(["H", "D", "A"])
    results["Prediction"] = labels[np.argmax(ensemble_probs, axis=1)]

    results = results.sort_values("Kickoff").reset_index(drop=True)

    output = PROCESSED_DATA_DIR / "current_predictions.csv"
    results.to_csv(output, index=False)

    print(f"LightGBM training matches: {len(train)}")
    print(f"Dixon-Coles training matches: {len(dc_train)}")
    print(f"Fixtures predicted: {len(results)}")
    print(f"Dixon-Coles rho: {dc_model['rho']:.4f}")
    print(f"Saved to: {output}")

    print("\nNext 10 fixtures:")
    print(
        results[
            ["Date", "HomeTeam", "AwayTeam",
             "HomeWin", "Draw", "AwayWin", "Prediction"]
        ].head(10).to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    return results


if __name__ == "__main__":
    generate_predictions()
