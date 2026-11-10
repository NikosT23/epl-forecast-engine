
import html

import numpy as np
import pandas as pd
import streamlit as st

from epl_predictor.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


# ----------------------------------------
# PAGE SETTINGS
# ----------------------------------------

st.set_page_config(
    page_title="EPL Forecast Engine",
    page_icon="⚽",
    layout="wide",
)


# ----------------------------------------
# STYLING
# ----------------------------------------

st.markdown("""
<style>
    .block-container {
        max-width: 1450px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .match-card, .prediction-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
    }

    .match-date {
        text-align: center;
        font-size: 12px;
        opacity: 0.65;
        margin-bottom: 10px;
    }

    .match-teams, .prediction-teams {
        display: grid;
        grid-template-columns: 1fr 35px 1fr;
        align-items: center;
        text-align: center;
        gap: 5px;
        margin-bottom: 12px;
    }

    .prediction-teams {
        max-width: 500px;
        margin: 12px auto 18px;
    }

    .team {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 5px;
        font-size: 13px;
        font-weight: 600;
    }

    .team img {
        width: 34px;
        height: 34px;
        object-fit: contain;
    }

    .prediction-teams .team img {
        width: 48px;
        height: 48px;
    }

    .vs {
        font-size: 12px;
        opacity: 0.5;
    }

    .probabilities, .prediction-bars {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        text-align: center;
        gap: 12px;
        border-top: 1px solid rgba(128,128,128,0.2);
        padding-top: 10px;
    }

    .prob-label, .stat-label {
        font-size: 11px;
        opacity: 0.65;
    }

    .prob-value {
        font-size: 15px;
        font-weight: 600;
    }

    .bar-label {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        margin-bottom: 6px;
    }

    .bar-background {
        height: 7px;
        background: rgba(128,128,128,0.18);
        border-radius: 10px;
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        background: #4285f4;
        border-radius: 10px;
    }

    .simulation-stats {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin: 12px 0;
    }

    .stat-box {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 10px;
        padding: 12px 8px;
        text-align: center;
    }

    .stat-value {
        font-size: 20px;
        font-weight: 600;
        margin-top: 4px;
    }

    .selected-team {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        margin: 12px 0;
    }

    .selected-team img {
        width: 42px;
        height: 42px;
        object-fit: contain;
    }

    .selected-team span {
        font-size: 19px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------
# LOAD CURRENT DATA
# ----------------------------------------

matches = pd.read_csv(
    RAW_DATA_DIR / "epl_2026_27.csv"
)

predictions = pd.read_csv(
    PROCESSED_DATA_DIR / "current_predictions.csv"
)

simulations = pd.read_csv(
    PROCESSED_DATA_DIR / "season_simulation.csv"
)

completed = matches[
    matches["Status"] == "FINISHED"
].copy()

teams = sorted(
    set(matches["HomeTeam"]) | set(matches["AwayTeam"])
)

crests = {}

for _, match in matches.iterrows():
    crests[match["HomeTeam"]] = match["HomeCrest"]
    crests[match["AwayTeam"]] = match["AwayCrest"]


# ----------------------------------------
# ACTUAL STANDINGS
# ----------------------------------------

standings = {
    team: {
        "Team": team,
        "P": 0, "W": 0, "D": 0, "L": 0,
        "GF": 0, "GA": 0, "Pts": 0,
    }
    for team in teams
}

for _, match in completed.iterrows():
    home = standings[match["HomeTeam"]]
    away = standings[match["AwayTeam"]]

    hg = int(match["FTHG"])
    ag = int(match["FTAG"])

    home["P"] += 1
    away["P"] += 1

    home["GF"] += hg
    home["GA"] += ag
    away["GF"] += ag
    away["GA"] += hg

    if hg > ag:
        home["W"] += 1
        away["L"] += 1
        home["Pts"] += 3

    elif hg < ag:
        away["W"] += 1
        home["L"] += 1
        away["Pts"] += 3

    else:
        home["D"] += 1
        away["D"] += 1
        home["Pts"] += 1
        away["Pts"] += 1


table = pd.DataFrame(standings.values())
table["GD"] = table["GF"] - table["GA"]

table = table.sort_values(
    ["Pts", "GD", "GF", "Team"],
    ascending=[False, False, False, True],
).reset_index(drop=True)

table.insert(0, "Pos", range(1, len(table) + 1))
table.insert(1, "Crest", table["Team"].map(crests))


# ----------------------------------------
# UPCOMING FIXTURES
# ----------------------------------------

KEYS = ["Date", "HomeTeam", "AwayTeam"]

upcoming = matches[
    matches["Status"] != "FINISHED"
].copy()

upcoming = upcoming.merge(
    predictions[KEYS + ["HomeWin", "Draw", "AwayWin"]],
    on=KEYS,
    how="left",
    validate="one_to_one",
)

upcoming["Kickoff"] = pd.to_datetime(
    upcoming["Kickoff"], utc=True
).dt.tz_convert("Europe/Athens")

upcoming = upcoming.sort_values("Kickoff")


# ----------------------------------------
# HTML HELPERS
# ----------------------------------------

def clean_html(code):
    return "".join(
        line.strip() for line in code.splitlines()
    )


def team_html(match, large=False):
    home = html.escape(str(match["HomeTeam"]))
    away = html.escape(str(match["AwayTeam"]))

    home_crest = html.escape(
        str(match["HomeCrest"]), quote=True
    )
    away_crest = html.escape(
        str(match["AwayCrest"]), quote=True
    )

    style = "prediction-teams" if large else "match-teams"

    return f"""
        <div class="{style}">
            <div class="team">
                <img src="{home_crest}">
                <span>{home}</span>
            </div>

            <div class="vs">VS</div>

            <div class="team">
                <img src="{away_crest}">
                <span>{away}</span>
            </div>
        </div>
    """


def probability_html(home, draw, away, bars=False):
    if pd.isna(home) or pd.isna(draw) or pd.isna(away):
        return "<div>Prediction unavailable</div>"

    values = [
        ("Home", home),
        ("Draw", draw),
        ("Away", away),
    ]

    if bars:
        result = '<div class="prediction-bars">'

        for label, value in values:
            result += f"""
                <div>
                    <div class="bar-label">
                        <span>{label}</span>
                        <strong>{value:.1%}</strong>
                    </div>
                    <div class="bar-background">
                        <div class="bar-fill"
                             style="width:{value * 100:.1f}%">
                        </div>
                    </div>
                </div>
            """

        return result + "</div>"

    result = '<div class="probabilities">'

    for label, value in values:
        result += f"""
            <div>
                <div class="prob-label">{label}</div>
                <div class="prob-value">{value:.1%}</div>
            </div>
        """

    return result + "</div>"


def match_card(match, large=False):
    date_format = (
        "%A, %d %B %Y · %H:%M"
        if large else "%a %d %b · %H:%M"
    )

    date = match["Kickoff"].strftime(date_format)

    card_class = "prediction-card" if large else "match-card"

    card = f"""
        <div class="{card_class}">
            <div class="match-date">{date}</div>
            {team_html(match, large)}
            {probability_html(
                match["HomeWin"],
                match["Draw"],
                match["AwayWin"],
                bars=large
            )}
        </div>
    """

    st.markdown(
        clean_html(card),
        unsafe_allow_html=True,
    )


# ----------------------------------------
# HISTORICAL MODEL DATA
# ----------------------------------------

HISTORICAL_KEYS = [
    "Season", "Date", "HomeTeam", "AwayTeam"
]

MODEL_COLUMNS = {
    "Elo": [
        ["EloH", "EloD", "EloA"],
        ["EloHomeWin", "EloDraw", "EloAwayWin"],
    ],
    "Dixon-Coles": [
        ["DCH", "DCD", "DCA"],
        ["DCHomeWin", "DCDraw", "DCAwayWin"],
    ],
    "LightGBM": [
        ["LGBMH", "LGBMD", "LGBMA"],
        ["LGBMHomeWin", "LGBMDraw", "LGBMAwayWin"],
    ],
    "Ensemble": [
        ["EnsembleH", "EnsembleD", "EnsembleA"],
        ["HomeWin", "Draw", "AwayWin"],
    ],
}

MODEL_FILES = [
    "ensemble_predictions.csv",
    "dixon_coles_predictions.csv",
    "lightgbm_predictions.csv",
    "elo_predictions.csv",
]


@st.cache_data
def load_historical_data():
    historical = pd.read_csv(
        PROCESSED_DATA_DIR / "epl_matches_2014_2026.csv"
    )

    historical["Date"] = pd.to_datetime(
        historical["Date"]
    ).dt.strftime("%Y-%m-%d")

    historical = historical[
        historical["Season"].isin([
            "2021-22", "2022-23", "2023-24",
            "2024-25", "2025-26",
        ])
    ]

    if historical.duplicated(HISTORICAL_KEYS).any():
        raise ValueError("Duplicate historical fixtures")

    data = historical[
        HISTORICAL_KEYS
        + ["FTR", "B365H", "B365D", "B365A"]
    ].copy()

    found_models = []

    for filename in MODEL_FILES:
        path = PROCESSED_DATA_DIR / filename

        if not path.exists():
            continue

        source = pd.read_csv(path)

        source["Date"] = pd.to_datetime(
            source["Date"]
        ).dt.strftime("%Y-%m-%d")

        if source.duplicated(HISTORICAL_KEYS).any():
            raise ValueError(f"Duplicates in {filename}")

        for name, options in MODEL_COLUMNS.items():
            if name in found_models:
                continue

            for columns in options:
                if all(c in source.columns for c in columns):
                    selected = source[
                        HISTORICAL_KEYS + columns
                    ].copy()

                    selected = selected.rename(
                        columns=dict(zip(
                            columns,
                            [f"{name}_{x}" for x in "HDA"]
                        ))
                    )

                    data = data.merge(
                        selected,
                        on=HISTORICAL_KEYS,
                        how="left",
                        validate="one_to_one",
                    )

                    found_models.append(name)
                    break

    # Normalize Bet365 implied probabilities
    odds = data[
        ["B365H", "B365D", "B365A"]
    ].apply(pd.to_numeric, errors="coerce")

    odds = odds.where((odds > 1).all(axis=1))

    implied = 1 / odds
    implied = implied.div(
        implied.sum(axis=1),
        axis=0,
    )

    data["Bet365_H"] = implied["B365H"]
    data["Bet365_D"] = implied["B365D"]
    data["Bet365_A"] = implied["B365A"]

    return data, found_models


def calculate_metrics(actual, probabilities):
    actual_ids = pd.Series(actual).map({
        "H": 0, "D": 1, "A": 2
    }).to_numpy(dtype=int)

    predictions_ids = probabilities.argmax(axis=1)

    accuracy = (predictions_ids == actual_ids).mean()

    correct = probabilities[
        np.arange(len(actual_ids)), actual_ids
    ]

    log_loss = -np.log(
        np.clip(correct, 1e-15, 1)
    ).mean()

    one_hot = np.eye(3)[actual_ids]

    brier = np.mean(
        np.sum((probabilities - one_hot) ** 2, axis=1)
    )

    return accuracy * 100, log_loss, brier


def compare_models(data, models):
    # Compare every model and bookmaker on identical fixtures
    names = models + ["Bet365"]

    columns = [
        f"{name}_{outcome}"
        for name in names
        for outcome in "HDA"
    ]

    common = data.dropna(
        subset=["FTR"] + columns
    ).copy()

    common = common[
        common["FTR"].isin(["H", "D", "A"])
    ]

    for name in names:
        cols = [f"{name}_{x}" for x in "HDA"]

        probs = common[cols].to_numpy(dtype=float)

        valid = (
            np.isfinite(probs).all(axis=1)
            & (probs >= 0).all(axis=1)
            & np.isclose(probs.sum(axis=1), 1, atol=0.01)
        )

        common = common.loc[valid].copy()

    if common.empty:
        return pd.DataFrame(), common

    rows = []

    for name in names:
        cols = [f"{name}_{x}" for x in "HDA"]

        probs = common[cols].to_numpy(dtype=float)

        probs /= probs.sum(axis=1, keepdims=True)

        accuracy, loss, brier = calculate_metrics(
            common["FTR"].to_numpy(),
            probs,
        )

        rows.append({
            "Model": name,
            "Matches": len(common),
            "Accuracy": accuracy,
            "Log Loss": loss,
            "Brier": brier,
        })

    results = pd.DataFrame(rows)

    bookmaker = results[
        results["Model"] == "Bet365"
    ].iloc[0]

    # Positive differences mean the model beats Bet365
    results["Accuracy Difference"] = (
        results["Accuracy"] - bookmaker["Accuracy"]
    )

    results["Log Loss Difference"] = (
        bookmaker["Log Loss"] - results["Log Loss"]
    )

    results["Brier Difference"] = (
        bookmaker["Brier"] - results["Brier"]
    )

    return results, common


# ----------------------------------------
# HEADER
# ----------------------------------------

st.title("⚽ EPL Forecast Engine")

st.caption(
    "Premier League 2026–27  |  "
    "Elo · Dixon–Coles · LightGBM"
)

st.divider()


# ----------------------------------------
# TABS
# ----------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "🏆 Overview",
    "⚽ Match Predictions",
    "📊 Season Simulation",
    "🧠 Model Performance",
])


# ----------------------------------------
# TAB 1: OVERVIEW
# ----------------------------------------

with tab1:
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("Premier League Standings")

        st.caption(
            f"Current table · {len(completed)} "
            "of 380 matches played"
        )

        st.dataframe(
            table[
                [
                    "Pos", "Crest", "Team",
                    "P", "W", "D", "L",
                    "GF", "GA", "GD", "Pts",
                ]
            ],
            hide_index=True,
            use_container_width=True,
            height=(len(table) + 1) * 33 + 4,
            row_height=33,
            column_config={
                "Pos": st.column_config.NumberColumn(
                    "#", width="small"
                ),
                "Crest": st.column_config.ImageColumn(
                    "", width="small"
                ),
                "Team": st.column_config.TextColumn(
                    "Club", width="medium"
                ),
            },
        )

        st.caption(
            "P: Played · W: Won · D: Drawn · "
            "L: Lost · GF: Goals For · "
            "GA: Goals Against · GD: Goal Difference"
        )

    with right:
        if upcoming.empty:
            st.info("No upcoming fixtures.")
        else:
            next_matchday = int(
                upcoming.iloc[0]["Matchday"]
            )

            weekly = upcoming[
                upcoming["Matchday"] == next_matchday
            ].sort_values("Kickoff")

            st.subheader(f"Matchweek {next_matchday}")
            st.caption("Upcoming fixtures · Athens time")

            for _, match in weekly.iterrows():
                match_card(match)


# ----------------------------------------
# TAB 2: MATCH PREDICTIONS
# ----------------------------------------

with tab2:
    st.subheader("Match Predictions")

    st.caption(
        "Upcoming fixtures and ensemble probabilities "
        "· Athens time"
    )

    if upcoming.empty:
        st.info("No remaining fixtures.")

    else:
        left_filter, right_filter = st.columns(2)

        with left_filter:
            weeks = sorted(
                upcoming["Matchday"].dropna().unique()
            )

            selected_week = st.selectbox(
                "Matchweek",
                weeks,
                format_func=lambda x: (
                    f"Matchweek {int(x)}"
                ),
            )

        with right_filter:
            selected_team = st.selectbox(
                "Team",
                ["All Teams"] + teams,
            )

        filtered = upcoming[
            upcoming["Matchday"] == selected_week
        ].copy()

        if selected_team != "All Teams":
            filtered = filtered[
                (filtered["HomeTeam"] == selected_team)
                | (filtered["AwayTeam"] == selected_team)
            ]

        filtered = filtered.sort_values("Kickoff")

        st.caption(
            f"{len(filtered)} fixture(s) · "
            f"Matchweek {int(selected_week)}"
        )

        if filtered.empty:
            st.info("No fixtures for this selection.")

        else:
            for start in range(0, len(filtered), 2):
                columns = st.columns(2, gap="medium")

                for i, column in enumerate(columns):
                    index = start + i

                    if index < len(filtered):
                        with column:
                            match_card(
                                filtered.iloc[index],
                                large=True,
                            )


# ----------------------------------------
# TAB 3: SEASON SIMULATION
# ----------------------------------------

with tab3:
    st.subheader("Monte Carlo Season Simulation")

    st.caption(
        "Projected final standings based on "
        "10,000 simulated seasons."
    )

    projected = simulations.sort_values(
        "ExpectedPosition"
    ).reset_index(drop=True).copy()

    projected.insert(
        0, "Pos", range(1, len(projected) + 1)
    )
    projected.insert(
        1, "Crest", projected["Team"].map(crests)
    )

    for col in ["Title", "Top4", "Relegation"]:
        projected[col] *= 100

    st.markdown("#### Projected Final Standings")

    st.dataframe(
        projected[
            [
                "Pos", "Crest", "Team",
                "ExpectedPoints", "ExpectedPosition",
                "Title", "Top4", "Relegation",
            ]
        ],
        hide_index=True,
        use_container_width=True,
        height=(len(projected) + 1) * 33 + 4,
        row_height=33,
        column_config={
            "Pos": st.column_config.NumberColumn(
                "#", width="small"
            ),
            "Crest": st.column_config.ImageColumn(
                "", width="small"
            ),
            "Team": st.column_config.TextColumn(
                "Club", width="medium"
            ),
            "ExpectedPoints": st.column_config.NumberColumn(
                "Expected Pts", format="%.1f"
            ),
            "ExpectedPosition": st.column_config.NumberColumn(
                "Expected Pos", format="%.1f"
            ),
            "Title": st.column_config.NumberColumn(
                "Title %", format="%.1f%%"
            ),
            "Top4": st.column_config.NumberColumn(
                "Top 4 %", format="%.1f%%"
            ),
            "Relegation": st.column_config.NumberColumn(
                "Relegation %", format="%.1f%%"
            ),
        },
    )

    st.divider()

    st.markdown("#### Explore a Team")

    selected = st.selectbox(
        "Choose a club",
        teams,
        key="simulation_team",
    )

    team_data = simulations[
        simulations["Team"] == selected
    ].iloc[0]

    team_html = f"""
        <div class="selected-team">
            <img src="{html.escape(str(crests[selected]), quote=True)}">
            <span>{html.escape(selected)}</span>
        </div>
    """

    st.markdown(
        clean_html(team_html),
        unsafe_allow_html=True,
    )

    current = table[
        table["Team"] == selected
    ].iloc[0]

    stats = [
        ("Current Points", str(int(current["Pts"]))),
        ("Expected Points", f"{team_data['ExpectedPoints']:.1f}"),
        ("Current Position", str(int(current["Pos"]))),
        ("Expected Position", f"{team_data['ExpectedPosition']:.1f}"),
    ]

    stats_html = '<div class="simulation-stats">'

    for label, value in stats:
        stats_html += f"""
            <div class="stat-box">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value}</div>
            </div>
        """

    stats_html += "</div>"

    st.markdown(
        clean_html(stats_html),
        unsafe_allow_html=True,
    )

    chances = [
        ("Title", team_data["Title"]),
        ("Top 4", team_data["Top4"]),
        ("Relegation", team_data["Relegation"]),
    ]

    chances_html = (
        '<div class="prediction-card">'
        '<div class="prediction-bars">'
    )

    for label, value in chances:
        chances_html += f"""
            <div>
                <div class="bar-label">
                    <span>{label}</span>
                    <strong>{value:.1%}</strong>
                </div>
                <div class="bar-background">
                    <div class="bar-fill"
                         style="width:{value * 100:.1f}%">
                    </div>
                </div>
            </div>
        """

    chances_html += "</div></div>"

    st.markdown(
        clean_html(chances_html),
        unsafe_allow_html=True,
    )

    st.download_button(
        "Download Simulation Results",
        data=simulations.to_csv(index=False),
        file_name="season_simulation.csv",
        mime="text/csv",
    )



# ----------------------------------------
# TAB 4: MODEL PERFORMANCE
# ----------------------------------------

with tab4:
    st.subheader("Historical Model Performance")

    st.caption(
        "Historical performance comparison between "
        "our prediction models and Bet365."
    )

    period = st.radio(
        "Evaluation period",
        [
            "Validation (2021–2025)",
            "2025–26",
        ],
        horizontal=True,
    )

    try:
        history, found_models = load_historical_data()

        if period == "2025–26":
            subset = history[
                history["Season"] == "2025-26"
            ].copy()
        else:
            subset = history[
                history["Season"].isin([
                    "2021-22", "2022-23",
                    "2023-24", "2024-25",
                ])
            ].copy()

        if not found_models:
            st.warning(
                "No historical model predictions were found."
            )

        else:
            results, compared = compare_models(
                subset, found_models
            )

            if results.empty:
                st.warning(
                    "No complete matches are available "
                    "for comparison."
                )

            else:
                bookmaker = results[
                    results["Model"] == "Bet365"
                ].iloc[0]

                st.caption(
                    f"Based on {len(compared)} "
                    "identical historical fixtures."
                )

                st.markdown("#### Model Comparison")

                display = results.copy()

                # Show Bet365 first, followed by our models
                display["Order"] = (
                    display["Model"] != "Bet365"
                ).astype(int)

                display = display.sort_values(
                    ["Order", "Log Loss"]
                ).drop(columns="Order")

                # Add differences in parentheses
                def format_accuracy(row):
                    value = f"{row['Accuracy']:.2f}%"

                    if row["Model"] == "Bet365":
                        return value

                    difference = (
                        row["Accuracy"]
                        - bookmaker["Accuracy"]
                    )

                    return f"{value} ({difference:+.2f} pp)"

                def format_loss(row):
                    value = f"{row['Log Loss']:.4f}"

                    if row["Model"] == "Bet365":
                        return value

                    difference = (
                        row["Log Loss"]
                        - bookmaker["Log Loss"]
                    )

                    return f"{value} ({difference:+.4f})"

                def format_brier(row):
                    value = f"{row['Brier']:.4f}"

                    if row["Model"] == "Bet365":
                        return value

                    difference = (
                        row["Brier"]
                        - bookmaker["Brier"]
                    )

                    return f"{value} ({difference:+.4f})"

                display["Accuracy"] = display.apply(
                    format_accuracy, axis=1
                )

                display["Log Loss"] = display.apply(
                    format_loss, axis=1
                )

                display["Brier"] = display.apply(
                    format_brier, axis=1
                )

                st.dataframe(
                    display[
                        [
                            "Model",
                            "Accuracy",
                            "Log Loss",
                            "Brier",
                        ]
                    ],
                    hide_index=True,
                    use_container_width=True,
                    height=(len(display) + 1) * 35 + 4,
                    row_height=35,
                    column_config={
                        "Model": st.column_config.TextColumn(
                            "Model", width="medium"
                        ),
                        "Accuracy": st.column_config.TextColumn(
                            "Accuracy", width="medium"
                        ),
                        "Log Loss": st.column_config.TextColumn(
                            "Log Loss", width="medium"
                        ),
                        "Brier": st.column_config.TextColumn(
                            "Brier Score", width="medium"
                        ),
                    },
                )

                st.caption(
                    "Values in parentheses show the difference "
                    "from Bet365. Higher accuracy and lower "
                    "log loss/Brier score are better. "
                    "Accuracy differences are percentage points."
                )

                st.download_button(
                    "Download Performance Results",
                    data=results.to_csv(index=False),
                    file_name="model_performance.csv",
                    mime="text/csv",
                )

    except (FileNotFoundError, ValueError, KeyError) as error:
        st.error(
            f"Could not load model performance: {error}"
        )



# ----------------------------------------
# FOOTER
# ----------------------------------------

st.divider()

st.caption(
    "EPL Forecast Engine · Predictions are probabilistic "
    "estimates, not guaranteed results."
)
