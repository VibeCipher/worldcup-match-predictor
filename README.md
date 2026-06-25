# World Cup Match Predictor

Predicts international football match outcomes using historical form, head-to-head record, and venue context — trained on 49,000+ **international matches** (covering all tournaments and friendlies, not just World Cups) from 1872 to 2026, validated live against the 2026 FIFA World Cup.

> Note: the FIFA World Cup itself started in 1930. The 1872 date marks the first ever official international football match (Scotland vs England) — the training data spans all international matches since then, while validation is specifically against World Cup 2026 results.

---

## What it does

- Trains 3 models (Logistic Regression, Random Forest, XGBoost) to predict Win / Draw / Loss probabilities for any international football match
- Engineers rolling form features (last 5 / last 10 matches), goals for/against, and head-to-head history between any two teams
- Validates predictions against **real World Cup 2026 results** using a walk-forward backtest — no hindsight, no data leakage
- Predicts any matchup, including teams that have never played each other before

---

## Live validation result

**64.6% ensemble accuracy across 48 real World Cup 2026 group stage matches** (vs. a 48% baseline of always predicting "Home Win") — using a true walk-forward backtest where each prediction was generated using only data available *before* that match was played.

The final prediction averages probabilities across all 3 trained models (Logistic Regression, Random Forest, XGBoost) into a single combined result, rather than reporting 3 separate numbers. Verified on the backtest: ensembling matches or beats every individual model and is never worse than the weakest one.

```
LIVE PREDICTION (genuinely unplayed match, made before kickoff):

Colombia vs Portugal — June 27, 2026, Miami Gardens
  Colombia Win: 37.9%  |  Portugal Win: 36.8%  |  Draw: 25.3%
  (essentially a toss-up — first ever meeting, zero head-to-head history)
```

The 48-match backtest includes one notable example: the model correctly predicted Argentina to beat Austria before checking the real result (Argentina won 2-0). Most misses across the backtest are the model predicting a winner when the actual result was a draw — a well-documented hard problem in football analytics; draws require very specific conditions that are difficult to capture from form data alone.

---

## How it works

```
49,000+ international matches (1872-2026)
            ↓
Convert to long format (1 row per team per match)
            ↓
Rolling form: avg goals for/against, points (last 5 & 10 matches)
shift(1) ensures only PAST matches are used — no leakage
            ↓
Head-to-head record between each specific pair of teams
built chronologically, no future meetings leak into past rows
            ↓
Train 3 models with a TIME-BASED split
(not random — prevents the model "seeing the future")
            ↓
Walk-forward backtest on real World Cup 2026 matches
            ↓
Predict any matchup using current team form
```

---

## Quickstart

Open `worldcup_predictor.py` in Google Colab and run cells top to bottom. No API key, no signup needed — the dataset downloads directly from a public GitHub mirror.

```python
predict_match("Colombia", "Portugal", neutral=True)
```

```
Prediction: Colombia vs Portugal (neutral venue: True)
Head-to-head meetings: 0

Combined prediction (average of Logistic Regression, Random Forest, XGBoost):
  Portugal Win: 36.8%
  Draw: 25.3%
  Colombia Win: 37.9%
```

---

## A real bug caught and fixed during development

The raw dataset has 4 rows (out of 49,445) that are genuine duplicate fixtures — e.g. a 1974 Tahiti vs New Caledonia double-header recorded twice. Merging features back using `date + team names` as the join key silently exploded these into duplicate rows during feature engineering. Fixed by assigning a unique `match_id` before any transformation and merging on that instead — guarantees a 1:1 merge regardless of data quality issues in the source.

This is exactly the kind of subtle bug that real-world messy data produces, and catching it is part of why a walk-forward backtest against real results matters more than any synthetic accuracy number.

---

## File structure

```
worldcup-predictor/
│
├── worldcup_predictor.py   # Full notebook — open in Colab
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── sample_data/
    └── wc2026_backtest.csv # Full backtest results, all 48 matches
```

---

## Tech stack

- **Python** — pandas, numpy, scikit-learn, XGBoost
- **Data source** — [martj42/international_results](https://github.com/martj42/international_results) (49,000+ matches, updated daily)
- **Google Colab** — free cloud runtime, no setup needed

---

## Known limitations

- Draws are hard to predict — the model tends to predict a winner even in close matches
- No player-level data (injuries, suspensions, lineup changes) — purely team-level historical form
- Early matches in a team's history have less reliable rolling stats (fewer data points to average)
- This is a portfolio/learning project, not a betting tool — sports outcomes carry genuine uncertainty

---

## Why this project

Built during the 2026 FIFA World Cup to apply proper ML methodology — time-based validation, no data leakage, and honest backtesting against real outcomes — to a genuinely fun, verifiable problem. The headline prediction (Colombia vs Portugal) was generated for a match that genuinely hadn't been played yet, and the same methodology is validated across all 48 completed group stage matches in the walk-forward backtest.

---

## License

MIT — free to use, modify, and share.
