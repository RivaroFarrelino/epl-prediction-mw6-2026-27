# Premier League Score Predictions

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RivaroFarrelino/epl-prediction-mw6-2026-27/blob/main/notebooks/epl_score_prediction.ipynb)

Predicting every scoreline in a Premier League matchweek using free data and a Dixon-Coles model.

I built this during the October 2026 international break to predict Matchweek 6 of the 2026/27 season. It works for any matchweek, though.

## How it works

Every team gets two numbers: how good they are at scoring and how good they are at keeping goals out. The model learns them from past results, then uses them to work out how many goals each side is likely to score in a given match. From there you get a probability for every possible scoreline.

A few things on top of that:

- **Recent games count more.** A match from last month says more about a team than one from two seasons ago, so older results are weighted down over time.
- **Home advantage** is part of the model.
- **Low scores get a small correction.** A plain Poisson model slightly misjudges how often 0-0, 1-0, 0-1 and 1-1 happen. Dixon-Coles fixes that.
- **Promoted teams aren't a blank slate.** Championship results go into the same model, and the teams that moved between divisions link the two leagues. That's how Coventry and Hull get a sensible rating after only five Premier League games.

Training data is the last two seasons (2024/25 and 2025/26) plus whatever has been played so far this season.

## Data

Everything is free and needs no API key.

| Source | Used for |
|---|---|
| [football-data.co.uk](https://www.football-data.co.uk) | Results and closing odds for the Premier League (`E0`) and Championship (`E1`) |
| [Fantasy Premier League API](https://fantasy.premierleague.com/api/fixtures/) | Fixtures and kickoff times |

## Running it

### Google Colab

Click the badge at the top, then Runtime > Run all. Nothing to install.

Change `MATCHWEEK` in the config cell to predict a different round. Set `USE_DRIVE = True` if you want the data cache and outputs saved to your Google Drive.

### Locally

```bash
git clone https://github.com/RivaroFarrelino/epl-prediction-mw6-2026-27.git
cd epl-prediction-mw6-2026-27
pip install -r requirements.txt
python src/predict.py --matchweek 6 --backtest
```

| Option | Default | What it does |
|---|---|---|
| `--matchweek` | `6` | Matchweek to predict |
| `--xi` | `0.0019` | Time decay per day (about a one-year half-life) |
| `--l2` | `1.0` | Regularization strength |
| `--backtest` | off | Test on the second half of 2025/26 first |
| `--refresh` | off | Re-download all CSVs |
| `--debug` | off | Verbose logging |

## Output

| File | Contents |
|---|---|
| `predictions_mw{N}.csv` | Expected goals, win/draw/loss probabilities, top 3 scorelines, over 2.5 and both-teams-to-score for each match |
| `team_ratings.csv` | Attack and defence rating for every Premier League team |
| `backtest_detail.csv` | Match-by-match backtest results |

The notebook also has an outcome chart for the whole matchweek and an interactive scoreline heatmap for each match.

## Checking the model

The backtest goes through the second half of 2025/26 one week at a time. Each week it refits using only the data available before that week, predicts the games, and compares the result with the bookmakers' closing odds.

Two metrics are reported, log loss and ranked probability score (RPS). Lower is better for both. Bookmaker odds are a tough baseline, so getting close to them is already a good sign.

## Limitations

- The model only knows results. It has no idea about injuries, suspensions, managers or transfers.
- The single most likely scoreline is almost always something like 1-0, 1-1 or 2-0, because low scores are the most common. The full probability spread is more useful than one number.
- Five matchweeks is a small sample, which is why older seasons are still in the training data.

## Project structure

```
epl-prediction-mw6-2026-27/
├── notebooks/
│   └── epl_score_prediction.ipynb
├── src/
│   └── predict.py
├── requirements.txt
├── LICENSE
└── README.md
```

## References

Dixon, M. J. and Coles, S. G. (1997). Modelling association football scores and inefficiencies in the football betting market. *Journal of the Royal Statistical Society: Series C*, 46(2), 265-280.

## License

MIT
