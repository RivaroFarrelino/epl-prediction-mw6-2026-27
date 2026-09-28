import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson

TRAIN_SEASONS = ["2425", "2526"]
CURRENT_SEASON = "2627"
LEAGUES = ["E0", "E1"]
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"
FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
HEADERS = {"User-Agent": "Mozilla/5.0 (epl-predictor)"}
MAX_GOALS = 10
XI = 0.0019
L2 = 1.0
RETRIES = 5
RETRY_WAIT = 5
STALE_SECONDS = 12 * 3600
TZ = "Asia/Jakarta"

FPL_TO_FD = {
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
    "Nott'm Forest": "Nott'm Forest",
}

FALLBACK_FIXTURES = {
    6: [
        ("Arsenal", "Leeds"),
        ("Aston Villa", "Brentford"),
        ("Chelsea", "Bournemouth"),
        ("Ipswich", "Fulham"),
        ("Sunderland", "Brighton"),
        ("Man United", "Tottenham"),
        ("Crystal Palace", "Nott'm Forest"),
        ("Hull", "Everton"),
        ("Liverpool", "Man City"),
        ("Coventry", "Newcastle"),
    ]
}

ODDS_PRIORITY = [
    ("PSCH", "PSCD", "PSCA"),
    ("AvgCH", "AvgCD", "AvgCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("PSH", "PSD", "PSA"),
    ("AvgH", "AvgD", "AvgA"),
    ("B365H", "B365D", "B365A"),
]

log = logging.getLogger("epl")


def setup_logging(debug):
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def http_get(url):
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
            log.debug("GET %s -> %d (%d bytes)", url, r.status_code, len(r.content))
            return r
        except requests.RequestException as e:
            log.warning("GET failed %s (attempt %d/%d): %s", url, attempt, RETRIES, e)
            if attempt == RETRIES:
                raise
            time.sleep(RETRY_WAIT)


def load_season(data_dir, league, season, refresh):
    path = data_dir / f"{league}_{season}.csv"
    stale = season == CURRENT_SEASON and path.exists() and time.time() - path.stat().st_mtime > STALE_SECONDS
    if refresh or stale or not path.exists():
        url = BASE_URL.format(season=season, league=league)
        try:
            path.write_bytes(http_get(url).content)
            log.info("Downloaded %s", url)
        except requests.RequestException:
            if not path.exists():
                raise
            log.warning("Download failed, using old cache %s", path)
    else:
        log.info("Using cache %s", path.name)

    raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime(raw["Date"], dayfirst=True, format="mixed"),
            "HomeTeam": raw["HomeTeam"].str.strip(),
            "AwayTeam": raw["AwayTeam"].str.strip(),
            "FTHG": raw["FTHG"].astype(int),
            "FTAG": raw["FTAG"].astype(int),
            "League": league,
            "Season": season,
        }
    )
    for side, i in (("OddsH", 0), ("OddsD", 1), ("OddsA", 2)):
        col = pd.Series(np.nan, index=raw.index)
        for triple in ODDS_PRIORITY:
            if all(c in raw.columns for c in triple):
                col = col.fillna(pd.to_numeric(raw[triple[i]], errors="coerce"))
        df[side] = col.to_numpy()
    log.info("%s %s: %d matches (%s to %s)", league, season, len(df), df.Date.min().date(), df.Date.max().date())
    return df


def load_all(data_dir, refresh):
    data_dir.mkdir(parents=True, exist_ok=True)
    frames = [load_season(data_dir, lg, s, refresh) for s in TRAIN_SEASONS + [CURRENT_SEASON] for lg in LEAGUES]
    df = pd.concat(frames, ignore_index=True).sort_values("Date", ignore_index=True)
    log.info("Total: %d matches, %d teams", len(df), len(set(df.HomeTeam) | set(df.AwayTeam)))
    return df


class DixonColes:
    def __init__(self, xi=XI, l2=L2):
        self.xi = xi
        self.l2 = l2

    def fit(self, df, ref_date, init=None):
        t0 = time.time()
        self.teams = sorted(set(df.HomeTeam) | set(df.AwayTeam))
        self.idx = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)
        h = df.HomeTeam.map(self.idx).to_numpy()
        a = df.AwayTeam.map(self.idx).to_numpy()
        x = df.FTHG.to_numpy(dtype=float)
        y = df.FTAG.to_numpy(dtype=float)
        champ = (df.League == "E1").to_numpy(dtype=float)
        days = (pd.Timestamp(ref_date) - df.Date).dt.days.clip(lower=0).to_numpy(dtype=float)
        w = np.exp(-self.xi * days)
        m = [(x == 0) & (y == 0), (x == 0) & (y == 1), (x == 1) & (y == 0), (x == 1) & (y == 1)]
        lgf = gammaln(x + 1) + gammaln(y + 1)
        l2 = self.l2

        def objective(p):
            att, dfn = p[:n], p[n:2 * n]
            home, mu0, off, rho = p[2 * n:]
            ll = mu0 + home + att[h] + dfn[a] + off * champ
            lm = mu0 + att[a] + dfn[h] + off * champ
            lam, mu = np.exp(ll), np.exp(lm)
            tau = np.select(m, [1 - lam * mu * rho, 1 + lam * rho, 1 + mu * rho, np.full_like(lam, 1 - rho)], 1.0)
            tau = np.maximum(tau, 1e-10)
            logl = np.log(tau) + x * ll - lam + y * lm - mu - lgf
            nll = -(w * logl).sum() + l2 * (att @ att + dfn @ dfn)

            d00 = -lam * mu * rho / tau
            dll = x - lam + np.select([m[0], m[1]], [d00, lam * rho / tau], 0.0)
            dlm = y - mu + np.select([m[0], m[2]], [d00, mu * rho / tau], 0.0)
            drho = np.select(m, [-lam * mu / tau, lam / tau, mu / tau, -1 / tau], 0.0)
            gl, gm = w * dll, w * dlm
            g_att = np.bincount(h, gl, n) + np.bincount(a, gm, n)
            g_def = np.bincount(a, gl, n) + np.bincount(h, gm, n)
            grad = -np.concatenate([g_att, g_def, [gl.sum(), gl.sum() + gm.sum(), ((gl + gm) * champ).sum(), (w * drho).sum()]])
            grad[:n] += 2 * l2 * att
            grad[n:2 * n] += 2 * l2 * dfn
            return nll, grad

        p0 = np.zeros(2 * n + 4)
        p0[2 * n:] = [0.25, np.log(max((x.mean() + y.mean()) / 2, 0.1)), 0.0, 0.0]
        if init is not None:
            for t, i in self.idx.items():
                if t in init.idx:
                    p0[i], p0[n + i] = init.att[init.idx[t]], init.dfn[init.idx[t]]
            p0[2 * n:] = [init.home, init.mu0, init.off, init.rho]

        bounds = [(None, None)] * (2 * n + 3) + [(-0.25, 0.25)]
        res = minimize(objective, p0, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000})
        if not res.success:
            log.warning("Optimizer did not converge: %s", res.message)
        p = res.x
        self.att, self.dfn = p[:n], p[n:2 * n]
        self.home, self.mu0, self.off, self.rho = p[2 * n:]
        log.debug(
            "Fit %d matches, %d teams, %d iterations, %.2fs | home=%.3f rho=%.3f offset_E1=%.3f",
            len(df), n, res.nit, time.time() - t0, self.home, self.rho, self.off,
        )
        return self

    def score_matrix(self, home, away):
        i, j = self.idx[home], self.idx[away]
        lam = np.exp(self.mu0 + self.home + self.att[i] + self.dfn[j])
        mu = np.exp(self.mu0 + self.att[j] + self.dfn[i])
        g = np.arange(MAX_GOALS + 1)
        mat = np.outer(poisson.pmf(g, lam), poisson.pmf(g, mu))
        mat[0, 0] *= 1 - lam * mu * self.rho
        mat[0, 1] *= 1 + lam * self.rho
        mat[1, 0] *= 1 + mu * self.rho
        mat[1, 1] *= 1 - self.rho
        return mat / mat.sum(), lam, mu

    def ratings(self):
        return pd.DataFrame(
            {"Team": self.teams, "Attack": np.exp(self.att), "Defence": np.exp(self.dfn)}
        ).assign(Rating=lambda d: d.Attack / d.Defence).sort_values("Rating", ascending=False, ignore_index=True)


def outcome_probs(mat):
    return np.tril(mat, -1).sum(), np.trace(mat), np.triu(mat, 1).sum()


def summarize(mat, lam, mu):
    ph, pd_, pa = outcome_probs(mat)
    flat = np.argsort(mat, axis=None)[::-1][:3]
    top = [f"{i}-{j} ({mat[i, j]:.1%})" for i, j in zip(*np.unravel_index(flat, mat.shape))]
    g = np.arange(MAX_GOALS + 1)
    total = g[:, None] + g[None, :]
    return {
        "xG_Home": round(lam, 2),
        "xG_Away": round(mu, 2),
        "P_Home": round(ph, 3),
        "P_Draw": round(pd_, 3),
        "P_Away": round(pa, 3),
        "Most_Likely": top[0].split(" ")[0],
        "Top3_Scores": " | ".join(top),
        "Over2.5": round(mat[total >= 3].sum(), 3),
        "BTTS": round(mat[1:, 1:].sum(), 3),
    }


def fetch_fixtures(mw):
    try:
        teams = {t["id"]: t["name"] for t in http_get(FPL_BOOTSTRAP_URL).json()["teams"]}
        rows = []
        for f in http_get(FPL_FIXTURES_URL).json():
            if f.get("event") != mw:
                continue
            kickoff = f.get("kickoff_time")
            rows.append(
                {
                    "Kickoff_WIB": pd.to_datetime(kickoff, utc=True).tz_convert(TZ).strftime("%a %d %b %H:%M") if kickoff else "",
                    "Home": FPL_TO_FD.get(teams[f["team_h"]], teams[f["team_h"]]),
                    "Away": FPL_TO_FD.get(teams[f["team_a"]], teams[f["team_a"]]),
                    "Actual": f"{f['team_h_score']}-{f['team_a_score']}" if f.get("finished") else "",
                }
            )
        if rows:
            log.info("Got %d fixtures for MW%d from the FPL API", len(rows), mw)
            return pd.DataFrame(rows)
        log.warning("FPL API returned no fixtures for MW%d", mw)
    except Exception as e:
        log.warning("Could not get fixtures from the FPL API: %s", e)
    if mw not in FALLBACK_FIXTURES:
        raise SystemExit(f"No fixtures for MW{mw}. Add them to FALLBACK_FIXTURES by hand.")
    log.info("Using FALLBACK_FIXTURES for MW%d", mw)
    return pd.DataFrame([{"Kickoff_WIB": "", "Home": h, "Away": a, "Actual": ""} for h, a in FALLBACK_FIXTURES[mw]])


def predict(df, mw, xi, l2, out_dir):
    fixtures = fetch_fixtures(mw)
    ref_date = pd.Timestamp(datetime.now(timezone.utc).date())
    model = DixonColes(xi, l2).fit(df[df.Date < ref_date], ref_date)
    log.info("Model: home_adv=x%.3f rho=%.3f offset_E1=x%.3f", np.exp(model.home), model.rho, np.exp(model.off))

    rows = []
    for fx in fixtures.itertuples(index=False):
        missing = [t for t in (fx.Home, fx.Away) if t not in model.idx]
        if missing:
            log.error("Unknown team %s. Add it to FPL_TO_FD. Known teams: %s", missing, ", ".join(model.teams))
            continue
        mat, lam, mu = model.score_matrix(fx.Home, fx.Away)
        rows.append({**fx._asdict(), **summarize(mat, lam, mu)})
    result = pd.DataFrame(rows)

    epl_teams = set(df.loc[(df.Season == CURRENT_SEASON) & (df.League == "E0"), "HomeTeam"]) | set(fixtures.Home) | set(fixtures.Away)
    ratings = model.ratings()
    ratings = ratings[ratings.Team.isin(epl_teams)].reset_index(drop=True)

    pred_path = out_dir / f"predictions_mw{mw}.csv"
    rating_path = out_dir / "team_ratings.csv"
    result.to_csv(pred_path, index=False)
    ratings.round(3).to_csv(rating_path, index=False)

    view = result[["Home", "Away", "xG_Home", "xG_Away", "P_Home", "P_Draw", "P_Away", "Most_Likely", "Over2.5", "BTTS"]]
    print(f"\n=== Premier League Matchweek {mw} predictions ===")
    print(view.to_string(index=False))
    print("\n=== Team ratings (attack > 1 scores more than average, defence < 1 concedes less) ===")
    print(ratings.round(3).to_string(index=False))
    log.info("Saved %s and %s", pred_path, rating_path)


def rps(p, o):
    cp, co = np.cumsum(p, axis=1)[:, :2], np.cumsum(o, axis=1)[:, :2]
    return ((cp - co) ** 2).sum(axis=1) / 2


def backtest(df, xi, l2, start, end, out_dir):
    test = df[(df.League == "E0") & (df.Date >= start) & (df.Date < end)]
    if test.empty:
        log.error("No test matches between %s and %s", start, end)
        return
    log.info("Backtesting %d EPL matches (%s to %s), refitting weekly", len(test), start, end)
    model, rows = None, []
    for _, block in test.groupby(test.Date.dt.to_period("W")):
        cutoff = block.Date.min()
        model = DixonColes(xi, l2).fit(df[df.Date < cutoff], cutoff, init=model)
        for r in block.itertuples(index=False):
            if r.HomeTeam not in model.idx or r.AwayTeam not in model.idx:
                log.debug("Skipping %s vs %s, team not in training data yet", r.HomeTeam, r.AwayTeam)
                continue
            mat, _, _ = model.score_matrix(r.HomeTeam, r.AwayTeam)
            i, j = np.unravel_index(mat.argmax(), mat.shape)
            ph, pd_, pa = outcome_probs(mat)
            rows.append(
                {
                    "Date": r.Date.date(), "Home": r.HomeTeam, "Away": r.AwayTeam,
                    "Score": f"{r.FTHG}-{r.FTAG}", "Predicted": f"{i}-{j}",
                    "pH": ph, "pD": pd_, "pA": pa,
                    "res": 0 if r.FTHG > r.FTAG else 1 if r.FTHG == r.FTAG else 2,
                    "OddsH": r.OddsH, "OddsD": r.OddsD, "OddsA": r.OddsA,
                }
            )
        log.debug("Week of %s done, %d matches so far", cutoff.date(), len(rows))

    bt = pd.DataFrame(rows)
    onehot = np.eye(3)[bt.res]
    pm = bt[["pH", "pD", "pA"]].to_numpy()
    report = {
        "Matches": len(bt),
        "1X2 accuracy": (pm.argmax(1) == bt.res).mean(),
        "Exact score accuracy": (bt.Score == bt.Predicted).mean(),
        "Log loss (model)": -np.log(pm[np.arange(len(bt)), bt.res]).mean(),
        "RPS (model)": rps(pm, onehot).mean(),
    }
    has_odds = bt[["OddsH", "OddsD", "OddsA"]].notna().all(axis=1).to_numpy()
    if has_odds.any():
        inv = 1 / bt.loc[has_odds, ["OddsH", "OddsD", "OddsA"]].to_numpy()
        pb = inv / inv.sum(axis=1, keepdims=True)
        res_b = bt.res.to_numpy()[has_odds]
        report["Log loss (model, matches with odds)"] = -np.log(pm[has_odds][np.arange(has_odds.sum()), res_b]).mean()
        report["Log loss (bookmakers)"] = -np.log(pb[np.arange(len(pb)), res_b]).mean()
        report["RPS (bookmakers)"] = rps(pb, onehot[has_odds]).mean()

    path = out_dir / "backtest_detail.csv"
    bt.drop(columns="res").round(3).to_csv(path, index=False)
    print("\n=== Backtest (lower log loss / RPS is better) ===")
    for k, v in report.items():
        print(f"{k:38s}: {v:.4f}" if isinstance(v, float) else f"{k:38s}: {v}")
    log.info("Backtest detail saved to %s", path)


def main():
    ap = argparse.ArgumentParser(description="Premier League score predictions with a Dixon-Coles model")
    ap.add_argument("--matchweek", type=int, default=6)
    ap.add_argument("--xi", type=float, default=XI)
    ap.add_argument("--l2", type=float, default=L2)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out-dir", type=Path, default=Path("output"))
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--bt-start", default="2026-01-01")
    ap.add_argument("--bt-end", default="2026-06-01")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    setup_logging(args.debug)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = load_all(args.data_dir, args.refresh)
    if args.backtest:
        backtest(df, args.xi, args.l2, args.bt_start, args.bt_end, args.out_dir)
    predict(df, args.matchweek, args.xi, args.l2, args.out_dir)
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
