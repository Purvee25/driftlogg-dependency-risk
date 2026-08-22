# DriftLogg

Predicts which of your open-source dependencies are likely to go silent in the
next 90 days — before the security patches quietly stop arriving.

## The problem

A typical project depends on hundreds of packages maintained by volunteers.
When a maintainer quits, the package doesn't break — it quietly stops receiving
security patches. Teams usually discover this months later, when a scanner
flags a vulnerability in something that went dark last year.

Abandonment isn't sudden. It's visible in public months ahead: commit velocity
falls, issue responses slow from a day to never, the backlog grows, the
contributor count drops to one. DriftLogg learns that decay pattern from
packages that already died and applies it to the ones you depend on now.

## Approach

Supervised binary classification on time-series features.

- **Label** — sustained silence: a package alive at `as_of` that then records
  no commits at all through a 180-day verdict window opening 90 days later.
  See "What the first model taught us" below for why this replaced the
  `archived` flag.
- **Features** — activity, contributor, responsiveness, backlog, and release
  signals, computed as *trends* rather than levels. A package slowing from
  weekly commits to nothing matters far more than one that was always quiet.
- **Model** — LightGBM over tabular features. Trees beat a neural net here on
  data this size, and the per-prediction attributions are what let the tool
  explain *why* a package was flagged.
- **Evaluation** — PR-AUC (not ROC-AUC, given the imbalance) and precision@k.

### The two things that decide whether this works

**1. Leakage.** Every feature is computed with an `as_of` cutoff and may only
use data from strictly before it. A feature that peeks past the cutoff produces
a model that scores brilliantly in testing and fails completely in production.
`FeatureWindow.filter_events` is the single chokepoint, and
`tests/test_features.py` asserts that adding future data changes nothing.

**2. Label quality.** A stable library quiet for a quarter is not abandoned,
and a dead package nobody archived is not alive. Getting this wrong in either
direction sinks the project — and the first version got it wrong in the second
direction. Details below.

### What the first model taught us

The first label was GitHub's `archived` flag: explicit, dated, machine-readable,
seemingly ideal. Trained on it, the model scored **PR-AUC 0.052 with
precision@20 of 0.000** — its twenty most confident predictions contained no
true abandonments.

Error analysis found the fault was in the label, not the model:

```
                         positives      negatives
days_since_last_commit   never (>1yr)   333 days
commits_trailing         0              1
```

The median package that got archived **had already been silent for over a year
before anyone archived it**. Archiving is not when a package dies; it is when a
maintainer eventually gets around to clicking a button. That is administrative
paperwork, and activity signals cannot forecast it. Worse, the top-scoring
"false positives" — `resume-cli`, `vue-meta`, `sm-crypto` — were all genuinely
dead. The model was right; ground truth called them alive because nobody had
archived them.

Relabelling on sustained silence, and excluding packages already dead at
prediction time, moved PR-AUC from 0.052 to **0.762** and precision@20 from
0.000 to **1.000** on the same collected data.

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env   # then add your GitHub token
```

A token is effectively required: without one you get 60 requests/hour instead
of 5,000. Create one at github.com/settings/tokens with `public_repo` scope.

## Pipeline

```bash
python scripts/01_collect_candidates.py --limit 2000   # living packages (negatives)
python scripts/01b_collect_archived.py                 # archived repos (positives)
python scripts/02_fetch_repos.py --limit 50            # smoke test first
python scripts/02_fetch_repos.py                       # then the full run (hours)
python scripts/03_build_dataset.py
python scripts/04_train.py --boundary 2025-01-01
```

Run **both** 01 and 01b. npm search is biased toward living packages — archived
repos rank poorly and barely surface — so 01 alone yields almost no positive
examples and the model has nothing to learn from. 01b goes after archived repos
directly, partitioning by star count to work around GitHub's 1000-result cap
per search query.

Every API response is cached to disk, so step 2 is safe to interrupt and rerun.

## Dashboard

```bash
uv pip install -e ".[api]"
uvicorn driftlogg.api.main:app --reload
```

Open http://localhost:8000. Paste package names or upload a manifest. With no
trained model on disk the service falls back to the baseline and says so rather
than presenting baseline output as model output.

## Tests

```bash
pytest
```

The leakage tests in `tests/test_features.py` are the ones that matter. Run
them after touching anything in `features.py`.

## Layout

```
driftlogg/
├── config.py          # settings via pydantic-settings
├── collect/github.py  # API client: disk cache + rate limiting
├── labels.py          # what counts as abandoned
├── features.py        # leakage-safe feature extraction
├── model.py           # baseline + LightGBM, temporal split
├── evaluate.py        # PR-AUC, precision@k
└── api/               # FastAPI service + dashboard
scripts/               # numbered pipeline stages
tests/                 # leakage and label tests
```

## Results

1,875 repositories, 7,330 labelled rows, 33.8% positive. Split temporally at
2025-01-01: train before, test on or after. Zero leakage errors.

| Model | PR-AUC | ROC-AUC | Precision@20 |
|---|---|---|---|
| Baseline (180d silent) | 0.475 | 0.666 | 0.750 |
| **LightGBM** | **0.762** | **0.867** | **1.000** |

Every correct positive gives 90 days of warning, which is the forecast horizon
by construction rather than a distribution to optimise.

Top features by gain: `days_since_last_commit`, `stars`, `fork_star_ratio`,
`commits_trailing`, `open_issues_count`.

**An honest caveat:** popularity features (`stars`, `forks`, `fork_star_ratio`)
rank high. Some of that is real — a package with more contributors is genuinely
less likely to go silent — but some is likely sampling artefact, since the
positives and negatives were drawn from different sources (GitHub archived
search versus npm search) whose popularity distributions differ. Confirming
which requires a matched-sampling run, and until that is done the headline
number should be read as an upper bound.

## Roadmap

- [x] FastAPI endpoint accepting `package.json` / `requirements.txt`
- [x] Dashboard with per-package feature attributions
- [ ] Matched sampling to separate real popularity signal from artefact
- [ ] Issue response-time features (needs per-issue comment timelines, sampled)
- [ ] GitHub Action that fails CI on high-risk new dependencies
