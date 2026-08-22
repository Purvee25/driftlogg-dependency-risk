# DriftLogg

Predicts which of your open-source dependencies are likely to be abandoned in
the next 90 days — before the maintainer archives the repo and the security
patches stop arriving.

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

- **Label** — a repo carrying GitHub's `archived` flag, which is explicit,
  dated, and machine-readable. Softer signals (deprecation notices, README
  wording) are captured but flagged for manual review rather than trusted.
- **Features** — activity, contributor, responsiveness, backlog, and release
  signals, computed as *trends* rather than levels. A package slowing from
  weekly commits to nothing matters far more than one that was always quiet.
- **Model** — LightGBM over tabular features. Trees beat a neural net here on
  data this size, and the per-prediction attributions are what let the tool
  explain *why* a package was flagged.
- **Evaluation** — PR-AUC (not ROC-AUC), precision@k, and median lead time.

### The two things that decide whether this works

**1. Leakage.** Every feature is computed with an `as_of` cutoff and may only
use data from strictly before it. A feature that peeks past the cutoff produces
a model that scores brilliantly in testing and fails completely in production.
`FeatureWindow.filter_events` is the single chokepoint, and
`tests/test_features.py` asserts that adding future data changes nothing.

**2. Label quality.** A stable, finished library with no recent commits is
**not** abandoned. If silence gets labelled as death, the model learns to flag
every mature package and the tool becomes noise. Negative controls — quiet but
demonstrably healthy packages — need hand-checking. This is the least glamorous
work in the project and the most important.

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
python scripts/01_collect_candidates.py --limit 2000
python scripts/02_fetch_repos.py --limit 50    # smoke test first
python scripts/02_fetch_repos.py               # then the full run (hours)
python scripts/03_build_dataset.py
python scripts/04_train.py --boundary 2025-01-01
```

Every API response is cached to disk, so step 2 is safe to interrupt and rerun.

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
└── evaluate.py        # PR-AUC, precision@k, lead time
scripts/               # numbered pipeline stages
tests/                 # leakage and label tests
```

## Status

Scaffolding complete; data collection not yet run. Results below get filled in
once the first full training run finishes.

| Model | PR-AUC | Precision@20 | Median lead time |
|---|---|---|---|
| Baseline (180d silent) | — | — | — |
| LightGBM | — | — | — |

## Roadmap

- [ ] Issue response-time features (needs per-issue comment timelines, sampled)
- [ ] FastAPI endpoint accepting `package.json` / `requirements.txt`
- [ ] React dashboard with per-package feature attributions
- [ ] GitHub Action that fails CI on high-risk new dependencies
