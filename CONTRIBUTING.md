# Contributing

Contributions are welcome. This document covers scope, setup, and what a good pull request
looks like here.

## Scope

**This engine validates strategies. It does not implement them.**

That boundary is deliberate and it decides what belongs in this repo:

**In scope**
- Validation methodology: walk-forward schemes, survival gating, overfit backstops, path risk
- Statistical correctness: estimator bias, deflation, distributional assumptions
- The adapter contract, and anything that makes plugging a strategy in easier
- Search backends, objectives, metrics
- Reporting, the Streamlit portal, CLI ergonomics
- Documentation, tests, typing, packaging

**Out of scope**
- Trading strategies, signals, indicators, or parameter sets
- Broker or exchange integrations
- Live execution, order routing, position management
- Data acquisition pipelines

If you want the engine to validate your strategy, you write an adapter in *your* repo. The engine
never imports a strategy directly, and keeping it that way is what makes the validation credible.

## Getting set up

```bash
git clone https://github.com/oager/walkforward-engine.git
cd walkforward-engine
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

You should see the full suite pass. If it does not on a clean checkout, that is a bug worth an
issue on its own.

## Pull requests

- **Tests are not optional.** Every behavioural change needs a test that fails before your change
  and passes after. This is a statistics library; an untested change to a formula is a liability.
- **One concern per PR.** A methodology fix and a refactor in the same diff is hard to review and
  harder to revert.
- **Explain the why.** For anything touching `objectives.py`, `metrics.py`, or `survival.py`, say
  what the formula should be and cite a source where one exists. "Looks wrong" is a good issue,
  not a good PR.
- **Keep the acausality guard green.** `tests/test_no_filtfilt.py` fails the build if lookahead is
  introduced. If your change trips it, that is the test doing its job.
- CI runs the suite on Python 3.11 and 3.12. Both must pass.

## Reporting bugs

For a statistical bug, the most useful report includes the input series, the value the engine
produced, the value you believe is correct, and why. A hand-computed counter-example is worth more
than a paragraph of description, and it usually converts straight into a regression test.

For a crash, include the traceback, your Python version, and the command you ran.

## A note on methodology disagreements

Reasonable people disagree about validation choices — how much to deflate, where a survival
threshold belongs, whether a breadth failure should disqualify an edge or reclassify it. Those
discussions are welcome; open an issue rather than a PR so the reasoning is on record before code
moves. Defaults here are opinionated on purpose, and documented so you can override them.

## License

Contributions are accepted under the MIT license that covers this project.
