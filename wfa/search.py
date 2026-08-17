"""Parameter search iterators for WFA fold optimisation.

All iterators yield dicts of {param_name: value}.

Search methods:
  RandomSearch  — random sampling without replacement up to budget N.
                  Better generalisation than full grid (Bergstra & Bengio 2012).
                  Default for all bots.
  FullGrid      — exhaustive Cartesian product. Use when total combos < 50.
  OptunaSearch  — Bayesian TPE search. Best for large/continuous param spaces.
                  Requires optuna to be installed.
"""
from __future__ import annotations

import itertools
import random
from collections.abc import Iterator

ParamSchema = dict[str, dict]  # {name: {type, choices, default, help}}


def _combo_at(index: int, choices_per_param: list) -> tuple:
    """Decode a flat index into the itertools.product-ordered combo (last param varies fastest)."""
    combo = []
    for choices in reversed(choices_per_param):
        index, r = divmod(index, len(choices))
        combo.append(choices[r])
    return tuple(reversed(combo))


class RandomSearch:
    """Random search over param schema, up to budget N trials.

    Samples without replacement up to min(budget, total_combos), then
    falls back to sampling with replacement for larger budgets.
    Fixed seed for reproducibility.
    """

    def __init__(self, schema: ParamSchema, budget: int, seed: int = 42) -> None:
        self.schema = schema
        self.budget = budget
        self.seed = seed

    def __iter__(self) -> Iterator[dict]:
        rng = random.Random(self.seed)
        names = list(self.schema.keys())
        choices_per_param = [self.schema[n]["choices"] for n in names]
        total = self.total_combos

        if total <= self.budget:
            all_combos = list(itertools.product(*choices_per_param))
            rng.shuffle(all_combos)
            combos = all_combos
        else:
            # Sample combo INDICES without replacement — never materialise the full
            # Cartesian product (large schemas previously stalled/OOM'd building
            # list(itertools.product(...)) before the first trial ran).
            indices = rng.sample(range(total), self.budget)
            combos = [_combo_at(i, choices_per_param) for i in indices]

        for combo in combos:
            yield dict(zip(names, combo))

    @property
    def total_combos(self) -> int:
        n = 1
        for spec in self.schema.values():
            n *= len(spec["choices"])
        return n


class FullGrid:
    """Exhaustive Cartesian product over all parameter choices."""

    def __init__(self, schema: ParamSchema) -> None:
        self.schema = schema

    def __iter__(self) -> Iterator[dict]:
        names = list(self.schema.keys())
        choices_per_param = [self.schema[n]["choices"] for n in names]
        for combo in itertools.product(*choices_per_param):
            yield dict(zip(names, combo))

    @property
    def total_combos(self) -> int:
        n = 1
        for spec in self.schema.values():
            n *= len(spec["choices"])
        return n


class OptunaSearch:
    """Bayesian TPE search via Optuna.

    Requires: pip install optuna
    """

    def __init__(
        self,
        schema: ParamSchema,
        budget: int,
        seed: int = 42,
        sampler: str = "tpe",
    ) -> None:
        self.schema = schema
        self.budget = budget
        self.seed = seed
        self.sampler_name = sampler
        self._study = None
        self._pending: dict[int, tuple] = {}

    def _build_study(self):
        try:
            import optuna
        except ImportError as e:
            raise ImportError("OptunaSearch requires optuna: pip install optuna") from e

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if self.sampler_name == "tpe":
            sampler = optuna.samplers.TPESampler(seed=self.seed)
        elif self.sampler_name == "random":
            sampler = optuna.samplers.RandomSampler(seed=self.seed)
        else:
            raise ValueError(f"Unknown optuna sampler '{self.sampler_name}'")

        self._study = optuna.create_study(direction="maximize", sampler=sampler)
        return self._study

    def suggest_params(self, trial) -> dict:
        return {
            name: trial.suggest_categorical(name, spec["choices"])
            for name, spec in self.schema.items()
        }

    def __iter__(self) -> Iterator[dict]:
        """Yield param dicts; caller must call .report_score(trial_number, score) after each."""
        self._build_study()
        for _ in range(self.budget):
            trial = self._study.ask()
            params = self.suggest_params(trial)
            self._pending[trial.number] = (trial, params)
            yield params

    def report_score(self, trial_number: int, score: float) -> None:
        """Tell Optuna the objective value for a completed trial."""
        import math
        if trial_number not in self._pending:
            return
        trial, _ = self._pending.pop(trial_number)
        value = score if math.isfinite(score) else float("-inf")
        self._study.tell(trial, value)


def make_search(
    method: str,
    schema: ParamSchema,
    budget: int,
    seed: int = 42,
) -> RandomSearch | FullGrid | OptunaSearch:
    """Return the appropriate search iterator."""
    if method == "random":
        return RandomSearch(schema, budget, seed)
    elif method == "grid":
        return FullGrid(schema)
    elif method == "optuna":
        return OptunaSearch(schema, budget, seed)
    else:
        raise ValueError(f"Unknown search method '{method}'. Choose: random, grid, optuna")
