#!/usr/bin/env python3
"""Deterministic predictive calibration for policy/search arbitration.

The deployable rule produced here is deliberately a scalar potential::

    score(a) = beta_search(ply) * search_q(a)
             + alpha_core(ply) * log(policy_core_probability(a))
             + alpha_draw(ply) * log(policy_draw_given_core(a))

Consequently every pairwise override threshold is transitive.  There is no
candidate-zero bonus or other pair-specific exception in the fitted model;
runtime confidence and minimum-advantage guards remain separate concerns.

Training observations predict an *independent* truth-panel action-value
difference from the search-panel difference and two log-prior gaps.  Search
is not assigned coefficient one: the fitted positive ``beta_search`` is the
predictive shrinkage (including winner's-curse correction) appropriate to one
deployed panel.  Target and design are standardized by
``sqrt(search_se**2 + truth_se**2 + se_floor**2)`` before Huber fitting.  This
is a fixed predictive precision weight, not a structural errors-in-variables
correction.
Every state has equal total base influence, however many action pairs and
primary/fresh search rows it contributes.  Source matches, rather than
individual states or action pairs, are the indivisible cross-validation unit.

This module intentionally depends only on the Python standard library and
NumPy so that a frozen campaign can transport it without SciPy or a solver
service.  Floating-point optimization can vary across NumPy/BLAS versions;
the emitted artifact records ``numpy.__version__`` and a locked campaign must
bind that version and its execution image rather than silently upgrading it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace as dataclass_replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "lc-policy-cost-calibration-v2"
STATISTICAL_FAILURE_REASON = (
    "authoritative_predictive_model_adequacy_gate_failed"
)
TRAIN_ALLOCATION_SCHEMA = "lc-policy-cost-train-allocation-v1"
CANONICAL_PAYLOAD_SHA256 = "canonical_payload_sha256"
PAIR_ORIENTATION = "canonical-left-minus-right"
PAIR_TYPES = ("different_core", "same_core_draw")
DEFAULT_PLY_ANCHORS = (
    0, 4, 8, 12, 16, 24, 32, 40, 48, 64,
)
DEFAULT_STANDARD_ERROR_FLOOR = 0.25
DEFAULT_MIN_SEARCH_BETA = 1.0e-6
DEFAULT_FOLD_SEED = "202704140101"
MAX_ROUND_PLY = 299
CAMPAIGN_CELL_QUOTA = 16
# Twenty-two two-ply bins preserve fine resolution through the ordinary part
# of a round; [44,48) ends at a runtime spline anchor and the sparse [48,64)
# natural tail is pooled.  nply >= 64 is census-only.
CAMPAIGN_PLY_BINS = tuple(
    (lower, lower + 2) for lower in range(0, 44, 2)
) + ((44, 48), (48, 64))
CAMPAIGN_RATIO_LOG_EDGES = tuple(
    math.log(value) for value in (1.0, 1.25, 2.0, 4.0, 8.0, 32.0)
)
MODEL_LACK_MAX_RELATIVE_IMPROVEMENT = 0.05
DESIGN_CONDITION_CEILING = 1.0e8
DEFAULT_SMOOTHNESS_GRID = (
    0.0, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0,
)
DEFAULT_GAP_SCENARIOS = (
    ("55_to_45", 0.55, 0.45),
    ("95_to_4", 0.95, 0.04),
    ("95_to_1", 0.95, 0.01),
    ("4_to_1", 0.04, 0.01),
)
RATIO_BAND_LABELS = (
    "[1,1.25)", "[1.25,2)", "[2,4)",
    "[4,8)", "[8,32)", "[32,inf)",
)
RATIO_BAND_LOWER_BOUNDS = (1.0, 1.25, 2.0, 4.0, 8.0, 32.0)


class CalibrationError(ValueError):
    """Raised when calibration input or numerical output is invalid."""


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise CalibrationError(f"{name} must be finite")
    return result


def _probability(value: float, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 < result <= 1.0:
        raise CalibrationError(f"{name} must be in (0, 1]")
    return result


def _validate_anchors(anchors: Sequence[int]) -> tuple[int, ...]:
    converted: list[int] = []
    for value in anchors:
        if isinstance(value, bool):
            raise CalibrationError("ply anchors must be integers")
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CalibrationError("ply anchors must be integers") from exc
        if integer != value:
            raise CalibrationError("ply anchors must be integers")
        converted.append(integer)
    result = tuple(converted)
    if len(result) < 2:
        raise CalibrationError("at least two ply anchors are required")
    if any(value < 0 for value in result):
        raise CalibrationError("ply anchors must be non-negative")
    if any(left >= right for left, right in zip(result, result[1:])):
        raise CalibrationError("ply anchors must be strictly increasing")
    return result


@dataclass(frozen=True)
class PairObservation:
    """One oriented action pair from one source match.

    ``*_delta`` and both log ratios are left action minus right action.  A
    fitted schedule predicts ``truth_delta`` from ``search_delta`` and its two
    log-prior gaps.  Reversing a pair negates all four quantities and
    therefore leaves the fit and decision rule coherent.  ``search_se`` and
    ``truth_se`` are the paired Monte Carlo standard errors of those action
    differences, not per-action standard errors.
    """

    source_match_id: str
    state_id: str
    pair_id: str
    ply: int
    search_delta: float
    truth_delta: float
    log_core_ratio: float
    log_draw_ratio: float
    search_se: float
    truth_se: float
    search_panel_id: str = "primary"
    truth_panel_id: str = "truth"
    orientation: str = PAIR_ORIENTATION
    state_weight: float = 1.0
    round_index: int = 0
    pair_type: str = "different_core"

    def validated(self, anchors: Sequence[int]) -> "PairObservation":
        checked_anchors = _validate_anchors(anchors)
        source = str(self.source_match_id)
        if not source:
            raise CalibrationError("source_match_id must be non-empty")
        state = str(self.state_id)
        if not state:
            raise CalibrationError("state_id must be non-empty")
        pair = str(self.pair_id)
        if not pair:
            raise CalibrationError("pair_id must be non-empty")
        orientation = str(self.orientation)
        if orientation != PAIR_ORIENTATION:
            raise CalibrationError(
                f"orientation must be {PAIR_ORIENTATION!r}"
            )
        if isinstance(self.ply, bool) or int(self.ply) != self.ply:
            raise CalibrationError("ply must be an integer")
        ply = int(self.ply)
        if not checked_anchors[0] <= ply <= MAX_ROUND_PLY:
            raise CalibrationError(
                f"ply {ply} lies outside the frozen round range"
            )
        search_panel = str(self.search_panel_id)
        truth_panel = str(self.truth_panel_id)
        if not search_panel or not truth_panel:
            raise CalibrationError("panel identifiers must be non-empty")
        if search_panel == truth_panel:
            raise CalibrationError(
                "truth panel must be independent of the search panel"
            )
        search_se = _finite(self.search_se, "search_se")
        truth_se = _finite(self.truth_se, "truth_se")
        if search_se < 0.0 or truth_se < 0.0:
            raise CalibrationError(
                "search_se and truth_se must be non-negative"
            )
        state_weight = _finite(self.state_weight, "state_weight")
        if state_weight <= 0.0:
            raise CalibrationError("state_weight must be strictly positive")
        if (isinstance(self.round_index, bool)
                or self.round_index not in (0, 1, 2)):
            raise CalibrationError("round_index must be 0, 1, or 2")
        pair_type = str(self.pair_type)
        if pair_type not in PAIR_TYPES:
            raise CalibrationError(
                f"pair_type must be one of {PAIR_TYPES!r}"
            )
        log_core = _finite(self.log_core_ratio, "log_core_ratio")
        log_draw = _finite(self.log_draw_ratio, "log_draw_ratio")
        if log_core < 0.0:
            raise CalibrationError(
                "canonical left action must not have lower semantic policy mass"
            )
        if pair_type == "same_core_draw":
            if log_core != 0.0 or log_draw < 0.0:
                raise CalibrationError(
                    "same-core pairs require equal core mass and the higher "
                    "conditional-draw prior on the canonical left"
                )
        return PairObservation(
            source_match_id=source,
            state_id=state,
            pair_id=pair,
            ply=ply,
            search_delta=_finite(self.search_delta, "search_delta"),
            truth_delta=_finite(self.truth_delta, "truth_delta"),
            log_core_ratio=log_core,
            log_draw_ratio=log_draw,
            search_se=search_se,
            truth_se=truth_se,
            search_panel_id=search_panel,
            truth_panel_id=truth_panel,
            orientation=orientation,
            state_weight=state_weight,
            round_index=int(self.round_index),
            pair_type=pair_type,
        )

    @classmethod
    def from_probabilities(
        cls,
        *,
        source_match_id: str,
        state_id: str,
        pair_id: str,
        ply: int,
        search_delta: float,
        truth_delta: float,
        left_core_probability: float,
        right_core_probability: float,
        left_draw_probability: float = 1.0,
        right_draw_probability: float = 1.0,
        search_se: float,
        truth_se: float,
        search_panel_id: str = "primary",
        truth_panel_id: str = "truth",
        orientation: str = PAIR_ORIENTATION,
        state_weight: float = 1.0,
        round_index: int = 0,
        pair_type: str = "different_core",
    ) -> "PairObservation":
        left_core = _probability(
            left_core_probability, "left_core_probability"
        )
        right_core = _probability(
            right_core_probability, "right_core_probability"
        )
        left_draw = _probability(
            left_draw_probability, "left_draw_probability"
        )
        right_draw = _probability(
            right_draw_probability, "right_draw_probability"
        )
        return cls(
            source_match_id=source_match_id,
            state_id=state_id,
            pair_id=pair_id,
            ply=ply,
            search_delta=search_delta,
            truth_delta=truth_delta,
            log_core_ratio=math.log(left_core / right_core),
            log_draw_ratio=math.log(left_draw / right_draw),
            search_se=search_se,
            truth_se=truth_se,
            search_panel_id=search_panel_id,
            truth_panel_id=truth_panel_id,
            orientation=orientation,
            state_weight=state_weight,
            round_index=round_index,
            pair_type=pair_type,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PairObservation":
        common_required = {
            "source_match_id", "state_id", "pair_id", "ply", "search_delta",
            "truth_delta", "search_se", "truth_se", "search_panel_id",
            "truth_panel_id", "orientation", "state_weight", "round",
            "pair_type",
        }
        direct = "log_core_ratio" in value or "log_draw_ratio" in value
        probabilities = any(
            key in value for key in (
                "left_core_probability", "right_core_probability",
                "left_draw_probability", "right_draw_probability",
            )
        )
        if direct and probabilities:
            raise CalibrationError(
                "use log ratios or probabilities, never both"
            )
        direct_keys = common_required | {"log_core_ratio", "log_draw_ratio"}
        probability_keys = common_required | {
            "left_core_probability", "right_core_probability",
            "left_draw_probability", "right_draw_probability",
        }
        expected = direct_keys if direct else probability_keys
        actual = set(value)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise CalibrationError(
                "observation keys differ "
                f"(missing={missing}, extra={extra})"
            )
        common = {
            "source_match_id": value["source_match_id"],
            "state_id": value["state_id"],
            "pair_id": value["pair_id"],
            "ply": value["ply"],
            "search_delta": value["search_delta"],
            "truth_delta": value["truth_delta"],
            "search_se": value["search_se"],
            "truth_se": value["truth_se"],
            "search_panel_id": value["search_panel_id"],
            "truth_panel_id": value["truth_panel_id"],
            "orientation": value["orientation"],
            "state_weight": value["state_weight"],
            "round_index": value["round"],
            "pair_type": value["pair_type"],
        }
        if direct:
            if "log_core_ratio" not in value or "log_draw_ratio" not in value:
                raise CalibrationError("both log ratios are required")
            return cls(
                **common,
                log_core_ratio=value["log_core_ratio"],
                log_draw_ratio=value["log_draw_ratio"],
            )
        return cls.from_probabilities(
            **common,
            left_core_probability=value["left_core_probability"],
            right_core_probability=value["right_core_probability"],
            left_draw_probability=value["left_draw_probability"],
            right_draw_probability=value["right_draw_probability"],
        )


@dataclass(frozen=True)
class FitConfig:
    anchors: tuple[int, ...] = DEFAULT_PLY_ANCHORS
    smoothness_grid: tuple[float, ...] = DEFAULT_SMOOTHNESS_GRID
    folds: int = 5
    fold_seed: str = DEFAULT_FOLD_SEED
    huber_delta: float = 1.345
    min_search_beta: float = DEFAULT_MIN_SEARCH_BETA
    min_core_alpha: float = 0.0
    min_draw_alpha: float = 0.0
    standard_error_floor: float = DEFAULT_STANDARD_ERROR_FLOOR
    max_irls_iterations: int = 500
    max_coordinate_iterations: int = 20_000
    tolerance: float = 1.0e-10
    require_campaign_design: bool = False
    model_lack_max_relative_improvement: float = (
        MODEL_LACK_MAX_RELATIVE_IMPROVEMENT
    )

    def validated(self) -> "FitConfig":
        anchors = _validate_anchors(self.anchors)
        grid = tuple(_finite(value, "smoothness")
                     for value in self.smoothness_grid)
        if not grid or any(value < 0.0 for value in grid):
            raise CalibrationError(
                "smoothness grid must contain non-negative values"
            )
        if len(set(grid)) != len(grid):
            raise CalibrationError("smoothness grid values must be unique")
        folds = int(self.folds)
        if isinstance(self.folds, bool) or folds != self.folds or folds < 2:
            raise CalibrationError("fold count must be an integer >= 2")
        delta = _finite(self.huber_delta, "huber_delta")
        if delta <= 0.0:
            raise CalibrationError("huber_delta must be strictly positive")
        beta_min = _finite(self.min_search_beta, "min_search_beta")
        core_min = _finite(self.min_core_alpha, "min_core_alpha")
        draw_min = _finite(self.min_draw_alpha, "min_draw_alpha")
        if beta_min <= 0.0:
            raise CalibrationError("search-beta lower bound must be positive")
        if core_min < 0.0 or draw_min < 0.0:
            raise CalibrationError("alpha lower bounds must be non-negative")
        se_floor = _finite(self.standard_error_floor, "standard_error_floor")
        if se_floor <= 0.0:
            raise CalibrationError(
                "standard_error_floor must be strictly positive"
            )
        if self.max_irls_iterations < 1 or self.max_coordinate_iterations < 1:
            raise CalibrationError("iteration limits must be positive")
        tolerance = _finite(self.tolerance, "tolerance")
        if tolerance <= 0.0:
            raise CalibrationError("tolerance must be strictly positive")
        if not isinstance(self.require_campaign_design, bool):
            raise CalibrationError("require_campaign_design must be boolean")
        model_lack_limit = _finite(
            self.model_lack_max_relative_improvement,
            "model_lack_max_relative_improvement",
        )
        if not 0.0 <= model_lack_limit <= 1.0:
            raise CalibrationError(
                "model-lack relative-improvement limit must be in [0, 1]"
            )
        seed = str(self.fold_seed)
        if not seed:
            raise CalibrationError("fold_seed must be non-empty")
        return FitConfig(
            anchors=anchors,
            smoothness_grid=grid,
            folds=folds,
            fold_seed=seed,
            huber_delta=delta,
            min_search_beta=beta_min,
            min_core_alpha=core_min,
            min_draw_alpha=draw_min,
            standard_error_floor=se_floor,
            max_irls_iterations=int(self.max_irls_iterations),
            max_coordinate_iterations=int(self.max_coordinate_iterations),
            tolerance=tolerance,
            require_campaign_design=self.require_campaign_design,
            model_lack_max_relative_improvement=model_lack_limit,
        )


def linear_spline_basis(ply: int, anchors: Sequence[int]) -> np.ndarray:
    """Return the clamped piecewise-linear interpolation basis for ``ply``."""

    checked = _validate_anchors(anchors)
    if isinstance(ply, bool) or int(ply) != ply:
        raise CalibrationError("ply must be an integer")
    value = int(ply)
    if not checked[0] <= value <= MAX_ROUND_PLY:
        raise CalibrationError("ply lies outside the frozen round range")
    result = np.zeros(len(checked), dtype=np.float64)
    if value >= checked[-1]:
        result[-1] = 1.0
        return result
    upper = int(np.searchsorted(checked, value, side="right"))
    lower = max(0, upper - 1)
    if checked[lower] == value:
        result[lower] = 1.0
        return result
    width = checked[upper] - checked[lower]
    fraction = (value - checked[lower]) / width
    result[lower] = 1.0 - fraction
    result[upper] = fraction
    return result


@dataclass(frozen=True)
class PolicyCostSchedule:
    anchors: tuple[int, ...]
    beta_search: tuple[float, ...]
    alpha_core: tuple[float, ...]
    alpha_draw: tuple[float, ...]

    def validated(self) -> "PolicyCostSchedule":
        anchors = _validate_anchors(self.anchors)
        beta = tuple(_finite(value, "beta_search")
                     for value in self.beta_search)
        core = tuple(_finite(value, "alpha_core")
                     for value in self.alpha_core)
        draw = tuple(_finite(value, "alpha_draw")
                     for value in self.alpha_draw)
        if (len(beta) != len(anchors) or len(core) != len(anchors)
                or len(draw) != len(anchors)):
            raise CalibrationError(
                "each coefficient schedule needs one value per anchor"
            )
        if any(value <= 0.0 for value in beta):
            raise CalibrationError("search betas must be strictly positive")
        if any(value < 0.0 for value in core + draw):
            raise CalibrationError("policy alphas must be non-negative")
        return PolicyCostSchedule(anchors, beta, core, draw)

    def coefficients_at(self, ply: int) -> tuple[float, float, float]:
        """Interpolate all raw coefficients before any normalization."""

        checked = self.validated()
        basis = linear_spline_basis(ply, checked.anchors)
        return (
            float(basis @ np.asarray(checked.beta_search)),
            float(basis @ np.asarray(checked.alpha_core)),
            float(basis @ np.asarray(checked.alpha_draw)),
        )

    def lambdas_at(self, ply: int) -> tuple[float, float]:
        checked = self.validated()
        beta, alpha_core, alpha_draw = checked.coefficients_at(ply)
        if beta <= 0.0 or not math.isfinite(beta):
            raise CalibrationError("interpolated search beta is not positive")
        return alpha_core / beta, alpha_draw / beta

    def score(
        self,
        *,
        ply: int,
        search_q: float,
        core_probability: float,
        draw_probability_given_core: float = 1.0,
    ) -> float:
        core = _probability(core_probability, "core_probability")
        draw = _probability(
            draw_probability_given_core, "draw_probability_given_core"
        )
        q = _finite(search_q, "search_q")
        beta, alpha_core, alpha_draw = self.coefficients_at(ply)
        return (
            beta * q
            + alpha_core * math.log(core)
            + alpha_draw * math.log(draw)
        )

    def required_search_advantage(
        self,
        *,
        ply: int,
        incumbent_core_probability: float,
        challenger_core_probability: float,
        incumbent_draw_probability: float = 1.0,
        challenger_draw_probability: float = 1.0,
    ) -> float:
        """Search-Q lead a challenger needs to tie the incumbent's score."""

        incumbent_core = _probability(
            incumbent_core_probability, "incumbent_core_probability"
        )
        challenger_core = _probability(
            challenger_core_probability, "challenger_core_probability"
        )
        incumbent_draw = _probability(
            incumbent_draw_probability, "incumbent_draw_probability"
        )
        challenger_draw = _probability(
            challenger_draw_probability, "challenger_draw_probability"
        )
        lambda_core, lambda_draw = self.lambdas_at(ply)
        return (
            lambda_core * math.log(incumbent_core / challenger_core)
            + lambda_draw * math.log(incumbent_draw / challenger_draw)
        )

    def choose(
        self,
        *,
        ply: int,
        search_q: Sequence[float],
        core_probabilities: Sequence[float],
        draw_probabilities_given_core: Sequence[float] | None = None,
    ) -> int:
        """Return the first maximum scalar score, with deterministic ties."""

        if len(search_q) != len(core_probabilities) or not search_q:
            raise CalibrationError("Q values and core probabilities must align")
        draws = (tuple(1.0 for _ in search_q)
                 if draw_probabilities_given_core is None
                 else tuple(draw_probabilities_given_core))
        if len(draws) != len(search_q):
            raise CalibrationError("draw probabilities must align with Q values")
        scores = [
            self.score(
                ply=ply,
                search_q=q,
                core_probability=core,
                draw_probability_given_core=draw,
            )
            for q, core, draw in zip(search_q, core_probabilities, draws)
        ]
        return max(range(len(scores)), key=lambda index: (scores[index], -index))


@dataclass(frozen=True)
class CrossValidationRow:
    smoothness: float
    mean_group_huber_loss: float
    group_standard_error: float
    paired_standard_error_vs_minimum: float
    fold_group_losses: tuple[tuple[float, ...], ...]
    fit_convergence: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "smoothness": self.smoothness,
            "mean_group_huber_loss": self.mean_group_huber_loss,
            "group_standard_error": self.group_standard_error,
            "paired_standard_error_vs_minimum": (
                self.paired_standard_error_vs_minimum
            ),
            "fold_group_losses": [list(values)
                                  for values in self.fold_group_losses],
            "fit_convergence": [dict(value)
                                for value in self.fit_convergence],
        }


@dataclass(frozen=True)
class CalibrationResult:
    calibration_passed: bool
    schedule: PolicyCostSchedule | None
    selected_smoothness: float
    cv_rows: tuple[CrossValidationRow, ...]
    huber_delta: float
    standard_error_floor: float
    observation_count: int
    source_match_count: int
    folds: int
    fold_seed: str
    fold_assignment_sha256: str
    fold_source_match_counts: tuple[int, ...]
    observation_input_sha256: str
    min_search_beta: float
    min_core_alpha: float
    min_draw_alpha: float
    campaign_design: Mapping[str, Any]
    model_adequacy: Mapping[str, Any]
    max_irls_iterations: int
    max_coordinate_iterations: int
    solver_tolerance: float
    final_fit_convergence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        adequacy = dict(self.model_adequacy)
        if adequacy.get("passed") is not self.calibration_passed:
            raise CalibrationError(
                "calibration decision/model-adequacy decision mismatch"
            )
        if self.calibration_passed:
            if self.schedule is None:
                raise CalibrationError(
                    "passed calibration is missing its deployable schedule"
                )
            schedule = self.schedule.validated()
        else:
            if self.schedule is not None:
                raise CalibrationError(
                    "failed calibration must not expose a deployable schedule"
                )
            if not (
                adequacy.get("required") is True
                and adequacy.get("evaluated") is True
                and adequacy.get("authoritative_pre_select_gate") is True
            ):
                raise CalibrationError(
                    "only an authoritative computed adequacy failure may be "
                    "serialized as a negative calibration result"
                )
        value: dict[str, Any] = {
            "schema": SCHEMA,
            "calibration_passed": self.calibration_passed,
            "status": (
                "passed" if self.calibration_passed
                else "failed_model_adequacy"
            ),
            "deployment": {
                "permitted": self.calibration_passed,
                "reason": (
                    None if self.calibration_passed
                    else STATISTICAL_FAILURE_REASON
                ),
            },
            "observation_input_sha256": self.observation_input_sha256,
            "model": {
                "score": (
                    "beta_search(ply)*Q + alpha_core(ply)*log(Pcore) + "
                    "alpha_draw(ply)*log(Pdraw_given_core)"
                ),
                "normalized_runtime_cost": (
                    "interpolate beta/alpha first, then lambda=alpha/beta"
                ),
                "candidate_zero_bonus": "absent",
                "runtime_confidence_guards": "external_unchanged",
                "interpolation": "piecewise_linear",
            },
            "fit": {
                "loss": "state-equal source-match-grouped variance-standardized Huber",
                "huber_delta": self.huber_delta,
                "variance_standardization": (
                    "sqrt(search_se^2 + truth_se^2 + "
                    "standard_error_floor^2)"
                ),
                "variance_interpretation": (
                    "frozen predictive precision weight; not structural "
                    "errors-in-variables correction"
                ),
                "standard_error_floor": self.standard_error_floor,
                "row_weighting": (
                    "state_weight divided equally over every pair x "
                    "search-panel row in that state; primary and fresh each "
                    "receive half of a pair's state-normalized base weight"
                ),
                "selected_smoothness": self.selected_smoothness,
                "selection_rule": (
                    "largest smoothness whose mean OOF source-match loss is "
                    "at most the minimum mean plus the source-match-cluster "
                    "SE of the minimum-loss model"
                ),
                "observation_count": self.observation_count,
                "source_match_count": self.source_match_count,
                "folds": self.folds,
                "fold_seed": self.fold_seed,
                "fold_assignment_sha256": self.fold_assignment_sha256,
                "fold_source_match_counts": list(self.fold_source_match_counts),
                "min_search_beta": self.min_search_beta,
                "min_core_alpha": self.min_core_alpha,
                "min_draw_alpha": self.min_draw_alpha,
                "solver": {
                    "max_irls_iterations": self.max_irls_iterations,
                    "max_coordinate_iterations": self.max_coordinate_iterations,
                    "tolerance": self.solver_tolerance,
                    "fail_closed_on_nonconvergence": True,
                    "final_fit_convergence": dict(
                        self.final_fit_convergence
                    ),
                },
                "cross_validation": [row.to_dict() for row in self.cv_rows],
            },
            "campaign_design": dict(self.campaign_design),
            "model_adequacy": adequacy,
            "runtime_dependencies": {
                "numpy_version": np.__version__,
                "version_policy": (
                    "campaign must bind NumPy and execution image; "
                    "cross-version byte identity is not assumed"
                ),
            },
        }
        if self.calibration_passed:
            assert self.schedule is not None
            schedule = self.schedule.validated()
            value["schedule"] = {
                "ply_anchors": list(schedule.anchors),
                "beta_search": list(schedule.beta_search),
                "alpha_core": list(schedule.alpha_core),
                "alpha_draw": list(schedule.alpha_draw),
            }
            value["derived_gap_thresholds"] = derived_gap_threshold_table(
                schedule
            )
        digest = hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
        value["calibration_sha256"] = digest
        return value

    def canonical_json(self) -> bytes:
        return _canonical_json_bytes(self.to_dict())


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) + "\n").encode("utf-8")


def derived_gap_threshold_table(
    schedule: PolicyCostSchedule,
    *,
    plies: Sequence[int] | None = None,
    scenarios: Sequence[tuple[str, float, float]] = DEFAULT_GAP_SCENARIOS,
) -> list[dict[str, Any]]:
    """Materialize auditable core-only Q thresholds at selected plies."""

    checked = schedule.validated()
    # The deployment input is integer State.nply, so the canonical evidence
    # materializes every admissible ply rather than leaving interpolation to
    # a reader.  Rows 64..299 intentionally repeat the clamped tail value.
    selected_plies = (
        tuple(range(MAX_ROUND_PLY + 1)) if plies is None else tuple(plies)
    )
    rows: list[dict[str, Any]] = []
    for ply in selected_plies:
        beta_search, alpha_core, alpha_draw = checked.coefficients_at(ply)
        lambda_core, lambda_draw = checked.lambdas_at(ply)
        thresholds: dict[str, float] = {}
        log_ratios: dict[str, float] = {}
        for name, incumbent, challenger in scenarios:
            ratio = math.log(
                _probability(incumbent, f"{name}.incumbent")
                / _probability(challenger, f"{name}.challenger")
            )
            log_ratios[str(name)] = ratio
            thresholds[str(name)] = lambda_core * ratio
        core_band_thresholds = {
            label: lambda_core * math.log(lower)
            for label, lower in zip(
                RATIO_BAND_LABELS, RATIO_BAND_LOWER_BOUNDS
            )
        }
        draw_band_thresholds = {
            label: lambda_draw * math.log(lower)
            for label, lower in zip(
                RATIO_BAND_LABELS, RATIO_BAND_LOWER_BOUNDS
            )
        }
        rows.append({
            "ply": int(ply),
            "beta_search": beta_search,
            "alpha_core": alpha_core,
            "alpha_draw": alpha_draw,
            "lambda_core": lambda_core,
            "lambda_draw": lambda_draw,
            "core_log_ratios": log_ratios,
            "required_search_advantage": thresholds,
            "ratio_band_lower_bound_required_search_advantage": {
                "semantic_core": core_band_thresholds,
                "conditional_draw": draw_band_thresholds,
                "interpretation": (
                    "minimum within-band cost at the closed lower ratio; "
                    "the [32,inf) row is a lower bound, not a cap"
                ),
            },
        })
    return rows


def make_group_folds(
    observations: Sequence[PairObservation],
    *,
    folds: int,
    seed: str,
    stratify_campaign_cells: bool = False,
) -> dict[str, int]:
    """Assign whole source matches to balanced, deterministic fixed folds."""

    groups = sorted({str(item.source_match_id) for item in observations})
    if folds < 2:
        raise CalibrationError("fold count must be at least two")
    if len(groups) < folds:
        raise CalibrationError(
            "source-match count must be at least the fold count"
        )
    def ranked(values: Iterable[str], cell: str = "") -> list[str]:
        return sorted(
            values,
            key=lambda group: (
                hashlib.sha256(
                    f"{seed}\0{cell}\0{group}".encode("utf-8")
                ).digest(),
                group,
            ),
        )

    if not stratify_campaign_cells:
        return {
            group: index % folds
            for index, group in enumerate(ranked(groups))
        }
    by_group: dict[str, tuple[int, int, int, str]] = {}
    for item in observations:
        cell = _campaign_cell(item)
        prior = by_group.setdefault(item.source_match_id, cell)
        if prior != cell:
            raise CalibrationError(
                "campaign source match spans multiple calibration cells"
            )
    by_cell: dict[tuple[int, int, int, str], list[str]] = {}
    for group, cell in by_group.items():
        by_cell.setdefault(cell, []).append(group)
    assignment: dict[str, int] = {}
    offset = 0
    for cell in sorted(by_cell):
        cell_label = ".".join(map(str, cell))
        members = ranked(by_cell[cell], cell_label)
        if len(members) < folds:
            raise CalibrationError(
                f"campaign cell {cell!r} has fewer members than folds"
            )
        for index, group in enumerate(members):
            assignment[group] = (offset + index) % folds
        # Carry each cell's remainder into the next fixed cell rather than
        # restarting at fold zero.  This preserves within-cell balance while
        # making the campaign-wide source counts differ by at most one.
        offset = (offset + len(members)) % folds
    return assignment


def _fold_digest(assignment: Mapping[str, int]) -> str:
    return hashlib.sha256(_canonical_json_bytes(dict(assignment))).hexdigest()


def _conventional_one_se_smoothness(
    rows: Sequence[tuple[float, float, float]],
) -> float:
    """Return the smoothest model within one SE of minimum mean CV loss.

    Each tuple is ``(smoothness, mean_group_loss, group_SE)``.  The threshold
    is the minimum-loss model's own source-cluster SE, as in the conventional
    one-standard-error rule; it is not a candidate-specific paired-difference
    threshold.
    """

    if not rows or any(
        not all(math.isfinite(value) for value in row) or row[2] < 0.0
        for row in rows
    ):
        raise CalibrationError("invalid one-SE summary")
    best = min(rows, key=lambda row: (row[1], -row[0]))
    limit = best[1] + best[2] + 1.0e-15
    return max(row[0] for row in rows if row[1] <= limit)


def _canonical_observations(
    observations: Iterable[PairObservation], anchors: Sequence[int],
) -> tuple[PairObservation, ...]:
    checked = [item.validated(anchors) for item in observations]
    if not checked:
        raise CalibrationError("at least one observation is required")
    state_sources: dict[str, str] = {}
    state_truth_panels: dict[str, str] = {}
    state_search_panels: dict[str, set[str]] = {}
    state_weights: dict[str, float] = {}
    pair_truth_rows: dict[tuple[str, str], tuple[Any, ...]] = {}
    pair_search_panels: dict[tuple[str, str], set[str]] = {}
    seen_search_rows: set[tuple[str, str, str]] = set()
    for item in checked:
        prior_source = state_sources.setdefault(
            item.state_id, item.source_match_id
        )
        if prior_source != item.source_match_id:
            raise CalibrationError(
                "a state_id cannot span multiple source matches"
            )
        prior_truth = state_truth_panels.setdefault(
            item.state_id, item.truth_panel_id
        )
        if prior_truth != item.truth_panel_id:
            raise CalibrationError(
                "primary/fresh rows for a state must share one truth panel"
            )
        prior_weight = state_weights.setdefault(
            item.state_id, item.state_weight
        )
        if prior_weight != item.state_weight:
            raise CalibrationError(
                "all rows for a state must bind the same state_weight"
            )
        state_search_panels.setdefault(item.state_id, set()).add(
            item.search_panel_id
        )
        pair_key = (item.state_id, item.pair_id)
        truth_row = (
            item.round_index,
            item.ply,
            item.pair_type,
            item.truth_delta,
            item.truth_se,
            item.log_core_ratio,
            item.log_draw_ratio,
            item.truth_panel_id,
            item.orientation,
        )
        prior_truth_row = pair_truth_rows.setdefault(pair_key, truth_row)
        if prior_truth_row != truth_row:
            raise CalibrationError(
                "primary/fresh rows disagree on shared truth or pair orientation"
            )
        search_key = (item.state_id, item.pair_id, item.search_panel_id)
        if search_key in seen_search_rows:
            raise CalibrationError(
                "duplicate state/pair/search-panel observation"
            )
        seen_search_rows.add(search_key)
        pair_search_panels.setdefault(pair_key, set()).add(
            item.search_panel_id
        )
    for state_id, truth_panel in state_truth_panels.items():
        if truth_panel in state_search_panels[state_id]:
            raise CalibrationError(
                "shared truth panel must be independent of every search panel"
            )
    required_search_panels = {"primary", "fresh"}
    for pair_key, panels in pair_search_panels.items():
        if panels != required_search_panels:
            raise CalibrationError(
                f"state/pair {pair_key!r} search panels differ "
                f"(required={sorted(required_search_panels)}, "
                f"actual={sorted(panels)})"
            )
    return tuple(sorted(checked, key=lambda item: (
        item.source_match_id,
        item.state_id,
        item.pair_id,
        item.round_index,
        item.ply,
        item.pair_type,
        item.search_panel_id,
        item.truth_panel_id,
        item.search_delta,
        item.truth_delta,
        item.log_core_ratio,
        item.log_draw_ratio,
        item.search_se,
        item.truth_se,
        item.state_weight,
    )))


def _observation_mapping(item: PairObservation) -> dict[str, Any]:
    """Return the one canonical, already-log-transformed evidence row."""

    return {
        "source_match_id": item.source_match_id,
        "state_id": item.state_id,
        "pair_id": item.pair_id,
        "round": item.round_index,
        "ply": item.ply,
        "pair_type": item.pair_type,
        "search_delta": item.search_delta,
        "truth_delta": item.truth_delta,
        "log_core_ratio": item.log_core_ratio,
        "log_draw_ratio": item.log_draw_ratio,
        "search_se": item.search_se,
        "truth_se": item.truth_se,
        "search_panel_id": item.search_panel_id,
        "truth_panel_id": item.truth_panel_id,
        "orientation": item.orientation,
        "state_weight": item.state_weight,
    }


def observation_input_sha256(
    observations: Sequence[PairObservation],
) -> str:
    """Hash the canonical normalized observation array used by the fit."""

    return hashlib.sha256(_canonical_json_bytes(
        [_observation_mapping(item) for item in observations]
    )).hexdigest()


def _campaign_cell(item: PairObservation) -> tuple[int, int, int, str]:
    """Return the one frozen TRAIN cell for an oriented pair."""

    ply_bin = -1
    for index, (lower, upper) in enumerate(CAMPAIGN_PLY_BINS):
        if lower <= item.ply < upper:
            ply_bin = index
            break
    if ply_bin < 0:
        raise CalibrationError(
            "campaign TRAIN observations must lie in ply bands [0, 64)"
        )
    relevant = (
        item.log_draw_ratio
        if item.pair_type == "same_core_draw"
        else item.log_core_ratio
    )
    if relevant < 0.0:
        raise CalibrationError("campaign pair ratio is not canonically oriented")
    ratio_bin = len(CAMPAIGN_RATIO_LOG_EDGES) - 1
    for index in range(len(CAMPAIGN_RATIO_LOG_EDGES) - 1):
        if relevant < CAMPAIGN_RATIO_LOG_EDGES[index + 1]:
            ratio_bin = index
            break
    return item.round_index, ply_bin, ratio_bin, item.pair_type


def _campaign_design_summary(
    observations: Sequence[PairObservation], *, required: bool,
) -> dict[str, Any]:
    if not required:
        return {
            "required": False,
            "validated": False,
            "reason": "standalone calibration without locked TRAIN design",
        }

    source_rows: dict[str, list[PairObservation]] = {}
    cells: dict[tuple[int, int, int, str], set[str]] = {}
    for item in observations:
        if item.state_weight != 1.0:
            raise CalibrationError(
                "locked TRAIN observations require unit state weights"
            )
        source_rows.setdefault(item.source_match_id, []).append(item)
        cells.setdefault(_campaign_cell(item), set()).add(item.source_match_id)

    for source, rows in source_rows.items():
        identities = {(row.state_id, row.pair_id) for row in rows}
        panels = {row.search_panel_id for row in rows}
        if (len(rows) != 2 or len(identities) != 1
                or panels != {"primary", "fresh"}):
            raise CalibrationError(
                "locked TRAIN requires exactly one pair and P/F rows per "
                f"distinct source match ({source!r})"
            )

    expected = {
        (round_index, ply_bin, ratio_bin, pair_type)
        for round_index in range(3)
        for ply_bin in range(len(CAMPAIGN_PLY_BINS))
        for ratio_bin in range(6)
        for pair_type in PAIR_TYPES
    }
    if set(cells) != expected:
        missing = sorted(expected - set(cells))
        extra = sorted(set(cells) - expected)
        raise CalibrationError(
            "locked TRAIN cell coverage differs "
            f"(missing={missing}, extra={extra})"
        )
    wrong = {
        str(cell): len(sources)
        for cell, sources in cells.items()
        if len(sources) != CAMPAIGN_CELL_QUOTA
    }
    if wrong:
        raise CalibrationError(
            f"locked TRAIN cells do not have quota {CAMPAIGN_CELL_QUOTA}: "
            f"{wrong}"
        )
    expected_sources = len(expected) * CAMPAIGN_CELL_QUOTA
    if len(source_rows) != expected_sources:
        raise CalibrationError(
            "locked TRAIN source matches are reused across cells "
            f"(expected={expected_sources}, actual={len(source_rows)})"
        )
    return {
        "required": True,
        "validated": True,
        "rounds": 3,
        "ply_bins": [list(pair) for pair in CAMPAIGN_PLY_BINS],
        "tail_policy": "nply>=64 census-only; excluded from fit",
        "ratio_bins": ["[1,1.25)", "[1.25,2)", "[2,4)",
                       "[4,8)", "[8,32)", "[32,inf)"],
        "pair_types": list(PAIR_TYPES),
        "quota_distinct_source_matches_per_cell": CAMPAIGN_CELL_QUOTA,
        "cells": len(expected),
        "source_matches": len(source_rows),
        "states": len(source_rows),
        "observations": len(observations),
        "source_match_reuse_across_cells": False,
    }


def _lower_sha256(value: Any, where: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise CalibrationError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _campaign_allocation_binding(
    observations: Sequence[PairObservation], raw: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise CalibrationError("TRAIN allocation manifest must be an object")
    required = {
        "schema", "source_reservoir_sha256",
        "eligible_pair_commitment_sha256", "allocation_rule_sha256",
        "ply_bins", "ratio_bins", "pair_types", "cell_quota",
        "selected_units", CANONICAL_PAYLOAD_SHA256,
    }
    if set(raw) != required:
        raise CalibrationError("TRAIN allocation manifest keys differ")
    if raw["schema"] != TRAIN_ALLOCATION_SCHEMA:
        raise CalibrationError("TRAIN allocation manifest schema mismatch")
    for field in (
        "source_reservoir_sha256", "eligible_pair_commitment_sha256",
        "allocation_rule_sha256",
    ):
        _lower_sha256(raw[field], f"TRAIN allocation {field}")
    claimed = _lower_sha256(
        raw[CANONICAL_PAYLOAD_SHA256],
        f"TRAIN allocation {CANONICAL_PAYLOAD_SHA256}",
    )
    payload = dict(raw)
    del payload[CANONICAL_PAYLOAD_SHA256]
    actual = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if claimed != actual:
        raise CalibrationError("TRAIN allocation manifest digest mismatch")
    if raw["ply_bins"] != [list(pair) for pair in CAMPAIGN_PLY_BINS]:
        raise CalibrationError("TRAIN allocation ply bins differ")
    if raw["ratio_bins"] != list(RATIO_BAND_LABELS):
        raise CalibrationError("TRAIN allocation ratio bins differ")
    if raw["pair_types"] != list(PAIR_TYPES):
        raise CalibrationError("TRAIN allocation pair types differ")
    if raw["cell_quota"] != CAMPAIGN_CELL_QUOTA:
        raise CalibrationError("TRAIN allocation cell quota differs")
    raw_units = raw["selected_units"]
    if not isinstance(raw_units, list):
        raise CalibrationError("TRAIN selected_units must be an array")
    units: dict[str, dict[str, Any]] = {}
    states: set[str] = set()
    priorities: set[str] = set()
    order: list[tuple[Any, ...]] = []
    cell_counts: dict[tuple[int, int, int, str], int] = {}
    for index, value in enumerate(raw_units):
        where = f"TRAIN selected unit {index}"
        if not isinstance(value, Mapping):
            raise CalibrationError(f"{where} must be an object")
        fields = {
            "source_match_id", "state_id", "pair_id", "state_sha256",
            "pair_sha256", "allocation_priority_sha256", "round",
            "ply_bin", "ratio_bin", "pair_type",
        }
        if set(value) != fields:
            raise CalibrationError(f"{where} keys differ")
        source = str(value["source_match_id"])
        state = str(value["state_id"])
        pair = str(value["pair_id"])
        if not source or not state or not pair:
            raise CalibrationError(f"{where} identity is empty")
        state_sha = _lower_sha256(value["state_sha256"], f"{where}.state")
        pair_sha = _lower_sha256(value["pair_sha256"], f"{where}.pair")
        priority = _lower_sha256(
            value["allocation_priority_sha256"], f"{where}.priority"
        )
        round_index = value["round"]
        ply_bin = value["ply_bin"]
        ratio_bin = value["ratio_bin"]
        pair_type = value["pair_type"]
        if (isinstance(round_index, bool) or round_index not in (0, 1, 2)
                or isinstance(ply_bin, bool) or not isinstance(ply_bin, int)
                or not 0 <= ply_bin < len(CAMPAIGN_PLY_BINS)
                or isinstance(ratio_bin, bool)
                or not isinstance(ratio_bin, int)
                or not 0 <= ratio_bin < 6
                or pair_type not in PAIR_TYPES):
            raise CalibrationError(f"{where} cell metadata is invalid")
        if source in units:
            raise CalibrationError("TRAIN source match is reused")
        if state_sha in states or priority in priorities:
            raise CalibrationError("TRAIN selected state/priority is duplicated")
        states.add(state_sha)
        priorities.add(priority)
        unit = {
            "source_match_id": source,
            "state_id": state,
            "pair_id": pair,
            "state_sha256": state_sha,
            "pair_sha256": pair_sha,
            "allocation_priority_sha256": priority,
            "round": round_index,
            "ply_bin": ply_bin,
            "ratio_bin": ratio_bin,
            "pair_type": pair_type,
        }
        units[source] = unit
        cell = (round_index, ply_bin, ratio_bin, pair_type)
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        order.append((*cell, priority, state_sha, source, state, pair))
    if order != sorted(order):
        raise CalibrationError(
            "TRAIN selected_units are not in canonical cell/priority order"
        )
    expected_cells = {
        (round_index, ply_bin, ratio_bin, pair_type)
        for round_index in range(3)
        for ply_bin in range(len(CAMPAIGN_PLY_BINS))
        for ratio_bin in range(6)
        for pair_type in PAIR_TYPES
    }
    if set(cell_counts) != expected_cells or any(
            count != CAMPAIGN_CELL_QUOTA for count in cell_counts.values()):
        raise CalibrationError("TRAIN selected-unit cell quotas differ")
    observed: dict[str, tuple[str, str, tuple[int, int, int, str]]] = {}
    for item in observations:
        identity = (item.state_id, item.pair_id, _campaign_cell(item))
        prior = observed.setdefault(item.source_match_id, identity)
        if prior != identity:
            raise CalibrationError(
                "TRAIN observation source has inconsistent allocation identity"
            )
    if set(observed) != set(units):
        raise CalibrationError(
            "TRAIN observations differ from sealed selected-unit sources"
        )
    for source, (state, pair, cell) in observed.items():
        unit = units[source]
        if (state != unit["state_id"] or pair != unit["pair_id"]
                or cell != (
                    unit["round"], unit["ply_bin"], unit["ratio_bin"],
                    unit["pair_type"],
                )):
            raise CalibrationError(
                "TRAIN observation identity differs from sealed allocation"
            )
    return {
        "required": True,
        "validated": True,
        "allocation_manifest_sha256": claimed,
        "source_reservoir_sha256": raw["source_reservoir_sha256"],
        "eligible_pair_commitment_sha256": (
            raw["eligible_pair_commitment_sha256"]
        ),
        "allocation_rule_sha256": raw["allocation_rule_sha256"],
        "selected_units": len(units),
    }


def _design_rank_diagnostics(
    observations: Sequence[PairObservation],
    assignment: Mapping[str, int],
    config: FitConfig,
) -> dict[str, Any]:
    """Fail closed on an unidentifiable or numerically singular spline."""

    if not config.require_campaign_design:
        return {"required": False, "validated": False}
    rows: list[dict[str, Any]] = []
    partitions: list[tuple[str, tuple[PairObservation, ...]]] = [
        ("full_train", tuple(observations))
    ]
    for fold in range(config.folds):
        partitions.append((
            f"cv_training_without_fold_{fold}",
            tuple(
                item for item in observations
                if assignment[item.source_match_id] != fold
            ),
        ))
        partitions.append((
            f"cv_validation_fold_{fold}",
            tuple(
                item for item in observations
                if assignment[item.source_match_id] == fold
            ),
        ))
    expected_rank = 3 * len(config.anchors)
    ceiling = DESIGN_CONDITION_CEILING
    for label, partition in partitions:
        design, _, weights = _design(
            partition, config.anchors, config.standard_error_floor
        )
        weighted = design * np.sqrt(weights)[:, None]
        norms = np.linalg.norm(weighted, axis=0)
        if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
            raise CalibrationError(f"{label} has an unobserved design column")
        scaled = weighted / norms[None, :]
        singular = np.linalg.svd(scaled, compute_uv=False)
        tolerance = (
            singular[0] * max(scaled.shape) * np.finfo(np.float64).eps
        )
        rank = int(np.sum(singular > tolerance))
        condition = float(singular[0] / singular[-1])
        if (rank != expected_rank or not math.isfinite(condition)
                or condition > ceiling):
            raise CalibrationError(
                f"{label} spline design is not identified "
                f"(rank={rank}/{expected_rank}, condition={condition})"
            )
        rows.append({
            "partition": label,
            "rows": len(partition),
            "rank_after_column_scaling": rank,
            "expected_rank": expected_rank,
            "condition_number_after_column_scaling": condition,
            "condition_number_ceiling": ceiling,
            "passed": True,
        })
    return {
        "required": True,
        "validated": True,
        "column_scaling": "weighted L2 norm",
        "partitions": rows,
    }


def _design(
    observations: Sequence[PairObservation], anchors: Sequence[int],
    standard_error_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    knot_count = len(anchors)
    design = np.zeros((len(observations), 3 * knot_count), dtype=np.float64)
    target = np.empty(len(observations), dtype=np.float64)
    weights = np.empty(len(observations), dtype=np.float64)
    state_rows: dict[str, int] = {}
    for item in observations:
        state_rows[item.state_id] = state_rows.get(item.state_id, 0) + 1
    for index, item in enumerate(observations):
        basis = linear_spline_basis(item.ply, anchors)
        variance_scale = math.hypot(
            item.search_se, item.truth_se, standard_error_floor
        )
        design[index, :knot_count] = (
            basis * item.search_delta / variance_scale
        )
        design[index, knot_count:2 * knot_count] = (
            basis * item.log_core_ratio / variance_scale
        )
        design[index, 2 * knot_count:] = (
            basis * item.log_draw_ratio / variance_scale
        )
        target[index] = item.truth_delta / variance_scale
        weights[index] = item.state_weight / state_rows[item.state_id]
    return design, target, weights


def _beta_design(
    observations: Sequence[PairObservation], anchors: Sequence[int],
    standard_error_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spline design for the nested search-only shrinkage comparator."""

    full, target, weights = _design(
        observations, anchors, standard_error_floor
    )
    return full[:, :len(anchors)], target, weights


def _roughness_matrix(
    anchors: Sequence[int], coefficient_blocks: int,
) -> np.ndarray:
    anchors_array = np.asarray(anchors, dtype=np.float64)
    knot_count = len(anchors)
    differences = np.zeros((knot_count - 2, knot_count), dtype=np.float64)
    for row in range(knot_count - 2):
        left_width = anchors_array[row + 1] - anchors_array[row]
        right_width = anchors_array[row + 2] - anchors_array[row + 1]
        differences[row, row] = 1.0 / left_width
        differences[row, row + 1] = -(1.0 / left_width + 1.0 / right_width)
        differences[row, row + 2] = 1.0 / right_width
    one = differences.T @ differences
    if coefficient_blocks < 1:
        raise CalibrationError("roughness needs at least one coefficient block")
    result = np.zeros(
        (coefficient_blocks * knot_count, coefficient_blocks * knot_count),
        dtype=np.float64,
    )
    for block in range(coefficient_blocks):
        lower = block * knot_count
        result[lower:lower + knot_count, lower:lower + knot_count] = one
    return result


def _solve_nonnegative_quadratic(
    hessian: np.ndarray,
    linear: np.ndarray,
    initial: np.ndarray,
    *,
    max_iterations: int,
    tolerance: float,
) -> np.ndarray:
    """Cyclic coordinate descent for a strictly convex non-negative QP."""

    result = np.maximum(np.asarray(initial, dtype=np.float64), 0.0).copy()
    diagonal = np.diag(hessian)
    if np.any(diagonal <= 0.0) or not np.all(np.isfinite(diagonal)):
        raise CalibrationError("invalid calibration Hessian")
    for _ in range(max_iterations):
        largest_change = 0.0
        largest_value = 0.0
        for column in range(len(result)):
            partial = (
                linear[column]
                - float(hessian[column] @ result)
                + diagonal[column] * result[column]
            )
            updated = max(0.0, partial / diagonal[column])
            largest_change = max(largest_change, abs(updated - result[column]))
            result[column] = updated
            largest_value = max(largest_value, abs(updated))
        if largest_change <= tolerance * (1.0 + largest_value):
            return result
    raise CalibrationError("non-negative quadratic solver did not converge")


def _fit_one(
    design: np.ndarray,
    target: np.ndarray,
    sample_weights: np.ndarray,
    *,
    anchors: Sequence[int],
    smoothness: float,
    config: FitConfig,
    convergence_log: list[dict[str, Any]] | None = None,
    fit_label: str = "schedule",
) -> np.ndarray:
    knot_count = len(anchors)
    if design.shape[1] == knot_count:
        coefficient_blocks = 1
        lower = np.full(
            knot_count, config.min_search_beta, dtype=np.float64
        )
    elif design.shape[1] == 3 * knot_count:
        coefficient_blocks = 3
        lower = np.concatenate((
            np.full(knot_count, config.min_search_beta, dtype=np.float64),
            np.full(knot_count, config.min_core_alpha, dtype=np.float64),
            np.full(knot_count, config.min_draw_alpha, dtype=np.float64),
        ))
    else:
        raise CalibrationError("unsupported spline design width")
    shifted_target = target - design @ lower
    roughness = _roughness_matrix(anchors, coefficient_blocks)
    weight_sum = float(np.sum(sample_weights))
    if weight_sum <= 0.0:
        raise CalibrationError("training weights sum to zero")
    result = np.zeros(design.shape[1], dtype=np.float64)
    previous_objective = math.inf
    final_change = math.inf
    final_objective_change = math.inf
    iterations = 0
    for iteration in range(config.max_irls_iterations):
        standardized = shifted_target - design @ result
        absolute = np.abs(standardized)
        robust_weights = np.ones_like(absolute)
        tail = absolute > config.huber_delta
        robust_weights[tail] = config.huber_delta / absolute[tail]
        effective = sample_weights * robust_weights / weight_sum
        hessian = design.T @ (effective[:, None] * design)
        hessian += smoothness * roughness
        # A tiny deterministic ridge resolves unobserved knots without
        # materially acting as a fitted policy penalty.
        hessian += np.eye(hessian.shape[0]) * 1.0e-12
        linear = design.T @ (effective * shifted_target)
        updated = _solve_nonnegative_quadratic(
            hessian,
            linear,
            result,
            max_iterations=config.max_coordinate_iterations,
            tolerance=config.tolerance,
        )
        residual = shifted_target - design @ updated
        losses = _huber_values(residual, config.huber_delta)
        objective = float(np.sum(sample_weights * losses) / weight_sum)
        objective += 0.5 * smoothness * float(updated @ roughness @ updated)
        change = float(np.max(np.abs(updated - result)))
        objective_change = abs(previous_objective - objective)
        result = updated
        iterations = iteration + 1
        final_change = change
        final_objective_change = objective_change
        if (change <= config.tolerance * (1.0 + float(np.max(result)))
                and objective_change
                <= config.tolerance * (1.0 + abs(objective))):
            break
        previous_objective = objective
    else:
        raise CalibrationError("Huber IRLS did not converge")
    fitted = result + lower
    if not np.all(np.isfinite(fitted)) or np.any(fitted < lower - 1.0e-10):
        raise CalibrationError("invalid constrained calibration result")
    fitted = np.maximum(fitted, lower)
    if np.any(fitted[:knot_count] <= 0.0):
        raise CalibrationError("fitted search beta is not strictly positive")
    if convergence_log is not None:
        convergence_log.append({
            "fit": fit_label,
            "converged": True,
            "irls_iterations": iterations,
            "max_irls_iterations": config.max_irls_iterations,
            "final_max_coefficient_change": final_change,
            "final_objective_change": final_objective_change,
        })
    return fitted


def _huber_values(residual: np.ndarray, delta: float) -> np.ndarray:
    absolute = np.abs(residual)
    return np.where(
        absolute <= delta,
        0.5 * np.square(residual),
        delta * (absolute - 0.5 * delta),
    )


def _fit_nonnegative_matrix(
    design: np.ndarray,
    target: np.ndarray,
    sample_weights: np.ndarray,
    config: FitConfig,
    lower_bounds: np.ndarray | None = None,
    convergence_log: list[dict[str, Any]] | None = None,
    fit_label: str = "model_adequacy_cell",
) -> np.ndarray:
    """Robust nonnegative fit for a frozen diagnostic design matrix."""

    if (design.ndim != 2 or target.shape != (design.shape[0],)
            or sample_weights.shape != target.shape or design.shape[0] < 1):
        raise CalibrationError("invalid model-adequacy design")
    lower = (
        np.zeros(design.shape[1], dtype=np.float64)
        if lower_bounds is None
        else np.asarray(lower_bounds, dtype=np.float64)
    )
    if (lower.shape != (design.shape[1],) or not np.all(np.isfinite(lower))
            or np.any(lower < 0.0)):
        raise CalibrationError("invalid model-adequacy lower bounds")
    shifted_target = target - design @ lower
    result = np.zeros(design.shape[1], dtype=np.float64)
    weight_sum = float(np.sum(sample_weights))
    if weight_sum <= 0.0:
        raise CalibrationError("model-adequacy weights sum to zero")
    previous_objective = math.inf
    final_change = math.inf
    final_objective_change = math.inf
    iterations = 0
    for iteration in range(config.max_irls_iterations):
        residual = shifted_target - design @ result
        absolute = np.abs(residual)
        robust = np.ones_like(absolute)
        tail = absolute > config.huber_delta
        robust[tail] = config.huber_delta / absolute[tail]
        effective = sample_weights * robust / weight_sum
        hessian = design.T @ (effective[:, None] * design)
        hessian += np.eye(design.shape[1]) * 1.0e-12
        linear = design.T @ (effective * shifted_target)
        updated = _solve_nonnegative_quadratic(
            hessian,
            linear,
            result,
            max_iterations=config.max_coordinate_iterations,
            tolerance=config.tolerance,
        )
        losses = _huber_values(
            shifted_target - design @ updated, config.huber_delta
        )
        objective = float(np.sum(sample_weights * losses) / weight_sum)
        change = float(np.max(np.abs(updated - result)))
        objective_change = abs(previous_objective - objective)
        result = updated
        iterations = iteration + 1
        final_change = change
        final_objective_change = objective_change
        if (change <= config.tolerance * (1.0 + float(np.max(result)))
                and objective_change
                <= config.tolerance * (1.0 + abs(objective))):
            break
        previous_objective = objective
    else:
        raise CalibrationError("model-adequacy IRLS did not converge")
    result = result + lower
    if not np.all(np.isfinite(result)) or np.any(result < lower):
        raise CalibrationError("invalid model-adequacy coefficients")
    if convergence_log is not None:
        convergence_log.append({
            "fit": fit_label,
            "converged": True,
            "irls_iterations": iterations,
            "max_irls_iterations": config.max_irls_iterations,
            "final_max_coefficient_change": final_change,
            "final_objective_change": final_objective_change,
        })
    return result


def _standardized_pair_design(
    observations: Sequence[PairObservation], standard_error_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = np.empty((len(observations), 3), dtype=np.float64)
    target = np.empty(len(observations), dtype=np.float64)
    weights = np.empty(len(observations), dtype=np.float64)
    state_rows: dict[str, int] = {}
    for item in observations:
        state_rows[item.state_id] = state_rows.get(item.state_id, 0) + 1
    for index, item in enumerate(observations):
        scale = math.hypot(
            item.search_se, item.truth_se, standard_error_floor
        )
        design[index] = (
            item.search_delta / scale,
            item.log_core_ratio / scale,
            item.log_draw_ratio / scale,
        )
        target[index] = item.truth_delta / scale
        weights[index] = item.state_weight / state_rows[item.state_id]
    return design, target, weights


def _cell_active_columns(pair_type: str) -> tuple[int, ...]:
    if pair_type == "same_core_draw":
        return (0, 2)
    if pair_type == "different_core":
        return (0, 1, 2)
    raise CalibrationError(f"unknown pair type {pair_type!r}")


def _cell_design_diagnostic(
    cell: tuple[int, int, int, str],
    observations: Sequence[PairObservation],
    config: FitConfig,
) -> dict[str, Any]:
    design, _, _ = _standardized_pair_design(
        observations, config.standard_error_floor
    )
    active = _cell_active_columns(cell[3])
    active_design = design[:, active]
    norms = np.linalg.norm(active_design, axis=0)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise CalibrationError(
            f"cell-saturated adequacy cell {cell!r} has a zero active column"
        )
    scaled = active_design / norms
    singular = np.linalg.svd(scaled, compute_uv=False)
    tolerance = (
        max(scaled.shape) * np.finfo(np.float64).eps * singular[0]
    )
    rank = int(np.sum(singular > tolerance))
    expected_rank = len(active)
    condition = (
        float(singular[0] / singular[-1])
        if singular[-1] > 0.0 else math.inf
    )
    if rank != expected_rank:
        raise CalibrationError(
            f"cell-saturated adequacy cell {cell!r} rank {rank} != "
            f"expected {expected_rank}"
        )
    if (not math.isfinite(condition)
            or condition > DESIGN_CONDITION_CEILING):
        raise CalibrationError(
            f"cell-saturated adequacy cell {cell!r} condition {condition} "
            f"exceeds {DESIGN_CONDITION_CEILING}"
        )
    return {
        "cell": [cell[0], cell[1], cell[2], cell[3]],
        "active_columns": [
            ("search_delta" if index == 0 else
             "log_core_ratio" if index == 1 else "log_draw_ratio")
            for index in active
        ],
        "expected_rank": expected_rank,
        "rank_after_column_scaling": rank,
        "condition_number_after_column_scaling": condition,
        "rows": len(observations),
    }


def _cell_saturated_identifiability(
    observations: Sequence[PairObservation], config: FitConfig,
) -> dict[str, Any]:
    by_cell: dict[tuple[int, int, int, str], list[PairObservation]] = {}
    for item in observations:
        by_cell.setdefault(_campaign_cell(item), []).append(item)
    diagnostics = [
        _cell_design_diagnostic(cell, tuple(by_cell[cell]), config)
        for cell in sorted(by_cell)
    ]
    canonical = json.dumps(
        diagnostics, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "validated": True,
        "cells": len(diagnostics),
        "expected_rank_by_pair_type": {
            "different_core": 3,
            "same_core_draw": 2,
        },
        "maximum_condition_number_after_column_scaling": max(
            row["condition_number_after_column_scaling"]
            for row in diagnostics
        ),
        "diagnostics_sha256": hashlib.sha256(canonical).hexdigest(),
        "diagnostics": diagnostics,
    }


def _predict_partition(
    training: Sequence[PairObservation],
    validation: Sequence[PairObservation],
    config: FitConfig,
    smoothness: float,
    model: str,
    convergence_log: list[dict[str, Any]] | None = None,
    fit_context: str = "adequacy",
) -> np.ndarray:
    """Fit one frozen model on training and predict untouched validation."""

    if not training or not validation:
        raise CalibrationError("empty model-adequacy partition")
    if model == "identity_search":
        return np.asarray([
            item.search_delta / math.hypot(
                item.search_se, item.truth_se,
                config.standard_error_floor,
            )
            for item in validation
        ], dtype=np.float64)
    if model == "beta_only":
        train_x, train_y, train_w = _beta_design(
            training, config.anchors, config.standard_error_floor
        )
        coefficient = _fit_one(
            train_x, train_y, train_w,
            anchors=config.anchors,
            smoothness=smoothness,
            config=config,
            convergence_log=convergence_log,
            fit_label=f"{fit_context}:beta-only",
        )
        validation_x, _, _ = _beta_design(
            validation, config.anchors, config.standard_error_floor
        )
        return validation_x @ coefficient
    if model == "deployable_gap":
        train_x, train_y, train_w = _design(
            training, config.anchors, config.standard_error_floor
        )
        coefficient = _fit_one(
            train_x, train_y, train_w,
            anchors=config.anchors,
            smoothness=smoothness,
            config=config,
            convergence_log=convergence_log,
            fit_label=f"{fit_context}:deployable-gap",
        )
        validation_x, _, _ = _design(
            validation, config.anchors, config.standard_error_floor
        )
        return validation_x @ coefficient
    if model == "round_specific":
        predictions = np.full(len(validation), np.nan, dtype=np.float64)
        for round_index in range(3):
            round_training = tuple(
                item for item in training if item.round_index == round_index
            )
            validation_indices = [
                index for index, item in enumerate(validation)
                if item.round_index == round_index
            ]
            round_validation = tuple(
                validation[index] for index in validation_indices
            )
            if not round_training or not round_validation:
                raise CalibrationError(
                    "round-specific adequacy partition lacks round coverage"
                )
            train_x, train_y, train_w = _design(
                round_training, config.anchors, config.standard_error_floor
            )
            coefficient = _fit_one(
                train_x, train_y, train_w,
                anchors=config.anchors,
                smoothness=smoothness,
                config=config,
                convergence_log=convergence_log,
                fit_label=f"{fit_context}:round-{round_index}",
            )
            validation_x, _, _ = _design(
                round_validation, config.anchors,
                config.standard_error_floor,
            )
            for index, prediction in zip(
                    validation_indices, validation_x @ coefficient):
                predictions[index] = prediction
        if not np.all(np.isfinite(predictions)):
            raise CalibrationError("incomplete round-specific predictions")
        return predictions
    if model == "cell_saturated":
        predictions = np.full(len(validation), np.nan, dtype=np.float64)
        train_by_cell: dict[
            tuple[int, int, int, str], list[PairObservation]
        ] = {}
        validation_by_cell: dict[
            tuple[int, int, int, str], list[int]
        ] = {}
        for item in training:
            train_by_cell.setdefault(_campaign_cell(item), []).append(item)
        for index, item in enumerate(validation):
            validation_by_cell.setdefault(_campaign_cell(item), []).append(index)
        if set(train_by_cell) != set(validation_by_cell):
            raise CalibrationError(
                "cell-saturated adequacy partition lacks complete cell coverage"
            )
        for cell in sorted(validation_by_cell):
            cell_training = tuple(train_by_cell[cell])
            validation_indices = validation_by_cell[cell]
            cell_validation = tuple(
                validation[index] for index in validation_indices
            )
            train_x, train_y, train_w = _standardized_pair_design(
                cell_training, config.standard_error_floor
            )
            active = _cell_active_columns(cell[3])
            _cell_design_diagnostic(cell, cell_training, config)
            coefficient = _fit_nonnegative_matrix(
                train_x[:, active], train_y, train_w, config,
                lower_bounds=np.asarray([
                    config.min_search_beta if column == 0 else 0.0
                    for column in active
                ], dtype=np.float64),
                convergence_log=convergence_log,
                fit_label=(
                    f"{fit_context}:cell-" + ".".join(map(str, cell))
                ),
            )
            validation_x, _, _ = _standardized_pair_design(
                cell_validation, config.standard_error_floor
            )
            for index, prediction in zip(
                    validation_indices,
                    validation_x[:, active] @ coefficient):
                predictions[index] = prediction
        if not np.all(np.isfinite(predictions)):
            raise CalibrationError("incomplete cell-saturated predictions")
        return predictions
    raise CalibrationError(f"unknown model-adequacy model {model!r}")


def _oof_group_losses(
    observations: Sequence[PairObservation],
    assignment: Mapping[str, int],
    config: FitConfig,
    smoothness: float,
    model: str,
) -> tuple[tuple[float, ...], np.ndarray, list[dict[str, Any]]]:
    predictions = np.full(len(observations), np.nan, dtype=np.float64)
    convergence: list[dict[str, Any]] = []
    for fold in range(config.folds):
        train_indices = [
            index for index, item in enumerate(observations)
            if assignment[item.source_match_id] != fold
        ]
        validation_indices = [
            index for index, item in enumerate(observations)
            if assignment[item.source_match_id] == fold
        ]
        if not train_indices or not validation_indices:
            raise CalibrationError("empty model-adequacy fold")
        training = tuple(observations[index] for index in train_indices)
        validation = tuple(observations[index] for index in validation_indices)
        fold_predictions = _predict_partition(
            training, validation, config, smoothness, model,
            convergence_log=convergence,
            fit_context=f"inner-fold-{fold}",
        )
        for index, prediction in zip(validation_indices, fold_predictions):
            predictions[index] = prediction

    if not np.all(np.isfinite(predictions)):
        raise CalibrationError("incomplete model-adequacy predictions")
    _, target, weights = _design(
        observations, config.anchors, config.standard_error_floor
    )
    losses = _huber_values(target - predictions, config.huber_delta)
    by_group: dict[str, list[tuple[float, float]]] = {}
    for item, loss, weight in zip(observations, losses, weights):
        by_group.setdefault(item.source_match_id, []).append(
            (float(loss), float(weight))
        )
    result = []
    for group in sorted(by_group):
        rows = by_group[group]
        denominator = math.fsum(weight for _, weight in rows)
        result.append(
            math.fsum(loss * weight for loss, weight in rows) / denominator
        )
    return tuple(result), losses, convergence


def _one_se_smoothness(
    observations: Sequence[PairObservation],
    config: FitConfig,
    *,
    seed: str,
    model: str,
) -> tuple[float, dict[str, Any]]:
    """Tune one model using only an outer fold's training sources.

    This is deliberately separate from the final whole-TRAIN smoothness
    choice.  Model-adequacy comparisons must not reuse a hyperparameter that
    was selected with the sources on which adequacy is scored.
    """

    if model not in ("beta_only", "deployable_gap", "round_specific"):
        raise CalibrationError("nested smoothness applies only to spline models")
    assignment = make_group_folds(
        observations,
        folds=config.folds,
        seed=seed,
        stratify_campaign_cells=True,
    )
    identifiability = _design_rank_diagnostics(
        observations, assignment, config
    )
    rows: list[dict[str, Any]] = []
    loss_vectors: dict[float, np.ndarray] = {}
    for smoothness in config.smoothness_grid:
        group_losses, _, fit_convergence = _oof_group_losses(
            observations, assignment, config, smoothness, model
        )
        vector = np.asarray(group_losses, dtype=np.float64)
        if vector.size < 1 or not np.all(np.isfinite(vector)):
            raise CalibrationError("invalid nested-CV loss vector")
        loss_vectors[smoothness] = vector
        rows.append({
            "smoothness": smoothness,
            "mean_group_huber_loss": float(np.mean(vector)),
            "group_standard_error": (
                float(np.std(vector, ddof=1) / math.sqrt(len(vector)))
                if len(vector) > 1 else 0.0
            ),
            "fit_convergence": fit_convergence,
        })
    best = min(
        rows,
        key=lambda row: (row["mean_group_huber_loss"], -row["smoothness"]),
    )
    best_losses = loss_vectors[best["smoothness"]]
    for row in rows:
        differences = loss_vectors[row["smoothness"]] - best_losses
        row["paired_standard_error_vs_minimum"] = (
            float(np.std(differences, ddof=1) / math.sqrt(len(differences)))
            if len(differences) > 1 else 0.0
        )
    selected_smoothness = _conventional_one_se_smoothness([
        (float(row["smoothness"]), float(row["mean_group_huber_loss"]),
         float(row["group_standard_error"]))
        for row in rows
    ])
    selected = next(
        row for row in rows if row["smoothness"] == selected_smoothness
    )
    return float(selected["smoothness"]), {
        "fold_seed": seed,
        "fold_assignment_sha256": _fold_digest(assignment),
        "source_match_count": len(assignment),
        "rows": rows,
        "selected_smoothness": selected["smoothness"],
        "deployable_design_identifiability": identifiability,
        "rule": (
            "greatest smoothness with mean grouped-CV loss <= minimum mean "
            "+ source-match-cluster SE of the minimum-loss model"
        ),
    }


def _nested_adequacy_predictions(
    observations: Sequence[PairObservation],
    assignment: Mapping[str, int],
    config: FitConfig,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Return outer-OOF predictions with all tuning nested inside training."""

    models = (
        "identity_search",
        "beta_only",
        "deployable_gap",
        "round_specific",
        "cell_saturated",
    )
    predictions = {
        model: np.full(len(observations), np.nan, dtype=np.float64)
        for model in models
    }
    nested_selections: list[dict[str, Any]] = []
    for outer_fold in range(config.folds):
        train_indices = [
            index for index, item in enumerate(observations)
            if assignment[item.source_match_id] != outer_fold
        ]
        validation_indices = [
            index for index, item in enumerate(observations)
            if assignment[item.source_match_id] == outer_fold
        ]
        if not train_indices or not validation_indices:
            raise CalibrationError("empty outer model-adequacy fold")
        training = tuple(observations[index] for index in train_indices)
        validation = tuple(observations[index] for index in validation_indices)
        selected_by_model: dict[str, float] = {}
        tuning_by_model: dict[str, Any] = {}
        for model in ("beta_only", "deployable_gap", "round_specific"):
            inner_seed = (
                f"{config.fold_seed}:nested-adequacy:{model}:"
                f"outer-{outer_fold}"
            )
            selected, tuning = _one_se_smoothness(
                training, config, seed=inner_seed, model=model
            )
            selected_by_model[model] = selected
            tuning_by_model[model] = tuning
        prediction_convergence_by_model: dict[
            str, list[dict[str, Any]]
        ] = {}
        for model in models:
            smoothness = selected_by_model.get(model, 0.0)
            prediction_convergence: list[dict[str, Any]] = []
            fold_predictions = _predict_partition(
                training, validation, config, smoothness, model,
                convergence_log=prediction_convergence,
                fit_context=f"outer-fold-{outer_fold}:{model}",
            )
            for index, prediction in zip(validation_indices, fold_predictions):
                predictions[model][index] = prediction
            prediction_convergence_by_model[model] = prediction_convergence
        cell_identifiability = _cell_saturated_identifiability(
            training, config
        )
        nested_selections.append({
            "outer_fold": outer_fold,
            "outer_training_source_matches": len({
                observations[index].source_match_id for index in train_indices
            }),
            "outer_validation_source_matches": len({
                observations[index].source_match_id
                for index in validation_indices
            }),
            "model_tuning": tuning_by_model,
            "outer_prediction_fit_convergence": (
                prediction_convergence_by_model
            ),
            "cell_saturated_training_identifiability": cell_identifiability,
        })
    for model, values in predictions.items():
        if not np.all(np.isfinite(values)):
            raise CalibrationError(
                f"incomplete nested model-adequacy predictions for {model}"
            )
    return predictions, nested_selections


def _group_losses_from_predictions(
    observations: Sequence[PairObservation],
    predictions: np.ndarray,
    config: FitConfig,
) -> tuple[tuple[float, ...], np.ndarray]:
    _, target, weights = _design(
        observations, config.anchors, config.standard_error_floor
    )
    losses = _huber_values(target - predictions, config.huber_delta)
    by_group: dict[str, list[tuple[float, float]]] = {}
    for item, loss, weight in zip(observations, losses, weights):
        by_group.setdefault(item.source_match_id, []).append(
            (float(loss), float(weight))
        )
    group_losses: list[float] = []
    for group in sorted(by_group):
        rows = by_group[group]
        denominator = math.fsum(weight for _, weight in rows)
        group_losses.append(
            math.fsum(loss * weight for loss, weight in rows) / denominator
        )
    return tuple(group_losses), losses


def _model_adequacy(
    observations: Sequence[PairObservation],
    assignment: Mapping[str, int],
    config: FitConfig,
) -> dict[str, Any]:
    if not config.require_campaign_design:
        return {
            "required": False,
            "evaluated": False,
            "passed": True,
            "reason": "standalone calibration",
        }
    models = (
        "identity_search",
        "beta_only",
        "deployable_gap",
        "round_specific",
        "cell_saturated",
    )
    predictions, nested_selections = _nested_adequacy_predictions(
        observations, assignment, config
    )
    evaluations = {
        model: _group_losses_from_predictions(
            observations, predictions[model], config
        )
        for model in models
    }
    losses = {model: value[0] for model, value in evaluations.items()}
    row_losses = {model: value[1] for model, value in evaluations.items()}
    means = {
        model: float(np.mean(np.asarray(values, dtype=np.float64)))
        for model, values in losses.items()
    }
    baseline = means["deployable_gap"]
    improvements: dict[str, float] = {}
    for model in ("round_specific", "cell_saturated"):
        improvements[model] = (
            0.0 if baseline == 0.0
            else (baseline - means[model]) / baseline
        )
    beta_loss = np.asarray(losses["beta_only"], dtype=np.float64)
    deployable_loss = np.asarray(losses["deployable_gap"], dtype=np.float64)
    paired_gain = beta_loss - deployable_loss
    gain_point = float(np.mean(paired_gain))
    gain_se = (
        float(np.std(paired_gain, ddof=1) / math.sqrt(len(paired_gain)))
        if len(paired_gain) > 1 else 0.0
    )
    gain_lcb = gain_point - 1.645 * gain_se
    _, _, row_weights = _design(
        observations, config.anchors, config.standard_error_floor
    )
    row_gain = (
        row_losses["beta_only"] - row_losses["deployable_gap"]
    )
    stratum_members: dict[str, list[int]] = {
        "pair:different_core": [],
        "pair:same_core_draw": [],
        "ply:early_0_15": [],
        "ply:mid_16_39": [],
        "ply:late_40_63": [],
    }
    for index, item in enumerate(observations):
        stratum_members[f"pair:{item.pair_type}"].append(index)
        if item.ply < 16:
            stratum_members["ply:early_0_15"].append(index)
        elif item.ply < 40:
            stratum_members["ply:mid_16_39"].append(index)
        else:
            stratum_members["ply:late_40_63"].append(index)
    stratum_points: dict[str, float] = {}
    for name, indices in stratum_members.items():
        if not indices:
            raise CalibrationError(f"model-adequacy stratum {name} is empty")
        denominator = float(np.sum(row_weights[indices]))
        stratum_points[name] = float(
            np.sum(row_weights[indices] * row_gain[indices]) / denominator
        )
    strata_passed = all(value >= 0.0 for value in stratum_points.values())
    richer_passed = all(
        improvement <= config.model_lack_max_relative_improvement + 1.0e-15
        for improvement in improvements.values()
    )
    passed = gain_lcb > 0.0 and richer_passed and strata_passed
    return {
        "required": True,
        "evaluated": True,
        "authoritative_pre_select_gate": True,
        "loss": "nested source-match OOF variance-standardized Huber",
        "hyperparameter_tuning": (
            "all beta-only, deployable, and round-specific smoothness "
            "choices are made by "
            "inner source-match-grouped CV inside each outer training fold"
        ),
        "nested_outer_folds": nested_selections,
        "deployable_model": (
            "round-shared ply splines for beta_search, alpha_core, alpha_draw"
        ),
        "comparators": [
            "identity search T=S",
            "ply-spline beta_search-only predictive shrinkage",
        ],
        "challengers": [
            "round-specific beta/alpha ply splines",
            "round x ply-bin x ratio-bin x pair-type saturated beta/alpha",
        ],
        "mean_group_losses": means,
        "gap_over_beta_only_comparison": {
            "paired_source_cluster_loss_reduction": gain_point,
            "paired_source_cluster_se": gain_se,
            "one_sided_lcb_z_1_645": gain_lcb,
            "criterion": "strictly greater than zero",
            "passed": gain_lcb > 0.0,
        },
        "pooled_beta_only_minus_gap_loss_reduction": {
            "points": stratum_points,
            "criterion": "every pair-type and early/mid/late point >= 0",
            "passed": strata_passed,
        },
        "relative_improvement_over_deployable": improvements,
        "maximum_allowed_relative_improvement": (
            config.model_lack_max_relative_improvement
        ),
        "rule": (
            "fail before SELECT unless the deployable gap model has a "
            "strictly positive one-sided source-cluster loss-reduction LCB "
            "versus beta-only, nonnegative reduction in both pair types and "
            "each early/mid/late phase, and no richer challenger reduces OOF "
            "Huber loss by more than the frozen relative limit"
        ),
        "richer_model_check_passed": richer_passed,
        "passed": passed,
    }


def _validation_group_losses(
    observations: Sequence[PairObservation],
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    coefficients: np.ndarray,
    delta: float,
) -> tuple[float, ...]:
    losses = _huber_values(target - design @ coefficients, delta)
    by_group: dict[str, list[tuple[float, float]]] = {}
    for item, loss, weight in zip(observations, losses, weights):
        by_group.setdefault(item.source_match_id, []).append(
            (float(loss), float(weight))
        )
    result: list[float] = []
    for group in sorted(by_group):
        rows = by_group[group]
        denominator = sum(weight for _, weight in rows)
        result.append(sum(loss * weight for loss, weight in rows) / denominator)
    return tuple(result)


def calibrate_policy_cost(
    observations: Iterable[PairObservation],
    config: FitConfig = FitConfig(),
    campaign_allocation_manifest: Mapping[str, Any] | None = None,
    campaign_evidence_binding: Mapping[str, Any] | None = None,
) -> CalibrationResult:
    """Fit and cross-validate a non-negative policy-cost schedule."""

    checked_config = config.validated()
    checked = _canonical_observations(observations, checked_config.anchors)
    campaign_design = _campaign_design_summary(
        checked, required=checked_config.require_campaign_design
    )
    campaign_design = dict(campaign_design)
    if campaign_allocation_manifest is None:
        campaign_design["allocation_binding"] = {
            "required": checked_config.require_campaign_design,
            "validated": False,
            "reason": "standalone calibration regression",
        }
    else:
        if not checked_config.require_campaign_design:
            raise CalibrationError(
                "allocation manifest requires locked campaign design"
            )
        campaign_design["allocation_binding"] = _campaign_allocation_binding(
            checked, campaign_allocation_manifest
        )
    if campaign_evidence_binding is None:
        campaign_design["evidence_binding"] = {
            "required": checked_config.require_campaign_design,
            "validated": False,
            "reason": "standalone calibration regression",
        }
    else:
        required_evidence = {
            "schema", "stage", "raw_verified", "execution_sha256",
            "evaluation_sha256", "evaluation_header_sha256", "allocation_sha256",
            "train_input_sha256",
        }
        if not checked_config.require_campaign_design or set(campaign_evidence_binding) != required_evidence or \
                campaign_evidence_binding.get("schema") != "lc-policy-cost-v6-train-evidence-binding-v1" or \
                campaign_evidence_binding.get("stage") != "TRAIN" or \
                campaign_evidence_binding.get("raw_verified") is not True:
            raise CalibrationError("TRAIN campaign evidence binding drift")
        for field in required_evidence - {"schema", "stage", "raw_verified"}:
            _lower_sha256(campaign_evidence_binding[field], f"TRAIN evidence {field}")
        campaign_design["evidence_binding"] = {
            "required": True, "validated": True,
            **dict(campaign_evidence_binding),
        }
    assignment = make_group_folds(
        checked,
        folds=checked_config.folds,
        seed=checked_config.fold_seed,
        stratify_campaign_cells=checked_config.require_campaign_design,
    )
    fold_counts = tuple(
        sum(value == fold for value in assignment.values())
        for fold in range(checked_config.folds)
    )
    if max(fold_counts) - min(fold_counts) > 1:
        raise CalibrationError("source-match fold allocation is globally imbalanced")
    campaign_design["fold_assignment"] = {
        "method": (
            "SHA256-ranked within fixed campaign cell; rotating cumulative "
            "cell offset modulo folds"
            if checked_config.require_campaign_design else
            "global SHA256 rank modulo folds"
        ),
        "folds": checked_config.folds,
        "source_match_counts": list(fold_counts),
        "assignment_sha256": _fold_digest(assignment),
    }
    campaign_design["design_identifiability"] = _design_rank_diagnostics(
        checked, assignment, checked_config
    )
    cv_rows: list[CrossValidationRow] = []
    for smoothness in checked_config.smoothness_grid:
        fold_losses: list[tuple[float, ...]] = []
        fit_convergence: list[dict[str, Any]] = []
        for fold in range(checked_config.folds):
            training = tuple(
                item for item in checked
                if assignment[item.source_match_id] != fold
            )
            validation = tuple(
                item for item in checked
                if assignment[item.source_match_id] == fold
            )
            if not training or not validation:
                raise CalibrationError("empty cross-validation partition")
            train_x, train_y, train_w = _design(
                training,
                checked_config.anchors,
                checked_config.standard_error_floor,
            )
            coefficients = _fit_one(
                train_x,
                train_y,
                train_w,
                anchors=checked_config.anchors,
                smoothness=smoothness,
                config=checked_config,
                convergence_log=fit_convergence,
                fit_label=f"outer-cv:smoothness-{smoothness}:fold-{fold}",
            )
            validation_x, validation_y, validation_w = _design(
                validation,
                checked_config.anchors,
                checked_config.standard_error_floor,
            )
            fold_losses.append(_validation_group_losses(
                validation,
                validation_x,
                validation_y,
                validation_w,
                coefficients,
                checked_config.huber_delta,
            ))
        all_group_losses = np.asarray(
            [loss for values in fold_losses for loss in values],
            dtype=np.float64,
        )
        mean = float(np.mean(all_group_losses))
        standard_error = (
            float(np.std(all_group_losses, ddof=1)
                  / math.sqrt(len(all_group_losses)))
            if len(all_group_losses) > 1 else 0.0
        )
        cv_rows.append(CrossValidationRow(
            smoothness=smoothness,
            mean_group_huber_loss=mean,
            group_standard_error=standard_error,
            paired_standard_error_vs_minimum=0.0,
            fold_group_losses=tuple(fold_losses),
            fit_convergence=tuple(fit_convergence),
        ))
    best = min(cv_rows, key=lambda row: (
        row.mean_group_huber_loss, -row.smoothness,
    ))
    best_losses = np.asarray(
        [loss for values in best.fold_group_losses for loss in values],
        dtype=np.float64,
    )
    paired_rows: list[CrossValidationRow] = []
    for row in cv_rows:
        row_losses = np.asarray(
            [loss for values in row.fold_group_losses for loss in values],
            dtype=np.float64,
        )
        if row_losses.shape != best_losses.shape:
            raise CalibrationError("cross-validation cluster rows do not align")
        loss_differences = row_losses - best_losses
        paired_se = (
            float(np.std(loss_differences, ddof=1)
                  / math.sqrt(len(loss_differences)))
            if len(loss_differences) > 1 else 0.0
        )
        paired_rows.append(CrossValidationRow(
            smoothness=row.smoothness,
            mean_group_huber_loss=row.mean_group_huber_loss,
            group_standard_error=row.group_standard_error,
            paired_standard_error_vs_minimum=paired_se,
            fold_group_losses=row.fold_group_losses,
            fit_convergence=row.fit_convergence,
        ))
    cv_rows = paired_rows
    best = min(cv_rows, key=lambda row: (
        row.mean_group_huber_loss, -row.smoothness,
    ))
    selected_smoothness = _conventional_one_se_smoothness([
        (row.smoothness, row.mean_group_huber_loss, row.group_standard_error)
        for row in cv_rows
    ])
    selected = next(
        row for row in cv_rows if row.smoothness == selected_smoothness
    )
    design, target, weights = _design(
        checked,
        checked_config.anchors,
        checked_config.standard_error_floor,
    )
    final_fit_convergence: list[dict[str, Any]] = []
    coefficients = _fit_one(
        design,
        target,
        weights,
        anchors=checked_config.anchors,
        smoothness=selected.smoothness,
        config=checked_config,
        convergence_log=final_fit_convergence,
        fit_label="final-whole-train",
    )
    knot_count = len(checked_config.anchors)
    schedule = PolicyCostSchedule(
        anchors=checked_config.anchors,
        beta_search=tuple(
            float(value) for value in coefficients[:knot_count]
        ),
        alpha_core=tuple(
            float(value) for value in coefficients[knot_count:2 * knot_count]
        ),
        alpha_draw=tuple(
            float(value) for value in coefficients[2 * knot_count:]
        ),
    ).validated()
    model_adequacy = _model_adequacy(
        checked, assignment, checked_config
    )
    calibration_passed = bool(model_adequacy["passed"])
    return CalibrationResult(
        calibration_passed=calibration_passed,
        schedule=schedule if calibration_passed else None,
        selected_smoothness=selected.smoothness,
        cv_rows=tuple(cv_rows),
        huber_delta=checked_config.huber_delta,
        standard_error_floor=checked_config.standard_error_floor,
        observation_count=len(checked),
        source_match_count=len(assignment),
        folds=checked_config.folds,
        fold_seed=checked_config.fold_seed,
        fold_assignment_sha256=_fold_digest(assignment),
        fold_source_match_counts=fold_counts,
        observation_input_sha256=observation_input_sha256(checked),
        min_search_beta=checked_config.min_search_beta,
        min_core_alpha=checked_config.min_core_alpha,
        min_draw_alpha=checked_config.min_draw_alpha,
        campaign_design=campaign_design,
        model_adequacy=model_adequacy,
        max_irls_iterations=checked_config.max_irls_iterations,
        max_coordinate_iterations=checked_config.max_coordinate_iterations,
        solver_tolerance=checked_config.tolerance,
        final_fit_convergence=final_fit_convergence[0],
    )


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise CalibrationError(f"non-finite JSON number: {value}")


def read_observation_jsonl(path: Path) -> list[PairObservation]:
    """Read strict JSONL, rejecting duplicates, non-finites, and extra keys."""

    result: list[PairObservation] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CalibrationError(f"cannot read observation JSONL: {exc}") from exc
    for line_number, raw in enumerate(lines, 1):
        if not raw.strip():
            raise CalibrationError(f"blank JSONL row at line {line_number}")
        try:
            value = json.loads(
                raw,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, CalibrationError) as exc:
            raise CalibrationError(
                f"invalid JSON at line {line_number}: {exc}"
            ) from exc
        if not isinstance(value, dict):
            raise CalibrationError(f"line {line_number} is not an object")
        result.append(PairObservation.from_mapping(value))
    return result


def read_strict_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_json_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise CalibrationError(f"strict JSON {path} must contain an object")
    return value


def write_result_no_clobber(path: Path, payload: bytes) -> None:
    """Publish complete bytes atomically without replacing any destination."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent,
    )
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o644)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, destination)
            linked = True
        except FileExistsError as exc:
            raise CalibrationError(
                f"refusing to replace existing output: {destination}"
            ) from exc
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        # If linking succeeded, the destination is already a complete inode;
        # leave it as valid evidence even if a subsequent directory fsync is
        # unavailable, and still surface the durability error to the caller.
        raise
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    if not linked:
        raise CalibrationError("calibration output was not published")


def _parse_csv_numbers(value: str, cast: Any) -> tuple[Any, ...]:
    try:
        return tuple(cast(item) for item in value.split(",") if item != "")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="immutable pair-observation JSONL")
    parser.add_argument("--output", required=True, type=Path,
                        help="canonical calibration JSON")
    parser.add_argument(
        "--anchors", default=",".join(map(str, DEFAULT_PLY_ANCHORS)),
        help="comma-separated frozen ply anchors",
    )
    parser.add_argument(
        "--smoothness-grid",
        default=",".join(map(str, DEFAULT_SMOOTHNESS_GRID)),
        help="comma-separated non-negative roughness penalties",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", default=DEFAULT_FOLD_SEED)
    parser.add_argument("--huber-delta", type=float, default=1.345)
    parser.add_argument(
        "--min-search-beta", type=float, default=DEFAULT_MIN_SEARCH_BETA,
    )
    parser.add_argument("--min-core-alpha", type=float, default=0.0)
    parser.add_argument("--min-draw-alpha", type=float, default=0.0)
    parser.add_argument(
        "--standard-error-floor", type=float,
        default=DEFAULT_STANDARD_ERROR_FLOOR,
    )
    parser.add_argument(
        "--require-campaign-design", action="store_true",
        help="enforce the exact 864-cell locked TRAIN design and model checks",
    )
    parser.add_argument(
        "--campaign-allocation-manifest", type=Path,
        help="sealed policy-only TRAIN allocation authority (required in campaign)",
    )
    parser.add_argument("--campaign-evidence-binding", type=Path)
    parser.add_argument(
        "--model-lack-max-relative-improvement", type=float,
        default=MODEL_LACK_MAX_RELATIVE_IMPROVEMENT,
    )
    arguments = parser.parse_args(argv)
    config = FitConfig(
        anchors=_parse_csv_numbers(arguments.anchors, int),
        smoothness_grid=_parse_csv_numbers(arguments.smoothness_grid, float),
        folds=arguments.folds,
        fold_seed=arguments.fold_seed,
        huber_delta=arguments.huber_delta,
        min_search_beta=arguments.min_search_beta,
        min_core_alpha=arguments.min_core_alpha,
        min_draw_alpha=arguments.min_draw_alpha,
        standard_error_floor=arguments.standard_error_floor,
        require_campaign_design=arguments.require_campaign_design,
        model_lack_max_relative_improvement=(
            arguments.model_lack_max_relative_improvement
        ),
    )
    if arguments.require_campaign_design != bool(
            arguments.campaign_allocation_manifest) or \
            arguments.require_campaign_design != bool(arguments.campaign_evidence_binding):
        parser.error(
            "--require-campaign-design and --campaign-allocation-manifest "
            "must be supplied together"
        )
    allocation = (
        read_strict_json(arguments.campaign_allocation_manifest)
        if arguments.campaign_allocation_manifest else None
    )
    evidence = (read_strict_json(arguments.campaign_evidence_binding)
                if arguments.campaign_evidence_binding else None)
    if evidence is not None:
        try:
            actual_input_sha = hashlib.sha256(arguments.input.read_bytes()).hexdigest()
        except OSError as exc:
            raise CalibrationError(f"cannot hash TRAIN input: {exc}") from exc
        if evidence.get("train_input_sha256") != actual_input_sha:
            raise CalibrationError("TRAIN evidence/input digest drift")
    result = calibrate_policy_cost(
        read_observation_jsonl(arguments.input), config, allocation, evidence
    )
    if (arguments.require_campaign_design
            and not result.campaign_design["allocation_binding"]["validated"]):
        raise CalibrationError("campaign allocation binding was not validated")
    write_result_no_clobber(arguments.output, result.canonical_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
