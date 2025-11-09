"""
NSGA-3 main engine implementation.

Phase 2 introduces the full evolutionary loop, reference-point niching, and
safe checkpoint/resume plumbing so future work can focus on refining operator
behaviour without worrying about infrastructure.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from crypto_trading_bot.nsga3 import objective_evaluator
from crypto_trading_bot.nsga3.individual import Individual
from crypto_trading_bot.nsga3.integration_hook import promote_to_learning_machine
from crypto_trading_bot.nsga3.jsonl_utils import load_jsonl
from crypto_trading_bot.nsga3.learning_bridge import get_adjustments
from crypto_trading_bot.nsga3.promotion_manager import run_promotion_cycle
from crypto_trading_bot.nsga3.reference_points import (
    OBJECTIVE_SPECS,
    associate_reference_points,
    generate_reference_points,
)
from crypto_trading_bot.nsga3.regime_utils import normalize_regime_label
from crypto_trading_bot.nsga3.shadow_simulation import run_shadow_tests

LOGGER = logging.getLogger(__name__)
OBJECTIVES: Tuple[Tuple[str, str], ...] = OBJECTIVE_SPECS
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config" / "nsga3.json"


def _build_evaluation_payloads(
    population: list["Individual"],
    cfg: dict[str, Any],
    gen: int,
    base_seed: int | None,
) -> list[dict[str, Any]]:
    """Create deterministic worker payloads for parallel evaluation."""
    safe_seed = int(base_seed or 0)
    payloads: list[dict[str, Any]] = []
    for index, individual in enumerate(population):
        payloads.append(
            {
                "index": index,
                "params": dict(individual.params),
                "seed": safe_seed + gen * 10_000 + index,
                "config": dict(cfg or {}),
            }
        )
    return payloads


def _evaluate_individual_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Process-pool compatible evaluation wrapper."""
    index = int(payload.get("index", 0))
    params = dict(payload.get("params") or {})
    seed = int(payload.get("seed", 0))
    random.seed(seed)
    try:
        objectives = objective_evaluator.evaluate_individual({"params": params})
    except Exception:  # pragma: no cover - executor safety net  # pylint: disable=broad-exception-caught
        LOGGER.exception("Parallel evaluator failed for index %s", index)
        objectives = {
            "roi": float("-inf"),
            "drawdown": 1.0,
            "win_rate": 0.0,
        }
    return {"index": index, "objectives": objectives, "params": params}


def evaluate_population_parallel(
    population: list["Individual"],
    cfg: dict[str, Any],
    gen: int,
    workers: int,
    base_seed: int | None,
) -> list["Individual"]:
    """Evaluate a population using multi-process execution."""
    if not population:
        return population
    payloads = _build_evaluation_payloads(population, cfg, gen, base_seed)
    max_workers = max(1, int(workers or 1))
    results: list[dict[str, Any]] = []
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_evaluate_individual_payload, payload): payload for payload in payloads}
            for future in as_completed(future_map):
                payload = future_map[future]
                try:
                    results.append(future.result())
                except (
                    Exception
                ) as exc:  # pragma: no cover - executor resilience  # pylint: disable=broad-exception-caught
                    LOGGER.exception("Evaluation failed for index %s: %s", payload["index"], exc)
                    results.append(
                        {
                            "index": payload["index"],
                            "objectives": {
                                "roi": float("-inf"),
                                "drawdown": 1.0,
                                "win_rate": 0.0,
                            },
                            "params": payload.get("params", {}),
                        }
                    )
    except (PermissionError, NotImplementedError, OSError, RuntimeError) as exc:
        LOGGER.warning("Process pool unavailable (%s); falling back to sequential evaluation.", exc)
        for payload in payloads:
            results.append(_evaluate_individual_payload(payload))
    for record in sorted(results, key=lambda item: int(item.get("index", 0))):
        idx = int(record.get("index", 0))
        if 0 <= idx < len(population):
            population[idx].objectives = dict(record.get("objectives") or {})
    return population


def _load_prior_promotions(path: str = "logs/nsga3_promotions_global.jsonl") -> list[dict[str, Any]]:
    """Load prior promotion records for Bayesian warm starts."""
    return load_jsonl(path)


# pylint: disable=too-many-locals,too-many-branches,too-many-statements


def _bayesian_warmstart(
    bounds: dict[str, dict[str, float]],
    prior: list[dict[str, Any]],
    n_suggest: int,
    seed: int | None,
) -> list[dict[str, float]]:
    """Generate Bayesian warm-start candidates using scikit-optimize when available."""
    if not prior or n_suggest <= 0:
        return []
    try:
        from skopt import Optimizer  # pylint: disable=import-outside-toplevel
    except ImportError:
        LOGGER.warning("scikit-optimize unavailable; skipping Bayesian warm-start suggestions.")
        return []

    param_keys = list(bounds.keys())
    dimensions = []
    for name in param_keys:
        spec = bounds.get(name, {})
        lower = float(spec.get("min", 0.0))
        upper = float(spec.get("max", 1.0))
        dimensions.append((lower, upper))
    optimizer = Optimizer(
        dimensions=dimensions,
        base_estimator="GP",
        acq_func="EI",
        random_state=int(seed or 0),
    )

    scored_entries: list[tuple[float, list[float]]] = []
    for entry in prior:
        params = entry.get("params")
        if not isinstance(params, dict):
            continue
        score_candidate = (
            entry.get("rar") or entry.get("rar_nsga3") or (entry.get("metrics") or {}).get("risk_adjusted_roi")
        )
        if score_candidate is None:
            objectives = entry.get("objectives") or {}
            for key in ("rar", "risk_adjusted_roi", "roi"):
                if key in objectives:
                    score_candidate = objectives.get(key)
                    break
        try:
            score_value = float(score_candidate)
        except (TypeError, ValueError):
            continue

        vector: list[float] = []
        valid = True
        for name in param_keys:
            spec = bounds.get(name, {})
            lower = float(spec.get("min", 0.0))
            upper = float(spec.get("max", 1.0))
            raw_value = params.get(name, lower)
            try:
                clamped_value = min(max(float(raw_value), lower), upper)
            except (TypeError, ValueError):
                valid = False
                break
            vector.append(clamped_value)
        if valid:
            scored_entries.append((score_value, vector))

    if not scored_entries:
        return []

    scored_entries.sort(key=lambda item: item[0], reverse=True)
    top_entries = scored_entries[: max(n_suggest * 3, n_suggest)]
    for score, vector in top_entries:
        optimizer.tell(vector, -score)

    suggestions: list[dict[str, float]] = []
    request_count = max(1, n_suggest)
    try:
        candidate_vectors = optimizer.ask(n_points=request_count)
    except TypeError:
        candidate_vectors = [optimizer.ask() for _ in range(request_count)]

    for vector in candidate_vectors[:n_suggest]:
        suggestion: dict[str, float] = {}
        for name, value in zip(param_keys, vector):
            spec = bounds.get(name, {})
            lower = float(spec.get("min", 0.0))
            upper = float(spec.get("max", 1.0))
            suggestion[name] = min(max(float(value), lower), upper)
        suggestions.append(suggestion)
    return suggestions


class NSGA3Engine:  # pylint: disable=too-many-instance-attributes
    """Primary NSGA-3 evolution driver."""

    def __init__(self, config_path: str | Path, regime: str = "global"):
        self.config_path = Path(config_path)
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.population_size = int(self.config.get("population", 200))
        self.generations = int(self.config.get("generations", 100))
        self.num_workers = int(self.config.get("cores", 8))

        base_mutation = float(self.config.get("mutation_rate", 0.15))
        self.cx_eta = float(self.config.get("cx_eta", 15))
        self.cx_prob = float(self.config.get("cx_prob", 0.9))
        self.mut_eta = float(self.config.get("mut_eta", 20))
        self.mut_prob = float(self.config.get("mut_prob", base_mutation))
        self.base_cx_eta = self.cx_eta
        self.base_cx_prob = self.cx_prob
        self.base_mut_eta = self.mut_eta
        self.base_mut_prob = self.mut_prob

        divisions = int(self.config.get("divisions", 4))
        self.reference_points = generate_reference_points(len(OBJECTIVES), divisions)

        self.constraints = self.config.get("constraints", {})
        self.population: List[Individual] = []
        self.regime = normalize_regime_label(regime)
        self.checkpoint_file = Path(f"logs/nsga3_checkpoint_{self.regime}.json")
        self.promotions_log_path = Path(f"logs/nsga3_promotions_{self.regime}.jsonl")
        self.warmstart_log_path = Path(f"logs/nsga3_warmstart_{self.regime}.log")
        raw_seed = self.config.get("seed")
        self.seed = int(raw_seed) if raw_seed is not None else 0
        self.rng = random.Random(self.seed)
        self.parallel_workers = int(self.config.get("parallel_workers", self.num_workers))
        self.shadow_config = self.config.get("shadow_test", {}) or {}
        self.objective_weights: Dict[str, float] = {"roi": 1.0, "drawdown": 1.0, "win_rate": 1.0}
        self.last_generation: int = -1

        os.makedirs("logs", exist_ok=True)

    # ------------------------------ Evolution Core ------------------------------

    def initialize_population(self) -> list[Individual]:
        """Return a warm-started population with Bayesian suggestions when available."""
        warmstart_cfg = self.config.get("warmstart", {}) or {}
        n_suggest = int(warmstart_cfg.get("suggestions", min(16, self.population_size)))
        prior_path = warmstart_cfg.get("prior_path") or self.promotions_log_path
        prior_records = _load_prior_promotions(str(prior_path))
        suggestions = _bayesian_warmstart(
            Individual.PARAM_BOUNDS,
            prior_records,
            min(n_suggest, self.population_size),
            self.seed,
        )
        population: list[Individual] = []
        for params in suggestions:
            individual = Individual(params, metadata={"origin": "bayesian_warmstart"})
            population.append(individual)

        while len(population) < self.population_size:
            candidate = Individual.random(rng=self.rng)
            candidate.metadata.setdefault("origin", "random")
            population.append(candidate)

        self._log_warmstart_summary(
            prior_count=len(prior_records),
            warmstart_count=len(suggestions),
            prior_path=str(prior_path),
        )
        if not suggestions:
            LOGGER.warning(
                "Bayesian warm-start unavailable (regime=%s, prior=%s); using random init.",
                self.regime,
                prior_path,
            )
        return population[: self.population_size]

    def _log_warmstart_summary(self, prior_count: int, warmstart_count: int, prior_path: str) -> None:
        """Persist a durable warm-start summary for auditing."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "regime": self.regime,
            "seed": self.seed,
            "prior_records": prior_count,
            "warmstart_candidates": warmstart_count,
            "population_size": self.population_size,
            "prior_path": prior_path,
        }
        self.warmstart_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.warmstart_log_path.open("a", encoding="utf-8") as handle:
            json.dump(entry, handle)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def apply_constraints(self, individual: Individual) -> None:
        """Update constraint violation score for an individual."""
        max_drawdown = float(self.constraints.get("max_drawdown", 1.0))
        min_win_rate = float(self.constraints.get("min_win_rate", 0.0))
        drawdown = float(individual.objectives.get("drawdown", 1.0))
        win_rate = float(individual.objectives.get("win_rate", 0.0))
        violation = 0.0
        if drawdown > max_drawdown:
            violation += drawdown - max_drawdown
        if win_rate < min_win_rate:
            violation += max(0.0, min_win_rate - win_rate)
        individual.constraint_violation = violation

    def evolve(
        self, *, dry_run: bool = False
    ) -> None:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        """
        Execute the NSGA-3 evolution loop with optional dry-run mode.
        """
        checkpoint = self._load_checkpoint()
        if checkpoint:
            self.population = checkpoint["population"]
            start_generation = checkpoint["generation"] + 1
            LOGGER.info("Loaded NSGA-3 checkpoint at generation %s", checkpoint["generation"])
        else:
            self.population = self.initialize_population()
            start_generation = 0

        self._evaluate_population(self.population, start_generation)
        self._apply_constraints_to_population(self.population)

        target_generations = 1 if dry_run else self.generations
        if start_generation >= target_generations:
            LOGGER.info(
                "NSGA-3 already completed %s generations; exiting.",
                target_generations,
            )
            return

        for generation in range(start_generation, target_generations):
            self._apply_learning_adjustments(generation)
            offspring = self._produce_offspring()
            self._evaluate_population(offspring, generation)
            self._apply_constraints_to_population(offspring)

            combined = self.population + offspring
            fronts = fast_non_dominated_sort(combined, weights=self.objective_weights)
            for front in fronts:
                calculate_crowding_distance(front, weights=self.objective_weights)

            next_population, pareto_front = self._select_next_population(fronts)
            self.save_checkpoint(next_population, pareto_front, generation)
            self._log_generation(generation, pareto_front)

            if pareto_front:
                promote_to_learning_machine(
                    [individual.to_dict() for individual in pareto_front[:5]],
                    generation=generation,
                    reason="pareto_front",
                    regime=self.regime,
                )

            shadow_results = self._run_shadow_tests(pareto_front, generation)
            run_promotion_cycle(
                population=[ind.to_dict() for ind in next_population],
                shadow_results=shadow_results,
                generation=generation,
                regime=self.regime,
            )

            self.population = next_population
            self.last_generation = generation
            if dry_run:
                break

    # ------------------------------ Persistence --------------------------------

    def save_checkpoint(
        self,
        population: list[Individual],
        pareto_front: list[Individual],
        gen: int,
    ) -> None:
        """Persist the current generation to disk."""
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "generation": gen,
            "regime": self.regime,
            "population": [ind.to_dict() for ind in population],
            "pareto_front": [ind.to_dict() for ind in pareto_front],
        }
        os.makedirs(self.checkpoint_file.parent, exist_ok=True)
        with self.checkpoint_file.open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        LOGGER.info("[NSGA-3] Checkpoint saved successfully.")

    def _load_checkpoint(self) -> dict | None:
        """Load the most recent checkpoint if present."""
        if not self.checkpoint_file.exists():
            return None
        try:
            data = json.loads(self.checkpoint_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            LOGGER.warning("Unable to read NSGA-3 checkpoint: %s", exc)
            return None
        population = [Individual.from_dict(entry) for entry in data.get("population", [])]
        generation = int(data.get("generation", 0))
        return {"population": population, "generation": generation}

    # ------------------------------ Selection Helpers ---------------------------

    def _apply_constraints_to_population(self, population: list[Individual]) -> None:
        for individual in population:
            self.apply_constraints(individual)

    def _apply_learning_adjustments(self, generation: int) -> dict[str, float]:
        """Incorporate continual learning feedback into objective weights and operators."""
        os.environ["NSGA3_ADAPT_SEED"] = str(self.seed)
        try:
            adjustments = get_adjustments()
        except Exception as exc:  # pragma: no cover - defensive bridge  # pylint: disable=broad-exception-caught
            LOGGER.exception("Learning bridge adjustments failed: %s", exc)
            adjustments = {}

        roi_weight = float(adjustments.get("roi_w", 1.0))
        drawdown_weight = float(adjustments.get("drawdown_w", 1.0))
        win_weight = float(adjustments.get("winrate_w", 1.0))
        self.objective_weights = {
            "roi": max(0.0, roi_weight),
            "drawdown": max(0.0, drawdown_weight),
            "win_rate": max(0.0, win_weight),
        }

        mutation_delta = float(adjustments.get("mutation_rate", 0.0))
        cx_prob_delta = float(adjustments.get("cx_prob", 0.0))
        cx_eta_scale = 1.0 + float(adjustments.get("cx_eta", 0.0))
        mut_eta_scale = 1.0 + float(adjustments.get("mut_eta", 0.0))

        self.mut_prob = self._clamp_probability(self.base_mut_prob + mutation_delta)
        self.cx_prob = self._clamp_probability(self.base_cx_prob + cx_prob_delta)
        self.cx_eta = max(1.0, self.base_cx_eta * cx_eta_scale)
        self.mut_eta = max(1.0, self.base_mut_eta * mut_eta_scale)

        LOGGER.info(
            "[NSGA-3] Applied learning adjustments | gen=%s weights=%s mut_prob=%.3f cx_prob=%.3f",
            generation,
            self.objective_weights,
            self.mut_prob,
            self.cx_prob,
        )
        return adjustments

    @staticmethod
    def _clamp_probability(value: float) -> float:
        """Clamp operator probabilities to [0, 1]."""
        return min(max(value, 0.0), 1.0)

    def _evaluate_population(self, population: list[Individual], generation: int) -> None:
        """Evaluate individuals deterministically via the process pool."""
        evaluate_population_parallel(
            population,
            self.config,
            generation,
            self.parallel_workers,
            self.seed,
        )

    def _run_shadow_tests(
        self,
        pareto_front: list[Individual],
        generation: int,
    ) -> list[dict[str, Any]]:
        """Execute shadow simulations for the top Pareto individuals."""
        if not self.shadow_config:
            return []
        max_trades = int(self.shadow_config.get("max_trades", 500))
        top_count = int(self.shadow_config.get("top_count", 5))
        candidates = pareto_front[:top_count] if top_count > 0 else pareto_front
        if not candidates:
            return []
        candidate_payloads = [
            individual.to_dict() if hasattr(individual, "to_dict") else individual for individual in candidates
        ]
        LOGGER.info(
            "[NSGA-3] Running shadow tests for %s individuals (generation %s)",
            len(candidates),
            generation,
        )
        try:
            return run_shadow_tests(candidate_payloads, max_trades=max_trades, regime=self.regime)
        # pylint: disable=broad-exception-caught
        except Exception as exc:  # pragma: no cover - downstream resilience
            LOGGER.exception("Shadow tests failed: %s", exc)
            return []

    def _produce_offspring(self) -> list[Individual]:
        """Create offspring using SBX crossover and polynomial mutation."""
        if not self.population:
            return []
        offspring: List[Individual] = []
        while len(offspring) < self.population_size:
            parent_a = self._tournament_selection()
            parent_b = self._tournament_selection()
            child_a, child_b = sbx_crossover(
                parent_a,
                parent_b,
                self.cx_eta,
                self.cx_prob,
                rng=self.rng,
            )
            polynomial_mutation(child_a, self.mut_eta, self.mut_prob, rng=self.rng)
            polynomial_mutation(child_b, self.mut_eta, self.mut_prob, rng=self.rng)
            child_a.clamp()
            child_b.clamp()
            offspring.extend([child_a, child_b])
        return offspring[: self.population_size]

    def _tournament_selection(self) -> Individual:
        """Binary tournament selection that respects constraints and rank."""
        if len(self.population) == 1:
            return self.population[0]
        contenders = self.rng.sample(self.population, 2)
        return self._prefer_individual(contenders[0], contenders[1])

    @staticmethod
    def _prefer_individual(
        candidate_a: Individual,
        candidate_b: Individual,
    ) -> Individual:
        """Return the better individual given rank, crowding, and violations."""
        result = candidate_a
        if candidate_a.constraint_violation != candidate_b.constraint_violation:
            result = candidate_a if candidate_a.constraint_violation < candidate_b.constraint_violation else candidate_b
        elif candidate_a.rank != candidate_b.rank:
            result = candidate_a if candidate_a.rank < candidate_b.rank else candidate_b
        elif candidate_a.crowding_distance != candidate_b.crowding_distance:
            result = candidate_a if candidate_a.crowding_distance > candidate_b.crowding_distance else candidate_b
        return result

    def _select_next_population(
        self,
        fronts: list[list[Individual]],
    ) -> tuple[list[Individual], list[Individual]]:
        """Select the next generation from ranked fronts."""
        next_population: List[Individual] = []
        reference_counts = {idx: 0 for idx in range(len(self.reference_points))}
        pareto_front = fronts[0] if fronts else []

        for front in fronts:
            if len(next_population) + len(front) <= self.population_size:
                next_population.extend(front)
                self._update_reference_counts(reference_counts, front)
                continue

            remaining_slots = self.population_size - len(next_population)
            if remaining_slots > 0:
                selected = self._niching_selection(front, remaining_slots, reference_counts)
                next_population.extend(selected)
            break

        return next_population, pareto_front

    def _update_reference_counts(
        self,
        reference_counts: dict[int, int],
        individuals: list[Individual],
    ) -> None:
        associations = associate_reference_points(
            individuals,
            self.reference_points,
            objective_specs=OBJECTIVES,
            rng=self.rng,
            weights=self.objective_weights,
        )
        for ref_idx, members in associations.items():
            reference_counts[ref_idx] = reference_counts.get(ref_idx, 0) + len(members)

    def _niching_selection(
        self,
        front: list[Individual],
        slots: int,
        reference_counts: dict[int, int],
    ) -> list[Individual]:
        """Fill the final slots using reference-point association."""
        associations = associate_reference_points(
            front,
            self.reference_points,
            objective_specs=OBJECTIVES,
            rng=self.rng,
            weights=self.objective_weights,
        )
        candidates = {idx: list(members) for idx, members in associations.items() if members}
        remaining = list(front)
        selected: List[Individual] = []

        while len(selected) < slots and candidates:
            best_idx = min(
                candidates.keys(),
                key=lambda ref: (reference_counts.get(ref, 0), self.rng.random()),
            )
            bucket = candidates[best_idx]
            if not bucket:
                del candidates[best_idx]
                continue
            _, individual = bucket.pop(0)
            if individual in remaining:
                selected.append(individual)
                remaining.remove(individual)
                reference_counts[best_idx] = reference_counts.get(best_idx, 0) + 1
            if not bucket:
                del candidates[best_idx]

        while len(selected) < slots and remaining:
            choice = self.rng.choice(remaining)
            remaining.remove(choice)
            selected.append(choice)

        return selected

    def _log_generation(self, generation: int, pareto_front: list[Individual]) -> None:
        """Emit a short summary for dashboards/logs."""
        if not pareto_front:
            LOGGER.info("[NSGA-3] Generation %s | Pareto front size: 0", generation)
            return
        best_roi = max(ind.objectives.get("roi", 0.0) for ind in pareto_front)
        LOGGER.info(
            "[NSGA-3] Generation %s | Pareto front size: %s | Best ROI: %.2f%%",
            generation,
            len(pareto_front),
            best_roi * 100,
        )


# ---------------------------------------------------------------------------
# Evolution helpers
# ---------------------------------------------------------------------------


def fast_non_dominated_sort(
    population: Iterable[Individual],
    weights: dict[str, float] | None = None,
) -> list[list[Individual]]:
    """Return Pareto fronts using Deb's fast non-dominated sort."""
    individuals = list(population)
    if not individuals:
        return []

    domination_counts: dict[Individual, int] = {}
    dominated: dict[Individual, list[Individual]] = {}
    fronts: list[list[Individual]] = [[]]

    for individual in individuals:
        dominated[individual] = []
        domination_counts[individual] = 0
        for other in individuals:
            if individual is other:
                continue
            if _dominates(individual, other, weights):
                dominated[individual].append(other)
            elif _dominates(other, individual, weights):
                domination_counts[individual] += 1
        if domination_counts[individual] == 0:
            individual.rank = 0
            fronts[0].append(individual)

    i = 0
    while fronts[i]:
        next_front: list[Individual] = []
        for individual in fronts[i]:
            for dominated_individual in dominated[individual]:
                domination_counts[dominated_individual] -= 1
                if domination_counts[dominated_individual] == 0:
                    dominated_individual.rank = i + 1
                    next_front.append(dominated_individual)
        i += 1
        fronts.append(next_front)

    if not fronts[-1]:
        fronts.pop()
    return fronts


def _dominates(
    candidate_a: Individual,
    candidate_b: Individual,
    weights: dict[str, float] | None = None,
) -> bool:
    """Return True if candidate_a dominates candidate_b."""
    violation_a = getattr(candidate_a, "constraint_violation", 0.0)
    violation_b = getattr(candidate_b, "constraint_violation", 0.0)
    if violation_a < violation_b:
        return True
    if violation_a > violation_b:
        return False

    better = False
    for name, direction in OBJECTIVES:
        value_a = _apply_weight(float(candidate_a.objectives.get(name, 0.0)), name, weights)
        value_b = _apply_weight(float(candidate_b.objectives.get(name, 0.0)), name, weights)
        if direction == "max":
            if value_a < value_b:
                return False
            if value_a > value_b:
                better = True
        else:
            if value_a > value_b:
                return False
            if value_a < value_b:
                better = True
    return better


def calculate_crowding_distance(
    front: list[Individual],
    weights: dict[str, float] | None = None,
) -> None:
    """Calculate crowding distance for individuals within a front."""
    if not front:
        return
    if len(front) <= 2:
        for individual in front:
            individual.crowding_distance = float("inf")
        return

    for individual in front:
        individual.crowding_distance = 0.0

    for objective, _ in OBJECTIVES:
        sorted_front = sorted(
            front,
            key=lambda ind, obj=objective: _apply_weight(float(ind.objectives.get(obj, 0.0)), obj, weights),
        )
        sorted_front[0].crowding_distance = float("inf")
        sorted_front[-1].crowding_distance = float("inf")
        min_value = _apply_weight(float(sorted_front[0].objectives.get(objective, 0.0)), objective, weights)
        max_value = _apply_weight(float(sorted_front[-1].objectives.get(objective, 0.0)), objective, weights)
        span = max_value - min_value
        if span == 0:
            continue
        for idx in range(1, len(sorted_front) - 1):
            prev_value = _apply_weight(
                float(sorted_front[idx - 1].objectives.get(objective, 0.0)),
                objective,
                weights,
            )
            next_value = _apply_weight(
                float(sorted_front[idx + 1].objectives.get(objective, 0.0)),
                objective,
                weights,
            )
            distance = (next_value - prev_value) / span
            if not sorted_front[idx].crowding_distance == float("inf"):
                sorted_front[idx].crowding_distance += distance


def sbx_crossover(
    parent_a: Individual,
    parent_b: Individual,
    eta: float,
    probability: float,
    *,
    rng: random.Random | None = None,
) -> tuple[Individual, Individual]:
    """Simulated binary crossover."""
    # pylint: disable=too-many-locals
    rng = rng or random.Random()
    child_a = parent_a.copy()
    child_b = parent_b.copy()
    if rng.random() > probability:
        return child_a, child_b

    for key, bounds in Individual.PARAM_BOUNDS.items():
        if rng.random() > 0.5:
            continue
        value_a = parent_a.params.get(key, bounds["min"])
        value_b = parent_b.params.get(key, bounds["min"])
        if abs(value_a - value_b) < 1e-12:
            continue
        y1, y2 = sorted([value_a, value_b])
        lower = bounds["min"]
        upper = bounds["max"]
        rand = rng.random()
        beta1 = 1.0 + (2.0 * (y1 - lower) / (y2 - y1))
        alpha1 = 2.0 - beta1 ** -(eta + 1)
        if rand <= 1.0 / alpha1:
            betaq = (rand * alpha1) ** (1.0 / (eta + 1))
        else:
            betaq = (1.0 / (2.0 - rand * alpha1)) ** (1.0 / (eta + 1))
        c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

        rand = rng.random()
        beta2 = 1.0 + (2.0 * (upper - y2) / (y2 - y1))
        alpha2 = 2.0 - beta2 ** -(eta + 1)
        if rand <= 1.0 / alpha2:
            betaq = (rand * alpha2) ** (1.0 / (eta + 1))
        else:
            betaq = (1.0 / (2.0 - rand * alpha2)) ** (1.0 / (eta + 1))
        c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

        c1 = min(max(c1, lower), upper)
        c2 = min(max(c2, lower), upper)
        if rng.random() <= 0.5:
            child_a.params[key] = c2
            child_b.params[key] = c1
        else:
            child_a.params[key] = c1
            child_b.params[key] = c2

    return child_a, child_b


def polynomial_mutation(
    individual: Individual,
    eta: float,
    probability: float,
    *,
    rng: random.Random | None = None,
) -> None:
    """Polynomial mutation applied in-place."""
    # pylint: disable=too-many-locals
    rng = rng or random.Random()
    for key, bounds in Individual.PARAM_BOUNDS.items():
        if rng.random() > probability:
            continue
        lower = bounds["min"]
        upper = bounds["max"]
        value = individual.params.get(key, lower)
        span = upper - lower
        if span <= 0:
            continue
        delta1 = (value - lower) / span
        delta2 = (upper - value) / span
        rand = rng.random()
        mut_pow = 1.0 / (eta + 1.0)
        if rand < 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1))
            deltaq = val**mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1))
            deltaq = 1.0 - val**mut_pow
        value += deltaq * span
        value = min(max(value, lower), upper)
        individual.params[key] = value


def _apply_weight(value: float, objective: str, weights: dict[str, float] | None) -> float:
    """Return weighted objective value if adjustments specify a scale."""
    if not weights:
        return value
    return value * float(weights.get(objective, 1.0))


def run_nsga3_cycle(
    config_path: str | Path | None = None,
    *,
    dry_run: bool = False,
    regime: str = "global",
) -> tuple[list[dict], int]:
    """Run one NSGA-III cycle and return the resulting population and generation."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    engine = NSGA3Engine(path, regime=regime)
    engine.evolve(dry_run=dry_run)
    last_generation = engine.last_generation if engine.last_generation >= 0 else 0
    return [ind.to_dict() for ind in engine.population], last_generation


def run_nsga3(config_path: str, *, dry_run: bool = False, regime: str = "global") -> None:
    """Legacy entrypoint retained for CLI compatibility."""
    run_nsga3_cycle(config_path, dry_run=dry_run, regime=regime)


def main(argv: list[str] | None = None) -> int:
    """CLI wrapper supporting --dry-run for quick validation."""
    parser = argparse.ArgumentParser(description="Run the NSGA-3 optimization engine.")
    parser.add_argument(
        "--config",
        default="src/crypto_trading_bot/nsga3/config/nsga3.json",
        help="Path to NSGA-3 configuration JSON.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Initialise engine only; skip extended evolution loop.",
    )
    parser.add_argument(
        "--regime",
        default="global",
        help="Regime label for checkpoint/log names.",
    )
    args = parser.parse_args(argv)
    LOGGER.info("[NSGA-3] Engine initialized — safe import context OK")
    run_nsga3(args.config, dry_run=args.dry_run, regime=args.regime)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
