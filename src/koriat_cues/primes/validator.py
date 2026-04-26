"""External-judge validation of prime quality.

Biggest validity threat (SPEC §Controls): cue-familiarity primes leak answer
information despite our efforts. For each prime of type `cue_familiarity_priming`,
ask an external judge whether it can answer the target question given only the
prime (no question). If it can more than `max_leak_rate` of the time, drop that item.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from ..eval.grader import grade_prediction
from .generator import PrimeSet


@dataclass
class PrimeJudge:
    model: str = "claude-opus-4-7"
    n_attempts: int = 3  # number of guesses per prime-only attempt
    client: object = None

    def _get_client(self):
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic()
        return self.client

    @retry(stop=stop_after_attempt(3), wait=wait_random_exponential(min=1, max=20))
    def _guess(self, prime_text: str, question: str) -> list[str]:
        client = self._get_client()
        system = (
            "You are an expert trivia solver. You will be given a short passage and a question. "
            "Using ONLY the passage (not your own knowledge), decide whether the passage "
            "contains or strongly implies the answer to the question. If it does, output the "
            "answer on a single line. If it does not, output exactly: UNKNOWN."
        )
        user = (
            f"Passage:\n{prime_text}\n\n"
            f"Question: {question}\n\n"
            "Remember: reply with the answer on a single line, or UNKNOWN."
        )
        resp = client.messages.create(
            model=self.model,
            max_tokens=50,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        guess = text.strip().splitlines()[0] if text.strip() else ""
        if guess.upper().startswith("UNKNOWN"):
            return []
        return [guess]


DEFAULT_LEAK_CHECK_CONDITIONS: tuple[str, ...] = (
    # Primes that claim not to reveal the answer; all need verification.
    "cue_familiarity_priming",
    "partial_accessibility",
    "illusory_tot",
)


def judge_prime_leakage(
    prime_sets: list[PrimeSet],
    judge: PrimeJudge,
    gold_lookup: dict[str, list[str]],
    conditions_to_check: Iterable[str] = DEFAULT_LEAK_CHECK_CONDITIONS,
) -> dict[str, dict[str, float]]:
    """Return a per-item, per-condition leak rate.

    leak_rate[item_id][condition] is the fraction of judge attempts on that item's
    prime that produced an answer matching a gold answer.
    """
    leaks: dict[str, dict[str, float]] = {}
    for ps in tqdm(prime_sets, desc="judge_leakage"):
        per_cond: dict[str, float] = {}
        for cond in conditions_to_check:
            prime = ps.primes.get(cond)
            if prime is None or prime.text is None:
                continue
            hits = 0
            for _ in range(judge.n_attempts):
                guesses = judge._guess(prime.text, ps.question)
                for g in guesses:
                    if grade_prediction(g, gold_lookup.get(ps.item_id, [ps.gold_answer])):
                        hits += 1
                        break
            per_cond[cond] = hits / judge.n_attempts
        leaks[ps.item_id] = per_cond
    return leaks
