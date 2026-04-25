"""Generate primes for every (item, condition) pair using a stronger LLM.

Uses the Anthropic SDK. Each prime is wrapped in a dataclass; a `PrimeSet` groups
primes per item. The `random_paragraph` yoked control is built by sampling
matched-length paragraphs from a neutral reference pool rather than generation.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import json
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

from ..data.loader import Item
from .conditions import CONDITION_SPECS


# A handful of neutral encyclopedic paragraphs used for the yoked `random_paragraph`
# control. Drawn from public-domain-style generic text; they are deliberately topic-
# unrelated to any TriviaQA question. Length-match is done by truncating / padding
# with the next sentence.
_RANDOM_POOL: list[str] = [
    "The city's public transit network operates across multiple fare zones. "
    "Timetables are published quarterly and adjusted for seasonal demand. "
    "Weekend service runs on a reduced schedule.",
    "Modern office furniture design emphasizes ergonomic support and modularity. "
    "Standing desks have become common in open-plan environments. "
    "Manufacturers offer configurable kits to suit different room layouts.",
    "Weather patterns in temperate climates follow predictable annual cycles. "
    "Spring rainfall supports early crop germination. "
    "Autumn brings cooler nights and shorter days.",
    "Technical writing prizes clarity and verifiable claims. "
    "Authors separate evidence from interpretation. "
    "Peer review catches factual errors before publication.",
    "Community libraries host a range of public programs throughout the year. "
    "Borrowing periods vary by item type. "
    "Many branches provide meeting rooms and quiet study spaces.",
    "Bicycle commuting is supported by dedicated lanes and secure parking. "
    "Helmet use is recommended in most jurisdictions. "
    "Maintenance includes regular tire inspection and brake checks.",
]


@dataclass
class Prime:
    condition: str
    # None for baseline / random_paragraph (pre-yoking) / generation failure.
    text: str | None
    # If True, prime generation failed and the trial should be skipped at runtime.
    failed: bool = False


@dataclass
class PrimeSet:
    item_id: str
    question: str
    gold_answer: str
    primes: dict[str, Prime] = field(default_factory=dict)
    # Prime-question ordering for this item. "prime_then_question" (default) or
    # "question_then_prime" (counterbalance control).
    order: str = "prime_then_question"

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "order": getattr(self, "order", "prime_then_question"),
            "primes": {
                k: {"condition": v.condition, "text": v.text, "failed": v.failed}
                for k, v in self.primes.items()
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PrimeSet":
        obj = cls(
            item_id=d["item_id"],
            question=d["question"],
            gold_answer=d["gold_answer"],
            primes={k: Prime(**v) for k, v in d.get("primes", {}).items()},
        )
        # `order` is stored on the PrimeSet for counterbalance; default if missing
        # for back-compat with pilot-era prime files.
        obj.order = d.get("order", "prime_then_question")
        return obj


@dataclass
class PrimeGenerator:
    model: str = "claude-opus-4-7"
    max_sentences: int = 3
    # If None, we instantiate anthropic.Anthropic() at call time.
    client: object = None

    def _get_client(self):
        if self.client is None:
            import anthropic
            self.client = anthropic.Anthropic()
        return self.client

    @retry(stop=stop_after_attempt(4), wait=wait_random_exponential(min=1, max=20))
    def _one_call(self, system: str, user: str) -> str:
        client = self._get_client()
        resp = client.messages.create(
            model=self.model,
            max_tokens=400,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text_parts = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        return "".join(text_parts).strip()

    def _generate_one(self, condition: str, question: str, answer: str) -> Optional[str]:
        """Return generated prime text, or None on irretrievable failure."""
        spec = CONDITION_SPECS[condition]
        if spec.yoked or spec.generator_instruction == "":
            return None

        system = (
            "You are assisting a research experiment on LLM metacognition. You will write a "
            "short priming passage according to a strict specification. Be concise, factual-"
            "sounding, and avoid questions or meta-commentary. Do NOT reveal the answer unless "
            "the instruction explicitly tells you to."
        )
        user = (
            f"{spec.generator_instruction.format(answer=answer)}\n\n"
            f"Question: {question}\n"
            f"Correct answer (do NOT reveal unless explicitly instructed above): {answer}\n\n"
            f"Length: about {self.max_sentences} sentences, max."
        )
        try:
            text = self._one_call(system=system, user=user)
        except Exception:
            return None
        text = re.sub(r"^\s*(passage|here.*?:)\s*", "", text, flags=re.IGNORECASE).strip()
        text = text.strip("\"'`")
        return text or None

    def generate_for_item(
        self, item: Item, conditions: Iterable[str]
    ) -> PrimeSet:
        """Generate primes for one item across the requested conditions."""
        gold = item.answers[0]
        ps = PrimeSet(item_id=item.id, question=item.question, gold_answer=gold)
        for cond in conditions:
            if cond in ("baseline", "random_paragraph"):
                ps.primes[cond] = Prime(condition=cond, text=None)
                continue
            text = self._generate_one(cond, item.question, gold)
            failed = (text is None) and not CONDITION_SPECS[cond].yoked
            ps.primes[cond] = Prime(condition=cond, text=text, failed=failed)
        return ps

    def generate_all(
        self, items: list[Item], conditions: list[str]
    ) -> list[PrimeSet]:
        out: list[PrimeSet] = []
        for it in tqdm(items, desc="generate_primes"):
            out.append(self.generate_for_item(it, conditions))
        return out


def yoke_random_paragraphs(prime_sets: list[PrimeSet], seed: int = 0) -> None:
    """For each item, fill `random_paragraph` prime with a length-matched neutral paragraph.

    Length is matched to whichever other prime that item already has, falling back to
    a default sentence count if none is present.
    """
    rng = random.Random(seed)
    for ps in prime_sets:
        if "random_paragraph" not in ps.primes:
            continue
        target_len = 0
        for cond, prime in ps.primes.items():
            if cond == "random_paragraph" or prime.text is None:
                continue
            target_len = max(target_len, len(prime.text.split()))
        if target_len == 0:
            target_len = 60
        # Sample and splice pool entries until we hit the target length.
        pool = list(_RANDOM_POOL)
        rng.shuffle(pool)
        words: list[str] = []
        for para in pool:
            for tok in para.split():
                words.append(tok)
                if len(words) >= target_len:
                    break
            if len(words) >= target_len:
                break
        ps.primes["random_paragraph"] = Prime(
            condition="random_paragraph",
            text=" ".join(words[:target_len]),
        )


def save_prime_sets(path: Path, prime_sets: list[PrimeSet]) -> None:
    with open(path, "w") as f:
        for ps in prime_sets:
            f.write(json.dumps(ps.to_dict()) + "\n")


def load_prime_sets(path: Path) -> list[PrimeSet]:
    out: list[PrimeSet] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(PrimeSet.from_dict(json.loads(line)))
    return out
