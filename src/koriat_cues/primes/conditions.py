"""The six Koriat cue-manipulation conditions + random-paragraph yoked control.

`assemble_prompt` composes the user turn given a question and (optional) prime.
Conditions vary only in how the prime is constructed; the assembly is shared.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    description: str
    # One-line instruction used by the prime-generator LLM.
    generator_instruction: str
    # If True, the condition is realized by rephrasing the question rather than adding a prime.
    rephrases_question: bool = False
    # If True, this is a yoked/random control; prime text is not model-generated.
    yoked: bool = False


CONDITION_SPECS: dict[str, ConditionSpec] = {
    "baseline": ConditionSpec(
        name="baseline",
        description="Unmodified question.",
        generator_instruction="",
    ),
    "cue_familiarity_priming": ConditionSpec(
        name="cue_familiarity_priming",
        description=(
            "Short passage that repeats surface features of the question (named entities, key "
            "terms) but carries no information that helps answer it. Schwartz & Metcalfe 1992."
        ),
        generator_instruction=(
            "Write a 2-3 sentence passage that mentions the key entities / terms in the "
            "question but contains NO information that helps answer it. The passage must not "
            "name or describe the target answer, directly or by paraphrase. Stay factual and "
            "generic. Do not ask questions. Return only the passage."
        ),
    ),
    "target_priming": ConditionSpec(
        name="target_priming",
        description=(
            "Passage that mentions the correct answer in an unrelated context without "
            "flagging its relevance. Should raise accuracy but not familiarity of the cue."
        ),
        generator_instruction=(
            "Write a 2-3 sentence passage that mentions the correct answer `{answer}` in an "
            "unrelated context, without flagging that it answers any question. Do not repeat "
            "the question's surface features. Return only the passage."
        ),
    ),
    "partial_accessibility": ConditionSpec(
        name="partial_accessibility",
        description=(
            "Plausible but wrong partial information, to induce Koriat accessibility without "
            "helping accuracy."
        ),
        generator_instruction=(
            "Write a 2-3 sentence passage that presents plausible-sounding information "
            "related to the question's topic but that does NOT support the correct answer "
            "`{answer}`. It may cite incorrect or tangential facts that sound confident and "
            "authoritative. Do not name the correct answer. Return only the passage."
        ),
    ),
    "illusory_tot": ConditionSpec(
        name="illusory_tot",
        description=(
            "Semantically related question content that doesn't match the target (Schwartz 1998)."
        ),
        generator_instruction=(
            "Write a 2-3 sentence passage on a topic semantically adjacent to the question "
            "(e.g., same era, same field, same category of entity) that does not answer the "
            "question and does not mention `{answer}`. Keep it dense with confident-sounding "
            "proper nouns. Return only the passage."
        ),
    ),
    "fluency_degradation": ConditionSpec(
        name="fluency_degradation",
        description=(
            "Rephrase the question with rare synonyms and awkward but grammatical syntax, "
            "holding semantic content constant."
        ),
        generator_instruction=(
            "Rephrase the following question using uncommon synonyms and awkward (but "
            "grammatical) syntax. Preserve the exact semantic content. Do not add or remove "
            "information. Return only the rephrased question."
        ),
        rephrases_question=True,
    ),
    "random_paragraph": ConditionSpec(
        name="random_paragraph",
        description="Yoked length-matched random paragraph (control).",
        generator_instruction="",
        yoked=True,
    ),
}


SYSTEM_PROMPT = (
    "You are a careful trivia assistant. When asked a question, respond with the answer on "
    "a single line, followed by a newline. Be concise: name the entity only, no explanation."
)


def assemble_prompt(
    question: str,
    prime: Optional[str],
    condition: str,
    order: str = "prime_then_question",
) -> list[dict]:
    """Assemble chat messages for a trial.

    `order` ∈ {"prime_then_question", "question_then_prime"}; the second is for the
    order-effects counterbalance control in SPEC §Controls.
    """
    spec = CONDITION_SPECS[condition]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if spec.rephrases_question:
        # Prime *is* the rephrased question; fall back to original if generator failed.
        user_text = (prime or question).strip()
        messages.append({"role": "user", "content": f"Question: {user_text}\nAnswer:"})
        return messages

    if prime is None or condition == "baseline":
        messages.append({"role": "user", "content": f"Question: {question}\nAnswer:"})
        return messages

    if order == "question_then_prime":
        body = f"Question: {question}\n\nContext: {prime}\nAnswer:"
    else:
        body = f"Context: {prime}\n\nQuestion: {question}\nAnswer:"
    messages.append({"role": "user", "content": body})
    return messages
