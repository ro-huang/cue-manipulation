This is a partial transcript of a conversation between a user and an AI assistant. Intermediate AI messages have been elided.

## User

Can you talk to me about how we vary prompts for the cue familiarity experiment? What do we do; if we use agents, how capable are the agents we use?

## Assistant

Here's the picture:

## How prompt variation works

Each item (a TriviaQA question with a gold answer) gets a **prime** generated for each cue-manipulation condition. The prime is then prepended (or, under counterbalance, appended) to the question in the user turn, with a fixed system prompt (`src/koriat_cues/primes/conditions.py:112`). The conditions are realized as different *instructions to a generator LLM* — not different code paths — except for the two yoked controls (`random_paragraph`, `whitespace`), which are length-matched filler with no model in the loop.

The six manipulations live as `ConditionSpec` records in `src/koriat_cues/primes/conditions.py:24`. Each carries a one-line `generator_instruction`, e.g.:
- **cue_familiarity_priming** — "mentions the key entities/terms in the question but contains NO information that helps answer it"
- **target_priming** — "mentions the correct answer `{answer}` in an unrelated context"
- **partial_accessibility** — "plausible-sounding but does NOT support the correct answer"
- **illusory_tot** — "semantically adjacent topic, dense with confident proper nouns, doesn't mention `{answer}`"
- **fluency_degradation** — rephrases the question itself with rare synonyms / awkward syntax (special-cased: replaces the question rather than prefixing it)

`assemble_prompt` (`conditions.py:118`) is shared across conditions, so the only thing that varies between trials of the same item is the prime text. The CAA experiment tests *the small Llama model under test* on these assembled prompts; the generator is a separate, stronger model.

## How "agentic" is the generator

Not very — it's a single, stateless Claude API call per (item, condition):

- `PrimeGenerator._one_call` (`generator.py:107`) is one `messages.create` with `max_tokens=400`, no tools, no multi-turn, no self-critique.
- Model is `claude-opus-4-7` (configurable in `configs/mve.yaml:26`).
- The system prompt frames it as a research assistant; the user turn is `instruction + question + answer + length cap`.
- Wrapped in a tenacity retry (4 attempts, exp backoff). Output is lightly post-processed (strip "Passage:" prefixes, quotes).

So "agent" overstates it — it's a one-shot prompted strong LLM, not a tool-using or planning agent.

## The one place that's more agent-like: validation

`PrimeJudge` in `src/koriat_cues/primes/validator.py` does a leak check: for the conditions that *claim* not to reveal the answer (`cue_familiarity_priming`, `partial_accessibility`, `illusory_tot`), it asks a separate Claude (also `claude-opus-4-7`) to try to answer the question *given only the prime*, three times per item. If it succeeds more than `max_leak_rate` (5% in the MVE), the item is dropped. This is still a one-shot call per attempt, just used as an external judge rather than a generator. It's the closest thing to a second "agent" in the loop, and its job is purely defensive — keeping the cue-familiarity condition from accidentally turning into target-priming.