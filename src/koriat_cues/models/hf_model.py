"""HuggingFace model wrapper supporting generation + hidden-state capture.

Designed to extract, for a single forward/generation:
  - the generated answer string,
  - the log-probability of the first generated token,
  - the hidden state at the "post-answer-newline" position at a configurable layer
    (Kumaran et al. finding: confidence is cached at that position).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from ..config import ModelConfig


@dataclass
class GenerationResult:
    text: str
    # Log-prob of the first generated token conditional on the prompt.
    first_token_logprob: float
    # Hidden state read at the post-answer-newline position at layer `caa_layer`.
    post_newline_hidden: torch.Tensor | None
    # Hidden state at the same position at EVERY layer (incl. embedding). Shape:
    # (n_layers+1, hidden_dim). Only populated when `capture_all_layers=True` is
    # passed to `generate()`. Used by the rigorous CAA-vector build script.
    post_newline_hidden_all_layers: torch.Tensor | None = None
    # Index of the newline token within the generated sequence (for debugging).
    newline_step: int | None = None


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(name, torch.bfloat16)


class HFModel:
    """Thin wrapper around AutoModelForCausalLM with chat-template support."""

    def __init__(self, cfg: ModelConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        load_kwargs: dict = {"device_map": cfg.device_map}
        if cfg.dtype in ("int8", "int4"):
            # Quantized load via bitsandbytes. Requires the `bnb` extra.
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=(cfg.dtype == "int8"),
                load_in_4bit=(cfg.dtype == "int4"),
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            load_kwargs["torch_dtype"] = _dtype(cfg.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.hf_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(cfg.hf_id, **load_kwargs)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.caa_layer = cfg.caa_layer

    def format_chat(self, messages: list[dict]) -> str:
        """Render `messages` via the tokenizer's chat template.

        `messages` is a list of {"role": "system"|"user"|"assistant", "content": str}.
        The returned string ends where the assistant's next turn would begin.
        """
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        do_sample: bool = False,
        temperature: float = 1.0,
        capture_post_newline: bool = True,
        capture_all_layers: bool = False,
    ) -> GenerationResult:
        """Generate a short completion, capturing first-token log-prob and an optional
        hidden-state readout at the post-answer-newline position.
        """
        max_new = max_new_tokens or self.cfg.max_new_tokens
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids: torch.Tensor = enc["input_ids"]
        prompt_len = input_ids.shape[1]

        out = self.model.generate(
            **enc,
            max_new_tokens=max_new,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=(capture_post_newline or capture_all_layers),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated_ids = out.sequences[0, prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Log-prob of the first generated token.
        first_logits = out.scores[0][0]  # (vocab,)
        first_logprobs = F.log_softmax(first_logits, dim=-1)
        first_tok_id = generated_ids[0].item() if generated_ids.numel() > 0 else -1
        first_token_logprob = (
            float(first_logprobs[first_tok_id].item()) if first_tok_id >= 0 else float("nan")
        )

        post_newline_hidden: torch.Tensor | None = None
        post_newline_hidden_all_layers: torch.Tensor | None = None
        newline_step: int | None = None

        want_hs = capture_post_newline or capture_all_layers
        if want_hs and generated_ids.numel() > 0:
            # out.hidden_states: tuple(len=n_generated_steps) of tuples(layers+1) of
            # tensors. Step 0 contains the hidden states up through the prompt + first
            # generated token; subsequent steps have only one token at seq_pos -1.
            newline_step = self._find_post_answer_newline_step(generated_ids)
            if newline_step is not None:
                step_hs = out.hidden_states[newline_step]
                if capture_post_newline:
                    layer_idx = self._resolve_layer_index()
                    h = step_hs[layer_idx]
                    post_newline_hidden = h[0, -1, :].detach().float().cpu()
                if capture_all_layers:
                    # (n_layers+1, hidden_dim) stack of last-position hidden states.
                    post_newline_hidden_all_layers = torch.stack(
                        [h[0, -1, :].detach().float().cpu() for h in step_hs]
                    )

        return GenerationResult(
            text=text,
            first_token_logprob=first_token_logprob,
            post_newline_hidden=post_newline_hidden,
            post_newline_hidden_all_layers=post_newline_hidden_all_layers,
            newline_step=newline_step,
        )

    def _resolve_layer_index(self) -> int:
        """`hidden_states` includes the embedding layer at index 0, so total length is
        n_layers + 1. Negative indexing counts from the end of that tuple.
        """
        n_hs = self.model.config.num_hidden_layers + 1
        if self.caa_layer < 0:
            return n_hs + self.caa_layer
        return min(self.caa_layer, n_hs - 1)

    def _find_post_answer_newline_step(self, generated_ids: torch.Tensor) -> int | None:
        """Return the generation step index at which the model emits the newline that
        terminates the answer (a one-liner per the system prompt).

        Heuristic:
          1. Find all newline-bearing steps after the first non-whitespace token.
          2. Prefer a newline that either (a) is followed by EOS / no more tokens,
             or (b) is followed by punctuation / another newline. A newline followed
             by alphabetic content is likely mid-answer ("I think\nthe answer is …")
             and is skipped in favor of a later one.
          3. Fallback: the LAST newline seen; failing that, the last generated step.
        """
        tokens = generated_ids.tolist()
        if not tokens:
            return None
        decoded_steps = [self.tokenizer.decode([t]) for t in tokens]
        seen_nonspace = False
        newline_steps: list[int] = []
        for step, s in enumerate(decoded_steps):
            if "\n" in s and seen_nonspace:
                newline_steps.append(step)
            if s.strip():
                seen_nonspace = True
        if not newline_steps:
            return len(tokens) - 1
        # Prefer a newline whose following token doesn't start with a letter.
        for step in newline_steps:
            if step == len(tokens) - 1:
                return step
            next_decoded = decoded_steps[step + 1].lstrip()
            if not next_decoded or not next_decoded[0].isalpha():
                return step
        # All newlines have letters after them → fall back to the last newline.
        return newline_steps[-1]

    @torch.no_grad()
    def score_next_token_words(
        self, prompt: str, words: Sequence[str]
    ) -> dict[str, float]:
        """Return the log-probability of each `word` as the very next token.

        For each word, both the leading-space and no-leading-space tokenizations are
        tried and the higher-probability single-token variant is reported (chat
        templates can leave the next token un-prefixed by a space, while many
        natural completions are space-prefixed). Multi-token words are skipped — the
        Kadavath P(True) elicitation is single-token by design.
        """
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model(**enc, return_dict=True)
        last_logits = out.logits[0, -1, :]
        log_softmax = F.log_softmax(last_logits, dim=-1)
        result: dict[str, float] = {}
        for w in words:
            best_lp = float("-inf")
            for variant in (w, " " + w):
                ids = self.tokenizer.encode(variant, add_special_tokens=False)
                if len(ids) == 1:
                    lp = float(log_softmax[ids[0]].item())
                    if lp > best_lp:
                        best_lp = lp
            result[w] = best_lp if best_lp != float("-inf") else float("nan")
        return result

    @torch.no_grad()
    def hidden_states_at(
        self, prompt: str, positions: Sequence[int], layers: Sequence[int]
    ) -> dict[tuple[int, int], torch.Tensor]:
        """Run a forward pass on `prompt` and return hidden states at (layer, position).

        `positions` are token indices (negative = from end). `layers` are layer indices
        into the hidden_states tuple (embedding layer is 0).
        """
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model(**enc, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states  # tuple of (n_layers+1) tensors, each (1, seq, hidden)
        seq_len = enc["input_ids"].shape[1]
        result: dict[tuple[int, int], torch.Tensor] = {}
        for lyr in layers:
            layer_hs = hs[lyr]
            for pos in positions:
                idx = pos if pos >= 0 else seq_len + pos
                result[(lyr, pos)] = layer_hs[0, idx, :].detach().float().cpu()
        return result
