"""Experiment configuration loaded from YAML."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


ConditionName = Literal[
    "baseline",
    "cue_familiarity_priming",
    "target_priming",
    "partial_accessibility",
    "illusory_tot",
    "fluency_degradation",
    "random_paragraph",
]

MeasureName = Literal["log_prob", "verbal_cat", "verbal_num", "caa_proj"]


class ModelConfig(BaseModel):
    name: str
    hf_id: str
    dtype: Literal["float16", "bfloat16", "float32", "int8", "int4"] = "bfloat16"
    device_map: str = "auto"
    max_new_tokens: int = 32
    # Which transformer block layer to read for CAA; negative indexes from the end.
    caa_layer: int = -12


class DataConfig(BaseModel):
    dataset: Literal["triviaqa", "natural_questions", "sciq"] = "triviaqa"
    split: str = "validation"
    n_items: int = 500
    # Oversample factor. ~12% of TriviaQA items pass the [0.2, 0.8] accuracy band on
    # Llama-3.1-8B, so 10× is a reasonable default for the full MVE.
    oversample: int = 10
    single_entity_only: bool = True
    baseline_acc_min: float = 0.2
    baseline_acc_max: float = 0.8
    baseline_acc_n_samples: int = 5
    seed: int = 0


class PrimeConfig(BaseModel):
    # Model used to generate primes.
    generator_model: str = "claude-opus-4-7"
    # Model used as external judge for leak detection.
    judge_model: str = "claude-opus-4-7"
    # Filter items whose cue_familiarity prime leaks the answer >= this rate.
    max_leak_rate: float = 0.05
    # Max prime length in sentences, used as a soft target during generation.
    max_sentences: int = 3
    # If true, include order-counterbalance (prime after question) control.
    counterbalance_order: bool = False


class CAAConfig(BaseModel):
    n_contrast_pairs: int = 200
    # Token position where we read the steering activation. "post_newline" uses
    # the newline immediately following the model's answer, per Kumaran et al.
    read_position: Literal["post_newline", "last_token"] = "post_newline"


class ExperimentConfig(BaseModel):
    run_name: str
    seed: int = 0
    output_dir: Path = Path("runs")
    models: list[ModelConfig]
    data: DataConfig
    primes: PrimeConfig
    caa: CAAConfig
    conditions: list[ConditionName]
    confidence_measures: list[MeasureName]

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = ExperimentConfig.model_validate(raw)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    return cfg
