"""Local-only language-model adapters.

The optional Transformers dependency is imported only when inference is requested,
so the market workflow remains usable without model weights or GPU libraries.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


class IncompleteGenerationError(RuntimeError):
    """Raised when local generation exhausts its token budget without ending."""


class PromptTooLongError(RuntimeError):
    """Raised when the current question and evidence exceed the input budget."""


class LocalLanguageModel(Protocol):
    """Small interface used by the agent's grounded report layer."""

    model_name: str

    def generate(self, messages: Sequence[Mapping[str, str]], **generation: Any) -> str:
        ...


@dataclass
class TransformersLocalModel:
    """Lazy Transformers text generator backed by local model files/cache."""

    model_id: str
    device: str = "auto"
    max_input_tokens: int = 4096
    model_name: str = "Arakandar"
    torch_dtype: str = "auto"

    def __post_init__(self) -> None:
        self._tokenizer = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers.utils.logging import disable_progress_bar
        except ImportError as exc:
            raise RuntimeError(
                "Local Transformers support requires optional torch and transformers packages"
            ) from exc

        model_source = self.model_id
        adapter_config_path = Path(self.model_id) / "adapter_config.json"
        adapter_config: dict[str, Any] | None = None
        if adapter_config_path.is_file():
            adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            model_source = str(adapter_config["base_model_name_or_path"])
        disable_progress_bar()
        self._tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
        load_kwargs: dict[str, Any] = {}
        if self.device in {"auto", "cuda"}:
            load_kwargs["device_map"] = "auto"
        if self.torch_dtype == "auto" and self.device in {"auto", "cuda"}:
            load_kwargs["dtype"] = torch.float16
        elif self.torch_dtype == "float16":
            load_kwargs["dtype"] = torch.float16
        elif self.torch_dtype == "bfloat16":
            load_kwargs["dtype"] = torch.bfloat16
        self._model = AutoModelForCausalLM.from_pretrained(model_source, local_files_only=True, **load_kwargs)
        if adapter_config is not None:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError(
                    "This model path is a LoRA adapter; install peft to load it"
                ) from exc
            self._model = PeftModel.from_pretrained(self._model, self.model_id, is_trainable=False)
        if self.device not in {"auto", "cuda"}:
            self._model.to(self.device)

    def generate(self, messages: Sequence[Mapping[str, str]], **generation: Any) -> str:
        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        import torch

        prompt_messages = list(messages)
        while True:
            prompt = self._tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._tokenizer(prompt, return_tensors="pt", truncation=False)
            if inputs["input_ids"].shape[1] <= self.max_input_tokens:
                break
            if len(prompt_messages) > 2:
                del prompt_messages[1]
                continue
            raise PromptTooLongError(
                f"Current question and evidence exceed the {self.max_input_tokens}-token input limit"
            )
        model_device = next(self._model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        generation.setdefault("pad_token_id", self._tokenizer.eos_token_id)
        chunk_limit = int(generation.pop("max_new_tokens", 768))
        total_limit = int(generation.pop("max_total_new_tokens", chunk_limit * 2))
        eos_token_id = generation.get("eos_token_id")
        if eos_token_id is None:
            eos_token_id = self._tokenizer.eos_token_id
        if eos_token_id is None:
            eos_token_id = getattr(self._model.generation_config, "eos_token_id", None)
        eos_token_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id])
        generated_tokens = []
        current_inputs = inputs

        while len(generated_tokens) < total_limit:
            requested = min(chunk_limit, total_limit - len(generated_tokens))
            output = self._model.generate(
                **current_inputs,
                **generation,
                max_new_tokens=requested,
            )
            new_tokens = output[0][current_inputs["input_ids"].shape[1] :]
            generated_tokens.extend(new_tokens.tolist())
            if any(token in eos_token_ids for token in new_tokens.tolist()):
                break
            if len(new_tokens) < requested:
                raise IncompleteGenerationError(
                    "Local model stopped before producing an end-of-sequence token"
                )
            if len(generated_tokens) >= total_limit:
                raise IncompleteGenerationError(
                    f"Local model reached its {total_limit}-token answer limit without ending"
                )

            current_inputs = {"input_ids": output}
            current_inputs["attention_mask"] = torch.ones_like(output)

        generated = torch.tensor(generated_tokens, device=model_device)
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()