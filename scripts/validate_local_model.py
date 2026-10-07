from __future__ import annotations

import time

import torch

from app.agent.llm.local_model import TransformersLocalModel


print(f"torch={torch.__version__}", flush=True)
print(f"cuda={torch.cuda.is_available()}", flush=True)
started = time.perf_counter()
model = TransformersLocalModel(
    "models/base/arakandar-1.5b",
    device="cuda",
    max_input_tokens=2048,
)
print("loading model", flush=True)
answer = model.generate(
    [
        {"role": "system", "content": "You are Arakandar, a concise local market research assistant."},
        {
            "role": "user",
            "content": (
                "Analyze this evidence: ticker BBCA.JK, close 9875, RSI 54, "
                "signal HOLD, news sentiment +0.20. Explain the result in two sentences."
            ),
        },
    ],
    max_new_tokens=80,
    do_sample=False,
)
print(f"model=Arakandar latency_seconds={time.perf_counter() - started:.2f}", flush=True)
print(answer, flush=True)
print(f"vram_mb={torch.cuda.memory_allocated() / 1024 / 1024:.0f}", flush=True)
