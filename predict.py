import json
import os
from typing import List

import torch
from cog import BasePredictor, Input, Path
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/santacoder"
device = "cuda"  # for GPU usage or "cpu" for CPU usage


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir="cache")
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint, trust_remote_code=True, cache_dir="cache"
        ).to(device)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
        ),
        max_new_tokens: int = Input(description="max tokens to generate", default=20),
        temperature: float = Input(description="0.1 to 2.5 temperature", default=0.2),
    ) -> str:

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(
            inputs, max_new_tokens=max_new_tokens, temperature=temperature
        )
        output = self.tokenizer.decode(outputs[0])
        print(output)

        return output
