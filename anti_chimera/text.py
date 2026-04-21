from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

TOKEN_RE = re.compile(r"[a-zA-Z']+")


@dataclass
class ParsedPrompt:
    tokens: List[str]
    entities: List[str]


class PromptParser:
    def __init__(self) -> None:
        self.stopwords = {
            'a', 'an', 'the', 'and', 'on', 'in', 'with', 'near', 'at', 'of', 'two', 'three'
        }
        self.entity_heads = {'circle', 'square', 'triangle', 'object', 'cat', 'dog', 'person'}

    def parse(self, prompt: str) -> ParsedPrompt:
        words = [w.lower() for w in TOKEN_RE.findall(prompt)]
        entities: List[str] = []
        current: List[str] = []
        for word in words:
            if word in self.stopwords:
                continue
            current.append(word)
            if word in self.entity_heads:
                entities.append(' '.join(current[-2:]) if len(current) >= 2 else word)
        if not entities:
            entities = ['unknown object']
        return ParsedPrompt(tokens=words, entities=entities)


class HFTextEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, freeze: bool = True) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        if freeze:
            self.model.requires_grad_(False)
            self.model.eval()

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def forward(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=min(getattr(self.tokenizer, 'model_max_length', 77), 77),
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = self.model(**encoded)
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        if isinstance(outputs, tuple) and len(outputs) > 0:
            return outputs[0]
        raise RuntimeError('Unsupported transformer text encoder output.')
