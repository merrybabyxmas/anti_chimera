from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

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
        self.entity_heads = {'circle', 'square', 'triangle', 'object'}

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


class TokenTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def _hash_token(self, token: str) -> int:
        return abs(hash(token)) % self.vocab_size

    def forward(self, prompts: List[str]) -> torch.Tensor:
        device = self.embedding.weight.device
        batch_embeddings = []
        for prompt in prompts:
            tokens = [t.lower() for t in TOKEN_RE.findall(prompt)]
            if not tokens:
                tokens = ['empty']
            ids = torch.tensor([self._hash_token(t) for t in tokens], device=device)
            emb = self.embedding(ids).mean(dim=0)
            batch_embeddings.append(emb)
        return self.proj(torch.stack(batch_embeddings, dim=0))
