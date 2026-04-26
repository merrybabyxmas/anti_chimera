from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

try:  # optional dependency for CogVideoX-backed runs
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - optional import
    AutoModel = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]

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


class SimplePromptEncoder(nn.Module):
    """A small, self-contained prompt encoder for the lite backend.

    It uses stable hashing over whitespace/regex tokens so that prompts map to the
    same ids across runs without requiring any external tokenizer files.
    """

    def __init__(self, vocab_size: int, hidden_size: int, max_tokens: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        if vocab_size < 8:
            raise ValueError('vocab_size must be at least 8')
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_tokens = max_tokens
        self.embedding = nn.Embedding(vocab_size + 2, hidden_size, padding_idx=0)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.parser = PromptParser()

    @property
    def hidden_dim(self) -> int:
        return self.hidden_size

    def _token_to_id(self, token: str) -> int:
        digest = hashlib.sha1(token.encode('utf-8')).digest()
        value = int.from_bytes(digest[:8], 'big', signed=False)
        return 2 + (value % self.vocab_size)

    def _encode_prompt(self, prompt: str) -> List[int]:
        tokens = self.parser.parse(prompt).tokens
        if not tokens:
            return []
        return [self._token_to_id(tok) for tok in tokens[: self.max_tokens]]

    def forward(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        token_ids: List[List[int]] = [self._encode_prompt(prompt) for prompt in prompts]
        max_len = max((len(ids) for ids in token_ids), default=0)
        if max_len == 0:
            return torch.zeros(len(prompts), self.hidden_size, device=device)

        max_len = min(max_len, self.max_tokens)
        batch = torch.zeros(len(prompts), max_len, dtype=torch.long, device=device)
        mask = torch.zeros(len(prompts), max_len, dtype=torch.float32, device=device)
        for i, ids in enumerate(token_ids):
            if not ids:
                continue
            ids = ids[:max_len]
            batch[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            mask[i, : len(ids)] = 1.0

        embedded = self.embedding(batch)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (embedded * mask[..., None]).sum(dim=1) / denom
        return self.proj(pooled)


class HFTextEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, freeze: bool = True) -> None:
        super().__init__()
        if AutoModel is None or AutoTokenizer is None:
            raise ImportError(
                'transformers is required for HFTextEncoder. Install transformers or use SimplePromptEncoder.'
            )
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
