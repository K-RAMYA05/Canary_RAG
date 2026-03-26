# from __future__ import annotations

# from dataclasses import dataclass

# try:
#     from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#     _TRANSFORMERS_AVAILABLE = True
# except ImportError:
#     _TRANSFORMERS_AVAILABLE = False


# @dataclass
# class LLMGenerator:
#     model_name: str
#     max_new_tokens: int = 256
#     temperature: float = 0.7

#     def __post_init__(self) -> None:
#         if _TRANSFORMERS_AVAILABLE:
#             tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#             model = AutoModelForCausalLM.from_pretrained(self.model_name)
#             self._pipe = pipeline(
#                 "text-generation",
#                 model=model,
#                 tokenizer=tokenizer,
#             )
#         else:
#             self._pipe = None

#     def generate(self, prompt: str) -> str:
#         if self._pipe is None:
#             # Fallback: return the prompt tail as a placeholder answer.
#             return (
#                 "LLM backend (transformers) is not installed; "
#                 "returning prompt tail as a placeholder answer.\n\n"
#                 f"{prompt[-1000:]}"
#             )
#         out = self._pipe(
#             prompt,
#             num_return_sequences=1,
#             max_new_tokens=self.max_new_tokens,
#             temperature=self.temperature,
#             do_sample=True,
#             top_p=0.9,
#             top_k=50,
#             repetition_penalty=1.15,
#         )
#         return out[0]["generated_text"]

from dataclasses import dataclass

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


@dataclass
class LLMGenerator:
    model_name: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = False
    top_p: float = 1.0
    top_k: int = 0

    def __post_init__(self) -> None:
        if _TRANSFORMERS_AVAILABLE:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self._tokenizer = tokenizer
            self._pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
        else:
            self._tokenizer = None
            self._pipe = None

    def generate(self, prompt: str) -> str:
        if self._pipe is None:
            return (
                "LLM backend (transformers) is not installed; "
                "returning prompt tail as a placeholder answer.\n\n"
                f"{prompt[-1000:]}"
            )

        out = self._pipe(
            prompt,
            num_return_sequences=1,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.do_sample,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=1.05,
            truncation=True,
            return_full_text=False,
            pad_token_id=self._tokenizer.eos_token_id if self._tokenizer is not None else None,
        )
        return out[0]["generated_text"].strip()