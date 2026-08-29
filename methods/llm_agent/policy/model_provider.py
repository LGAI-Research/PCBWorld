"""Pluggable LLM/API providers for cadagent evaluation.

Encapsulates the two action-source backends that ``rollout/cadagent.py`` (batch
eval) and the interactive REPL share:

    VLLMProvider  — local vLLM engine (batched generation)
    APIProvider   — OpenAI / Anthropic / Google chat completions

Both expose the same minimal interface so callers can swap them by mode:

    provider = make_provider("llm" | "api", args)
    responses, token_counts = provider.generate(prompt_pairs)
    provider.close()

``token_counts`` is a list of ``(system_tokens, user_tokens, output_tokens)``
per prompt-pair. APIProvider estimates the system/user split proportionally
by character length (most chat APIs only return aggregate input tokens).
"""

from __future__ import annotations

import time

# The Agent contract and its type aliases live in methods._shared.agent; the
# providers below structurally satisfy Agent. PromptPair/TokenCounts are used
# in the provider method signatures.
from methods._shared.agent import PromptPair, TokenCounts


# Substrings that mark a transient (retryable) API failure. Matched against
# the exception's type name + message, case-insensitively, so it works across
# provider SDKs (OpenAI / Anthropic / Google / Together) without importing
# each one's exception classes. Permanent errors (bad request, auth, context
# length, model-not-found) are intentionally absent → they fall straight
# through to the idle fallback rather than wasting retries.
_TRANSIENT_API_MARKERS = (
    "rate limit", "ratelimit", "429",
    "timeout", "timed out",
    "connection", "connectionerror", "apiconnection",
    "overloaded", "503", "502", "500", "504",
    "service unavailable", "serviceunavailable",
    "temporarily", "try again", "internalservererror", "server error",
)


def _is_transient_api_error(exc: Exception) -> bool:
    """True if ``exc`` looks like a transient API error worth retrying."""
    blob = f"{type(exc).__name__} {exc}".lower()
    return any(m in blob for m in _TRANSIENT_API_MARKERS)


# ---------------------------------------------------------------------------
# vLLM provider
# ---------------------------------------------------------------------------

class VLLMProvider:
    """Local vLLM engine. Heavy import is deferred to construction so that
    callers in ``--mode manual`` / ``--mode api`` never load vLLM."""

    def __init__(self, args):
        from vllm import LLM
        from transformers import AutoTokenizer

        print("\n" + "=" * 60)
        print(f"  Loading model (vLLM): {args.model_path}")
        print("=" * 60)
        t0 = time.time()

        self.llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=args.enable_prefix_caching,
            enable_chunked_prefill=args.enable_chunked_prefill,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=args.max_model_len,
            enforce_eager=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True,
        )
        self.max_new_tokens = args.max_new_tokens
        self.temperature = args.temperature

        print(f"  Loaded in {time.time() - t0:.1f}s")
        print(
            f"  tensor_parallel_size={args.tensor_parallel_size}  "
            f"max_model_len={args.max_model_len}  "
            f"prefix_caching={args.enable_prefix_caching}  "
            f"chunked_prefill={args.enable_chunked_prefill}"
        )

    def generate(self, prompt_pairs, observations=None, **_kw):
        # ``observations`` / ``raw_tokens`` are ignored by the text-prompt
        # providers — only RLPolicyProvider reads them. Accept-and-drop
        # so worker.py can forward the kwarg uniformly.
        del observations, _kw
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

        prompts = []
        for system_prompt, user_prompt in prompt_pairs:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            prompts.append(text)

        outputs = self.llm.generate(prompts, sampling_params)

        responses: list[str] = []
        token_counts: list[TokenCounts] = []
        for idx, output in enumerate(outputs):
            system_prompt, user_prompt = prompt_pairs[idx]
            sys_tok = len(self.tokenizer.encode(system_prompt))
            usr_tok = len(self.tokenizer.encode(user_prompt))
            gen_text = output.outputs[0].text
            gen_len = len(output.outputs[0].token_ids)
            responses.append(gen_text)
            token_counts.append((sys_tok, usr_tok, gen_len))

        return responses, token_counts

    def close(self) -> None:
        # vLLM's LLM has no explicit close API; drop refs and let GC reclaim.
        self.llm = None
        self.tokenizer = None


# ---------------------------------------------------------------------------
# API provider
# ---------------------------------------------------------------------------

_DEFAULT_API_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "google": "gemini-2.0-flash",
}


def _call_openai(system_prompt, user_prompt, model, temperature, max_tokens,
                 api_key=None, base_url=None):
    from openai import OpenAI
    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        # Used by RemoteLLMProvider to point at vLLM serve / any OpenAI-
        # compatible HTTP endpoint. None preserves the default (api.openai.com).
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

    is_reasoning = any(model.startswith(p) for p in ("o1", "o3", "o4", "gpt-5"))
    token_param = (
        {"max_completion_tokens": max_tokens} if is_reasoning
        else {"max_tokens": max_tokens}
    )

    if is_reasoning:
        messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        extra = {}
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        extra = {"temperature": temperature}

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        **extra,
        **token_param,
    )
    choice = resp.choices[0]
    return choice.message.content, (resp.usage.prompt_tokens, resp.usage.completion_tokens)


def _call_anthropic(system_prompt, user_prompt, model, temperature, max_tokens, api_key=None):
    from anthropic import Anthropic
    client = Anthropic(**({"api_key": api_key} if api_key else {}))
    resp = client.messages.create(
        model=model,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = resp.content[0].text
    return text, (resp.usage.input_tokens, resp.usage.output_tokens)


def _call_google(system_prompt, user_prompt, model, temperature, max_tokens, api_key=None):
    from google import genai
    client = genai.Client(**({"api_key": api_key} if api_key else {}))
    resp = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    text = resp.text
    in_tok = resp.usage_metadata.prompt_token_count
    out_tok = resp.usage_metadata.candidates_token_count
    return text, (in_tok, out_tok)


def _call_together(system_prompt, user_prompt, model, temperature, max_tokens,
                   api_key=None, base_url=None):
    """Together AI — OpenAI-compatible chat completions.

    Defaults to disabling thinking mode (Qwen3-thinking, etc.) so the response
    arrives in ``message.content`` rather than the separate ``reasoning`` field
    that the parser doesn't read. Override via ``TOGETHER_ENABLE_THINKING=1``
    if you want chain-of-thought back in the bill.
    """
    import os
    from openai import OpenAI
    key = api_key or os.environ.get("TOGETHER_API_KEY")
    client = OpenAI(api_key=key, base_url=base_url or "https://api.together.xyz/v1")
    enable_think = os.environ.get("TOGETHER_ENABLE_THINKING", "0") == "1"
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user",   "content": user_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": enable_think}},
    )
    msg = resp.choices[0].message
    text = msg.content or ""
    # cadagent_projection requires a well-formed (non-empty) ``<think>...</think>``
    # block in front of the ``<action>...</action>``. With thinking disabled the
    # model returns just the action tag and the parser would fall back to idle
    # every step (see wrappers/action_converter.py:_has_valid_think), so
    # synthesise a non-empty think prefix when the model didn't emit one.
    # Reasoning models (thinking on) already include ``<think>...</think>`` and
    # pass through unchanged.
    if not enable_think and "<think>" not in text:
        text = "<think>(thinking disabled)</think>" + text
    return text, (resp.usage.prompt_tokens, resp.usage.completion_tokens)


_API_CALLERS = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "google": _call_google,
    "together": _call_together,
}


class APIProvider:
    """Provider-API rollout. One synchronous call per (system, user) pair."""

    def __init__(self, args):
        if args.api_provider not in _API_CALLERS:
            raise ValueError(
                f"unsupported api_provider: {args.api_provider!r} "
                f"(supported: {list(_API_CALLERS)})"
            )
        self.caller = _API_CALLERS[args.api_provider]
        self.model = args.api_model or _DEFAULT_API_MODELS[args.api_provider]
        self.temperature = args.temperature
        self.max_new_tokens = args.max_new_tokens
        self.api_key = args.api_key

    def generate(self, prompt_pairs, observations=None, **_kw):
        del observations, _kw  # see VLLMProvider.generate docstring
        responses: list[str] = []
        token_counts: list[TokenCounts] = []
        for system_prompt, user_prompt in prompt_pairs:
            max_retries = 6
            for attempt in range(max_retries):
                try:
                    text, counts = self.caller(
                        system_prompt, user_prompt, self.model,
                        self.temperature, self.max_new_tokens,
                        api_key=self.api_key,
                    )
                    break
                except Exception as e:  # noqa: BLE001
                    # Two retry classes:
                    #  - transient (rate limit / timeout / connection / 5xx):
                    #    full exponential backoff up to ``max_retries``.
                    #  - content-filter "flagged": ~deterministic for a given
                    #    prompt, and empirically the next rollout step recovers
                    #    on its own (flagged steps never cascade), so a single
                    #    quick retry is enough — don't burn the full backoff
                    #    budget re-sending an identically-flagged prompt.
                    # Anything else → idle fallback immediately.
                    is_flagged = "flagged" in str(e)
                    is_transient = _is_transient_api_error(e)
                    retry_cap = max_retries if is_transient else (2 if is_flagged else 1)
                    if attempt < retry_cap - 1 and (is_flagged or is_transient):
                        backoff = min(2.0 ** attempt, 30.0) if is_transient else 0.5
                        print(
                            f"  [WARN] API error ({type(e).__name__}: {e}); "
                            f"retry {attempt + 1}/{retry_cap} in {backoff:.1f}s"
                        )
                        time.sleep(backoff)
                        continue
                    print(f"  [WARN] API error: {e}. Using idle fallback.")
                    # No ``<action>`` tag → cadagent_projection returns its
                    # ``_FALLBACK_ACTION`` (action_type=6, idle) so the env
                    # applies a no-op step rather than an arbitrary
                    # ``net_select 1``. See wrappers/action_converter.py.
                    text, counts = (
                        "<think>API error — falling back to idle.</think>",
                        (0, 0),
                    )
                    break
            in_tok, out_tok = counts
            sys_chars = len(system_prompt)
            usr_chars = len(user_prompt)
            total_chars = sys_chars + usr_chars
            if total_chars > 0:
                sys_tok = int(in_tok * sys_chars / total_chars)
                usr_tok = in_tok - sys_tok
            else:
                sys_tok, usr_tok = 0, 0
            responses.append(text)
            token_counts.append((sys_tok, usr_tok, out_tok))

        return responses, token_counts

    def close(self) -> None:
        # No persistent connection to release.
        pass


# ---------------------------------------------------------------------------
# Remote LLM provider
# ---------------------------------------------------------------------------

class RemoteLLMProvider:
    """OpenAI-compatible HTTP provider.

    The server side is just ``vllm.entrypoints.openai.api_server`` (or
    any other OpenAI-shaped endpoint — TGI, llama.cpp server, an ollama
    + openai-adapter combo, ...). The visualizer never imports vLLM in
    this path; it only uses the ``openai`` SDK over HTTP.

    Selected when ``mode == "llm"`` AND ``args.remote == True``.

    ``model``: ``api_model`` wins if set; otherwise we fall back to
    ``args.model_path`` so a single value drives both local-vLLM and
    remote-vLLM paths.

    Authentication: the OpenAI SDK sends ``Authorization: Bearer
    <api_key>`` on every request. Pass an explicit ``--api_key``
    (hosted OpenAI, real tokens, vLLM serve --api-key), or
    ``--remote_no_auth`` ("EMPTY" placeholder for a vLLM serve
    without --api-key).
    """

    def __init__(self, args):
        from openai import OpenAI
        if not getattr(args, "remote_url", None):
            raise ValueError(
                "RemoteLLMProvider requires remote_url (e.g. "
                "'http://host:8000/v1'). Set --remote_url."
            )
        self.base_url = args.remote_url
        # Resolve model name. ``api_model`` (API-mode field) takes priority
        # over ``model_path`` (LLM-mode field) so a remote run can override
        # the served name without touching local-vLLM settings. Empty / None
        # is rejected outright — OpenAI's API requires an explicit model
        # name and silently sending ``""`` to vLLM serve yields a 404 with
        # an opaque error.
        api_model = getattr(args, "api_model", None)
        model_path = getattr(args, "model_path", None)
        model = (api_model or model_path or "").strip()
        if not model:
            raise ValueError(
                "RemoteLLMProvider requires a model name. Pass "
                "--model_path / --api_model — it must match the name the "
                "remote server is serving."
            )
        self.model = model
        # API key resolution, in priority order:
        #   1. Explicit ``args.api_key`` (user-provided — hosted OpenAI,
        #      TGI with token, vLLM serve --api-key, …).
        #   2. ``remote_no_auth=True`` opt-out → "EMPTY" (vLLM serve
        #      without --api-key flag accepts any non-empty key).
        explicit = getattr(args, "api_key", None)
        if explicit:
            self.api_key = explicit
        elif bool(getattr(args, "remote_no_auth", False)):
            self.api_key = "EMPTY"
        else:
            raise ValueError(
                "RemoteLLMProvider: no API key. Either set --api_key "
                "explicitly, or pass --remote_no_auth for an "
                "unauthenticated vLLM serve."
            )
        self.temperature = float(getattr(args, "temperature", 0.7))
        self.max_new_tokens = int(getattr(args, "max_new_tokens", 256))
        # Construct an OpenAI client once so connection pooling is reused
        # across calls. The actual chat completion is dispatched via
        # ``_call_openai`` to share the reasoning-model / token-accounting
        # branches with APIProvider.
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        print("\n" + "=" * 60)
        print(f"  Remote LLM endpoint: {self.base_url}  model={self.model!r}")
        print("=" * 60)

    def generate(self, prompt_pairs, observations=None, **_kw):
        del observations, _kw  # ignored — text-prompt provider
        responses: list[str] = []
        token_counts: list[TokenCounts] = []
        for system_prompt, user_prompt in prompt_pairs:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    text, counts = _call_openai(
                        system_prompt, user_prompt, self.model,
                        self.temperature, self.max_new_tokens,
                        api_key=self.api_key,
                        base_url=self.base_url,
                    )
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt < max_retries - 1 and "flagged" in str(e):
                        print(
                            f"  [WARN] content filter triggered, "
                            f"retrying ({attempt + 1}/{max_retries})..."
                        )
                        time.sleep(1)
                        continue
                    print(f"  [WARN] remote LLM error: {e}. "
                          f"Using idle fallback.")
                    text, counts = (
                        "<think>remote LLM error — falling back to idle.</think>",
                        (0, 0),
                    )
                    break
            in_tok, out_tok = counts
            sys_chars = len(system_prompt)
            usr_chars = len(user_prompt)
            total_chars = sys_chars + usr_chars
            if total_chars > 0:
                sys_tok = int(in_tok * sys_chars / total_chars)
                usr_tok = in_tok - sys_tok
            else:
                sys_tok, usr_tok = 0, 0
            responses.append(text)
            token_counts.append((sys_tok, usr_tok, out_tok))
        return responses, token_counts

    def close(self) -> None:
        # ``OpenAI`` clients are stateless wrappers around httpx — drop ref.
        self._client = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
#
# Intentional split:
#   * LLM-flavored providers (this file)  : VLLMProvider, RemoteLLMProvider,
#                                           APIProvider
#   * RL policy provider                  : methods/rl_agent/policy/rl_provider.py
#
# This file therefore exports only the LLM-side providers + the shared
# ``Provider`` / ``PromptPair`` / ``TokenCounts`` interface types.


def make_provider(mode: str, args) -> Provider:
    """Build the LLM-side provider for ``mode`` ('llm' or 'api').

    Calling this with ``'rl'`` is rejected so RL imports never leak
    into LLM-only call sites.
    """
    if mode == "llm":
        if getattr(args, "remote", False):
            return RemoteLLMProvider(args)
        return VLLMProvider(args)
    if mode == "api":
        return APIProvider(args)
    if mode == "rl":
        raise ValueError(
            "mode 'rl' is not handled here; instantiate "
            "methods.rl_agent.policy.rl_provider.RLPolicyProvider directly."
        )
    raise ValueError(
        f"unsupported provider mode: {mode!r} (expected 'llm' or 'api')"
    )
