"""Shared Groq LLM client and chat helpers used across the agent."""
import os
import re
import uuid
from types import SimpleNamespace

from groq import Groq, GroqError

DEFAULT_MODEL = "llama-3.3-70b-versatile"
WHISPER_MODEL = "whisper-large-v3"

# Llama on Groq occasionally emits tool calls as plain text in the form
# `<function=name{json args}</function>` instead of the structured tool_calls
# field, which makes Groq return a 400 `tool_use_failed`. We recover by parsing
# the intended call(s) out of the failed generation and replaying them.
_FUNCTION_CALL_RE = re.compile(r"<function=([a-zA-Z_]\w*)>?\s*(\{.*?\})\s*(?:</function>)?", re.DOTALL)

_client = None


class LLMError(Exception):
    """Raised when a Groq API call fails or the API key is missing."""


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise LLMError("GROQ_API_KEY ist nicht gesetzt.")
        _client = Groq(api_key=api_key)
    return _client


def chat(messages: list, model: str = DEFAULT_MODEL, temperature: float = 0.7, json_mode: bool = False) -> str:
    """Send a chat completion request to Groq and return the response text.

    Raises LLMError if the API key is missing or the request fails.
    """
    client = get_client()

    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
    except GroqError as e:
        raise LLMError(f"Groq API Fehler: {e}") from e

    return response.choices[0].message.content


def _build_tool_call(name: str, arguments: str):
    """Build a tool_call shaped like the SDK's, so callers can treat it uniformly."""
    return SimpleNamespace(
        id=f"call_{uuid.uuid4().hex[:12]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _recover_tool_calls_from_error(error: GroqError):
    """If Groq rejected a text-formatted tool call, parse the intended call(s) out of it.

    Returns a synthetic message (with .content and .tool_calls) or None if nothing recoverable.
    """
    failed_generation = None
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            failed_generation = err.get("failed_generation")

    # Fall back to scanning the raw error text for the distinctive <function=...> pattern.
    haystack = failed_generation or str(error)
    calls = [_build_tool_call(m.group(1), m.group(2)) for m in _FUNCTION_CALL_RE.finditer(haystack)]
    if not calls:
        return None
    return SimpleNamespace(content="", tool_calls=calls)


def chat_with_tools(messages: list, tools: list, model: str = DEFAULT_MODEL, temperature: float = 0.4):
    """Send a chat completion request with tool definitions and return the raw message.

    The returned message may carry `.tool_calls`; the caller is responsible for
    executing them and continuing the conversation. Raises LLMError on failure.
    """
    client = get_client()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice="auto",
        )
    except GroqError as e:
        recovered = _recover_tool_calls_from_error(e)
        if recovered is not None:
            return recovered
        raise LLMError(f"Groq API Fehler: {e}") from e

    return response.choices[0].message


def transcribe_audio(file_path: str, language: str = "de") -> str:
    """Transcribe an audio file with Groq Whisper. Raises LLMError on failure."""
    client = get_client()

    try:
        with open(file_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=f,
                language=language,
            )
    except GroqError as e:
        raise LLMError(f"Groq Transkriptions-Fehler: {e}") from e
    except OSError as e:
        raise LLMError(f"Audiodatei konnte nicht gelesen werden: {e}") from e

    return transcription.text
