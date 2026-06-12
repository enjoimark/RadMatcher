"""
Provider-agnostic LLM code suggestion for low-confidence matches.

The matcher works fully offline; this module is an *optional* fallback that
asks a large language model to pick the best code when the local match is
weak. It supports several backends behind one ``suggest()`` call:

    provider = "openai"             -> OpenAI Chat Completions API
    provider = "anthropic"          -> Anthropic (Claude) Messages API
    provider = "gemini"             -> Google Gemini generateContent API
    provider = "openai_compatible"  -> any OpenAI-compatible server, e.g.
                                       Ollama, LM Studio, vLLM, LocalAI,
                                       Together, Groq, OpenRouter, ...

All backends return the same shape on success::

    {"suggested_code": "...", "suggested_description": "...", "confidence": 0.0-1.0}

and ``{"error": "<specific reason>"}`` on any failure (also logged to the
server console), so problems are diagnosable instead of silent.

No third-party packages are required — everything uses the standard library
``urllib`` so the dependency footprint stays small.
"""

import json
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional, Tuple, List, Dict, Any


# Default API endpoints. base_url in the config overrides these (and is
# required for the self-hosted "openai_compatible" provider).
OPENAI_DEFAULT_BASE = "https://api.openai.com/v1"
ANTHROPIC_DEFAULT_BASE = "https://api.anthropic.com/v1"
GEMINI_DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta"

MAX_OUTPUT_TOKENS = 4096
HTTP_TIMEOUT = 60


def _fail(reason: str) -> Dict[str, Any]:
    """Log why a suggestion could not be produced and return it as an error dict."""
    print(f"[llm] suggestion unavailable: {reason}")
    return {"error": reason}


def _norm_mod(m: str) -> str:
    """Normalize a modality for filtering (CR/DX are X-ray)."""
    u = (m or "").upper()
    if u in ("CR", "DX"):
        return "XR"
    return u


# ---------------------------------------------------------------------------
# Shared candidate selection + prompt building (provider-independent)
# ---------------------------------------------------------------------------

def _build_candidates(
    codes: List[Dict[str, str]],
    inferred_modality: Optional[str],
) -> List[Dict[str, str]]:
    """Filter the catalog down to plausible candidates to send to the model."""
    # Exclude UNCLASSIFIED / generic catch-all codes — never the right answer.
    filtered = [
        c for c in codes
        if c.get("description", "").upper() not in ("UNCLASSIFIED", "PRIOR", "OTHER PRIOR")
        and "UNCLASSIFIED" not in (c.get("description") or "").upper()
    ]
    # When we have an inferred modality, only send codes with that modality so
    # the model cannot pick CT when the query is clearly XR.
    if inferred_modality and inferred_modality.upper() not in ("UNKNOWN", ""):
        mod_upper = _norm_mod(inferred_modality)
        narrowed = [c for c in filtered if _norm_mod(c.get("modality") or "") == mod_upper]
        if narrowed:
            return narrowed
    if filtered:
        return filtered
    return [c for c in codes if c.get("description", "").upper() not in ("UNCLASSIFIED", "PRIOR", "OTHER PRIOR")]


def _build_prompts(
    query: str,
    code_list: List[Dict[str, str]],
    inferred_modality: Optional[str],
) -> Tuple[str, str]:
    """Return (system_prompt, user_message) shared across providers."""
    code_list_text = "\n".join(
        f"{c.get('code', '')}\t{c.get('description', '')}\t({(c.get('modality') or '?')})"
        for c in code_list
    )

    modality_hint = ""
    if inferred_modality and inferred_modality.upper() not in ("UNKNOWN", ""):
        modality_hint = (
            f" CRITICAL: The exam description suggests modality {inferred_modality.upper()} "
            "(e.g. sinus/nasal -> XR, not CT). STRONGLY prefer codes with this modality. "
        )

    system_prompt = (
        "You are a radiology coding expert. Given an exam description and a list of codes "
        "(code, description, modality), pick the single best matching code. "
        + modality_hint +
        "Choose a SPECIFIC code that matches the exam (e.g. sinus exam -> sinus X-ray code, not CT sinus). "
        "Never suggest UNCLASSIFIED or generic codes when a specific match exists. "
        'Reply in JSON only: {"code": "...", "confidence": 0.0-1.0, "reason": "..."}.'
    )
    user_msg = (
        "Code list (code TAB description TAB modality):\n"
        + code_list_text
        + '\n\nExam description to match: "'
        + query.replace('"', '\\"')
        + '"\n\nReply with JSON only: {"code": "...", "confidence": 0.0-1.0, "reason": "..."}'
    )
    return system_prompt, user_msg


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, headers: dict) -> Tuple[Optional[dict], Optional[str]]:
    """POST JSON and return (parsed_body, None) or (None, error_message)."""
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            return json.loads(resp.read().decode()), None
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace").strip()
        except Exception:
            pass
        hint = ""
        if exc.code == 401:
            hint = " (check the API key in Settings)"
        elif exc.code == 403:
            hint = " (key rejected or model not permitted)"
        elif exc.code == 429:
            hint = " (rate limited or out of quota)"
        elif exc.code == 404:
            hint = " (endpoint or model not found — check the model name / base URL)"
        return None, (f"HTTP {exc.code} {exc.reason}{hint}"
                      + (f" - {detail[:200]}" if detail else ""))
    except urllib.error.URLError as exc:
        return None, f"could not reach the LLM endpoint (network error or timeout): {exc.reason}"
    except (ValueError, OSError) as exc:
        return None, f"LLM response could not be read: {exc}"


# ---------------------------------------------------------------------------
# Provider callers — each returns (reply_text, None) or (None, error_message)
# ---------------------------------------------------------------------------

def _call_openai(cfg: dict, system_prompt: str, user_msg: str) -> Tuple[Optional[str], Optional[str]]:
    """OpenAI and any OpenAI-compatible server (Ollama, LM Studio, vLLM, ...)."""
    provider = (cfg.get("provider") or "openai").lower()
    base = (cfg.get("base_url") or "").strip().rstrip("/") or OPENAI_DEFAULT_BASE
    if provider == "openai_compatible" and not cfg.get("base_url"):
        return None, "openai_compatible provider needs a base URL (e.g. http://localhost:11434/v1)"
    model = (cfg.get("model") or "gpt-4o").strip()
    api_key = (cfg.get("api_key") or "").strip()

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    }
    # Official OpenAI's newer/reasoning models require max_completion_tokens;
    # most self-hosted OpenAI-compatible servers expect the classic max_tokens.
    if provider == "openai":
        payload["max_completion_tokens"] = MAX_OUTPUT_TOKENS
    else:
        payload["max_tokens"] = MAX_OUTPUT_TOKENS

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key

    body, err = _post_json(base + "/chat/completions", payload, headers)
    if err:
        return None, err
    if not isinstance(body, dict):
        return None, "response had an unexpected shape"
    choices = body.get("choices", [])
    if not choices:
        api_err = body.get("error")
        if api_err:
            msg = api_err.get("message", api_err) if isinstance(api_err, dict) else api_err
            return None, f"API error: {msg}"
        return None, "no choices returned"
    content = (choices[0].get("message", {}) or {}).get("content", "")
    if not content or not content.strip():
        finish = choices[0].get("finish_reason", "") or "unknown"
        extra = " - hit the token limit" if finish == "length" else ""
        return None, f"empty reply (finish_reason={finish}){extra}"
    return content, None


def _call_anthropic(cfg: dict, system_prompt: str, user_msg: str) -> Tuple[Optional[str], Optional[str]]:
    """Anthropic Claude Messages API."""
    base = (cfg.get("base_url") or "").strip().rstrip("/") or ANTHROPIC_DEFAULT_BASE
    model = (cfg.get("model") or "claude-sonnet-4-6").strip()
    api_key = (cfg.get("api_key") or "").strip()
    if not api_key:
        return None, "no Anthropic API key configured"

    payload = {
        "model": model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_msg}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    body, err = _post_json(base + "/messages", payload, headers)
    if err:
        return None, err
    if not isinstance(body, dict):
        return None, "response had an unexpected shape"
    if body.get("error"):
        api_err = body["error"]
        msg = api_err.get("message", api_err) if isinstance(api_err, dict) else api_err
        return None, f"API error: {msg}"
    parts = body.get("content") or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    if not text.strip():
        return None, f"empty reply (stop_reason={body.get('stop_reason', 'unknown')})"
    return text, None


def _call_gemini(cfg: dict, system_prompt: str, user_msg: str) -> Tuple[Optional[str], Optional[str]]:
    """Google Gemini generateContent API."""
    base = (cfg.get("base_url") or "").strip().rstrip("/") or GEMINI_DEFAULT_BASE
    model = (cfg.get("model") or "gemini-1.5-flash").strip()
    api_key = (cfg.get("api_key") or "").strip()
    if not api_key:
        return None, "no Gemini API key configured"

    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_msg}]}],
        "generationConfig": {
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "responseMimeType": "application/json",
        },
    }
    headers = {"Content-Type": "application/json"}
    # Key is passed as a query param per Gemini's REST convention.
    url = f"{base}/models/{model}:generateContent?key={urllib.parse.quote(api_key)}"
    body, err = _post_json(url, payload, headers)
    if err:
        return None, err
    if not isinstance(body, dict):
        return None, "response had an unexpected shape"
    if body.get("error"):
        api_err = body["error"]
        msg = api_err.get("message", api_err) if isinstance(api_err, dict) else api_err
        return None, f"API error: {msg}"
    candidates = body.get("candidates") or []
    if not candidates:
        return None, "no candidates returned (possibly blocked by safety filters)"
    parts = (candidates[0].get("content", {}) or {}).get("parts", []) or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
    if not text.strip():
        return None, "empty reply"
    return text, None


_PROVIDERS = {
    "openai": _call_openai,
    "openai_compatible": _call_openai,
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def suggest(
    query: str,
    codes: List[Dict[str, str]],
    llm_cfg: Dict[str, Any],
    inferred_modality: Optional[str] = None,
) -> Dict[str, Any]:
    """Ask the configured LLM to suggest a matching code.

    ``llm_cfg`` is the "llm" block from app settings:
        {enabled, provider, model, api_key, base_url}

    Returns {suggested_code, suggested_description, confidence} on success or
    {"error": "..."} on any failure.
    """
    if not isinstance(llm_cfg, dict):
        return _fail("LLM is not configured")

    provider = (llm_cfg.get("provider") or "openai").lower()
    caller = _PROVIDERS.get(provider)
    if caller is None:
        return _fail(f"unknown LLM provider '{provider}'")

    code_list = _build_candidates(codes, inferred_modality)
    if not code_list:
        return _fail("no candidate codes available to send to the model")

    system_prompt, user_msg = _build_prompts(query, code_list, inferred_modality)

    content, err = caller(llm_cfg, system_prompt, user_msg)
    if err:
        return _fail(f"{provider}: {err}")

    code, desc, confidence = _parse_suggest_response(content, code_list)
    if not code:
        return _fail(f"could not find a JSON code in the reply: {content.strip()[:160]!r}")

    # Safety: never accept UNCLASSIFIED.
    if desc and "UNCLASSIFIED" in (desc or "").upper():
        return _fail(f"model suggested an UNCLASSIFIED code ({code})")

    # Reject codes the model invented that aren't in the candidate list.
    if not any(str(c.get("code", "")).strip() == code for c in code_list):
        return _fail(f"model suggested code '{code}', which is not in the candidate list (hallucination)")

    # Reject a wrong modality when we had an inferred modality.
    if inferred_modality and inferred_modality.upper() not in ("UNKNOWN", ""):
        suggested = next((c for c in code_list if str(c.get("code", "")).strip() == code), None)
        if suggested:
            sugg_mod = _norm_mod(suggested.get("modality") or "")
            inf_mod = _norm_mod(inferred_modality)
            if sugg_mod and sugg_mod != inf_mod:
                return _fail(f"model picked {code} ({sugg_mod}) but the query modality is {inf_mod}")

    return {
        "suggested_code": code,
        "suggested_description": desc or "",
        "confidence": confidence,
    }


def _parse_suggest_response(
    content: str,
    codes: List[Dict[str, str]],
) -> Tuple[Optional[str], Optional[str], float]:
    """Extract (code, description, confidence) from a model reply."""
    try:
        json_str = _extract_json_object(content)
        if not json_str:
            return (None, None, 0.0)

        obj = json.loads(json_str)
        code = obj.get("code")
        if not code:
            return (None, None, 0.0)

        code = str(code).strip()
        confidence = 0.5
        if "confidence" in obj and isinstance(obj["confidence"], (int, float)):
            confidence = float(obj["confidence"])

        desc = None
        for c in codes:
            if str(c.get("code", "")).strip() == code:
                desc = c.get("description", "")
                break

        return (code, desc, confidence)
    except (json.JSONDecodeError, TypeError):
        return (None, None, 0.0)


def _extract_json_object(text: str) -> Optional[str]:
    """Extract the first {...} JSON object from text."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None
