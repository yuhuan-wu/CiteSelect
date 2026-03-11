"""
Lightweight LLM client abstraction using the official OpenAI Python SDK.

Supports both OpenAI and OpenRouter through the OpenAI-compatible Chat Completions API.
"""

from typing import List, Dict, Any, Optional
import logging
import os

try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None  # type: ignore
    logging.getLogger(__name__).warning(f"Failed to import OpenAI SDK: {e}")


class OpenAIChatClient:
    """
    Minimal OpenAI Chat Completions client (official SDK).

    Example:
        client = OpenAIChatClient(api_key="...", model="gpt-5")
        content = client.chat(messages=[{"role":"user","content":"hi"}],
                              max_completion_tokens=512, temperature=None)
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        provider: str = "openai",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        default_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.provider = (provider or "openai").lower()
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.default_headers = default_headers or {}
        if OpenAI is None:
            logging.error("OpenAI SDK 未安装，请先 `pip install openai`")
            self._client = None
        else:
            # The SDK accepts base_url and api_key
            try:
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    default_headers=self.default_headers or None,
                    timeout=self.timeout,
                )
            except TypeError:
                # Fallback for older SDK that may not accept base_url
                self._client = OpenAI(api_key=self.api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Invoke chat completions and return content string.
        Returns None when call fails.
        """
        if not self.api_key:
            logging.error("OpenAI API key missing")
            return None
        if self._client is None:
            logging.error("OpenAI SDK client 未初始化")
            return None

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        # Map to chat.completions API parameter

        # Some models (e.g., gpt-5 family) do not allow temperature override.
        allow_temperature = True
        try:
            m = (self.model or "").lower()
            if m.startswith("gpt-5") or m.startswith("openai/gpt-5"):
                allow_temperature = False
        except Exception:
            pass
        if allow_temperature and temperature is not None:
            try:
                params["temperature"] = float(temperature)
            except Exception:
                pass

        # Append any extra payload options
        if extra_payload:
            params.update(extra_payload)

        try:
            completion = self._client.chat.completions.create(**params)
            # Prefer choices[0].message.content
            content: Optional[str] = None
            try:
                choice0 = completion.choices[0]
                msg = getattr(choice0, "message", None)
                content = getattr(msg, "content", None) if msg is not None else None
                if content is None:
                    # Some SDKs may expose `text` for compatibility
                    content = getattr(choice0, "text", None)
            except Exception:
                content = None
            if not content:
                logging.warning("OpenAI response has empty content")
                return None
            return content
        except Exception as e:
            logging.exception(f"OpenAI 请求异常: {e}")
            return None


def create_llm_client_from_config(cfg: Dict[str, Any]) -> OpenAIChatClient:
    llm_cfg = cfg.get("llm") or {}
    provider = str(llm_cfg.get("provider") or "openai").lower()
    model = str(llm_cfg.get("model") or ("openai/gpt-5.4" if provider == "openrouter" else "gpt-5.4"))
    timeout = float(llm_cfg.get("timeout", 60.0))

    if provider == "openrouter":
        api_key = (
            cfg.get("openrouter_api_key")
            or os.environ.get("OPENROUTER_API_KEY")
            or cfg.get("openai_api_key")
            or os.environ.get("OPENAI_API_KEY")
        )
        base_url = str(llm_cfg.get("base_url") or "https://openrouter.ai/api/v1")
        app_name = str(llm_cfg.get("app_name") or "CiteSelect")
        site_url = str(llm_cfg.get("site_url") or "https://github.com/")
        default_headers = {
            "HTTP-Referer": site_url,
            "X-Title": app_name,
        }
    else:
        api_key = cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
        base_url = str(llm_cfg.get("base_url") or "https://api.openai.com/v1")
        default_headers = {}

    return OpenAIChatClient(
        api_key=str(api_key or ""),
        model=model,
        provider=provider,
        base_url=base_url,
        timeout=timeout,
        default_headers=default_headers,
    )
