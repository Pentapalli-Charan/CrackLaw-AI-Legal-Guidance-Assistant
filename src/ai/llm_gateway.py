import os
import json
import time
import logging
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import requests
from src.config import Config
from src.ai.exceptions import LLMProviderError

logger = logging.getLogger("CrackLaw.AI.LLMGateway")

class BaseLLMProvider(ABC):
    """Abstract base class for all external LLM provider integrations."""

    def __init__(self, config: Config, model_name: str):
        self.config = config
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        top_p: float
    ) -> str:
        """Executes the synchronous request to the provider API."""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API client implementation using standard HTTP."""

    def generate(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int, top_p: float) -> str:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise LLMProviderError("OPENAI_API_KEY environment variable is not set.")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code != 200:
                raise LLMProviderError(f"OpenAI error status {response.status_code}: {response.text}")
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if isinstance(e, LLMProviderError):
                raise e
            raise LLMProviderError(f"OpenAI connection error: {e}") from e


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude API client implementation using standard HTTP."""

    def generate(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int, top_p: float) -> str:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMProviderError("ANTHROPIC_API_KEY environment variable is not set.")

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code != 200:
                raise LLMProviderError(f"Anthropic error status {response.status_code}: {response.text}")
            
            data = response.json()
            return data["content"][0]["text"]
        except Exception as e:
            if isinstance(e, LLMProviderError):
                raise e
            raise LLMProviderError(f"Anthropic connection error: {e}") from e


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API client implementation using standard HTTP."""

    def generate(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int, top_p: float) -> str:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise LLMProviderError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set.")

        # API expects endpoint formatted with the model name
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={api_key}"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "topP": top_p
            }
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            if response.status_code != 200:
                raise LLMProviderError(f"Gemini error status {response.status_code}: {response.text}")
            
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            if isinstance(e, LLMProviderError):
                raise e
            raise LLMProviderError(f"Gemini connection error: {e}") from e


class OllamaProvider(BaseLLMProvider):
    """Local Ollama client using the OpenAI compatibility layer."""

    def generate(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int, top_p: float) -> str:
        base_url = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        url = f"{base_url}/v1/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code != 200:
                raise LLMProviderError(f"Ollama error status {response.status_code}: {response.text}")
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if isinstance(e, LLMProviderError):
                raise e
            raise LLMProviderError(f"Ollama connection error (check if Ollama is running): {e}") from e


class OpenRouterProvider(BaseLLMProvider):
    """OpenRouter client using standard HTTP."""

    def generate(self, prompt: str, system_prompt: str, temperature: float, max_tokens: int, top_p: float) -> str:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMProviderError("OPENROUTER_API_KEY environment variable is not set.")

        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://cracklaw.ai",
            "X-Title": "CrackLaw"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=45)
            if response.status_code != 200:
                raise LLMProviderError(f"OpenRouter error status {response.status_code}: {response.text}")
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if isinstance(e, LLMProviderError):
                raise e
            raise LLMProviderError(f"OpenRouter connection error: {e}") from e


class LLMGateway:
    """Unified API Gateway for LLM provider requests. Handles backoff, timeouts, and file-based caching."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.cache_file = os.path.join(self.config.cache_dir, "llm_cache.json")
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, str]:
        """Loads cached responses from the disk cache file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load LLM response cache: %s", str(e))
        return {}

    def _save_cache(self) -> None:
        """Saves current memory cache to the disk cache file."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Failed to save LLM response cache: %s", str(e))

    def _compute_hash(self, provider: str, model: str, prompt: str, system_prompt: str, temp: float) -> str:
        """Generates a unique MD5 hash for the request characteristics to use as cache keys."""
        raw_key = f"{provider}:{model}:{temp}:{system_prompt}:{prompt}"
        return hashlib.md5(raw_key.encode("utf-8")).hexdigest()

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a legal assistant.",
        provider_name: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        top_p: float = 0.95,
        use_cache: bool = True
    ) -> str:
        """Dispatches text generation requests to the active provider with automatic retry backoff."""
        # Load from config defaults if parameters are omitted
        ai_settings = self.config.get("ai", {})
        provider_name = provider_name or ai_settings.get("llm_provider", "gemini")
        model_name = model_name or ai_settings.get("model_name", "gemini-1.5-flash")
        
        # 1. Lookup in Cache
        cache_key = self._compute_hash(provider_name, model_name, prompt, system_prompt, temperature)
        if use_cache and cache_key in self.cache:
            logger.info("LLM Gateway cache hit for provider '%s' (model: %s)", provider_name, model_name)
            return self.cache[cache_key]

        # 2. Get provider instance (import dynamically to avoid circular references)
        from src.ai.provider_factory import ProviderFactory
        provider = ProviderFactory.get_provider(provider_name, model_name, self.config)

        # 3. Request execution loop with exponential retry backoff
        max_retries = ai_settings.get("max_retries", 3)
        base_delay = 1.0  # seconds
        last_error = None

        logger.info("Calling LLM Gateway (%s: %s)", provider_name, model_name)
        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.time()
                result = provider.generate(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p
                )
                latency = (time.time() - t0) * 1000
                logger.info("LLM generation successful in %.2f ms (attempt %d)", latency, attempt)

                # Store response in cache
                if use_cache:
                    self.cache[cache_key] = result
                    self._save_cache()

                return result

            except LLMProviderError as e:
                last_error = e
                logger.warning("LLM provider attempt %d failed: %s", attempt, str(e))
                if attempt < max_retries:
                    sleep_time = base_delay * (2 ** (attempt - 1))
                    logger.info("Sleeping for %.2f seconds before retrying...", sleep_time)
                    time.sleep(sleep_time)

        raise LLMProviderError(f"All {max_retries} attempts failed. Last error: {last_error}")

    def clear_cache(self) -> None:
        """Evicts cache records from memory and filesystem."""
        self.cache.clear()
        if os.path.exists(self.cache_file):
            try:
                os.remove(self.cache_file)
            except Exception as e:
                logger.warning("Failed to delete LLM cache file: %s", str(e))
        logger.info("LLM Gateway cache cleared.")
