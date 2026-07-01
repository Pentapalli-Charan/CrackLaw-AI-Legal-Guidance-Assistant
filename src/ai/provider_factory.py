import logging
from src.config import Config
from src.ai.exceptions import LLMProviderError
from src.ai.llm_gateway import (
    BaseLLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    GeminiProvider,
    OllamaProvider,
    OpenRouterProvider
)

logger = logging.getLogger("CrackLaw.AI.ProviderFactory")

class ProviderFactory:
    """Factory class to instantiate specific LLM provider clients based on runtime configuration."""

    @staticmethod
    def get_provider(provider_name: str, model_name: str, config: Config) -> BaseLLMProvider:
        """Instantiates and returns the requested provider subclass."""
        provider_name = provider_name.lower().strip()
        
        logger.debug("Factory resolving provider client for: '%s' (model: %s)", provider_name, model_name)

        if provider_name == "openai":
            return OpenAIProvider(config, model_name)
        elif provider_name == "anthropic":
            return AnthropicProvider(config, model_name)
        elif provider_name == "gemini" or provider_name == "google":
            return GeminiProvider(config, model_name)
        elif provider_name == "ollama":
            return OllamaProvider(config, model_name)
        elif provider_name == "openrouter":
            return OpenRouterProvider(config, model_name)
        else:
            raise LLMProviderError(
                f"Unsupported LLM Provider requested: '{provider_name}'. "
                f"Choose from: openai, anthropic, gemini, ollama, openrouter."
            )
