from langchain_groq import ChatGroq
from src.config import get_settings
from src.utils import get_logger, LLMError
import os
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)


class LLMService:

    def __init__(
        self,
        repo_id: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        settings = get_settings()
        self._repo_id = repo_id or settings.repo_id
        self._max_new_tokens = max_new_tokens or settings.max_new_tokens
        self._temperature = (
            temperature if temperature is not None else settings.temperature
        )
        self._chat_model: ChatGroq | None = None

    def get_chat_model(self) -> ChatGroq:
        if self._chat_model is None:
            logger.info("Loading Groq model: %s", self._repo_id)
            try:
                api_key = os.getenv("GROQ_API_KEY", "")
                self._chat_model = ChatGroq(
                    model=self._repo_id,
                    max_tokens=self._max_new_tokens,
                    temperature=self._temperature,
                    api_key=api_key,
                )
                logger.info("Groq model loaded successfully.")
            except Exception as exc:
                raise LLMError(
                    f"Failed to initialise Groq model '{self._repo_id}': {exc}"
                ) from exc
        return self._chat_model


from functools import lru_cache

@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    return LLMService()