import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str,
                 max_tokens: int = 512, temperature: float = 0.7,
                 system_prompt_zh: str = "", system_prompt_en: str = "",
                 timeout: float = 30.0):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.default_prompts = {
            "zh": system_prompt_zh,
            "en": system_prompt_en,
        }
        logger.info(f"LLM client initialized: {base_url} model={model}")

    def chat_stream(self, messages: list[dict], language: str = "zh",
                    skill_prompt: str = ""):
        """Stream chat completion tokens.

        Args:
            messages: conversation history
            language: 'zh' or 'en'
            skill_prompt: if provided, replaces the default system prompt
        """
        if skill_prompt:
            system_content = skill_prompt
        else:
            system_content = self.default_prompts.get(language, self.default_prompts["zh"])

        full_messages = [{"role": "system", "content": system_content}] + messages

        logger.info(f"LLM call: {len(full_messages)} messages, "
                    f"system_prompt={len(system_content)} chars")

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
