
import os
import json

from deepeval.models import DeepEvalBaseLLM

from backend.config import get_groq_client


class GroqJudge(DeepEvalBaseLLM):

    def __init__(self):

        self.model_name = os.getenv(
            "EVALUATION_MODEL",
            "openai/gpt-oss-120b"
        )

        self.client = get_groq_client()

    def load_model(self):

        return self.client

    def generate(self, prompt: str, schema=None):

        messages = []

        if schema is not None:

            messages.append({
                "role": "system",
                "content": (
                    "Return only valid JSON matching "
                    "the following JSON schema:\n"
                    + json.dumps(
                        schema.model_json_schema()
                    )
                )
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        parameters = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            "reasoning_effort": "low",
            "include_reasoning": False,
            "max_completion_tokens": 8192
        }

        if schema is not None:

            parameters["response_format"] = {
                "type": "json_object"
            }

        response = (
            self.client.chat.completions.create(
                **parameters
            )
        )

        content = response.choices[0].message.content

        if schema is not None:

            return schema.model_validate_json(content)

        return content

    async def a_generate(self, prompt: str, schema=None):

        return self.generate(prompt, schema)

    def get_model_name(self):

        return self.model_name


if __name__ == "__main__":

    judge = GroqJudge()

    print("Judge:", judge.get_model_name())

    print(
        judge.generate(
            "Reply with the word READY."
        )
    )
