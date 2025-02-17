# deep-research-python/deep_research_lib/ai/providers.py
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file if it exists

import os
import re
import json
from typing import Optional, Dict, Any, Type, List
from pydantic import BaseModel
import openai
import tiktoken
import httpx

from json_repair import repair_json

from deep_research_lib.ai.text_splitter import RecursiveCharacterTextSplitter

def log_provider(*args): # Create a separate log function for provider to distinguish logs
    print("[Provider Log]", *args)

# OpenAI Provider Implementation
class OpenAIProvider:
    def __init__(self, api_key: str, base_url: Optional[str] = None, model_name: str = "gpt-3.5-turbo-0125"): # Default model updated
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self.model_name = model_name
        openai.api_key = self.api_key
        openai.api_base = self.base_url

    async def generate_object(self, prompt: str, system: str, response_model: Type[BaseModel], **kwargs) -> BaseModel:
        log_provider("OpenAIProvider.generate_object: Starting") # Provider Log
        try:
            log_provider("OpenAIProvider.generate_object: Calling openai.chat.completions.create...") # Provider Log
            api_params = { # Logged parameters
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                **kwargs # Include any extra parameters like temperature, etc.
            }
            log_provider(f"OpenAIProvider.generate_object: API Request Parameters: {api_params}") # Log API parameters

            response = await openai.chat.completions.create( # Using openai library's async client
                **api_params
            )
            log_provider("OpenAIProvider.generate_object: openai.chat.completions.create call completed") # Provider Log

            log_provider(f"OpenAIProvider.generate_object: Raw API Response: {response}") # Log raw response

            # Assuming response.choices[0].message.content is the JSON string
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("No content in OpenAI API response")

            log_provider(f"OpenAIProvider.generate_object: Content received from API: {content}") # Log content

            # try:
            #     repaired_content = json.loads(content)
            #     log_provider(f"OpenAIProvider.generate_object: Successfully parsed JSON without repair.")
            # except json.JSONDecodeError:
            #     repaired_content = repair_json(content)
            #     log_provider(f"OpenAIProvider.generate_object: Repaired Content: {repaired_content}") # Log repaired content

            repaired_content = repair_json(content)
            log_provider(f"OpenAIProvider.generate_object: Repaired Content: {repaired_content}") # Log repaired content
            log_provider(f"OpenAIProvider.generate_object: Repaired Content type: {type(repaired_content)}") # Log repaired content

            try:
                parsed_object = response_model.model_validate_json(repaired_content) # Use Pydantic to parse JSON and validate
            except Exception as e:
                repaired_content = f"""{{
                "reportMarkdown": {repaired_content}
                }}"""
                parsed_object = response_model.model_validate_json(repaired_content) # Fallback to original content if repair fails
            log_provider(f"OpenAIProvider.generate_object: Parsed Object: {parsed_object}") # Log parsed object
            return parsed_object


        except Exception as e:
            log_provider(f"OpenAIProvider.generate_object: OpenAI API Error: {e}") # Provider Log - Detailed error
            raise


# Ollama Provider Implementation
class OllamaProvider:
    def __init__(self, base_url: str, model_name: str):
        self.base_url = base_url
        self.model_name = model_name
        # self.async_client = httpx.AsyncClient(base_url=self.base_url) # No need to initialize here anymore

    async def generate_object(self, prompt: str, system: str, response_model: Type[BaseModel], **kwargs) -> BaseModel:
        log_provider("OllamaProvider.generate_object: Starting") # Provider Log
        try:
            log_provider("OllamaProvider.generate_object: Calling client.post('/chat/completions')...") # Provider Log
            payload = {
                "model": self.model_name,
                "format": response_model.model_json_schema(),
                "messages": [ # Use "messages" list
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                "stream": False, # Set to False for non-streaming response for now
                **kwargs # Include any extra parameters
            }
            log_provider(f"OllamaProvider.generate_object: API Payload: {payload}") # Log API payload

            async with httpx.AsyncClient(base_url=self.base_url) as client: # Create a NEW client here
                response = await client.post("/chat/completions", json=payload, timeout=240.0) # Adjust timeout as needed
                log_provider("OllamaProvider.generate_object: client.post('/chat/completions') call completed") # Provider Log
                log_provider(f"OllamaProvider.generate_object: Raw API Response (status code): {response.status_code}") # Log status code
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                log_provider(f"OllamaProvider.generate_object: API Response Headers: {response.headers}") # Log headers

                raw_content = response.text # Get raw text content
                log_provider(f"OllamaProvider.generate_object: Raw Ollama Response Content:\n---\n{raw_content}\n---")
                
                response_json = response.json()
                # Assuming response_json['choices'][0]['message']['content'] contains the JSON string
                content = response_json['choices'][0]['message']['content']
                log_provider(f"OllamaProvider.generate_object: Content from JSON Response: {content}")

                # content = repair_json(content_from_json)
                # log_provider(f"OllamaProvider.generate_object: Repaired Content: {content}")

                if content is None:
                    raise ValueError("No content in Ollama API response")
                # log_provider(f"OllamaProvider.generate_object: Content received from API: {content}") # Log content

                # try:
                #     repaired_content = json.loads(content)
                #     log_provider(f"OllamaProvider.generate_object: Successfully parsed JSON without repair.")
                # except json.JSONDecodeError:
                #     repaired_content = repair_json(content)
                #     log_provider(f"OllamaProvider.generate_object: Repaired Content: {repaired_content}") # Log repaired content
                # repaired_content = repair_json(content)
                # log_provider(f"OllamaProvider.generate_object: Repaired Content: {repaired_content}") # Log repaired content


                try:
                    repaired_content = repair_json(content)
                    log_provider(f"OllamaProvider.generate_object: Repaired Content: {repaired_content}") # Log repaired content
                    parsed_object = response_model.model_validate_json(repaired_content) # Use Pydantic to parse JSON and validate
                except Exception as e:
                    log_provider(f"OllamaProvider.generate_object: Entered into exception") # Log repaired content
                    repaired_content = f"""{{
                    "reportMarkdown": {json.dumps(content)}
                    }}"""
                    parsed_object = response_model.model_validate_json(repaired_content) # Fallback to original content if repair fails
                log_provider(f"OllamaProvider.generate_object: Parsed Object: {parsed_object}") # Log parsed object
                return parsed_object


        except httpx.HTTPError as e:
            log_provider(f"OllamaProvider.generate_object: Ollama API HTTP Error: {e}, Response content: {e.response.text if hasattr(e, 'response') and e.response else 'No response content'}") # Safe access to response.text
            raise
        except httpx.ReadTimeout as e: # Catch ReadTimeout specifically
            log_provider(f"OllamaProvider.generate_object: Ollama API ReadTimeout Error: {e}") # Log Timeout Error
            raise
        except Exception as e:
            log_provider(f"OllamaProvider.generate_object: Ollama API Error: {e}")
            raise


# Provider Selection Logic and Initialization
def get_llm_provider():
    provider_type = os.environ.get("LLM_PROVIDER", "openai").lower() # Default to openai if not set

    if provider_type == "openai":
        api_key = os.environ.get("OPENAI_KEY")
        base_url = os.environ.get("OPENAI_ENDPOINT")
        model_name = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo-0125") # Updated default model
        if not api_key:
            raise ValueError("OPENAI_KEY environment variable must be set when using OpenAI provider.")
        return OpenAIProvider(api_key=api_key, base_url=base_url, model_name=model_name) # Return OpenAIProvider instance
    elif provider_type == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1") # Default Ollama URL
        model_name = os.environ.get("OLLAMA_MODEL", "llama3") # Default Ollama model
        return OllamaProvider(base_url=base_url, model_name=model_name) # Return OllamaProvider instance
    else:
        raise ValueError(f"Unknown LLM provider: {provider_type}. Please set LLM_PROVIDER to 'openai' or 'ollama'.")


# Initialize LLM Provider and Model - Directly initialize o3_mini_model
llm_provider = get_llm_provider()

# Directly initialize o3_mini_model to the generate_object method of the selected provider
o3_mini_model = llm_provider.generate_object


# Text Splitting and Trimming (Ported from text-splitter.ts - assuming text_splitter.py is created)
MIN_CHUNK_SIZE = 140
ENCODER = tiktoken.get_encoding('o200k_base') # Assuming this encoding is relevant, otherwise adjust

def trim_prompt(prompt: str, context_size: int = int(os.environ.get("CONTEXT_SIZE", "128000"))) -> str:
    if not prompt:
        return ""

    tokens = ENCODER.encode(prompt)
    length = len(tokens)
    log_provider(f"trim_prompt: Initial prompt tokens: {length}, context_size: {context_size}") # LOG - added logging
    if length <= context_size:
        log_provider("trim_prompt: Prompt within context size, no trimming needed.") # LOG - added logging
        return prompt

    overflow_tokens = length - context_size
    chunk_size_chars = len(prompt) - overflow_tokens * 3  # Rough char estimate
    if chunk_size_chars < MIN_CHUNK_SIZE:
        trimmed_prompt = prompt[:MIN_CHUNK_SIZE]
        log_provider(f"trim_prompt: Prompt trimmed to MIN_CHUNK_SIZE due to overflow. New length: {len(ENCODER.encode(trimmed_prompt))} tokens.") # LOG - added logging
        return trimmed_prompt
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_chars, chunk_overlap=0)
        trimmed_prompt = splitter.split_text(prompt)[0] or "" # Handle potential None return from split_text
        log_provider(f"trim_prompt: Prompt trimmed by RecursiveCharacterTextSplitter. New length: {len(ENCODER.encode(trimmed_prompt))} tokens.") # LOG - added logging

        if len(trimmed_prompt) == len(prompt): # Handle edge case where trimmed prompt is same length
            return trim_prompt(prompt[:chunk_size_chars], context_size)

        return trim_prompt(trimmed_prompt, context_size)


# Example usage (for testing - remove or comment out in final version)
async def test_provider():
    provider = get_llm_provider() # Get the configured provider

    class TestResponse(BaseModel):
        message: str

    try:
        # Now you need to call generate_object on the provider instance directly for testing
        response = await provider.generate_object(
            prompt="Write a short joke in JSON format.",
            system="You are a helpful joke-telling assistant.",
            response_model=TestResponse
        )
        print("LLM Response:", response)
    except Exception as e:
        print("Test Error:", e)

# Run test if script is executed directly (for development)
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_provider())