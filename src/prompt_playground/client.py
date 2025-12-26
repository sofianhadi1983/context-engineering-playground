import os
from typing import Optional, List, Dict, Any
from anthropic import Anthropic
from dotenv import load_dotenv
import tiktoken


def create_client(api_key: Optional[str] = None) -> Anthropic:
    """
    Creates and returns an Anthropic client.

    Args:
        api_key: Optional API key. If not provided, loads from ANTHROPIC_API_KEY environment variable.

    Returns:
        Anthropic client instance.

    Raises:
        ValueError: If no API key is provided and none is found in environment.
    """
    load_dotenv()

    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        raise ValueError(
            "No API key provided. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
        )

    return Anthropic(api_key=api_key)


def send_prompt(
    prompt: str,
    model: str = "claude-sonnet-4-5-20250929",
    system: Optional[str] = None,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    client: Optional[Anthropic] = None,
) -> Dict[str, Any]:
    """
    Sends a single prompt to Claude and returns the response with token usage information.

    Args:
        prompt: The user prompt to send.
        model: The Claude model to use.
        system: Optional system prompt.
        temperature: Sampling temperature (0.0 to 1.0).
        max_tokens: Maximum tokens in response.
        client: Optional Anthropic client. If not provided, creates a new one.

    Returns:
        Dictionary containing response text, token usage, and metadata.

    Raises:
        Exception: If the API call fails.
    """
    if client is None:
        client = create_client()

    try:
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)

        return {
            "text": response.content[0].text,
            "model": response.model,
            "role": response.role,
            "stop_reason": response.stop_reason,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }
    except Exception as e:
        raise Exception(f"Failed to send prompt: {str(e)}")


def send_batch(
    prompts: List[str],
    model: str = "claude-sonnet-4-5-20250929",
    client: Optional[Anthropic] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Sends multiple prompts and returns a list of responses.

    Args:
        prompts: List of prompts to send.
        model: The Claude model to use.
        client: Optional Anthropic client. If not provided, creates a new one.
        **kwargs: Additional arguments to pass to send_prompt (system, temperature, max_tokens).

    Returns:
        List of response dictionaries.

    Raises:
        Exception: If any API call fails.
    """
    if client is None:
        client = create_client()

    responses = []
    for prompt in prompts:
        response = send_prompt(prompt, model=model, client=client, **kwargs)
        responses.append(response)

    return responses


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Counts tokens in the given text using tiktoken.

    Args:
        text: The text to count tokens for.
        encoding_name: The encoding to use (default: cl100k_base for Claude).

    Returns:
        Number of tokens in the text.

    Raises:
        Exception: If token counting fails.
    """
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        return len(encoding.encode(text))
    except Exception as e:
        raise Exception(f"Failed to count tokens: {str(e)}")


def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str,
) -> Dict[str, float]:
    """
    Estimates API cost based on token usage.

    Args:
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        model: The model name.

    Returns:
        Dictionary with cost breakdown (input_cost, output_cost, total_cost in USD).
    """
    pricing = {
        "claude-opus-4-5": {
            "input": 15.00 / 1_000_000,
            "output": 75.00 / 1_000_000,
        },
        "claude-sonnet-4-5": {
            "input": 3.00 / 1_000_000,
            "output": 15.00 / 1_000_000,
        },
        "claude-sonnet-4": {
            "input": 3.00 / 1_000_000,
            "output": 15.00 / 1_000_000,
        },
        "claude-3-7-sonnet": {
            "input": 3.00 / 1_000_000,
            "output": 15.00 / 1_000_000,
        },
        "claude-3-5-haiku": {
            "input": 1.00 / 1_000_000,
            "output": 5.00 / 1_000_000,
        },
        "claude-haiku-4": {
            "input": 1.00 / 1_000_000,
            "output": 5.00 / 1_000_000,
        },
    }

    model_key = None
    for key in pricing.keys():
        if key in model.lower():
            model_key = key
            break

    if model_key is None:
        model_key = "claude-sonnet-4-5"

    rates = pricing[model_key]
    input_cost = input_tokens * rates["input"]
    output_cost = output_tokens * rates["output"]
    total_cost = input_cost + output_cost

    return {
        "input_cost": round(input_cost, 6),
        "output_cost": round(output_cost, 6),
        "total_cost": round(total_cost, 6),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
