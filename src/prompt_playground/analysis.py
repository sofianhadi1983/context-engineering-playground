import re
from typing import List, Dict, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
from prompt_playground.client import count_tokens, estimate_cost


def calculate_metrics(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a single response.

    Args:
        response: Response dictionary from send_prompt containing text and token usage.

    Returns:
        Dictionary with calculated metrics.
    """
    text = response.get("text", "")

    char_count = len(text)
    word_count = len(text.split())

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    paragraph_count = len(paragraphs)

    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    input_tokens = response.get("input_tokens", 0)
    output_tokens = response.get("output_tokens", 0)
    model = response.get("model", "claude-sonnet-4-5-20250929")

    cost_info = estimate_cost(input_tokens, output_tokens, model)

    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    avg_paragraph_length = word_count / paragraph_count if paragraph_count > 0 else 0

    return {
        "char_count": char_count,
        "word_count": word_count,
        "token_count": output_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost": cost_info["total_cost"],
        "paragraph_count": paragraph_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "avg_paragraph_length": round(avg_paragraph_length, 2),
        "model": model,
    }


def compare_responses(responses: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare multiple responses and return a DataFrame with metrics.

    Args:
        responses: List of response dictionaries from send_prompt.

    Returns:
        DataFrame with comparison metrics for each response.
    """
    metrics_list = []

    for idx, response in enumerate(responses):
        metrics = calculate_metrics(response)
        metrics["response_id"] = f"Response {idx + 1}"
        metrics_list.append(metrics)

    df = pd.DataFrame(metrics_list)

    columns_order = ["response_id"] + [col for col in df.columns if col != "response_id"]
    df = df[columns_order]

    return df


def visualize_comparison(
    responses: List[Dict[str, Any]],
    metric: str = "length",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Create matplotlib visualizations comparing responses.

    Args:
        responses: List of response dictionaries.
        metric: Metric to visualize ('length', 'tokens', 'cost', 'structure').
        figsize: Figure size as (width, height) tuple.

    Returns:
        Matplotlib figure object.
    """
    df = compare_responses(responses)

    fig, ax = plt.subplots(figsize=figsize)

    if metric == "length":
        df.plot(
            x="response_id",
            y=["char_count", "word_count"],
            kind="bar",
            ax=ax,
            title="Response Length Comparison",
        )
        ax.set_ylabel("Count")
        ax.legend(["Characters", "Words"])

    elif metric == "tokens":
        df.plot(
            x="response_id",
            y=["input_tokens", "output_tokens"],
            kind="bar",
            ax=ax,
            title="Token Usage Comparison",
            stacked=True,
        )
        ax.set_ylabel("Tokens")
        ax.legend(["Input Tokens", "Output Tokens"])

    elif metric == "cost":
        df.plot(
            x="response_id",
            y="estimated_cost",
            kind="bar",
            ax=ax,
            title="Estimated Cost Comparison",
            color="green",
        )
        ax.set_ylabel("Cost (USD)")

    elif metric == "structure":
        df.plot(
            x="response_id",
            y=["paragraph_count", "sentence_count"],
            kind="bar",
            ax=ax,
            title="Response Structure Comparison",
        )
        ax.set_ylabel("Count")
        ax.legend(["Paragraphs", "Sentences"])

    else:
        raise ValueError(
            f"Unknown metric: {metric}. Choose from 'length', 'tokens', 'cost', 'structure'"
        )

    ax.set_xlabel("Response")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    return fig


def extract_key_points(response: str, max_points: int = 5) -> List[str]:
    """
    Extract main points from a response.

    Args:
        response: Response text to analyze.
        max_points: Maximum number of key points to extract.

    Returns:
        List of key points extracted from the response.
    """
    sentences = re.split(r"[.!?]+", response)
    sentences = [s.strip() for s in sentences if s.strip()]

    key_indicators = [
        r"\b(first|second|third|finally|importantly|key|main|primary)\b",
        r"\b(therefore|thus|consequently|as a result)\b",
        r"\b(in conclusion|to summarize|overall)\b",
        r"^\d+[\.\)]\s+",
        r"^[-•]\s+",
    ]

    scored_sentences = []
    for sentence in sentences:
        score = 0

        for pattern in key_indicators:
            if re.search(pattern, sentence, re.IGNORECASE):
                score += 2

        if len(sentence.split()) >= 10 and len(sentence.split()) <= 30:
            score += 1

        if sentence[0].isupper() and len(sentence) > 20:
            score += 0.5

        scored_sentences.append((sentence, score))

    scored_sentences.sort(key=lambda x: x[1], reverse=True)

    key_points = [sentence for sentence, score in scored_sentences[:max_points]]

    return key_points


def analyze_tone(response: str) -> Dict[str, Any]:
    """
    Analyze response tone and characteristics.

    Args:
        response: Response text to analyze.

    Returns:
        Dictionary with tone analysis results.
    """
    word_count = len(response.split())
    char_count = len(response)

    sentences = re.split(r"[.!?]+", response)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = len(sentences)

    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    exclamation_count = response.count("!")
    question_count = response.count("?")

    formal_words = [
        "furthermore",
        "moreover",
        "consequently",
        "therefore",
        "however",
        "nevertheless",
        "accordingly",
    ]
    informal_words = [
        "gonna",
        "wanna",
        "yeah",
        "cool",
        "awesome",
        "stuff",
        "things",
    ]

    formal_count = sum(
        1 for word in formal_words if word in response.lower()
    )
    informal_count = sum(
        1 for word in informal_words if word in response.lower()
    )

    formality_score = "neutral"
    if formal_count > informal_count:
        formality_score = "formal"
    elif informal_count > formal_count:
        formality_score = "informal"

    enthusiasm_score = "neutral"
    if exclamation_count > sentence_count * 0.2:
        enthusiasm_score = "high"
    elif exclamation_count == 0:
        enthusiasm_score = "low"

    complexity_score = "medium"
    if avg_sentence_length > 25:
        complexity_score = "high"
    elif avg_sentence_length < 15:
        complexity_score = "low"

    first_person = len(re.findall(r"\b(I|me|my|mine|we|us|our)\b", response, re.IGNORECASE))
    second_person = len(re.findall(r"\b(you|your|yours)\b", response, re.IGNORECASE))
    third_person = len(re.findall(r"\b(he|she|it|they|them|their)\b", response, re.IGNORECASE))

    perspective = "third-person"
    if first_person > max(second_person, third_person):
        perspective = "first-person"
    elif second_person > max(first_person, third_person):
        perspective = "second-person"

    return {
        "formality": formality_score,
        "enthusiasm": enthusiasm_score,
        "complexity": complexity_score,
        "perspective": perspective,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "exclamation_count": exclamation_count,
        "question_count": question_count,
        "first_person_count": first_person,
        "second_person_count": second_person,
        "third_person_count": third_person,
        "formal_word_count": formal_count,
        "informal_word_count": informal_count,
    }
