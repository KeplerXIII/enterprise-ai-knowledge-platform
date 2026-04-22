#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================
# Бенч на OpenAI-compatible API Ollama
# со structured output через Pydantic
# =========================

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ValidationError


# =========================
# Настройки
# =========================

BASE_URL = "http://127.0.0.1:11434/v1/"
API_KEY = "ollama"  # Ollama игнорирует, но клиенту нужен
LOG_FILE = Path("ollama_openai_benchmark.log")

MODELS = [
    "qwen3:1.7b",
    "gemma3:4b",
    "qwen3:8b",
    "qwen3:14b",
]

TEMPERATURE = 0
MAX_TOKENS = 500


# =========================
# Pydantic-схема ответа
# =========================

class ModelAnswer(BaseModel):
    answer: str
    confidence: int


# =========================
# Клиент
# =========================

client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
)


# =========================
# Вспомогательные функции
# =========================

def safe_get_usage(resp: Any) -> tuple[int, int, int]:
    """
    Возвращает (prompt_tokens, completion_tokens, total_tokens)
    даже если usage отсутствует.
    """
    usage = getattr(resp, "usage", None)
    if not usage:
        return 0, 0, 0

    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or 0
    return prompt_tokens, completion_tokens, total_tokens


def safe_tps(tokens: int, seconds: float) -> float:
    return tokens / seconds if seconds > 0 else 0.0


def extract_raw_content(resp: Any) -> str:
    """
    Пытаемся достать сырой текст ответа.
    """
    try:
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


def call_model(model: str, prompt: str) -> dict:
    """
    Запрос через OpenAI SDK с auto-parse в Pydantic.
    """
    started = time.perf_counter()

    response = client.beta.chat.completions.parse(
        model=model,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "system",
                "content": (
                    "Отвечай строго в соответствии со схемой. "
                    "Поле confidence должно быть целым числом от 0 до 100."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        response_format=ModelAnswer,
    )

    elapsed_wall = time.perf_counter() - started

    message = response.choices[0].message
    parsed = getattr(message, "parsed", None)
    refusal = getattr(message, "refusal", None)

    prompt_tokens, completion_tokens, total_tokens = safe_get_usage(response)
    raw_content = extract_raw_content(response)

    result = {
        "model": model,
        "wall_time": elapsed_wall,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "raw_content": raw_content,
        "parsed": parsed,
        "refusal": refusal,
        "response": response,
    }
    return result


def format_stats(result: dict) -> str:
    model = result["model"]
    wall_time = result["wall_time"]
    prompt_tokens = result["prompt_tokens"]
    completion_tokens = result["completion_tokens"]
    total_tokens = result["total_tokens"]
    raw_content = result["raw_content"]
    parsed = result["parsed"]
    refusal = result["refusal"]

    prompt_tps = safe_tps(prompt_tokens, wall_time)
    gen_tps = safe_tps(completion_tokens, wall_time)

    lines = [
        "=" * 100,
        f"Модель: {model}",
        f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 100,
        "СТАТИСТИКА",
        f"prompt_tokens         : {prompt_tokens}",
        f"completion_tokens     : {completion_tokens}",
        f"total_tokens          : {total_tokens}",
        f"wall_time_s           : {wall_time:.3f}",
        f"prompt_tokens_per_sec : {prompt_tps:.2f}",
        f"gen_tokens_per_sec    : {gen_tps:.2f}",
        "-" * 100,
    ]

    if refusal:
        lines.extend([
            "REFUSAL",
            str(refusal).strip(),
            "-" * 100,
        ])

    if parsed is not None:
        lines.extend([
            "PARSED_OBJECT",
            parsed.model_dump_json(indent=2, ensure_ascii=False),
            "-" * 100,
        ])
    else:
        lines.extend([
            "PARSED_OBJECT",
            "Не удалось автоматически распарсить ответ в Pydantic-модель.",
            "-" * 100,
        ])

    if raw_content:
        lines.extend([
            "RAW_RESPONSE",
            raw_content.strip(),
            "=" * 100,
            "",
        ])
    else:
        lines.extend([
            "RAW_RESPONSE",
            "<пусто>",
            "=" * 100,
            "",
        ])

    return "\n".join(lines)


def main():
    prompt = input("Введи промпт: ").strip()
    if not prompt:
        print("Пустой промпт, выхожу.")
        return

    LOG_FILE.write_text("", encoding="utf-8")

    header = [
        "OLLAMA OPENAI-COMPAT BENCHMARK LOG",
        f"Создан: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"BASE_URL: {BASE_URL}",
        f"Модели: {', '.join(MODELS)}",
        f"Pydantic schema: {ModelAnswer.__name__}",
        "",
        "PROMPT",
        prompt,
        "",
    ]
    LOG_FILE.write_text("\n".join(header), encoding="utf-8")

    summary_rows = []

    print("\nСтартую прогон по моделям...\n")

    for model in MODELS:
        print(f"[...] {model}")
        try:
            result = call_model(model, prompt)
            block = format_stats(result)

            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(block)

            parsed = result["parsed"]
            completion_tokens = result["completion_tokens"]
            wall_time = result["wall_time"]
            gen_tps = safe_tps(completion_tokens, wall_time)

            summary_rows.append({
                "model": model,
                "prompt_tokens": result["prompt_tokens"],
                "output_tokens": completion_tokens,
                "total_tokens": result["total_tokens"],
                "wall_time": wall_time,
                "gen_tps": gen_tps,
                "ok": parsed is not None,
            })

            status = "OK" if parsed is not None else "NO_PARSE"
            print(
                f"[{status}] {model} | "
                f"output_tokens={completion_tokens} | "
                f"{gen_tps:.2f} tok/s"
            )

        except ValidationError as e:
            err_block = "\n".join([
                "=" * 100,
                f"Модель: {model}",
                "ОШИБКА ВАЛИДАЦИИ Pydantic",
                str(e),
                "=" * 100,
                "",
            ])
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(err_block)
            print(f"[ERR] {model} | ValidationError: {e}")

        except Exception as e:
            err_block = "\n".join([
                "=" * 100,
                f"Модель: {model}",
                f"ОШИБКА: {e}",
                "=" * 100,
                "",
            ])
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(err_block)
            print(f"[ERR] {model} | {e}")

    if summary_rows:
        summary_lines = [
            "",
            "#" * 100,
            "ИТОГОВАЯ СВОДКА",
            "#" * 100,
        ]

        for row in summary_rows:
            summary_lines.append(
                f"{row['model']:<14} | "
                f"time={row['wall_time']:.3f}s | "
                f"in={row['prompt_tokens']:<5} | "
                f"out={row['output_tokens']:<5} | "
                f"total={row['total_tokens']:<5} | "
                f"gen={row['gen_tps']:.2f} tok/s | "
                f"parsed={row['ok']}"
            )

        summary_lines.append("")

        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))

    print(f"\nГотово. Лог: {LOG_FILE.resolve()}")


if __name__ == "__main__":
    main()