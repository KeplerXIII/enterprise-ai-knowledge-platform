#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================
# Бенч на ollama.chat(...)
# c format=schema
# и валидацией через Pydantic
# =========================

import json
import time
from datetime import datetime
from pathlib import Path

import ollama
from pydantic import BaseModel, ConfigDict, ValidationError


# =========================
# Настройки
# =========================

LOG_FILE = Path("ollama_chat_schema_benchmark.log")

MODELS = [
    "qwen3:1.7b",
    "gemma3:4b",
    "qwen3:8b",
    "qwen3:14b",
]

THINK = False

OPTIONS = {
    # "temperature": 0,
    # "num_predict": 300,
}


# =========================
# Pydantic-схема ответа
# =========================

class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    confidence: int


# =========================
# Вспомогательные функции
# =========================

def safe_div(a, b):
    return a / b if b else 0.0


def ns_to_s(value):
    if not value:
        return 0.0
    return value / 1_000_000_000


def get_json_schema(model_cls: type[BaseModel]) -> dict:
    return model_cls.model_json_schema()


def call_ollama(model: str, prompt: str, schema: dict) -> dict:
    started = time.perf_counter()

    result = ollama.chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Отвечай строго JSON-объектом по переданной схеме. "
                    "Без markdown, без пояснений, без лишнего текста."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        format=schema,
        think=THINK,
        options=OPTIONS,
    )

    elapsed_wall = time.perf_counter() - started
    result["_wall_time"] = elapsed_wall
    return result


def parse_structured_response(result: dict) -> tuple[ModelAnswer | None, str]:
    message = result.get("message", {}) or {}
    content = message.get("content", "") or ""

    if not content.strip():
        return None, "Пустой message.content"

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return None, f"message.content не является валидным JSON: {e}"

    try:
        validated = ModelAnswer.model_validate(data)
        return validated, ""
    except ValidationError as e:
        return None, f"Ошибка валидации Pydantic:\n{e}"


def format_stats(result: dict, parsed_obj: ModelAnswer | None, parse_error: str) -> str:
    model = result.get("model", "unknown")
    created_at = result.get("created_at", "")
    message = result.get("message", {}) or {}

    response_text = message.get("content", "") or ""
    thinking_text = message.get("thinking", "") or ""

    total_duration = ns_to_s(result.get("total_duration", 0))
    load_duration = ns_to_s(result.get("load_duration", 0))
    prompt_eval_duration = ns_to_s(result.get("prompt_eval_duration", 0))
    eval_duration = ns_to_s(result.get("eval_duration", 0))
    wall_time = result.get("_wall_time", 0.0)

    prompt_eval_count = result.get("prompt_eval_count", 0)
    eval_count = result.get("eval_count", 0)

    gen_tps = safe_div(eval_count, eval_duration)
    prompt_tps = safe_div(prompt_eval_count, prompt_eval_duration)

    lines = [
        "=" * 100,
        f"Модель: {model}",
        f"created_at: {created_at}",
        f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 100,
        "СТАТИСТИКА",
        f"prompt_eval_count      : {prompt_eval_count}",
        f"eval_count             : {eval_count}",
        f"prompt_eval_duration_s : {prompt_eval_duration:.3f}",
        f"eval_duration_s        : {eval_duration:.3f}",
        f"load_duration_s        : {load_duration:.3f}",
        f"total_duration_s       : {total_duration:.3f}",
        f"wall_time_s            : {wall_time:.3f}",
        f"prompt_tokens_per_sec  : {prompt_tps:.2f}",
        f"gen_tokens_per_sec     : {gen_tps:.2f}",
        "-" * 100,
    ]

    if thinking_text:
        lines.extend([
            "THINKING",
            thinking_text.strip(),
            "-" * 100,
        ])

    if parsed_obj is not None:
        lines.extend([
            "PARSED_OBJECT",
            parsed_obj.model_dump_json(indent=2, ensure_ascii=False),
            "-" * 100,
        ])
    else:
        lines.extend([
            "PARSE_ERROR",
            parse_error.strip() if parse_error else "Неизвестная ошибка парсинга",
            "-" * 100,
        ])

    lines.extend([
        "RAW_RESPONSE",
        response_text.strip(),
        "=" * 100,
        "",
    ])

    return "\n".join(lines)


def main():
    user_prompt = input("Введи промпт: ").strip()
    if not user_prompt:
        print("Пустой промпт, выхожу.")
        return

    schema = get_json_schema(ModelAnswer)

    LOG_FILE.write_text("", encoding="utf-8")

    header = [
        "OLLAMA CHAT BENCHMARK LOG (format=schema)",
        f"Создан: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Модели: {', '.join(MODELS)}",
        f"Pydantic model: {ModelAnswer.__name__}",
        "",
        "JSON SCHEMA",
        json.dumps(schema, indent=2, ensure_ascii=False),
        "",
        "PROMPT",
        user_prompt,
        "",
    ]
    LOG_FILE.write_text("\n".join(header), encoding="utf-8")

    summary_rows = []

    print("\nСтартую прогон по моделям...\n")

    for model in MODELS:
        print(f"[...] {model}")
        try:
            result = call_ollama(model, user_prompt, schema)
            parsed_obj, parse_error = parse_structured_response(result)

            block = format_stats(result, parsed_obj, parse_error)
            with LOG_FILE.open("a", encoding="utf-8") as f:
                f.write(block)

            prompt_eval_count = result.get("prompt_eval_count", 0)
            eval_count = result.get("eval_count", 0)

            prompt_eval_duration = ns_to_s(result.get("prompt_eval_duration", 0))
            eval_duration = ns_to_s(result.get("eval_duration", 0))
            wall_time = result.get("_wall_time", 0.0)

            gen_tps = safe_div(eval_count, eval_duration)
            prompt_tps = safe_div(prompt_eval_count, prompt_eval_duration)

            summary_rows.append({
                "model": model,
                "prompt_tokens": prompt_eval_count,
                "output_tokens": eval_count,
                "wall_time": wall_time,
                "gen_tps": gen_tps,
                "prompt_tps": prompt_tps,
                "parsed_ok": parsed_obj is not None,
            })

            status = "OK" if parsed_obj is not None else "BAD_JSON"
            print(f"[{status}] {model} | output_tokens={eval_count} | {gen_tps:.2f} tok/s")

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
                f"gen={row['gen_tps']:.2f} tok/s | "
                f"prompt={row['prompt_tps']:.2f} tok/s | "
                f"parsed={row['parsed_ok']}"
            )

        summary_lines.append("")

        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))

    print(f"\nГотово. Лог: {LOG_FILE.resolve()}")


if __name__ == "__main__":
    main()