#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================
# Бенч на /api/generate без входных моделей JSON 
# =========================

import json
import time
from datetime import datetime
from pathlib import Path
from urllib import request, error

# =========================
# Настройки
# =========================

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
API_KEY = "ollama"  # обязателен для клиента, Ollama его игнорирует
LOG_FILE = Path("ollama_benchmark.log")

MODELS = [
    "qwen3:1.7b",
    "gemma3:4b",
    "qwen3:8b",
    "qwen3:14b",
]

THINK = False

OPTIONS = {
    # "num_predict": 300,
    # "temperature": 0.7,
}


def ns_to_s(value):
    if not value:
        return 0.0
    return value / 1_000_000_000


def safe_div(a, b):
    return a / b if b else 0.0


def call_ollama(model: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": THINK,
        "options": OPTIONS,
    }

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        OLLAMA_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with request.urlopen(req, timeout=3600) as resp:
            raw = resp.read().decode("utf-8")
    except error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    except Exception as e:
        raise RuntimeError(f"Ошибка запроса к Ollama: {e}") from e

    elapsed_wall = time.perf_counter() - started

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Не удалось распарсить JSON: {raw[:500]}") from e

    result["_wall_time"] = elapsed_wall
    return result


def format_stats(result: dict) -> str:
    model = result.get("model", "unknown")
    response_text = result.get("response", "")
    thinking_text = result.get("thinking", "")

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

    lines.extend([
        "RESPONSE",
        response_text.strip(),
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
        "OLLAMA BENCHMARK LOG",
        f"Создан: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"URL: {OLLAMA_URL}",
        f"Модели: {', '.join(MODELS)}",
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
            result = call_ollama(model, prompt)
            block = format_stats(result)
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
            })

            print(f"[OK] {model} | output_tokens={eval_count} | {gen_tps:.2f} tok/s")
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
                f"prompt={row['prompt_tps']:.2f} tok/s"
            )

        summary_lines.append("")

        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write("\n".join(summary_lines))

    print(f"\nГотово. Лог: {LOG_FILE.resolve()}")


if __name__ == "__main__":
    main()