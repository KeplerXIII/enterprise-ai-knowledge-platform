#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from datetime import datetime
from pathlib import Path

import ollama


MODELS = [
    "qwen3:1.7b",
    "gemma3:4b",
    "qwen3:8b",
    "qwen3:14b",
]

LOG_FILE = Path("benchmark_results.log")

OPTIONS = {
    "temperature": 0,
}

TASKS = [
    {
        "name": "qa",
        "prompt": "Что такое RAG и зачем он нужен?",
    },
    {
        "name": "summary",
        "prompt": (
            "Сделай краткое резюме:\n"
            "RAG — это метод, при котором модель использует внешнюю базу знаний "
            "для повышения точности ответа."
        ),
    },
    {
        "name": "classification",
        "prompt": (
            "Категоризируй обращение (IT / Billing / Support):\n"
            "Не могу войти в личный кабинет."
        ),
    },
]


def ns_to_s(v):
    return v / 1_000_000_000 if v else 0.0


def safe_div(a, b):
    return a / b if b else 0.0


def call(model: str, prompt: str):
    start = time.perf_counter()

    response = ollama.generate(
        model=model,
        prompt=prompt,
        stream=False,
        options=OPTIONS,
    )

    wall_time = time.perf_counter() - start
    return response, wall_time


def get_field(response, name, default=None):
    """
    Поддержка и dict-подобного, и объектного доступа.
    """
    if isinstance(response, dict):
        return response.get(name, default)
    return getattr(response, name, default)


def main():
    LOG_FILE.write_text("", encoding="utf-8")

    print("\nСтарт benchmark...\n")

    header = [
        "OLLAMA BENCHMARK RESULTS",
        f"Создан: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Модели: {', '.join(MODELS)}",
        f"Задач: {len(TASKS)}",
        "",
    ]
    LOG_FILE.write_text("\n".join(header), encoding="utf-8")

    summary = []

    for task in TASKS:
        print(f"\n=== {task['name']} ===")

        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"\n=== TASK: {task['name']} ===\n")

        for model in MODELS:
            print(f"[...] {model}")

            try:
                res, wall_time = call(model, task["prompt"])

                output_tokens = get_field(res, "eval_count", 0)
                prompt_tokens = get_field(res, "prompt_eval_count", 0)
                eval_duration = ns_to_s(get_field(res, "eval_duration", 0))
                prompt_eval_duration = ns_to_s(get_field(res, "prompt_eval_duration", 0))
                response_text = get_field(res, "response", "") or ""

                gen_tps = safe_div(output_tokens, eval_duration)
                prompt_tps = safe_div(prompt_tokens, prompt_eval_duration)

                summary.append({
                    "task": task["name"],
                    "model": model,
                    "time": wall_time,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "prompt_tps": prompt_tps,
                    "gen_tps": gen_tps,
                    "response": response_text.strip(),
                })

                with LOG_FILE.open("a", encoding="utf-8") as f:
                    f.write(
                        "\n".join([
                            "-" * 80,
                            f"Модель: {model}",
                            f"task: {task['name']}",
                            f"wall_time: {wall_time:.3f}s",
                            f"prompt_tokens: {prompt_tokens}",
                            f"output_tokens: {output_tokens}",
                            f"prompt_tps: {prompt_tps:.2f}",
                            f"gen_tps: {gen_tps:.2f}",
                            "RESPONSE:",
                            response_text.strip(),
                            "",
                        ])
                    )

                print(f"[OK] {model} | {wall_time:.2f}s | {gen_tps:.2f} tok/s")

            except Exception as e:
                with LOG_FILE.open("a", encoding="utf-8") as f:
                    f.write(
                        "\n".join([
                            "-" * 80,
                            f"Модель: {model}",
                            f"task: {task['name']}",
                            f"ERROR: {e}",
                            "",
                        ])
                    )
                print(f"[ERR] {model} | {e}")

    print("\n=== SUMMARY ===\n")

    for row in summary:
        print(
            f"{row['task']:<15} | "
            f"{row['model']:<12} | "
            f"{row['time']:.2f}s | "
            f"{row['gen_tps']:.2f} tok/s"
        )

    print(f"\nЛог: {LOG_FILE.resolve()}")


if __name__ == "__main__":
    main()