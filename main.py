#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys


def run_requests_bench():
    try:
        from bench.benchmark_requests import main as run
        run()
    except ImportError as e:
        print(f"[ERR] Не удалось запустить benchmark_requests: {e}")


def run_openai_bench():
    try:
        from bench.benchmark_openai import main as run
        run()
    except ImportError as e:
        print(f"[ERR] Не удалось запустить benchmark_openai: {e}")


def run_ollama_bench():
    try:
        from bench.benchmark_ollama import main as run
        run()
    except ImportError as e:
        print(f"[ERR] Не удалось запустить benchmark_ollama: {e}")


def run_chat_bot():
    try:
        from chat_bot.chat_bot import main as run
        run()
    except ImportError as e:
        print(f"[ERR] Не удалось запустить chat_bot: {e}")


def print_menu():
    print("\n" + "=" * 50)
    print("ВЫБЕРИ РЕЖИМ")
    print("=" * 50)
    print("1 — Benchmark (native / requests)")
    print("2 — Benchmark (OpenAI API)")
    print("3 — Benchmark (Ollama API)")
    print("4 — Chat бот")
    print("0 — Выход")
    print("=" * 50)


def main():
    while True:
        print_menu()
        choice = input("Ввод: ").strip()

        if choice == "1":
            print("\n>>> Запуск benchmark_requests\n")
            run_requests_bench()

        elif choice == "2":
            print("\n>>> Запуск benchmark_openai\n")
            run_openai_bench()

        elif choice == "3":
            print("\n>>> Запуск benchmark_ollama\n")
            run_ollama_bench()

        elif choice == "4":
            print("\n>>> Запуск chat_bot\n")
            run_chat_bot()

        elif choice == "0":
            print("Выход.")
            sys.exit(0)

        else:
            print("Неверный выбор.")


if __name__ == "__main__":
    main()