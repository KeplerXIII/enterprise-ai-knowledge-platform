#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import ollama
from pydantic import BaseModel, ConfigDict, ValidationError


class Sentiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentiment: str
    score: int
    keywords: list[str]


def main():
    text = input("Введите отзыв: ").strip()

    schema = Sentiment.model_json_schema()

    response = ollama.chat(
        model="enterprise-assistant",
        messages=[
            {
                "role": "system",
                "content": "Проанализируй тональность и верни JSON.",
            },
            {"role": "user", "content": text},
        ],
        format=schema,
    )

    raw = response["message"]["content"]

    print("\nRAW:")
    print(raw)

    try:
        data = json.loads(raw)
        validated = Sentiment.model_validate(data)

        print("\nPARSED:")
        print(validated.model_dump_json(indent=2, ensure_ascii=False))

    except Exception as e:
        print("Ошибка:", e)


if __name__ == "__main__":
    main()