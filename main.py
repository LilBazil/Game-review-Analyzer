import os
import json
import time
import random
from enum import Enum
from datetime import datetime
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise ValueError("API ключ не найден. Создайте файл .env с OPENROUTER_API_KEY")

MODEL = "minimax/minimax-m2.5:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json; charset=utf-8",
    "HTTP-Referer": "http://localhost",
    "X-Title": "Review Analyzer"
}

CSV_FILE = "reviews.csv"
TXT_FILE = "review.txt"
OUTPUT_JSON = "analysis_results.json"
OUTPUT_TEXT = "analysis_report.txt"

REQUEST_DELAY = 2
MAX_RETRIES = 8
MAX_BACKOFF = 60


# классификация -------------------------------------------------------------------------------------------------------

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class Topic(str, Enum):
    GRAPHICS = "graphics"
    GAMEPLAY = "gameplay"
    BUGS = "bugs"
    PERFORMANCE = "performance"
    STORY = "story"
    CONTROLS = "controls"
    MUSIC = "music"
    OTHER = "other"

def find_text_column(df):
    for col in df.columns:
        if "review" in col.lower() or "text" in col.lower():
            return col
    return df.select_dtypes(include=["object"]).columns[0]

def clean_json(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()

def validate(data):
    sentiment = data.get("sentiment", "neutral")
    topics = data.get("topics", [])
    summary = data.get("summary", "")
    confidence = float(data.get("confidence", 0.5))

    valid_sentiments = [s.value for s in Sentiment]
    valid_topics = [t.value for t in Topic]

    if sentiment not in valid_sentiments:
        sentiment = "neutral"

    if not isinstance(topics, list):
        topics = ["other"]

    cleaned_topics = []
    for t in topics:
        if t in valid_topics:
            cleaned_topics.append(t)

    if not cleaned_topics:
        cleaned_topics = ["other"]

    summary = str(summary).strip()
    if len(summary) > 200:
        summary = summary[:200]

    return {
        "sentiment": sentiment,
        "topics": cleaned_topics,
        "summary": summary,
        "confidence": round(max(0, min(confidence, 1)), 3)
    }


# загрузка данных --------------------------------------------------------------------------------------------------------

def load_reviews():

    reviews = []

    if os.path.exists(CSV_FILE):
        print(f"Загрузка данных из {CSV_FILE}")
        df = pd.read_csv(CSV_FILE, encoding='utf-8')
        text_col = find_text_column(df)
        for idx, row in df.iterrows():
            reviews.append({
                "id": idx,
                "text": str(row[text_col])
            })
        return reviews

    elif os.path.exists(TXT_FILE):
        print(f"Загрузка данных из {TXT_FILE}")
        with open(TXT_FILE, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        if text:
            reviews.append({
                "id": 0,
                "text": text
            })
        return reviews

    else:
        print(f"Ошибка: файл не найден.")
        print(f"Создайте {CSV_FILE} с колонкой reviews или {TXT_FILE} с одним отзывом")
        return None


# промпт ----------------------------------------------------------------------------------------------------------------

def analyze(review: str):

    system_prompt = """
Ты — эксперт по анализу отзывов об играх.

Верни ТОЛЬКО JSON:

{
  "sentiment": "positive | negative | neutral",
  "topics": ["graphics", "gameplay", "bugs", "performance", "story", "controls", "music", "other"],
  "summary": "одно короткое предложение на русском",
  "confidence": 0.0-1.0
}

ПРАВИЛА:

1. sentiment:
- positive - если есть общая похвала
- negative - если преобладают жалобы
- neutral - если смешано или нет эмоций

2. topics:
- выбирай ВСЕ подходящие категории, даже если только упомянули
- other только если ничего не подходит

3. summary:
- одно короткое предложение до 10 слов, которое отражает ОБЩЕЕ мнение

4. строго JSON без текста
"""

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": review[:3000]}
        ],
        "temperature": 0,
        "max_tokens": 200
    }

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)

            if r.status_code == 429:
                wait = min(MAX_BACKOFF, (2 ** attempt) + random.random())
                time.sleep(wait)
                continue

            if r.status_code != 200:
                print(f"Ошибка API: {r.status_code}")
                time.sleep(3)
                continue

            data = r.json()["choices"][0]["message"]["content"]
            data = clean_json(data)
            parsed = json.loads(data)
            return validate(parsed)

        except Exception as e:
            print(f"Ошибка: {e}")
            time.sleep(2)

    return None


# отчёт ---------------------------------------------------------------------------------------------------------------

def save_readable_report(results, output_file):

    topics_ru = {
        "graphics": "графика",
        "gameplay": "геймплей",
        "bugs": "баги",
        "performance": "производительность",
        "story": "сюжет",
        "controls": "управление",
        "music": "музыка",
        "other": "прочее"
    }

    sentiment_ru = {
        "positive": "положительный",
        "negative": "отрицательный",
        "neutral": "нейтральный"
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("ОТЧЁТ ПО АНАЛИЗУ ОТЗЫВОВ\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Модель: {MODEL}\n")
        f.write(f"Всего отзывов: {len(results)}\n")
        f.write("=" * 70 + "\n\n")

        sentiments = [r["sentiment"] for r in results]
        positive_count = sentiments.count("positive")
        negative_count = sentiments.count("negative")
        neutral_count = sentiments.count("neutral")

        f.write("СТАТИСТИКА:\n")
        f.write(f"  Положительных: {positive_count}\n")
        f.write(f"  Отрицательных: {negative_count}\n")
        f.write(f"  Нейтральных:   {neutral_count}\n\n")

        f.write("ОТдЕЛЬНЫЙ РАЗБОР ОТЗЫВОВ:\n")
        f.write("-" * 70 + "\n\n")

        for i, r in enumerate(results):
            ru_sentiment = sentiment_ru.get(r["sentiment"], r["sentiment"])
            ru_topics = [topics_ru.get(t, t) for t in r["topics"]]

            f.write(f"Отзыв #{i + 1} (ID: {r['row_id']})\n")
            f.write(f"  Тональность: {ru_sentiment}\n")
            f.write(f"  Темы: {', '.join(ru_topics)}\n")
            f.write(f"  Краткое содержание: {r['summary']}\n")
            f.write(f"  Текст отзыва:\n")
            f.write(f"    {r['review'][:200]}...\n")
            f.write("\n")


# ======================================================================================================================

def main():
    print("\n"+"АНАЛИТИКА ОТЗЫВОВ ОБ ИГРАХ")
    print("*"*60+"\n")

    reviews = load_reviews()
    if reviews is None:
        return
    print(f"Найдено отзывов: {len(reviews)}\n")

    # если уже есть обработанные ---------------------------------------------------------------------------------------
    results = []
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            old = json.load(f)
            results = old.get("results", [])

    done_ids = {r["row_id"] for r in results}

    # словарь ----------------------------------------------------------------------------------------------------------
    topics_ru = {
        "graphics": "графика",
        "gameplay": "геймплей",
        "bugs": "баги",
        "performance": "производительность",
        "story": "сюжет",
        "controls": "управление",
        "music": "музыка",
        "other": "прочее"
    }
    sentiment_ru = {
        "positive": "положительный",
        "negative": "отрицательный",
        "neutral": "нейтральный"
    }

    for review in reviews:
        idx = review["id"]

        if idx in done_ids:
            print(f"Пропуск обработаного отзыва {idx}")
            continue

        text = review["text"]
        print(f"Обработка отзыва {idx}:")

        res = analyze(text)

        while res is None:
            print("Повторная попытка...")
            time.sleep(10)
            res = analyze(text)

        # вывод -------------------------------------------------------------------------------------------------------
        ru_sentiment = sentiment_ru.get(res["sentiment"], res["sentiment"])
        ru_topics = [topics_ru.get(t, t) for t in res["topics"]]

        print(f"  Тональность: {ru_sentiment}")
        print(f"  Темы: {', '.join(ru_topics)}")
        print(f"  Кратко: {res['summary']}")
        print()

        # результат ------------------------------------------------------------------------------------------------
        results.append({
            "row_id": idx,
            "review": text,
            **res,
            "processed_at": datetime.now().isoformat()
        })

        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump({
                "model": MODEL,
                "total": len(results),
                "results": results
            }, f, ensure_ascii=False, indent=2)

        save_readable_report(results, OUTPUT_TEXT)
        time.sleep(REQUEST_DELAY)

    print("\n"+"="*80)
    print("АНАЛИЗ ЗАВЕРШЁН")
    print("="*80)

    sentiments = [r["sentiment"] for r in results]
    print(f"\nСтатистика по {len(results)} отзывам:")
    print(f"  Положительных: {sentiments.count('positive')}")
    print(f"  Отрицательных: {sentiments.count('negative')}")
    print(f"  Нейтральных:   {sentiments.count('neutral')}")

    print(f"\nРезультаты сохранены в:")
    print(f"  {OUTPUT_JSON} файл JSON")
    print(f"  {OUTPUT_TEXT} файл TXT")


if __name__ == "__main__":
    main()