import json
import os
import time

import wikipedia

ARTICLES = [
    "Artificial intelligence",
    "World War II",
    "Climate change",
    "Python (programming language)",
    "Renaissance",
]


def fetch_articles(articles: list, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results = []

    for title in articles:
        print(f"Fetching: {title}...")
        try:
            page = wikipedia.page(title, auto_suggest=False)
            results.append({
                "title": page.title,
                "url": page.url,
                "content": page.content,
                "char_count": len(page.content),
                "word_count": len(page.content.split()),
            })
            print(f"  Done — {len(page.content):,} chars, {len(page.content.split()):,} words")
            time.sleep(0.5)  # polite delay between requests

        except wikipedia.exceptions.DisambiguationError as e:
            print(f"  Disambiguation error for '{title}': {e.options[:3]}")
        except wikipedia.exceptions.PageError:
            print(f"  Page not found: {title}")
        except Exception as e:
            print(f"  Error fetching '{title}': {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(results)} articles to {output_path}")
    total_chars = sum(a["char_count"] for a in results)
    total_words = sum(a["word_count"] for a in results)
    print(f"Total: {total_chars:,} chars | {total_words:,} words")


if __name__ == "__main__":
    fetch_articles(ARTICLES, "data/articles.json")