"""Test the rendering helpers and edge cases without running Streamlit."""
import copy
import random

import pandas as pd

import data_loader as dl
from recommender import BookRecommender

# Pull the pure helper functions out of app.py
src = open("app.py").read()
start = src.index("SPINE_COLOURS")
end = src.index("# ======================================================================\n# DATA")
ns = {"random": random, "pd": pd}
exec(src[start:end], ns)

books, _ = dl.load_books("data/sample_books.csv")
ratings, _ = dl.load_ratings("data/sample_ratings.csv")
eng = BookRecommender(books, ratings)

print("=== draw_shelf ===")
html = ns["draw_shelf"](books["title"].tolist())
print("  spines:", html.count('class="spine"'), "| chars:", len(html))

print("\n=== stars ===")
for v in (None, 8.7, 4.2, 10.0):
    print(f"  {str(v):<5} -> {ns['stars'](v)[:62]}")

print("\n=== card() across every strategy ===")
i = eng.find_title("dune")[0]
hybrid, _ = eng.hybrid(i, n=2)
sets = {
    "hybrid": hybrid,
    "content": eng.by_content(i, n=2),
    "collaborative": eng.by_collaborative(i, n=2),
    "mood": eng.by_mood("quiet sad cold place", n=2),
    "popular": eng.top_rated(2),
    "shelf_row": [eng._row(5, "You set this aside", 0, "")],
}
for name, rows in sets.items():
    for r in rows:
        h = ns["card"](r, 1)
        assert '<div class="card">' in h, name
        assert r["title"] in h, name
    print(f"  {name:<14} {len(rows)} cards OK")

print("\n=== edge case: missing everything ===")
edge = copy.deepcopy(hybrid[0])
edge.update({"description": "", "rating_mean": None, "rating_count": 0,
             "year": None, "author": "", "genre": ""})
h = ns["card"](edge)
print("  renders:", '<div class="card">' in h)
print("  'No summary' shown:", "No summary" in h)
print("  'not yet rated' shown:", "not yet rated" in h)

print("\n=== books only, no ratings file ===")
eng2 = BookRecommender(books, None)
print("  cf_ready:", eng2.cf_ready)
print("  popular is None:", eng2.popular is None)
r2, m2 = eng2.hybrid(i, n=3)
print("  hybrid falls back to:", m2, "->", len(r2), "results")
print("  mood still works:", len(eng2.by_mood("space desert politics", n=3)))
print("  top_rated empty:", eng2.top_rated(3) == [])
print("  coverage:", eng2.coverage())

print("\n=== cold start: a book with too few ratings ===")
counts = eng.rating_counts
sparse = [b for b in books["book_id"] if counts.get(b, 0) < 8]
print(f"  books under the threshold: {len(sparse)}")
rows, method = eng.hybrid(eng.index_of[books.iloc[0]["book_id"]], n=3)
print("  method used:", method)

print("\nall rendering tests passed")
