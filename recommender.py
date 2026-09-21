"""
The recommendation engine.

Three strategies, plus a hybrid that chooses between them:

  content       TF-IDF over title, author, genre and description.
                Works for any book, including ones nobody has rated.
                This is what solves the cold start problem.

  collaborative Item-item similarity over the user-rating matrix.
                "Readers who rated this highly also rated..."
                Only works where enough ratings exist.

  hybrid        Blends the two, weighted by how much rating evidence a
                book actually has. Sparse data leans on content, dense
                data leans on collaborative.

  popular       Weighted rating (the IMDB formula) for when we know
                nothing about the reader yet.

Every recommendation carries a reason, so the interface can explain
itself rather than presenting a black box.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel


class BookRecommender:

    def __init__(self, books, ratings=None, min_ratings_per_book=8,
                 min_ratings_per_user=3):
        self.books = books.reset_index(drop=True)
        self.ratings = ratings
        self.min_ratings_per_book = min_ratings_per_book
        self.min_ratings_per_user = min_ratings_per_user

        self.index_of = {bid: i for i, bid in enumerate(self.books["book_id"])}
        self.title_lookup = {
            str(t).lower().strip(): i
            for i, t in enumerate(self.books["title"])
        }

        self._build_content()
        self._build_collaborative()
        self._build_popularity()

    # ------------------------------------------------------------------
    # content
    # ------------------------------------------------------------------
    def _build_content(self):
        """TF-IDF over everything textual we know about each book."""
        parts = []
        for column, weight in (("title", 2), ("author", 2),
                               ("genre", 2), ("description", 1)):
            if column in self.books.columns:
                text = self.books[column].fillna("").astype(str)
                parts.append(pd.concat([text] * weight, axis=1)
                             .agg(" ".join, axis=1))

        corpus = parts[0]
        for extra in parts[1:]:
            corpus = corpus + " " + extra

        self.corpus = corpus.str.lower()

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=40_000,
            ngram_range=(1, 2),
            min_df=1,
        )
        self.content_matrix = self.vectorizer.fit_transform(self.corpus)

    # ------------------------------------------------------------------
    # collaborative
    # ------------------------------------------------------------------
    def _build_collaborative(self):
        """
        Sparse book x user matrix, restricted to books and users with
        enough activity to be meaningful.
        """
        self.cf_ready = False
        if self.ratings is None or len(self.ratings) < 50:
            return

        r = self.ratings

        book_counts = r["book_id"].value_counts()
        keep_books = book_counts[book_counts >= self.min_ratings_per_book].index
        r = r[r["book_id"].isin(keep_books)]

        user_counts = r["user_id"].value_counts()
        keep_users = user_counts[user_counts >= self.min_ratings_per_user].index
        r = r[r["user_id"].isin(keep_users)]

        if len(r) < 50 or r["book_id"].nunique() < 10:
            return

        self.cf_books = sorted(r["book_id"].unique())
        self.cf_users = sorted(r["user_id"].unique())
        book_pos = {b: i for i, b in enumerate(self.cf_books)}
        user_pos = {u: i for i, u in enumerate(self.cf_users)}

        rows = r["book_id"].map(book_pos).to_numpy()
        cols = r["user_id"].map(user_pos).to_numpy()
        vals = r["rating"].astype(float).to_numpy()

        self.cf_matrix = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(self.cf_books), len(self.cf_users)),
        )
        self.cf_index = book_pos
        self.rating_counts = r["book_id"].value_counts().to_dict()
        self.cf_ready = True

    # ------------------------------------------------------------------
    # popularity
    # ------------------------------------------------------------------
    def _build_popularity(self):
        """
        Weighted rating, the IMDB formula:

            WR = (v / (v + m)) * R  +  (m / (v + m)) * C

        A book with 3 perfect scores should not outrank one with 400
        ratings averaging 8.5. This pulls low-count books toward the
        global mean.
        """
        self.popular = None
        if self.ratings is None or len(self.ratings) < 20:
            return

        stats = (self.ratings.groupby("book_id")["rating"]
                 .agg(["count", "mean"]).reset_index())
        stats.columns = ["book_id", "rating_count", "rating_mean"]

        C = stats["rating_mean"].mean()
        m = max(stats["rating_count"].quantile(0.90), 5)

        v = stats["rating_count"]
        R = stats["rating_mean"]
        stats["weighted"] = (v / (v + m)) * R + (m / (v + m)) * C

        self.popular = (stats.merge(self.books, on="book_id", how="inner")
                             .sort_values("weighted", ascending=False)
                             .reset_index(drop=True))
        self.global_mean = float(C)
        self.vote_threshold = float(m)

    # ------------------------------------------------------------------
    # lookups
    # ------------------------------------------------------------------
    def find_title(self, query, limit=8):
        """Loose title search, for the picker."""
        q = str(query).lower().strip()
        if not q:
            return []
        titles = self.books["title"].str.lower()
        exact = self.books.index[titles == q].tolist()
        starts = self.books.index[titles.str.startswith(q)].tolist()
        contains = self.books.index[titles.str.contains(q, regex=False)].tolist()

        ordered, seen = [], set()
        for group in (exact, starts, contains):
            for i in group:
                if i not in seen:
                    seen.add(i)
                    ordered.append(i)
                if len(ordered) >= limit:
                    return ordered
        return ordered

    def _row(self, i, reason, score, method):
        book = self.books.iloc[i]
        book_id = book["book_id"]
        count = getattr(self, "rating_counts", {}).get(book_id, 0)

        mean = None
        if self.ratings is not None:
            subset = self.ratings[self.ratings["book_id"] == book_id]["rating"]
            if len(subset):
                mean = round(float(subset.mean()), 1)

        return {
            "index": int(i),
            "book_id": book_id,
            "title": book["title"],
            "author": book.get("author", ""),
            "year": book.get("year"),
            "publisher": book.get("publisher", ""),
            "description": book.get("description", ""),
            "genre": book.get("genre", ""),
            "image": book.get("image", ""),
            "rating_mean": mean,
            "rating_count": int(count),
            "score": round(float(score), 3),
            "reason": reason,
            "method": method,
        }

    # ------------------------------------------------------------------
    # strategies
    # ------------------------------------------------------------------
    def by_content(self, book_index, n=8):
        """Books that resemble this one in subject and style."""
        sims = linear_kernel(self.content_matrix[book_index],
                             self.content_matrix).ravel()
        sims[book_index] = -1
        order = np.argsort(sims)[::-1][:n]

        source_title = self.books.iloc[book_index]["title"]
        out = []
        for i in order:
            if sims[i] <= 0:
                continue
            out.append(self._row(
                i,
                f"Shares subject matter and language with {source_title}",
                sims[i], "content"))
        return out

    def by_collaborative(self, book_index, n=8):
        """Books rated similarly by the same readers."""
        if not self.cf_ready:
            return []

        book_id = self.books.iloc[book_index]["book_id"]
        if book_id not in self.cf_index:
            return []

        position = self.cf_index[book_id]
        sims = cosine_similarity(self.cf_matrix[position], self.cf_matrix).ravel()
        sims[position] = -1
        order = np.argsort(sims)[::-1][: n * 3]

        out = []
        for pos in order:
            if sims[pos] <= 0:
                continue
            other_id = self.cf_books[pos]
            if other_id not in self.index_of:
                continue
            shared = self.rating_counts.get(other_id, 0)
            out.append(self._row(
                self.index_of[other_id],
                f"Readers who rated this book highly also rated it "
                f"({shared} ratings)",
                sims[pos], "collaborative"))
            if len(out) >= n:
                break
        return out

    def hybrid(self, book_index, n=8):
        """
        Blend both signals. The weighting is not fixed: it depends on how
        much rating evidence the source book actually has. A book with
        four ratings should not have its neighbours decided by those four
        readers.
        """
        book_id = self.books.iloc[book_index]["book_id"]
        evidence = getattr(self, "rating_counts", {}).get(book_id, 0)

        if not self.cf_ready or evidence < self.min_ratings_per_book:
            results = self.by_content(book_index, n)
            for r in results:
                r["method"] = "content (cold start)"
            return results, "content"

        # Confidence in the collaborative signal grows with the number of
        # ratings, flattening out around 60.
        cf_weight = min(evidence / 60.0, 0.75)
        content_weight = 1 - cf_weight

        scores = {}
        for r in self.by_content(book_index, n * 2):
            scores[r["index"]] = {
                "row": r,
                "score": r["score"] * content_weight,
                "content": r["score"],
                "cf": 0.0,
            }
        for r in self.by_collaborative(book_index, n * 2):
            entry = scores.get(r["index"])
            if entry:
                entry["score"] += r["score"] * cf_weight
                entry["cf"] = r["score"]
            else:
                scores[r["index"]] = {
                    "row": r,
                    "score": r["score"] * cf_weight,
                    "content": 0.0,
                    "cf": r["score"],
                }

        ranked = sorted(scores.values(), key=lambda e: e["score"], reverse=True)

        out = []
        for entry in ranked[:n]:
            row = dict(entry["row"])
            row["score"] = round(entry["score"], 3)
            if entry["cf"] > 0 and entry["content"] > 0:
                row["reason"] = ("Similar in subject, and rated highly by the "
                                 "same readers")
                row["method"] = "hybrid"
            elif entry["cf"] > 0:
                row["method"] = "collaborative"
            else:
                row["method"] = "content"
            out.append(row)

        return out, f"hybrid ({int(cf_weight*100)}% collaborative)"

    def by_mood(self, text, n=8):
        """
        Free-text search over the content space. Describe what you feel
        like reading and this finds the closest books.
        """
        if not text.strip():
            return []
        vector = self.vectorizer.transform([text.lower()])
        sims = linear_kernel(vector, self.content_matrix).ravel()
        order = np.argsort(sims)[::-1][:n]

        out = []
        for i in order:
            if sims[i] <= 0:
                continue
            out.append(self._row(
                i, "Matches what you described", sims[i], "mood"))
        return out

    def top_rated(self, n=10):
        """Popular fallback for a reader we know nothing about."""
        if self.popular is None:
            return []
        out = []
        for _, row in self.popular.head(n).iterrows():
            i = self.index_of.get(row["book_id"])
            if i is None:
                continue
            entry = self._row(
                i,
                f"Weighted rating {row['weighted']:.2f} from "
                f"{int(row['rating_count'])} ratings",
                row["weighted"], "popularity")
            entry["rating_mean"] = round(float(row["rating_mean"]), 1)
            entry["rating_count"] = int(row["rating_count"])
            out.append(entry)
        return out

    # ------------------------------------------------------------------
    # insight
    # ------------------------------------------------------------------
    def coverage(self):
        """
        How much of the library collaborative filtering can actually
        serve. This number is the argument for building a hybrid.
        """
        total = len(self.books)
        if not self.cf_ready:
            return {"total": total, "covered": 0, "percent": 0.0}
        covered = sum(1 for b in self.books["book_id"] if b in self.cf_index)
        return {
            "total": total,
            "covered": covered,
            "percent": round(covered / total * 100, 1) if total else 0.0,
        }
