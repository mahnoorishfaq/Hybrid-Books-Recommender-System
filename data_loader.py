"""
Loading and normalising book data.

Book datasets come in many shapes. Book-Crossing calls the title column
"Book-Title", Goodreads calls it "title", others use "name". Rather than
forcing one format, this module detects the columns and maps them onto a
standard schema.

Standard schema:
    book_id, title, author, year, publisher, image, description
    user_id, book_id, rating
"""

import io
import re

import pandas as pd

BOOK_COLUMNS = {
    "book_id":     ["isbn", "book_id", "bookid", "id", "book-id"],
    "title":       ["book-title", "title", "book_title", "name", "booktitle"],
    "author":      ["book-author", "author", "authors", "book_author", "writer"],
    "year":        ["year-of-publication", "year", "publication_year",
                    "original_publication_year", "publishyear"],
    "publisher":   ["publisher", "publishing_house"],
    "image":       ["image-url-l", "image-url-m", "image_url", "image-url-s",
                    "coverimg", "cover", "img", "image"],
    "description": ["description", "summary", "synopsis", "overview",
                    "book_description", "plot"],
    "genre":       ["genre", "genres", "category", "categories", "subject"],
}

RATING_COLUMNS = {
    "user_id": ["user-id", "user_id", "userid", "uid", "reader_id"],
    "book_id": ["isbn", "book_id", "bookid", "id", "book-id"],
    "rating":  ["book-rating", "rating", "book_rating", "score", "stars"],
}


def _normalise(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def detect_columns(df, mapping):
    """
    Work out which of the dataframe's columns correspond to our schema.
    Returns {standard_name: actual_column_name} for whatever was found.
    """
    lookup = {_normalise(c): c for c in df.columns}
    found = {}
    for standard, candidates in mapping.items():
        for candidate in candidates:
            key = _normalise(candidate)
            if key in lookup:
                found[standard] = lookup[key]
                break
    return found


def read_csv(file_like):
    """
    Read a CSV defensively. Book datasets are notoriously messy: odd
    separators, bad encodings, stray quote characters.
    """
    raw = file_like.read() if hasattr(file_like, "read") else open(file_like, "rb").read()

    for encoding in ("utf-8", "latin-1", "cp1252"):
        for separator in (",", ";", "\t"):
            try:
                df = pd.read_csv(
                    io.BytesIO(raw),
                    encoding=encoding,
                    sep=separator,
                    on_bad_lines="skip",
                    low_memory=False,
                )
                if df.shape[1] > 1:
                    return df, None
            except Exception:
                continue

    return None, ("Could not read that CSV. Check it is a valid comma or "
                  "semicolon separated file.")


def load_books(file_like):
    """Read a books CSV and map it onto the standard schema."""
    df, error = read_csv(file_like)
    if error:
        return None, error

    found = detect_columns(df, BOOK_COLUMNS)

    if "title" not in found:
        return None, ("No title column found. The file needs a column named "
                      "something like 'title', 'Book-Title' or 'name'. "
                      f"Columns found: {', '.join(map(str, df.columns[:8]))}")

    books = pd.DataFrame()
    for standard, actual in found.items():
        books[standard] = df[actual]

    if "book_id" not in books.columns:
        books["book_id"] = books.index.astype(str)
    books["book_id"] = books["book_id"].astype(str).str.strip()

    for column in ("author", "publisher", "description", "genre", "image"):
        if column not in books.columns:
            books[column] = ""
        books[column] = books[column].fillna("").astype(str)

    if "year" in books.columns:
        books["year"] = pd.to_numeric(books["year"], errors="coerce")
    else:
        books["year"] = pd.NA

    books["title"] = books["title"].astype(str).str.strip()
    books = books[books["title"].str.len() > 0]
    books = books.drop_duplicates(subset="book_id", keep="first")
    books = books.reset_index(drop=True)

    return books, None


def load_ratings(file_like):
    """Read a ratings CSV and map it onto the standard schema."""
    df, error = read_csv(file_like)
    if error:
        return None, error

    found = detect_columns(df, RATING_COLUMNS)
    missing = [k for k in ("user_id", "book_id", "rating") if k not in found]
    if missing:
        return None, (f"Ratings file is missing: {', '.join(missing)}. "
                      f"Columns found: {', '.join(map(str, df.columns[:8]))}")

    ratings = pd.DataFrame({
        "user_id": df[found["user_id"]].astype(str),
        "book_id": df[found["book_id"]].astype(str).str.strip(),
        "rating": pd.to_numeric(df[found["rating"]], errors="coerce"),
    }).dropna(subset=["rating"])

    ratings = ratings[ratings["rating"] > 0]

    return ratings.reset_index(drop=True), None


def dataset_summary(books, ratings):
    """Facts about the loaded data, for the interface to display."""
    summary = {
        "books": len(books),
        "has_descriptions": int((books["description"].str.len() > 20).sum()),
        "has_authors": int((books["author"].str.len() > 0).sum()),
        "ratings": 0,
        "users": 0,
        "rated_books": 0,
        "mean_rating": None,
    }
    if ratings is not None and len(ratings):
        summary["ratings"] = len(ratings)
        summary["users"] = ratings["user_id"].nunique()
        summary["rated_books"] = ratings["book_id"].nunique()
        summary["mean_rating"] = round(float(ratings["rating"].mean()), 2)
    return summary
