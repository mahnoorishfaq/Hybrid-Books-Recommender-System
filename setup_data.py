"""
Load your dataset into the app's backend.

Run this BEFORE starting the app:

    python setup_data.py

A file picker opens. Choose your books CSV first, then your ratings CSV
(press Cancel to skip ratings). The files are copied into the data/
folder, which is where the app reads from. Nothing appears in the
website itself.
"""

import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox

import data_loader as dl

DATA_DIR = "data"
BOOKS_TARGET = os.path.join(DATA_DIR, "Books.csv")
RATINGS_TARGET = os.path.join(DATA_DIR, "Ratings.csv")


def choose(root, title):
    return filedialog.askopenfilename(
        parent=root,
        title=title,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )


def copy_into_place(source, target):
    if os.path.abspath(source) == os.path.abspath(target):
        return  
    shutil.copyfile(source, target)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    # ---- books (required) ----
    books_file = choose(root, "Step 1 of 2 - choose your BOOKS file (e.g. Books.csv)")
    if not books_file:
        print("No books file chosen. Nothing was changed.")
        return

    print("Checking the books file...")
    books, error = dl.load_books(books_file)
    if error:
        messagebox.showerror("Problem with the books file", error, parent=root)
        return
    copy_into_place(books_file, BOOKS_TARGET)
    print(f"  {len(books):,} books saved to {BOOKS_TARGET}")

    # ---- ratings (optional) ----
    ratings_file = choose(root, "Step 2 of 2 - choose your RATINGS file (Cancel to skip)")
    rating_note = "no ratings (content-based recommendations only)"

    if ratings_file:
        print("Checking the ratings file...")
        ratings, error = dl.load_ratings(ratings_file)
        if error:
            messagebox.showwarning("Problem with the ratings file",
                                   error + "\n\nContinuing without ratings.",
                                   parent=root)
        else:
            copy_into_place(ratings_file, RATINGS_TARGET)
            rating_note = f"{len(ratings):,} ratings"
            print(f"  {len(ratings):,} ratings saved to {RATINGS_TARGET}")
    elif os.path.exists(RATINGS_TARGET):

        os.remove(RATINGS_TARGET)
        print("  Removed old ratings file (it belonged to a different dataset)")

    messagebox.showinfo(
        "Library ready",
        f"{len(books):,} books and {rating_note}.\n\n"
        "Now start the app:\n    streamlit run app.py",
        parent=root,
    )
    print("\nDone. Start the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
