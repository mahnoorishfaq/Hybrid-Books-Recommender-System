"""
The Shelf - a book recommender.

Design: a library card catalog. Recommendations arrive as catalogue
cards with call numbers, on a dark walnut ground. The shelf across the
top is drawn in CSS from the actual loaded books, so it changes with
your data.
"""

import html
import os
import random
import pandas as pd
import streamlit as st

import data_loader as dl
from recommender import BookRecommender

st.set_page_config(
    page_title="Hybrid Book Recommender System",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# STYLE
# ======================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,500&family=Public+Sans:wght@400;500;600&display=swap');

:root{
  --wood:#131A2A; --parchment:#F3E9D6; --ink:#2A1E16; --muted:#BFAF95;
  --gold:#D9A441; --gold-hi:#F2CF7A; --gold-lo:#A9762A;
  --emerald:#2E6B4F; --sapphire:#2C4A7A; --ruby:#8E2F3B; --plum:#5B3A63;
  --rule:rgba(217,164,65,.22);
}

/* remove Streamlit's own toolbar: Deploy / Run / Stop */
[data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stAppDeployButton"],
.stAppDeployButton, #MainMenu, footer { display:none !important; }
header[data-testid="stHeader"]{ background:transparent; }

/* the reading room: lamplight on dark wood */
.stApp{
  background:
    radial-gradient(900px 440px at 15% -10%, rgba(242,180,90,.22), transparent 70%),
    radial-gradient(700px 400px at 95% 5%, rgba(46,107,79,.25), transparent 70%),
    radial-gradient(600px 500px at 50% 110%, rgba(91,58,99,.22), transparent 70%),
    repeating-linear-gradient(90deg, rgba(255,255,255,.012) 0 2px, transparent 2px 11px),
    var(--wood);
  background-attachment:fixed;
}
html,body,[class*="css"]{ font-family:'Public Sans',system-ui,sans-serif; color:var(--parchment); }
h1,h2,h3,h4{ font-family:'Fraunces',Georgia,serif; color:var(--parchment); }
.block-container{ padding-top:1.6rem; max-width:1240px; }

/* gold-leaf lettering */
.wordmark{
  font-family:'Fraunces',serif; font-weight:700; font-size:1.55rem; line-height:1.08;
  background:linear-gradient(180deg,var(--gold-hi) 0%,var(--gold) 55%,var(--gold-lo) 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.wordmark em{ font-style:italic; font-weight:500; }
.wordmark.big{ font-size:clamp(2rem,3.6vw,2.9rem); }
.eyebrow{ font-size:.74rem; letter-spacing:.18em; text-transform:uppercase; color:#8FC7A8; margin-bottom:6px; }

/* the shelf: jewel-toned leather with gold bands */
.shelf-wrap{ margin:0 0 4px; }
.shelf{
  display:flex; align-items:flex-end; gap:3px; height:104px; padding:0 12px;
  border-bottom:11px solid #5A3F2C;
  box-shadow:0 10px 0 -3px #3B281C, 0 26px 32px -16px rgba(0,0,0,.9);
  border-radius:3px; overflow:hidden;
}
.spine{ position:relative; flex-shrink:0; border-radius:2px 2px 0 0;
  box-shadow:inset -3px 0 0 rgba(0,0,0,.28), inset 2px 0 0 rgba(255,255,255,.08); }
.spine::before,.spine::after{ content:""; position:absolute; left:0; right:0; height:3px;
  background:linear-gradient(90deg,var(--gold-lo),var(--gold-hi),var(--gold-lo)); opacity:.9; }
.spine::before{ top:10px; }
.spine::after{ bottom:12px; }
.spine:not(.lean){ transition:transform .2s ease; }
.spine:not(.lean):hover{ transform:translateY(-8px); }
.spine.lean{ transform:rotate(-10deg); transform-origin:bottom right; margin:0 6px 0 14px; }

/* masthead */
.masthead{ display:flex; align-items:flex-end; justify-content:space-between; gap:22px; flex-wrap:wrap;
  padding:22px 2px 16px; border-bottom:1px solid var(--rule); margin-bottom:24px; }
.tagline{ color:var(--muted); font-size:.98rem; max-width:56ch; margin-top:10px; line-height:1.55; }
.figure{ display:flex; gap:10px; flex-wrap:wrap; }
.figure div{ background:rgba(255,255,255,.035); border:1px solid var(--rule); border-radius:6px; padding:10px 14px; min-width:98px; }
.figure div span{ display:block; font-family:'Fraunces',serif; font-weight:700; font-size:1.5rem; line-height:1.1; }
.figure div:nth-child(1) span{ color:var(--gold-hi); }
.figure div:nth-child(2) span{ color:#8FC7A8; }
.figure div:nth-child(3) span{ color:#9DB6E0; }
.figure div:nth-child(4) span{ color:#E4A0A8; }
.figure div small{ color:var(--muted); font-size:.74rem; }

/* section headings with a fleuron */
.section{ display:flex; align-items:center; gap:12px; font-family:'Fraunces',serif; font-weight:600;
  font-size:1.32rem; color:var(--parchment); margin:26px 0 16px; }
.section::before{ content:"❦"; color:var(--gold); font-size:1.05rem; }
.section::after{ content:""; flex:1; height:1px; background:linear-gradient(90deg,var(--rule),transparent); }

/* catalogue cards, edge colour set by the model that chose the book */
.card{
  position:relative; color:var(--ink); border-radius:4px; margin-bottom:14px;
  padding:16px 18px 14px 22px; border-left:6px solid var(--ruby);
  background:
    radial-gradient(120% 90% at 0% 0%, rgba(255,255,255,.6), transparent 60%),
    linear-gradient(170deg,#F7EEDD 0%,#E9DAC0 100%);
  box-shadow:0 1px 0 rgba(0,0,0,.25), 0 16px 28px -18px rgba(0,0,0,.95);
  transition:transform .18s ease, box-shadow .18s ease;
}
.card:hover{ transform:translateY(-3px); box-shadow:0 1px 0 rgba(0,0,0,.25), 0 24px 36px -18px rgba(0,0,0,.95); }
.card:has(.tag.hybrid){ border-left-color:var(--gold); }
.card:has(.tag.content){ border-left-color:var(--emerald); }
.card:has(.tag.collaborative){ border-left-color:var(--sapphire); }
.card:has(.tag.mood){ border-left-color:var(--plum); }
.card:has(.tag.popularity){ border-left-color:var(--ruby); }
.call{ font-size:.68rem; letter-spacing:.1em; font-weight:600; color:#9A7A45; margin-bottom:2px; }
.card h4{ font-family:'Fraunces',serif; font-weight:600; color:var(--ink); font-size:1.16rem; line-height:1.25; margin:0 0 2px; }
.byline{ font-size:.87rem; color:#6B5B45; margin-bottom:12px; }
.blurb{ font-size:.92rem; line-height:1.55; color:#3A2E22; margin:8px 0 12px; }
.card-foot{ display:flex; align-items:center; gap:12px; flex-wrap:wrap; border-top:1px solid rgba(42,30,22,.14); padding-top:10px; }
.stars{ color:#B07A12; font-size:.95rem; letter-spacing:1px; }
.votes{ color:#6B5B45; font-size:.8rem; }
.why{ font-size:.81rem; color:#5A4A36; font-style:italic; flex:1 1 220px; min-width:180px; }
.tag{ font-size:.7rem; font-weight:600; letter-spacing:.05em; padding:2px 8px; border-radius:3px; }
.tag.hybrid{ background:#F0DDAE; color:#6A4A12; }
.tag.content{ background:#CFE5D8; color:#1F4D38; }
.tag.collaborative{ background:#D2DDF1; color:#223E6A; }
.tag.mood{ background:#E6D6EB; color:#4E2F57; }
.tag.popularity{ background:#F3D3D7; color:#7A2430; }

/* panels and small text */
.panel{ background:rgba(255,255,255,.035); border:1px solid var(--rule); border-radius:6px; padding:18px 20px; margin-bottom:16px; }
.panel h3{ font-size:1.06rem; margin:0 0 6px; color:var(--gold-hi); }
.panel p{ color:var(--muted); font-size:.9rem; margin:0; line-height:1.6; }
.empty{ border:1px dashed var(--rule); border-radius:6px; padding:40px 24px; text-align:center; color:var(--muted); background:rgba(255,255,255,.02); }
.fine{ color:var(--muted); font-size:.84rem; line-height:1.55; }

/* tabs as leather labels */
.stTabs [data-baseweb="tab-list"]{ gap:8px; border-bottom:1px solid var(--rule); }
.stTabs [data-baseweb="tab"]{
  padding:8px 18px !important; margin:0 !important;
  background:rgba(255,255,255,.05) !important;
  border:1px solid var(--rule) !important; border-bottom:none !important;
  border-radius:8px 8px 0 0 !important;
}
.stTabs [data-baseweb="tab"] p{ font-family:'Fraunces',serif; font-size:.98rem; color:var(--muted); }
.stTabs [data-baseweb="tab"]:hover p{ color:var(--gold-hi); }
.stTabs [aria-selected="true"]{ background:linear-gradient(180deg,var(--gold-hi),var(--gold)) !important; border-color:var(--gold) !important; }
.stTabs [aria-selected="true"] p{ color:#2A1B08 !important; font-weight:600; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{ display:none; }

/* gold-leaf buttons */
.stButton>button, .stDownloadButton>button{
  background:linear-gradient(180deg,var(--gold-hi),var(--gold)); border:none; border-radius:5px;
  font-family:'Fraunces',serif; font-weight:600; padding:.45rem 1.1rem;
  box-shadow:0 2px 0 #8A6121; transition:transform .12s ease, filter .12s ease;
}
.stButton>button p, .stDownloadButton>button p{ color:#2A1B08; }
.stButton>button:hover, .stDownloadButton>button:hover{ filter:brightness(1.07); transform:translateY(-1px); }

/* inputs */
.stTextInput input, .stTextArea textarea{
  background:rgba(255,255,255,.045) !important; color:var(--parchment) !important; border-radius:6px !important;
}

/* sidebar: green reading-room felt and a ruled library card */
section[data-testid="stSidebar"]{ background:linear-gradient(180deg,#1A2438 0%,#0F1522 70%); border-right:1px solid var(--rule); }
.libcard{
  color:var(--ink); border-radius:4px; padding:12px 14px 8px; margin:14px 0 18px;
  background-color:#F3E9D6; border-top:5px solid var(--ruby);
  background-image:repeating-linear-gradient(180deg, transparent 0 22px, rgba(142,47,59,.22) 22px 23px);
  background-position:0 30px; box-shadow:0 12px 22px -12px rgba(0,0,0,.9);
}
.libcard h5{ font-family:'Fraunces',serif; font-size:.86rem; letter-spacing:.14em; color:var(--ruby); margin:0 0 4px; }
.libcard .row{ display:flex; justify-content:space-between; font-size:.84rem; line-height:23px; }

@media (prefers-reduced-motion: reduce){
  .card,.spine,.stButton>button{ transition:none !important; }
  .card:hover,.spine:not(.lean):hover{ transform:none !important; }
}
</style>
""", unsafe_allow_html=True)


# ======================================================================
# THE SHELF GRAPHIC
# ======================================================================
SPINE_COLOURS = ["#8E2F3B", "#2E6B4F", "#2C4A7A", "#5B3A63", "#A0662A",
                 "#1F5C5C", "#6E2A2A", "#3B4F8A", "#4F6B2E", "#7A3D6B"]


def draw_shelf(titles, seed=7):
    rng = random.Random(seed)
    lean_at = rng.randint(8, 30)
    spines = []
    for i, title in enumerate(titles[:48]):
        colour = SPINE_COLOURS[rng.randrange(len(SPINE_COLOURS))]
        width = rng.randint(12, 28)
        height = rng.randint(58, 90)
        lean = " lean" if i == lean_at else ""
        spines.append(
            f'<div class="spine{lean}" style="width:{width}px;height:{height}px;'
            f'background:linear-gradient(90deg,rgba(0,0,0,.25) 0%,{colour} 22%,'
            f'{colour} 70%,rgba(0,0,0,.35) 100%)" '
            f'title="{html.escape(str(title)[:60])}"></div>'
        )
    return (f'<div class="shelf-wrap"><div class="shelf">'
            f'{"".join(spines)}</div></div>')


def stars(value):
    """Render a 0-10 rating as five characters."""
    if value is None:
        return '<span class="votes">not yet rated</span>'
    filled = int(round(value / 2))
    return (f'<span class="stars">{"★" * filled}{"☆" * (5 - filled)}</span>'
            f'<span class="votes">{value}/10</span>')


def card(book, position=None):
    """One catalogue card. Built as one line of HTML on purpose: a blank
    line inside HTML makes Streamlit print the rest as raw code.
    Parts with no data are left out instead of showing a placeholder."""
    esc = lambda v: html.escape(html.unescape(str(v)))

    parts = []
    if position:
        parts.append(f"№ {position:02d}")
    genre = str(book.get("genre") or "").strip()
    if genre:
        parts.append(genre[:3].upper())
    author = str(book.get("author") or "").strip()
    if author:
        parts.append(author[:3].upper())
    year = book.get("year")
    try:
        if pd.notna(year) and int(float(year)) > 0:
            parts.append(str(int(float(year))))
    except (ValueError, TypeError):
        pass

    pieces = ['<div class="card">']
    if parts:
        pieces.append(f'<div class="call">{esc(" · ".join(parts))}</div>')
    pieces.append(f'<h4>{esc(book["title"])}</h4>')
    if author:
        pieces.append(f'<div class="byline">{esc(author)}</div>')

    blurb = str(book.get("description") or "").strip()
    if blurb:
        if len(blurb) > 340:
            blurb = blurb[:337].rsplit(" ", 1)[0] + "…"
        pieces.append(f'<div class="blurb">{esc(blurb)}</div>')

    foot = []
    if book.get("rating_mean") is not None:
        foot.append(stars(book["rating_mean"]))
    if book.get("rating_count"):
        foot.append(f'<span class="votes">{book["rating_count"]} ratings</span>')
    method_text = str(book.get("method") or "")
    method = method_text.split(" ")[0]
    if method:
        foot.append(f'<span class="tag {method}">{esc(method_text)}</span>')
    if book.get("reason"):
        foot.append(f'<span class="why">{esc(book["reason"])}</span>')
    if foot:
        pieces.append('<div class="card-foot">' + "".join(foot) + '</div>')

    pieces.append('</div>')
    return "".join(pieces)

# ======================================================================
# DATA — loaded from the project folder, not uploaded by the user
# ======================================================================
BOOKS_PATH = "data/Books.csv"
RATINGS_PATH = "data/Ratings.csv"

SAMPLE_BOOKS = "data/sample_books.csv"
SAMPLE_RATINGS = "data/sample_ratings.csv"


@st.cache_resource(show_spinner=False)
def load_library(books_path, ratings_path, stamp):
    """Read the dataset once and build the models. Cached across reruns."""
    books, error = dl.load_books(books_path)
    if error:
        return None, None, None, error
    ratings = None
    if os.path.exists(ratings_path):
        ratings, rating_error = dl.load_ratings(ratings_path)
        if rating_error:
            ratings = None
    engine = BookRecommender(books, ratings)
    return books, ratings, engine, None


if os.path.exists(BOOKS_PATH):
    books_path, ratings_path, source = BOOKS_PATH, RATINGS_PATH, "full"
else:
    books_path, ratings_path, source = SAMPLE_BOOKS, SAMPLE_RATINGS, "sample"

with st.spinner("Opening the library… the first load takes a minute on the full dataset"):
    stamp = tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in (books_path, ratings_path))
    books, ratings, engine, error_message = load_library(books_path, ratings_path, stamp)

if error_message:
    st.error(error_message)
    st.stop()

if "shelf" not in st.session_state:
    st.session_state.shelf = []

summary = dl.dataset_summary(books, ratings)
coverage = engine.coverage()


# ---------------- sidebar: library card ----------------
with st.sidebar:
    st.markdown('<div class="wordmark">Hybrid <em>Book</em><br>Recommender</div>',
                unsafe_allow_html=True)

    collection = "Full dataset" if source == "full" else "Sample library"
    st.markdown(f"""
    <div class="libcard">
      <h5>LIBRARY CARD</h5>
      <div class="row"><span>Collection</span><b>{collection}</b></div>
      <div class="row"><span>Books</span><b>{summary['books']:,}</b></div>
      <div class="row"><span>Ratings</span><b>{summary['ratings']:,}</b></div>
      <div class="row"><span>Readers</span><b>{summary['users']:,}</b></div>
      <div class="row"><span>Collaborative reach</span><b>{coverage['percent']}%</b></div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("✦ Surprise me"):
        if engine.popular is not None and len(engine.popular):
            pick = engine.popular.head(300).sample(1).iloc[0]["book_id"]
            st.session_state.surprise = engine.index_of.get(pick)
        else:
            st.session_state.surprise = random.randrange(len(books))

    if st.session_state.get("surprise") is not None:
        st.markdown(card(engine._row(st.session_state.surprise,
                                     "Pulled from the stacks at random", 0, "")),
                    unsafe_allow_html=True)

    with st.expander("How the recommendations work"):
        st.markdown(
            "**Content-based** TF-IDF over title, author, genre and "
            "description. Works even for books nobody has rated.\n\n"
            "**Collaborative** books rated highly by the same readers.\n\n"
            "**Hybrid** blends both, trusting ratings more when a book has "
            "more of them. This is how rarely-rated books still get "
            "recommended.\n\n"
            "**Most loved** weighted rating, so a few perfect scores "
            "can't beat hundreds of good ones."
        )


# ======================================================================
# HEADER
# ======================================================================
shelf_titles = books["title"].sample(min(48, len(books)), random_state=11).tolist()
st.markdown(draw_shelf(shelf_titles), unsafe_allow_html=True)

st.markdown(f"""
<div class="masthead">
  <div>
    <div class="eyebrow">A reading room for your next book</div>
    <div class="wordmark big">Hybrid <em>Book</em> Recommender</div>
    <div class="tagline">Name a book you loved, or describe the mood you're in. Four models read the whole catalogue, and every suggestion comes with the reason it was chosen.</div>
  </div>
  <div class="figure">
    <div><span>{summary['books']:,}</span><small>books</small></div>
    <div><span>{summary['ratings']:,}</span><small>ratings</small></div>
    <div><span>{summary['users']:,}</span><small>readers</small></div>
    <div><span>{coverage['percent']}%</span><small>collaborative reach</small></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ======================================================================
# TABS
# ======================================================================
tab_book, tab_mood, tab_top, tab_shelf, tab_data = st.tabs([
    "From a book you loved", "By mood", "Most loved",
    f"My shelf ({len(st.session_state.shelf)})", "The data",
])


# ---------------------------------------------------------------- book
with tab_book:
    left, right = st.columns([2, 1])

    with left:
        query = st.text_input(
            "A book you loved",
            placeholder="Start typing a title…",
            key="title_query",
        )

    matches = engine.find_title(query) if query else []

    with right:
        strategy = st.selectbox(
            "Method",
            ["Hybrid (recommended)", "Content only", "Collaborative only"],
            help="Hybrid weighs the two signals by how much rating "
                 "evidence the book actually has.",
        )

    if query and not matches:
        st.markdown('<div class="empty">No title matches that. Try fewer '
                    'words, or check the spelling.</div>',
                    unsafe_allow_html=True)

    elif matches:
        options = {
            f"{books.iloc[i]['title']} — {books.iloc[i]['author'] or 'Unknown'}": i
            for i in matches
        }
        chosen_label = st.radio("Which one?", list(options),
                                horizontal=False, key="which_book")
        chosen = options[chosen_label]

        source = books.iloc[chosen]
        st.markdown(f'<div class="section">Because you liked '
                    f'{source["title"]}</div>', unsafe_allow_html=True)

        if strategy.startswith("Hybrid"):
            results, method_note = engine.hybrid(chosen, n=8)
            st.markdown(f'<p class="fine">Method: {method_note}. '
                        f'This book carries '
                        f'{engine.rating_counts.get(source["book_id"], 0) if engine.cf_ready else 0} '
                        f'ratings, which is what set that balance.</p>',
                        unsafe_allow_html=True)
        elif strategy.startswith("Content"):
            results = engine.by_content(chosen, n=8)
        else:
            results = engine.by_collaborative(chosen, n=8)
            if not results:
                st.warning("Not enough ratings for this book to use "
                           "collaborative filtering. This is the cold start "
                           "problem try Hybrid, which falls back to "
                           "content.")

        cols = st.columns(2)
        for n, book in enumerate(results):
            with cols[n % 2]:
                st.markdown(card(book, n + 1), unsafe_allow_html=True)
                if st.button("Add to my shelf", key=f"add_b_{book['index']}"):
                    if book["index"] not in st.session_state.shelf:
                        st.session_state.shelf.append(book["index"])
                        st.rerun()
    else:
        st.markdown('<div class="empty">Type a title above to begin. '
                    'Try Harry Potter, The Da Vinci Code or The Lovely '
                    'Bones.</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- mood
with tab_mood:
    st.markdown('<div class="section">Describe what you feel like '
                'reading</div>', unsafe_allow_html=True)
    st.markdown('<p class="fine">Plain language. Themes, setting, tone, '
                'subject whatever comes to mind. This searches the '
                'content space rather than matching keywords in titles.</p>',
                unsafe_allow_html=True)

    mood = st.text_area(
        "Mood", height=90, label_visibility="collapsed",
        placeholder="something quiet and sad set in a cold place\n"
                    "a clever murder with an unreliable narrator\n"
                    "practical book to actually learn machine learning",
    )

    if st.button("Find these books", key="mood_go"):
        results = engine.by_mood(mood, n=8)
        if not results:
            st.markdown('<div class="empty">Nothing matched. Try '
                        'describing it differently, or using more words.'
                        '</div>', unsafe_allow_html=True)
        else:
            cols = st.columns(2)
            for n, book in enumerate(results):
                with cols[n % 2]:
                    st.markdown(card(book, n + 1), unsafe_allow_html=True)
                    if st.button("Add to my shelf", key=f"add_m_{book['index']}"):
                        if book["index"] not in st.session_state.shelf:
                            st.session_state.shelf.append(book["index"])
                            st.rerun()


# ----------------------------------------------------------------- top
with tab_top:
    st.markdown('<div class="section">Most loved in this library</div>',
                unsafe_allow_html=True)

    if engine.popular is None:
        st.markdown('<div class="empty">This needs a ratings file. Upload '
                    'one in the sidebar to see weighted rankings.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f"""<p class="fine">Ranked by weighted rating, not raw
        average. Three perfect scores should not outrank four hundred
        ratings averaging 8.5, so each book is pulled toward the library
        mean of {engine.global_mean:.2f} in proportion to how few ratings
        it has. The threshold here is {engine.vote_threshold:.0f} ratings.</p>""",
        unsafe_allow_html=True)

        cols = st.columns(2)
        for n, book in enumerate(engine.top_rated(10)):
            with cols[n % 2]:
                st.markdown(card(book, n + 1), unsafe_allow_html=True)
                if st.button("Add to my shelf", key=f"add_t_{book['index']}"):
                    if book["index"] not in st.session_state.shelf:
                        st.session_state.shelf.append(book["index"])
                        st.rerun()


# --------------------------------------------------------------- shelf
with tab_shelf:
    st.markdown('<div class="section">Books you set aside</div>',
                unsafe_allow_html=True)

    if not st.session_state.shelf:
        st.markdown('<div class="empty">Nothing here yet. Add books from '
                    'any tab and they collect here for the session.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(draw_shelf(
            [books.iloc[i]["title"] for i in st.session_state.shelf], seed=3),
            unsafe_allow_html=True)

        cols = st.columns(2)
        for n, index in enumerate(list(st.session_state.shelf)):
            entry = engine._row(index, "You set this aside", 0, "")
            with cols[n % 2]:
                st.markdown(card(entry, n + 1), unsafe_allow_html=True)
                if st.button("Remove", key=f"rm_{index}"):
                    st.session_state.shelf.remove(index)
                    st.rerun()

        export = books.iloc[st.session_state.shelf][
            ["title", "author", "year", "genre"]]
        st.download_button(
            "Download as CSV", export.to_csv(index=False),
            "my-shelf.csv", "text/csv",
        )


# ---------------------------------------------------------------- data
with tab_data:
    st.markdown('<div class="section">What the recommender is working '
                'with</div>', unsafe_allow_html=True)

    a, b, c = st.columns(3)
    with a:
        st.markdown(f"""<div class="panel"><h3>Catalogue</h3>
        <p>{summary['books']:,} books<br>
        {summary['has_authors']:,} with an author<br>
        {summary['has_descriptions']:,} with a usable description</p></div>""",
        unsafe_allow_html=True)
    with b:
        mean_text = (f"{summary['mean_rating']} average"
                     if summary["mean_rating"] else "no ratings loaded")
        st.markdown(f"""<div class="panel"><h3>Ratings</h3>
        <p>{summary['ratings']:,} ratings<br>
        {summary['users']:,} readers<br>
        {mean_text}</p></div>""", unsafe_allow_html=True)
    with c:
        st.markdown(f"""<div class="panel"><h3>Collaborative reach</h3>
        <p>{coverage['covered']:,} of {coverage['total']:,} books have
        enough ratings<br>
        <strong>{coverage['percent']}% coverage</strong></p></div>""",
        unsafe_allow_html=True)

    st.markdown(f"""<p class="fine">That coverage figure is the whole
    argument for the hybrid. Collaborative filtering only works for books
    with enough ratings here, {coverage['percent']}% of the catalogue.
    Every other book is invisible to it. Content-based similarity reaches
    100%, because it only needs the text. The hybrid uses whichever signal
    a given book actually supports.</p>""", unsafe_allow_html=True)

    if ratings is not None and len(ratings):
        st.markdown('<div class="section">How ratings are '
                    'distributed</div>', unsafe_allow_html=True)

        per_book = ratings["book_id"].value_counts()
        d1, d2 = st.columns(2)
        with d1:
            st.markdown('<p class="fine">Ratings per book. Most datasets '
                        'have a long tail of books rated only once or '
                        'twice.</p>', unsafe_allow_html=True)
            dist = per_book.value_counts().sort_index().head(30)
            dist.index.name = "ratings per book"
            dist.name = "number of books"
            st.bar_chart(dist, color="#D9A441")
        with d2:
            st.markdown('<p class="fine">Scores given. Readers rate '
                        'generously most ratings cluster high.</p>',
                        unsafe_allow_html=True)
            scores = ratings["rating"].value_counts().sort_index()
            scores.index.name = "score"
            scores.name = "number of ratings"
            st.bar_chart(scores, color="#2E6B4F")

    if "genre" in books.columns and books["genre"].str.len().gt(0).any():
        st.markdown('<div class="section">What the library holds</div>',
                    unsafe_allow_html=True)
        genres = books["genre"].value_counts().head(15)
        genres.index.name = "genre"
        genres.name = "books"
        st.bar_chart(genres, color="#8E2F3B")
    with st.expander("A sample of the catalogue"):
        st.dataframe(
            books[["title", "author", "year", "genre"]].head(40),
            use_container_width=True, hide_index=True,
        )

st.markdown('<p class="fine" style="margin-top:34px;padding-top:16px;'
            'border-top:1px solid var(--rule)">Hybrid Book Recommender · '
            'content-based TF-IDF, item-item collaborative filtering and a '
            'weighted hybrid for cold start. Built with scikit-learn and '
            'Streamlit.</p>', unsafe_allow_html=True)
