# Tech News Curator

A daily tech-news feed that ranks stories against *your* interests instead of
the crowd's. It pulls from Hacker News, Reddit, Lobsters, Bluesky, GitHub,
arXiv and ~20 publication feeds, merges the copies of each story into one
row, normalizes popularity across sources that count in different units, and
then asks Gemini to score every item against a plain-English profile you
write in [`profile.md`](profile.md).

A GitHub Action runs it every morning and pushes the result to GitHub Pages.
It is a Twitter feed for one person, organized, with the noise ranked down.

Built as a sibling to
[sfevents](https://github.com/watakandai/sfevents) and shares
its shape: standard-library Python, a cached SQLite database, a two-layer
ranker, and a static page committed to `docs/`.

---

## How it works

```
fetch  ->  27 sources  ->  SQLite (cached between runs)
rank   ->  popularity (per-source normalization, shared across duplicates)
       ->  categories (keyword pass)
       ->  heuristic score (free, always)
       ->  LLM score + category (Gemini, only for items never seen before)
export ->  docs/data/items.json  ->  GitHub Pages
```

### The sources

| Source | What it adds | Popularity signal |
| --- | --- | --- |
| Hacker News (Algolia) | the tech front page | points + comments |
| Reddit (8 subreddits) | breadth, practitioner talk | upvotes (with app credentials) or feed rank |
| Lobsters | small, heavily moderated, good corroboration | points + comments |
| GitHub | repos that got popular this week | stars |
| arXiv | robotics, ML, formal methods, control, optimization | none - ranked by profile alone |
| Bluesky | the microblog layer, link-carrying posts only | likes + reposts |
| X / Twitter | same, **opt-in** - see below | likes + reposts |
| ~20 RSS feeds | publications, labs, company blogs, newsletters | none |

Adding a publication is one line in
[`technews/feeds.json`](technews/feeds.json). Adding a new
*kind* of source is one file in `fetchers/` plus one line in `cli.py`.

**On X/Twitter:** the v2 search endpoint this would need is not on X's free
tier, so an unattended daily job can't rely on it. The fetcher is written and
tested - set `X_BEARER_TOKEN` and it joins the rotation. Left unset it
reports itself disabled and is skipped. `bluesky.py` covers the same ground
for free, and filters to posts that carry a link, which is what makes them
news rather than commentary.

### Merging duplicates

The same article shows up on Hacker News, Reddit, Lobsters and the
publisher's own feed, wearing four different URLs. `normalize.py` reduces a
URL to what identifies the document (tracking parameters, AMP paths, `www.`,
fragments and shortener redirects all removed), and rows sharing that
identity are one story. In the export they collapse into a single row that
keeps every discussion link:

> **New quadruped controller** — ieee_spectrum · ▲ 94 · *also: hackernews 300 comments, reddit*

That merge is also what lets a publisher's copy inherit the attention its
Hacker News copy got, and what makes "four sources ran this" a usable signal.

### Popularity

900 Hacker News points, 1,900 Reddit upvotes and 8,000 GitHub stars are not
the same number. Every score is computed **relative to the item's own
source**: half where it falls in that source's current distribution, half how
far above that source's typical item it reached (log-scaled, referenced to
the 90th percentile so one freak story doesn't flatten the day).

Sources that publish no counts get **no popularity at all** rather than an
invented one, and sort last under "Popular". What they get instead is the
inherited score from any aggregator that ran the same link.

### Ranking

Two layers, the same split as the sibling project:

- **Heuristic** (`rank.heuristic_scores`) — free, deterministic, runs every
  time. Popularity, cross-source agreement, exponential freshness decay
  (halving every 36h), source trust, filler penalties. It is the floor when
  there is no key, no network, or a spent quota.
- **LLM** (`rank.llm_scores`) — reads `profile.md` and answers the question
  that actually matters: *would this person open this?* It assigns the
  category in the same call, because it has already read the item.

Scores are cached by `sha256(profile + model)`, so a daily run only pays for
items it has never seen — typically 100-200 rather than 900. Editing
`profile.md` changes the hash and re-scores the backlog once, which is the
intended way to retune the feed.

**The heuristic alone cannot surface a quiet paper.** An arXiv preprint on
Lyapunov certificates has no votes anywhere, so the free ranker has nothing
to go on. That is exactly what the profile ranker is for, and why the page
looks generic until you add a key.

---

## Setup

### 1. Write your profile

```bash
cp profile.example.md profile.md   # then edit it
```

This file **is** the prompt. Be concrete — name subfields, tools, companies.
HTML comments are stripped before sending, so notes-to-self are safe.

### 2. Run it locally

```bash
python -m technews.cli --db /tmp/news.db fetch
```

```bash
python -m technews.cli --db /tmp/news.db rank
```

```bash
python -m technews.cli --db /tmp/news.db export --out docs/data/items.json
```

Then open `docs/index.html` through any static server:

```bash
python -m http.server 8777 --directory docs
```

To rank against your profile you need a key (free tier is plenty):

```bash
GEMINI_API_KEY=... python -m technews.cli --db /tmp/news.db rank --llm --provider gemini
```

### 3. Publish it

Push to GitHub, then **Settings → Pages → Source: `main` / `/docs`**.

Add under **Settings → Secrets and variables → Actions**:

| Kind | Name | Needed? | Why |
| --- | --- | --- | --- |
| Variable | `RANKER_PROVIDER` | for profile ranking | `gemini` or `anthropic`. Unset = heuristic only |
| Secret | `GEMINI_API_KEY` | for profile ranking | [aistudio.google.com](https://aistudio.google.com/apikey) |
| Secret | `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` | recommended | Reddit 429s anonymous CI traffic; a free "script" app at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) fixes it and adds real upvote counts |
| Secret | `X_BEARER_TOKEN` | optional | Enables X. Requires a paid X API tier |

Optional tuning variables: `RANKER_LIMIT` (cap items sent per run),
`RANKER_BATCH_SIZE` (default 40), `RANKER_MIN_INTERVAL` (default 8s),
`EXPORT_DAYS` (default 10), `PRUNE_DAYS` (default 45).

The workflow runs at 13:00 UTC (6am PT) and on every push to `main`. Run it
by hand from the Actions tab — it takes a provider override, a
"re-score everything" toggle, and an item cap.

### Cost

Gemini Flash on a day of ~900 items, of which ~150 are new: roughly 4
requests of 40 items each. Comfortably inside the free tier. `RANKER_LIMIT`
caps it if you want a hard ceiling.

---

## The page

`docs/index.html` is one file, no build step, no dependencies. Built for a
daily scan rather than a browse:

- **Today / 3 days / Week / All**, and **For you / Popular / Newest**
- Grouped by category, with the day's strongest subject first — each
  category shows its top 10, ungrouped shows 20, and a **show more** button
  opens the rest. A day is ~200 rows, which is a wall rather than a page
- Read state per-browser — click or press <kbd>m</kbd>, then **hide seen**
  to make yesterday disappear
- <kbd>j</kbd>/<kbd>k</kbd> to move, <kbd>o</kbd> to open, <kbd>space</kbd>
  to expand
- Dark mode, and a usable phone layout

Everything it remembers lives in `localStorage` — nothing is uploaded, and
every read is guarded so a private window still works.

---

## Tests

```bash
python -m pytest tests/ -q
```

85 tests, no network. Parser tests run against captured real API responses in
`tests/fixtures/` so they assert against the shapes these services actually
return; the LLM tests use a stub provider and cover batching, a failed batch,
a per-minute 429 retry, a daily quota stopping the run, and the caching that
keeps a second run free.

## Layout

```
technews/
  cli.py            fetch / rank / export / list / prune
  db.py             SQLite schema, migrations, upserts (peak counts win)
  models.py         one Item shape for every source
  normalize.py      URL canonicalization and story identity
  popularity.py     per-source normalization + cross-source propagation
  categorize.py     the fixed taxonomy and its keyword pass
  rank.py           heuristic + LLM rankers, provider plumbing
  feeds.json        the RSS sources - add outlets here
  fetchers/         one file per kind of source
docs/               the published site (index.html + data/items.json)
profile.md          your interests. This file is the prompt.
```
