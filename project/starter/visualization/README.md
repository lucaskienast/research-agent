# UdaPlay Retrieval Console — visualization

An interactive, single-file dashboard visualizing **both** the UdaPlay agent's
retrieval process and its knowledge base — the extra "Visualization" challenge.

Open **`udaplay_dashboard.html`** in any browser (double-click it). No server, no
build step, no API key.

## What it shows

- **KPI strip** — 15 games · 1,536-D embeddings · 15 platforms · genres/publishers · release span · closest embedding pair.
- **The retrieval pipeline** — the `retrieve_game → evaluate_retrieval → (game_web_search) → generate` control flow, including the LLM-judge branch that makes it agentic.
- **Vector search, live** — pick any game to use its stored embedding as the query (exactly what `retrieve_game` does) and see the true ranked nearest neighbours by cosine similarity.
- **Embedding space** — the 1,536-D vectors projected to 2-D (PCA), colored by console family, with the selected game's neighbours linked; plus the full 15×15 cosine-similarity heatmap.
- **Strongest semantic links** — the sequel/franchise pairs the embeddings pull closest together.
- **Composition** — games by genre, publisher and release year.
- **The full catalogue** — a sortable, searchable table of all 15 documents.

Everything is driven by one embedded `DATA` blob and is theme-aware (light/dark).

## How it was built (honest notes)

- The embeddings (OpenAI, 1,536-D) are read **directly from the persistent ChromaDB
  collection `udaplay`** built in `Udaplay_01`. Reading stored vectors needs no API key.
- PCA (to 2-D) and cosine similarity are computed offline with NumPy. PCA explains
  only ~29% of variance combined — normal for high-dimensional text embeddings — so
  the map shows **relative** structure, not absolute axes.
- The "vector search, live" panel queries with each game's **stored** embedding, so it
  shows genuine nearest neighbours over the existing 15 items. Embedding an arbitrary
  new text query would require a live OpenAI key.

## Files

| File | Purpose |
|---|---|
| `udaplay_dashboard.html` | The self-contained dashboard (open this). |
| `dashboard_template.html` | HTML/CSS/JS template with a `"__UDAPLAY_DATA__"` placeholder. |
| `dashboard_data.json` | Data extracted from ChromaDB (games, PCA coords, similarity matrix, neighbours, aggregates). |
| `generate_dashboard.py` | Regenerates the JSON from ChromaDB and rebuilds the HTML. |

## Regenerate

Run `Udaplay_01` first (so `../chromadb/` exists), then:

```bash
python generate_dashboard.py
```

This reads `../chromadb`, recomputes everything, and rewrites `dashboard_data.json`
and `udaplay_dashboard.html`.
