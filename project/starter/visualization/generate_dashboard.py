#!/usr/bin/env python3
"""Regenerate the UdaPlay retrieval dashboard from the ChromaDB knowledge base.

Reads the stored 1536-D OpenAI embeddings from the persistent ``udaplay`` collection
(no API key needed to READ stored vectors), computes a 2-D PCA projection, the
cosine-similarity matrix, per-game nearest neighbours and category aggregates with
NumPy, then injects the data into ``dashboard_template.html`` to produce a fully
self-contained ``udaplay_dashboard.html``.

Prerequisite: run Udaplay_01 first so the ``chromadb/`` collection exists.

Usage:
    python generate_dashboard.py
"""
import json
import os

import numpy as np
import chromadb

HERE = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.environ.get("UDAPLAY_CHROMA_PATH", os.path.join(HERE, "..", "chromadb"))
COLLECTION = "udaplay"


def counts(metas, key):
    d = {}
    for m in metas:
        d[m[key]] = d.get(m[key], 0) + 1
    return dict(sorted(d.items(), key=lambda kv: (-kv[1], str(kv[0]))))


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    col = client.get_collection(COLLECTION)  # reading stored vectors needs no embedding fn
    res = col.get(include=["embeddings", "metadatas", "documents"])

    ids = res["ids"]
    embs = np.array(res["embeddings"], dtype=float)
    metas = res["metadatas"]

    # deterministic order by id (001..015)
    order = np.argsort(ids)
    ids = [ids[i] for i in order]
    embs = embs[order]
    metas = [metas[i] for i in order]

    # PCA -> 2D (numpy only)
    X = embs - embs.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    coords = X @ Vt[:2].T
    explained = (S[:2] ** 2) / (S ** 2).sum()

    # cosine similarity
    norm = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    sim = norm @ norm.T

    # nearest neighbours (leave-one-out, top 4)
    neighbors = {}
    for i, gid in enumerate(ids):
        s = sim[i].copy()
        s[i] = -1
        top = np.argsort(s)[::-1][:4]
        neighbors[gid] = [
            {"id": ids[j], "name": metas[j]["Name"],
             "platform": metas[j]["Platform"], "similarity": round(float(sim[i][j]), 4)}
            for j in top
        ]

    games = []
    for i, gid in enumerate(ids):
        m = metas[i]
        games.append({
            "id": gid, "Name": m["Name"], "Platform": m["Platform"], "Genre": m["Genre"],
            "Publisher": m["Publisher"], "YearOfRelease": m["YearOfRelease"],
            "Description": m["Description"],
            "x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4),
        })

    data = {
        "count": len(ids),
        "embedding_dim": int(embs.shape[1]),
        "pca_explained": [round(float(e), 4) for e in explained],
        "games": games,
        "by_platform": counts(metas, "Platform"),
        "by_genre": counts(metas, "Genre"),
        "by_publisher": counts(metas, "Publisher"),
        "by_year": counts(metas, "YearOfRelease"),
        "similarity_matrix": [[round(float(v), 4) for v in row] for row in sim],
        "neighbors": neighbors,
        "labels": [m["Name"] for m in metas],
    }

    with open(os.path.join(HERE, "dashboard_data.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    # inject into the template (ensure_ascii keeps the output charset-proof)
    tpl = open(os.path.join(HERE, "dashboard_template.html"), encoding="utf-8").read()
    inject = dict(data)
    inject.pop("neighbors", None)  # recomputed in JS from the matrix
    html = tpl.replace('"__UDAPLAY_DATA__"', json.dumps(inject, ensure_ascii=True))
    with open(os.path.join(HERE, "udaplay_dashboard.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Indexed {data['count']} games, {data['embedding_dim']}-D embeddings.")
    print("Wrote dashboard_data.json and udaplay_dashboard.html")


if __name__ == "__main__":
    main()
