from dataclasses import dataclass
from collections import Counter
import math, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .query import expand_query

@dataclass
class Hit:
    clause_id: str
    score: float
    semantic: float
    lexical: float


def toks(s: str):
    return re.findall(r"(?u)\b\w+\b", s.lower())


class HybridRetriever:
    """Contract-scoped sparse retriever with benchmark-tuned, calibrated fusion.

    The default weights were selected on the development split only and then
    frozen before evaluating the held-out split. Query expansion is additive and
    deterministic. This avoids tuning directly on the reported holdout set.
    """
    DEFAULT_WEIGHTS = (0.10, 0.15, 0.75)  # word, char, BM25

    def __init__(self, clauses, weights=None):
        self.clauses = clauses
        self.texts = [c.text for c in clauses]
        self.weights = tuple(weights or self.DEFAULT_WEIGHTS)
        if len(self.weights) != 3 or abs(sum(self.weights) - 1.0) > 1e-9:
            raise ValueError("retrieval weights must be three non-negative values summing to 1")
        if any(w < 0 for w in self.weights):
            raise ValueError("retrieval weights cannot be negative")
        self.word = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, sublinear_tf=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)
        self.W = self.word.fit_transform(self.texts) if self.texts else None
        self.C = self.char.fit_transform(self.texts) if self.texts else None
        self.docs = [toks(t) for t in self.texts]
        self.df = Counter()
        for d in self.docs:
            self.df.update(set(d))
        self.avgdl = sum(map(len, self.docs)) / len(self.docs) if self.docs else 1
        self.N = len(self.docs)

    def _bm25(self, q):
        q = toks(q); scores = []; k1 = 1.5; b = .75
        for d in self.docs:
            tf = Counter(d); dl = len(d); s = 0.0
            for term in q:
                if term not in tf:
                    continue
                df = self.df[term]
                idf = math.log(1 + (self.N - df + .5) / (df + .5))
                s += idf * (tf[term] * (k1 + 1)) / (tf[term] + k1 * (1 - b + b * dl / self.avgdl))
            scores.append(s)
        return scores

    @staticmethod
    def norm(xs):
        if not xs:
            return []
        lo, hi = min(xs), max(xs)
        if hi == lo:
            return [1.0 if hi > 0 else 0.0 for _ in xs]
        return [(x - lo) / (hi - lo) for x in xs]

    def _signals(self, q):
        w = cosine_similarity(self.word.transform([q]), self.W)[0].tolist()
        c = cosine_similarity(self.char.transform([q]), self.C)[0].tolist()
        b = self._bm25(q)
        return w, c, b

    def search(self, q, k=8, expand=True):
        if not self.clauses:
            return []
        q = expand_query(q) if expand else q
        w, c, b = self._signals(q)
        wn, cn, bn = self.norm(w), self.norm(c), self.norm(b)
        ww, wc, wb = self.weights
        hits = [Hit(cl.id, ww * wn[i] + wc * cn[i] + wb * bn[i], w[i], b[i])
                for i, cl in enumerate(self.clauses)]
        return sorted(hits, key=lambda h: (-h.score, h.clause_id))[:k]

    def search_rrf(self, q, k=8, rrf_k=60):
        if not self.clauses:
            return []
        q = expand_query(q)
        w, c, b = self._signals(q)
        rankings = [sorted(range(len(self.clauses)), key=lambda i: scores[i], reverse=True)
                    for scores in (w, c, b)]
        fused = {i: 0.0 for i in range(len(self.clauses))}
        for ranking in rankings:
            for r, i in enumerate(ranking):
                fused[i] += 1 / (rrf_k + r + 1)
        return [Hit(self.clauses[i].id, fused[i], w[i], b[i])
                for i in sorted(fused, key=lambda i: (-fused[i], self.clauses[i].id))[:k]]


class OptionalDenseRetriever:
    """Optional sentence-transformer retriever; never required for offline CI."""
    def __init__(self, clauses, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.clauses = clauses
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.matrix = self.model.encode([c.text for c in clauses], normalize_embeddings=True, show_progress_bar=False)

    def search(self, query, k=8):
        import numpy as np
        q = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = np.asarray(self.matrix) @ np.asarray(q)
        idx = np.argsort(-scores)[:k]
        return [Hit(self.clauses[i].id, float(scores[i]), float(scores[i]), 0.0) for i in idx]
