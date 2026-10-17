"""
RAG retriever over the care-planning guideline notes in genai/guidelines/*.md.

Each "## " section of a note is one chunk. Ranking is TF-IDF + cosine similarity in plain numpy:
about 30 short chunks don't need an embedding model, and nothing extra has to build on s390x.

Note format:
    # Title
    Source: <where the guidance comes from>
    ## Section heading
    Text in our own words...
"""
import glob, math, os, re
from collections import Counter
import numpy as np

GUIDE_DIR = os.environ.get("TTT_GUIDELINES", os.path.join(os.path.dirname(os.path.abspath(__file__)), "guidelines"))

STOP = set("""a an and are as at be but by can do does for from has have how i if in into is it its
me my not of on or our so than that the their them then there these they this to too was we were
what when where which who will with you your yourself about any all also been more most other some
such only very just should would could may might""".split())


def tokenize(text):
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in STOP]


class Retriever:
    def __init__(self, folder=GUIDE_DIR):
        self.chunks = []
        for path in sorted(glob.glob(os.path.join(folder, "*.md"))):
            text = open(path, encoding="utf-8").read()
            m = re.search(r"^Source:\s*(.+)$", text, re.M)
            source = m.group(1).strip() if m else os.path.basename(path)
            for part in re.split(r"^## ", text, flags=re.M)[1:]:
                title, _, body = part.partition("\n")
                body = " ".join(body.split())
                if body:
                    self.chunks.append({"title": title.strip(), "text": body,
                                        "file": os.path.basename(path), "source": source})
        if not self.chunks:
            raise FileNotFoundError(f"no guideline notes found in {folder}")

        docs = [Counter(tokenize(c["title"] + " " + c["title"] + " " + c["text"])) for c in self.chunks]
        self.vocab = {w: i for i, w in enumerate(sorted(set().union(*docs)))}
        df = Counter(w for d in docs for w in d)
        n = len(docs)
        self.idf = np.array([math.log((1 + n) / (1 + df[w])) + 1 for w in self.vocab])
        self.M = np.vstack([self._vec(d) for d in docs])

    def _vec(self, counts):
        v = np.zeros(len(self.vocab))
        for w, c in counts.items():
            if w in self.vocab:
                v[self.vocab[w]] = 1 + math.log(c)
        v *= self.idf
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def search(self, query, k=4):
        sims = self.M @ self._vec(Counter(tokenize(query)))
        top = np.argsort(-sims)[:k]
        return [dict(self.chunks[i], score=float(sims[i])) for i in top if sims[i] > 0]


_default = None


def retrieve(query, k=4):
    global _default
    if _default is None:
        _default = Retriever()
    return _default.search(query, k)
