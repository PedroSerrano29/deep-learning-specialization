# Module 1 — Foundations

Three practicals covering the three families of classical machine learning, before any
neural network shows up. All three share one constraint set by the course: **data cleaning
had to be written with plain Python**, not with a cleaning library — the point being to
understand what those libraries do rather than call them.

| Practical | Problem | Approach | Dataset |
|---|---|---|---|
| [2 — Cardiotocography](pratica-2-cardiotocografia/) | Predict foetal state from cardiotocography readings | Supervised: Random Forest | [UCI Cardiotocography](https://archive.ics.uci.edu/ml/datasets/Cardiotocography) |
| [3 — Flower types](pratica-3-flores-kmeans/) | Group flowers by their measurements, with no labels | Unsupervised: K-means | [UCI Iris](https://archive.ics.uci.edu/ml/datasets/iris) |
| [4 — Machine failure](pratica-4-falha-maquina/) | Predict failures while learning in batches, not all at once | Incremental learning | [Kaggle: machine-failure-prediction](https://www.kaggle.com/c/machine-failure-prediction/data) |

## Practical 2 — Cardiotocography

A Random Forest trained to classify foetal state. Beyond the metrics, the exercise asked
for the resulting tree to be rendered twice: the whole thing, and cut at a single level of
depth.

- `arvore-global.png` / `.svg` — the full tree
- `arvore-profundidade-1.png` — depth 1, which is the one you can actually read and explain
  to someone

That pair is the real lesson: the accurate model is unreadable, the readable model is the
one you can defend to whoever has to act on it.

## Practical 3 — Flower types

K-means on the iris measurements, with the clusters plotted against the true species to see
where unsupervised grouping agrees with reality and where it does not.

## Practical 4 — Machine failure

Incremental learning: the model is updated batch by batch instead of being trained once on
everything. The exercise also required generating artificial data to feed that process.

**The dataset is not in this repository.** It comes from a Kaggle competition, so it is
theirs to distribute, not mine. Download `train.csv` and `test.csv` from the
[competition page](https://www.kaggle.com/c/machine-failure-prediction/data) and put them
next to the notebook.

## Running any of them

```bash
pip install scikit-learn pandas numpy matplotlib seaborn ucimlrepo graphviz
jupyter lab
```

Practicals 2 and 3 fetch their data at runtime with `ucimlrepo`. All notebooks are committed
with their outputs.

> These three practicals previously lived in their own repositories
> (`Cardiotocography_DL`, `FlowerType_DL`, `PredictMachineFailure_DL`). They were brought
> together here so the specialization reads as one piece of work.
