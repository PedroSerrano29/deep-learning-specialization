# Deep Learning Specialization

Coursework for the Deep Learning specialization at Tokio School, taken after its Python
course. Three modules so far, each one a self-contained piece of work with its
notebook, its figures and its report.

Everything here ran on a laptop CPU — 8 logical cores, no GPU. That constraint shaped the
results more than anything else, and it is the thread running through the whole thing.

## The result worth reading first

Module 3 compares four architectures on the same image-classification problem
(Fashion-MNIST, 10 classes). The interesting part is not which one wins — it is the price
of winning:

| Model | Test accuracy | Parameters | Training time | Weights |
|---|---|---|---|---|
| **CNN (built from scratch)** | **92.6 %** | 422 K | 4 min 49 s | 1.6 MB |
| ResNet50 (transfer learning) | 92.1 % | 23.6 M | 1 h 16 min | 90 MB |
| GRU (image read as a sequence) | 88.6 % | 62 K | 4 min 5 s | 0.2 MB |
| LSTM (image read as a sequence) | 88.5 % | 82 K | 3 min 49 s | 0.3 MB |

A 422 K-parameter CNN written by hand matches a pre-trained ResNet50 that is **56× larger**
and takes **16× longer to train**. The confidence intervals overlap, so the honest word is
*matches*, not *beats* — which makes the cost gap the whole story. On 28×28 greyscale
images, the capacity of a network pre-trained on 224×224 colour photographs has nowhere to
go — and its cost shows up in full.

The recurrent models are there to answer a different question: what happens if you read an
image one row at a time, as if it were a sequence? Four accuracy points worse, with a
fraction of the parameters.

## Modules

| | What it covers | Tools |
|---|---|---|
| [Module 1](modulo-1/) | Classical machine learning foundations: supervised, unsupervised and incremental learning | scikit-learn, pandas |
| [Module 2](modulo-2/) | First neural networks: classification and regression, and why feature scaling decides whether they train at all | TensorFlow, Keras |
| [Module 3](modulo-3/) | Computer-vision architectures compared head to head: CNN, ResNet50, LSTM, GRU | TensorFlow, Keras |

The final project — intent classification for a European-Portuguese voice assistant — lives
in its own repository, which is not public yet.

## Running any of it

Each module has its own `requirements.txt`.

```bash
python -m venv .venv
.venv\Scripts\activate        # Linux/macOS: source .venv/bin/activate
pip install -r modulo-2/requirements.txt
jupyter lab
```

The notebooks are committed **with their outputs**, so every chart, table and metric is
visible on GitHub without running anything.

## Notes

- **The notebooks and the reports are written in Portuguese** — they are the work as it was
  submitted. The READMEs are in English.
- **Datasets are not committed.** Modules 1 to 3 pull theirs at runtime (`ucimlrepo`,
  `keras.datasets`, scikit-learn). The one exception is the machine-failure practical, whose
  data comes from a Kaggle competition — the README there explains where to get it.
- Course materials from Tokio School are not included here; they are not mine to publish.
