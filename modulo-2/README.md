# Module 2 — First neural networks: classification, regression, and scaling

Two problems with Keras, plus a controlled experiment on the one preprocessing step that
decides whether either of them trains at all.

## Classification — Fashion-MNIST

Dense network: `Flatten → 256 → Dropout → 128 → Dropout → 10`. 235 146 parameters,
54 000 / 6 000 / 10 000 split, 22 epochs with early stopping (best at epoch 16), 63 seconds.

| | |
|---|---|
| Test accuracy | **89.16 %** |
| Test loss | 0.313 |
| Macro F1 | 0.891 |

The errors are concentrated and they make sense: *Shirt → T-shirt/top* (133 cases),
*Pullover → Coat* (95), *T-shirt/top → Shirt* (95). Three classes that are genuinely
similar in a 28×28 greyscale image. Module 3 shows that a CNN pushes this to 92.6 %, and
that these same three classes stay the hard ones.

## Regression — California Housing

Dense network, 11 521 parameters, 13 209 / 3 303 / 4 128 split, 94 epochs (best at 74),
26 seconds. Compared against two baselines, which is the part that matters:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Training mean | 0.905 | 1.145 | −0.000 |
| Linear regression | 0.533 | 0.745 | 0.577 |
| **Neural network (MLP)** | **0.340** | **0.513** | **0.800** |

**31 % lower RMSE than linear regression.** A baseline is what turns "the model works" into
a number — without the linear regression there, 0.80 R² means nothing on its own.

## The scaling experiment

The same two networks, trained twice: once on properly scaled features, once on raw data.
Nothing else changed.

| | Properly scaled | Raw data |
|---|---|---|
| Classification — best validation accuracy | **89.80 %** | 84.45 % |
| Regression — best validation MAE | **0.365** | 0.531 |

Five accuracy points on classification, and 45 % worse error on regression — from a
preprocessing step, not from the architecture. See `figuras/cmp_01_escala.png`.

## Running it

```bash
pip install -r requirements.txt
jupyter lab classificacao-e-regressao.ipynb
```

Both datasets download at runtime (`keras.datasets` and scikit-learn). The notebook is
committed with its outputs. Full write-up (Portuguese) in [`relatorio.pdf`](relatorio.pdf);
all metrics in [`resultados/metricas.json`](resultados/metricas.json).

Environment: Python 3.12.6, TensorFlow 2.21, Keras 3.15, scikit-learn 1.8, CPU only.
