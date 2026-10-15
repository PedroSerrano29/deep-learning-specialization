# Module 3 — Computer-vision architectures, compared

Four architectures on the same problem, under the same budget: classify Fashion-MNIST
(28×28 greyscale, 10 clothing classes) on a laptop CPU.

The point of the exercise is not accuracy. It is the trade-off between accuracy, size and
the hours it costs you.

## Results

| Model | Test accuracy | 95 % CI | Parameters | Epochs | Training time | Weights | Inference |
|---|---|---|---|---|---|---|---|
| **CNN** | **92.62 %** | ± 0.51 | 421 642 | 19 | 4 min 49 s | 1.6 MB | 0.10 ms/img |
| ResNet50 | 92.14 % | ± 0.53 | 23 608 202 | 7 | 1 h 15 min 38 s | 90.1 MB | 1.61 ms/img |
| GRU | 88.61 % | ± 0.62 | 61 962 | 15 | 4 min 5 s | 0.24 MB | 0.09 ms/img |
| LSTM | 88.45 % | ± 0.63 | 81 674 | 17 | 3 min 49 s | 0.31 MB | 0.16 ms/img |

Full numbers in [`resultados/tabela_comparativa.csv`](resultados/tabela_comparativa.csv)
and [`resultados/resultados.json`](resultados/resultados.json).

## What the numbers say

**The hand-built CNN wins, and it is not close on cost.** It is 56× smaller than ResNet50,
trains 16× faster, and runs inference 16× faster — for slightly *better* accuracy. The
confidence intervals overlap, so the honest reading is "equal accuracy at 1/56 of the
size", which is the same conclusion from the point of view of anyone paying for the
hardware.

**Transfer learning is not free.** ResNet50 was pre-trained on 224×224 colour photographs.
Fed 32×32 upscaled greyscale images, most of that capacity has nothing to represent. Two
phases were trained — frozen base, then fine-tuning the whole network — and the second
phase alone took 72 minutes of the 76.

**Reading an image as a sequence costs about four accuracy points.** LSTM and GRU take the
28 rows as 28 time steps. They lose the two-dimensional structure a convolution exploits,
and the gap is visible but smaller than one might expect. The GRU matches the LSTM with 24 %
fewer parameters.

**The errors are the same ones for every model.** Shirt, T-shirt and pullover are the hard
classes — they are hard for all four architectures, which says the limit is the dataset,
not the network.

## Figures

Training curves, confusion matrices and activation maps per model are in
[`resultados/`](resultados/). Worth a look:

- `04_cnn_mapas_ativacao.png` — what the CNN's filters actually respond to
- `07_rnn_imagem_como_sequencia.png` — how an image becomes a sequence of rows
- `12_rnn_acuracia_por_linhas.png` — accuracy as the recurrent model reads more rows
- `14_acuracia_vs_parametros.png` — the whole argument of this module in one chart

## Running it

```bash
pip install -r requirements.txt
jupyter lab arquiteturas-visao-computacional.ipynb
```

Keras downloads Fashion-MNIST on first run. The notebook is committed with its outputs, so
everything is readable without running it. Full write-up (Portuguese) in
[`relatorio.pdf`](relatorio.pdf).

Environment: Python 3.12.6, TensorFlow 2.21, Keras 3.15, Windows 11, CPU only.
