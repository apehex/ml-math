# Datasets

Usually, no analytical expression of the target functions are known.

Instead of a full map, it is defined by a finite dataset of evaluations $(\mathbf{x}^{(k)}, \mathbf{y}^{(k)}))$ of the function.

## Notations

| Symbol                                                                    | Meaning                                                                           |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| $r\_x$                                                                    | The rank of all inputs (number of axes)                                           |
| $r\_y$                                                                    | The rank of all targets                                                           |
| $n = \vert \mathcal{D} \vert$                                             | The cardinality of $\mathcal{D}$                                                  |
| $\mathcal{D} = \lbrace \mathbf{s}^{(1)}, \dots, \mathbf{s}^{(n)} \rbrace$ | The dataset, composed of $n$ samples                                              |
| $\mathbf{s}^{(k)} = (\mathbf{x}^{(k)}, \mathbf{y}^{(k)})$                 | A sample of matching input and target                                             |
| $\mathbf{x}^{(k)} = [x^{(k)}\_1, x^{(k)}\_2, \dots, x^{(k)}\_{r\_x}]$     | An input tensor of rank $r\_x$                                                    |
| $\mathbf{y}^{(k)} = [y^{(k)}\_1, y^{(k)}\_2, \dots, y^{(k)}\_{r\_y}]$     | A target ransor of rank $r\_y$                                                    |

## Data Types

The datasets can be sorted by:

- the nature of the input data $x_i$:
    - textual data: UTF bytes, tokenizer ids, etc
    - image data: RGB bytes, grayscale ratio, etc
    - raw bytes: sound encodings, bytecode, etc
    - etc
- the nature of the target data $y_i$:
    - classification task: identify the class id(s) of the input
    - generation task: the next input in a sequence
    - denoising task: scrambled data
    - masking task: missing / corrupted data
    - encoding task: the original input for reconstruction
    - etc

As neural networks operate on tensors, these data types have to be formated / preprocessed accordingly.
