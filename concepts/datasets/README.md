# Datasets

Usually, no analytical expression of the target functions are known.

Instead of a full map, it is defined by a finite dataset of evaluations $(x_i, y_i)$ of the function.

Hence, the datasets can be sorted by:

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
