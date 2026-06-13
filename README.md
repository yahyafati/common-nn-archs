# Common Neural Network Architectures

A modular, reusable PyTorch-based framework for implementing and training common neural network architectures such as
MobileNet, ResNet, and more — without rewriting boilerplate code.

## Motivation

Implementing standard architectures repeatedly across projects is inefficient and error-prone. This repository provides:

* Reusable architecture definitions
* Unified training interface
* CLI-driven experimentation
* Clean separation of concerns (models, training, config, utils)

The goal is to enable workflows like:

```bash
python app.py --arch mobilenetv1,resnet --epochs 100
```

## Features

* 🔧 Plug-and-play architectures (e.g. MobileNet, ResNet)
* 🧱 Modular design for easy extension
* 🧪 Experiment-friendly CLI interface
* 📊 Built-in metrics and logging utilities
* 🔁 Reproducibility via seed control
* ⚙️ Minimal boilerplate for training loops

## Project Structure

```
.
├── app.py                 # Entry point
├── configs/               # Configuration files (optional)
├── models/                # Model definitions
│   ├── mobilenet.py
│   ├── resnet.py
│   └── __init__.py
├── trainers/              # Training logic
│   └── trainer.py
├── utils/
│   ├── logger.py
│   ├── metrics.py
│   └── seed.py
├── datasets/              # Dataset loaders
├── experiments/           # Experiment configs / outputs
└── README.md
```

## Installation

```bash
git clone https://github.com/yahyafati/common-nn-archs.git
cd common-nn-archs
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python app.py --arch resnet --epochs 50
```

### Multiple Architectures

```bash
python app.py --arch mobilenetv1,resnet --epochs 100
```

### Additional Options

```bash
python app.py \
    --arch resnet \
    --epochs 100 \
    --batch-size 64 \
    --lr 0.001 \
    --seed 42
```

## Adding a New Architecture

1. Create a new file in `models/`:

```python
# models/my_model.py
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        ...

    def forward(self, x):
        return ...
```

2. Register it in `models/__init__.py`

3. Ensure it's selectable via CLI

## Logging

Logging is handled via `utils/logger.py`:

* Console + optional file logging
* Structured output for experiments

## Metrics

Implemented in `utils/metrics.py`:

* Accuracy
* Top-k accuracy
* Extendable for custom metrics

## Reproducibility

Set seeds using:

```python
from utils.seed import set_seed

set_seed(42)
```

Ensures deterministic behavior where possible.

## Design Principles

* **Modularity**: decouple models, training, and utilities
* **Reusability**: avoid duplication across experiments
* **Extensibility**: easily add new architectures or datasets
* **Simplicity**: minimal abstraction overhead

## Roadmap

* [ ] Add more architectures (EfficientNet, ViT)
* [ ] Config system (YAML / TOML)
* [ ] Experiment tracking (TensorBoard / WandB)
* [ ] Distributed training support
* [ ] Pretrained weights support

## Contributing

PRs are welcome. Please keep contributions modular and consistent with existing structure.

## License

MIT License
