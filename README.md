# Hails

## Instructions

### Requirements:

- Python >=3.11
- [uv](https://docs.astral.sh/uv/)

### Installation

1. Download M5 Dataset following the instructions in [Reamde](dataset/M5/README.md)
2. Run `./dataset/M5/extract.sh`
3. Run `uv sync` to download the required dependencies
4. Run `uv run pretrainm5.py` for pre-training (atleast for 10 ep)
5. Run `rye run trainm5.py` for training (atleast for 100 epochs)
  - Requires atleast 70 GB VRAM with Mixed precision
  - Toggle `SCALE_PREC = False` in `trainm5.py` to use FP16 to run on GPUs of less than 40 GB VRAM
