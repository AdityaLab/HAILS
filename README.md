# Hails

## Instructions

### Requirements:

- Python >=3.11
- [Rye](https://rye-up.com/guide/installation/)

### Installation

1. Download M5 Dataset following the instructions in [Reamde](dataset/M5/README.md)
2. Run `./dataset/M5/extract.sh`
3. Run `rye sync` to download the required dependencies
4. Run `rye run python pretrainm5.py` for pre-training (atleast for 10 epochs)
5. Run `rye run python trainm5.py` for training (atleast for 100 epochs)
  - Requires atleast 70 GB VRAM with Mixed precision
  - Toggle `SCALE_PREC = False` in `trainm5.py` to use FP16 to run on GPUs of less than 40 GB VRAM
