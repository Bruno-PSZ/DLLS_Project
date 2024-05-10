# CellSlighter 0.7
(Slightly worse results than CellSighter, but just slightly more chaotic codebase)

## Requirements:
1. Use `/requirements.txt`

## How to run:
1. `cellslighter/train.py` - run this for training:
   - training data should be put unzipped in `train/`
   - 3 best model checkpoints are saved in `cellslighter/checkpoints/`
   - checkpoint filenames are written in `cellslighter/checkpoints_path.txt`
2. `cellslighter/test.py` - run this for evaluation on training-like data:
    - evaluation data should be put unzipped in `test/`
    - checkpoint filenames are read from `cellslighter/checkpoints_path.txt`
    - all checkpoints are read from `cellslighter/checkpoints/`
    - to run a validation on pretrained weights download checkpoints from <https://mega.nz/file/oWtkjZKB#ZEzWKBLw61Ac3QmVh3GR1Nhul9HGBkP0Mn3-1VQ7nHc> to `cellslighter/checkpoints/`
  