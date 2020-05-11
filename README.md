# Spanning Attack

The implementation of algorithms proposed in the paper “Spanning attack: reinforce black-box attacks with unlabeled data”

## Implemented algorithms

- **score_baseline**: the baseline score-based black-box attack: the RGF attack.
- **score_subspace**: the subspace attack for **score_baseline**, including the original spanning attack, top subspace attack and bottom subspace attack.
- **decision_baseline**: the baseline decision-based black-box attack: the boundary attack.
- **decision_subspace**: the subspace attack for **decision_baseline**, including the original spanning attack, top subspace attack and bottom subspace attack.

## Getting started with the code

Our program is tested on Python 3.7.
The required packages are

- numpy
- pytorch
- pandas (only used to collect results)

Configuration files for all algorithms are in the `config` directory.

For example, if you want to run the bottom subspace attack of **score_subspace**,

1. Edit the `torch_home` (based on the pytorch installation) and `data_dir` (directory of the imagenet dataset) fields in `config/score.ini`;
2. Edit the `pool_size` (maybe 1,000), `subspace_size` (maybe 800) and `position` (for BSA, it should be set `bottom`);
3. Run `python main_score.py`.

