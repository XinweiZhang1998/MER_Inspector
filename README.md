# Code of MP-Inspector
Implementation of the paper "**MP-Inspector: Quantifying Model Privacy Risks from An Attack-Agnostic Perspective**".
We implement this project with reference to the implementation of [MAZE](https://github.com/sanjaykariyappa/MAZE).

## Environment
```
conda env create -f environment.yaml
```

## Generate victim model (e.g., CIFAR-10, res20)
```
python defender.py --dataset=cifar10 --epochs=200 --model_tgt=res20_2
```

## Generate surrogate model to give the ground-truth model privacy risk
```
python attacker.py --dataset=cifar10 --dataset_sur=cifar10 --epochs=100 --model_tgt=res20_2 --model_clone=res20_2
```

## Computer metric
```
python Risk/Metrics.py --dataset=cifar10  --model_tgt=res20_2
```
