# assignment3
# BCHW <-> B x (CHW) Flatten / Reconstruct (AI Accelerator Design, Assignment 3)

| File | Purpose |
|---|---|
| `flatten_core.py` | From-scratch flatten, reconstruct, error metrics (no reshape/view/flatten/ravel) |
| `datasets.py` | MNIST / CIFAR-10 loaders (torchvision, optional) + synthetic feature maps |
| `manual_example.py` | Prints the B=1,C=2,H=2,W=3 hand-worked mapping |
| `run_experiments.py` | Runs all cases -> `results.csv`, `results.md` |
| `test_flatten.py` | Tests (bijection, round trip, numpy reference check) |
| `make_report.py` | Builds `report.pdf` from `results.csv` |


## Run
```
pip install -r requirements.txt
python test_flatten.py
python manual_example.py
python run_experiments.py
python make_report.py
```
Mapping: `j = c*H*W + h*W + w`; inverse: `c=j//(HW); r=j%(HW); h=r//W; w=r%W`.
`np.reshape` appears only once, in `run_experiments.py`, in a block marked REFERENCE CHECK ONLY.
