# Getting started

You'll need a conda environment with all the usual scientific computing packages. You can use the `environment.yml` file to generate an environment with all the packages you should need by running

```
conda env create -f environment.yml
```

and then

```
conda activate adasketch
```

The figures from [Adaptive Sketch-and-Project Methods for Solving Linear Systems](https://arxiv.org/abs/1909.03604) for synthetic matrices can be generated by running

```
python run_this.py
```

The figures will be saved in a `figures` subfolder.

If you use this code, please cite:

Gower, Robert, Denali Molitor, Jacob Moorman, and Deanna Needell. "Adaptive Sketch-and-Project Methods for Solving Linear Systems." arXiv preprint arXiv:1909.03604 (2019).
