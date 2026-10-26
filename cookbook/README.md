# LoongForge Cookbook

The cookbook collects projects built or maintained by the LoongForge team on top
of LoongForge. Each project is a self-contained recipe that provides what is
needed to reproduce its results, such as training scripts and configs.

## Cookbook vs. examples

| Directory | Purpose |
|---|---|
| [`examples/`](../examples/) | Launch scripts for the models LoongForge supports, one directory per model. |
| `cookbook/` | End-to-end project recipes (for example a research method or a model variant) with step-by-step reproduction instructions. |

## What you will find in a project

Each project lives in its own directory, `cookbook/<project-name>/`. Contents
vary by project, but typically include:

- A `README.md` describing the project, its environment and data requirements,
  and how to reproduce the results.
- Training scripts or configs.
- Evaluation scripts or configs, when the project reports benchmark results.

Projects typically build on the model and training code in this repository. Any
extra dependency, patch, or external asset (datasets, model weights) a project
needs is documented in its own README.
