# Instructions for CSCS cluster

## Install uv if not already installed
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Activate `uenv`

([uenv docs](https://docs.cscs.ch/software/uenv/using/#how-uenv-works))

```bash
uenv image pull prgenv-gnu/25.6:v2
uenv start --view=default prgenv-gnu/25.6:v2
```

## Install dependencies and build the project

This needs to be done on a compute node or using `uenv` (see above) so that CUDA toolkit is available.

```bash
rm -rf build/
uv sync --all-extras
```

## Run tests
```bash
pytest tests/ -v
```
or use makefile to run on a (non interactive) compute node

```bash
make test
```
