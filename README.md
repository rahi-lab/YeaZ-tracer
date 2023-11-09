# ``yeaz-trace`` -- determining lineage of budding yeast cells

## Installation

Recommended : create a virtual environment :

```sh
conda create -n yeaz_tracer python=3.9
conda activate yeaz_tracer
python -m pip install pip --upgrade
```

For development :

```sh
# Install the package in development mode (installs dependencies and creates symlink)
git clone https://github.com/rahi-lab/YeaZ-tracer
# alternatively, download and extract the zip of the repository
cd yeaz-tracer
pip install -e .
```

## Graphical user interface

Launch the GUI using ``python -m bread.gui``.

![bread gui](assets/gui.png)

![algorithm parameter window](assets/param_budlum.png)


## Command-line interface

Launch the CLI using ``python -m bread.cli --help``.

## Raw data

The testing data can be downloaded at https://drive.google.com/file/d/1XcXTDPyDiBBlLeQpNFmugn3O6xVFfleU/view?usp=sharing.