# Reconstruction of the torso surface based on 3D camera recordings

A method for reconstructing 3D torso geometry from partial 3D scans using the
STAR human body model.

## Running the application

The development operating system was **Manjaro Linux (x86_64)**. The commands below assume a Linux or another Unix-like shell. The project uses Python 3.8;
Run all commands from the repository root.

### 1. Initialize the STAR submodule

If the repository has already been cloned, download the STAR submodule with:

```bash
git submodule update --init --recursive
```

Alternatively, include `--recurse-submodules` when cloning the repository:

```bash
git clone --recurse-submodules <repository-url>
cd 3d-torso-reconstruction
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

The application has been tested with the CPU-only version of PyTorch specified
in `requirements.txt`.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The STAR submodule creates some tensors explicitly on CUDA. When using the
CPU-only PyTorch package, replace those constructors with device-independent
CPU tensor constructors:

```bash
# Required when using the CPU-only PyTorch package from requirements.txt.
find STAR -type f -name '*.py' -exec sed -i \
  's/torch\.cuda\.FloatTensor/torch.FloatTensor/g' {} +
```

To use an NVIDIA GPU instead, first install the other dependencies from
`requirements.txt`, then replace its CPU-only PyTorch package with a CUDA wheel.
For example, install PyTorch 2.4.1 for CUDA 12.4 with:

```bash
python -m pip uninstall -y torch
python -m pip install torch==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

Choose the CUDA wheel compatible with your system (`cu118`, `cu121`, or
`cu124`). The commands for each variant are listed in the
[official PyTorch 2.4.1 installation instructions](https://docs.pytorch.org/get-started/previous-versions/#v241).

### 4. Download and configure the STAR body models

1. Register or sign in at the [STAR website](https://star.is.tue.mpg.de/),
   accept its license, and download the STAR body models.
2. Extract the downloaded files to a location of your choice.
3. Open `STAR/star/config.py` and set the three paths to the corresponding
   downloaded `.npz` model files. Absolute paths are recommended:

```python
path_male_star = "/absolute/path/to/male/model.npz"
path_female_star = "/absolute/path/to/female/model.npz"
path_neutral_star = "/absolute/path/to/neutral/model.npz"
```

### 5. Start the GUI

Start without a scan and select one in the application:

```bash
python src/main.py
```

You can also provide an OBJ, PLY, or STL scan on the command line and optionally
choose the output path:

```bash
python src/main.py /path/to/scan.obj
python src/main.py /path/to/scan.obj --output out/result.ply
```

When `--output` is omitted, the fitted mesh is saved as
`out/<scan_name>.ply`.
