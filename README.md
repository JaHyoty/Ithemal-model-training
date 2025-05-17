# **Setting Up Ithemal Environment & Running `train.py`**
This guide helps you **set up the Conda environment**, properly activate it, and run `train.py` with either a **small dataset** or the **full dataset**. Additionally, it includes steps to **generate training data** if needed.

## ✅ **1. Install Conda**
If you don’t already have Conda installed, download and install **Miniconda** or **Anaconda**:

- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (lightweight)
- [Anaconda](https://www.anaconda.com/products/distribution) (includes additional packages)

## ✅ **2. Set Up the Ithemal Environment**
Run the following in your terminal to **create and set up** the environment:

```bash
conda create --name Ithemal python=3.9 -y
conda activate Ithemal
pip install -r requirements.txt
```

📌 **Notes**:
- The Python version can be adjusted if needed.
- `requirements.txt` contains all necessary dependencies.

## ✅ **3. Ensure Conda is Initialized Properly**
If you encounter an error such as _"Run 'conda init' before 'conda activate'"_, initialize Conda by running:

```bash
conda init
```
Then **restart your terminal**, and re-run:
```bash
conda activate Ithemal
```

## ✅ **4. (OPTIONAL) Generate Training Data**
You can generate the training data using the `data_preprocessing.py` but we recommend you use the data we already preprocessed in /data.</br>
To generate training data yourself, you need to download the BHive repo that is included as a submodule. Then run the python script.

```bash
git submodule update --init --recursive
python data_preprocessing.py
```

---

## ✅ **5. Running `train.py`**
After setting up the environment and generating data, you can run `train.py` for each microarchitecture:

### 🔹 **Run for Haswell (HSW)**
```bash
python train.py --arch HSW
```

### 🔹 **Run for Ivy Bridge (IVB)**
```bash
python train.py --arch IVB
```

### 🔹 **Run for Skylake (SKL)**
```bash
python train.py --arch SKL
```

If you want to **train using the full dataset**, add `--full_dataset`:
```bash
python train.py --arch HSW --full_dataset
python train.py --arch IVB --full_dataset
python train.py --arch SKL --full_dataset
```

---

## ✅ **6. Evaluate and run `visualize.py` (Only After Training All Architectures)**
Once `train.py` has been executed for **HSW, IVB, and SKL**, you can visualize the evaluation results using:

```bash
python visualize.py
```

📌 **This step should only be run after all training tasks have completed**, ensuring that visualization includes data from **all microarchitectures**.

---

## ✅ **7. Troubleshooting**
### 🔹 **Environment Activation Issues**
If `conda activate Ithemal` doesn’t work, try:
```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate Ithemal
```

---
