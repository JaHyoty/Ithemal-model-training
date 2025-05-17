import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

MICROARCHITECTURES = ["HSW", "IVB", "SKL"]

# ### Generate visualizations
# Make sure to run the evaluations for all microarchitectures before generating visualizations

# Load CSV files
PE_HSW = np.loadtxt(f"models/evaluation/{MICROARCHITECTURES[0]}_PE.csv", delimiter=",")
PE_IVB = np.loadtxt(f"models/evaluation/{MICROARCHITECTURES[1]}_PE.csv", delimiter=",")
PE_SKL = np.loadtxt(f"models/evaluation/{MICROARCHITECTURES[2]}_PE.csv", delimiter=",")


df = pd.DataFrame({
    "Microarchitecture": ["HSW"] * len(PE_HSW) + ["IVB"] * len(PE_IVB) + ["SKL"] * len(PE_SKL),
    "Signed Percentage Error": np.concatenate([PE_HSW, PE_IVB, PE_SKL])
})

# Generate Box-and-Whisker Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x="Microarchitecture", y="Signed Percentage Error", data=df, showfliers=False, width=0.5)
plt.title("Box-and-Whisker Plot of Signed Percentage Errors by Microarchitecture")
plt.ylim(-25, 20)
increment = 5
for y in np.arange(-25, 20 + increment, increment):
    plt.axhline(y=y, color="gray", linestyle="-", alpha=0.5, zorder=0)

plt.ylabel("Signed Percentage Error (%)")
plt.xlabel("Microarchitecture")
plt.show(block=True)



PE_HSW_arcsinh = np.arcsinh(PE_HSW)
PE_IVB_arcsinh = np.arcsinh(PE_IVB)
PE_SKL_arcsinh = np.arcsinh(PE_SKL)

def sinh_formatter(x_arcsinh, pos):
    return f"{np.sinh(x_arcsinh):.2f}"

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(PE_HSW_arcsinh, bins=50, kde=True, color="blue")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(sinh_formatter))
ax.set_title("Signed Percentage Error Histogram for Haswell Throughput Predictions")
ax.set_xlabel("Signed Percentage Error")
ax.set_ylabel("Frequency")
plt.show(block=True)



fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(PE_IVB_arcsinh, bins=50, kde=True, color="blue")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(sinh_formatter))
ax.set_title(
    "Signed Percentage Error Histogram for Ivy Bridge Throughput Predictions"
)
ax.set_xlabel("Signed Percentage Error")
ax.set_ylabel("Frequency")
plt.show(block=True)


fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(PE_SKL_arcsinh, bins=50, kde=True, color="blue")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(sinh_formatter))
ax.set_title(
    "Signed Percentage Error Histogram for Skylake Throughput Predictions"
)
ax.set_xlabel("Signed Percentage Error")
ax.set_ylabel("Frequency")
plt.show(block=True)
