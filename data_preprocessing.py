import pandas as pd
import torch
import subprocess
import os
from multiprocessing import Pool, cpu_count

# Function to call the tokenizer C program
def tokenize_block(hex_code):
    """Runs the tokenizer executable and returns tokenized XML."""
    try:
        hex_code = str(hex_code)  # Ensure hex_code is a string
        process = subprocess.Popen(["data/tokenizer", hex_code, "--token"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            xml_output = stdout.decode("utf-8").strip()
            return xml_output if xml_output else None  # Ensure it's not empty
        else:
            return None  # Handle tokenizer errors
    except Exception as e:
        return None

# Function to process a single row efficiently
def process_row(params):
    index, row, start_id = params
    hex_code = str(row[0])  # Ensure hex_code is a string
    timing = float(row[1])  # Convert timing to float

    code_id = start_id + index  # Generate unique incremental identifier
    code_intel = u''  # Leave this empty
    code_xml = tokenize_block(hex_code)  # Tokenize using C program

    return (code_id, timing, code_intel, code_xml)

# Function to process BHive CSV files using multiprocessing
def process_bhive_csv(file_path, start_id=1400000):
    df = pd.read_csv(file_path, names=['hex_code', 'throughput'], dtype={'hex_code': str})
    params = [(index, row, start_id) for index, row in enumerate(df.values)]

    # Use multiprocessing pool for parallel execution
    pool = Pool(cpu_count())
    results = pool.map(process_row, params)
    pool.close()
    pool.join()

    return results

save_dir = "data/"
os.makedirs(save_dir, exist_ok=True)


# Process BHive CSV files efficiently
print("Processing HSW...")
final_data = process_bhive_csv('external/bhive/benchmark/throughput/hsw.csv', start_id=1000000)
save_path = os.path.join(save_dir, "bhive_hsw.data")
torch.save(final_data, save_path)

print("Processing IVB...")
final_data = process_bhive_csv('external/bhive/benchmark/throughput/ivb.csv', start_id=2000000)
save_path = os.path.join(save_dir, "bhive_ivb.data")
torch.save(final_data, save_path)

print("Processing SKL...")
final_data = process_bhive_csv('external/bhive/benchmark/throughput/skl.csv', start_id=3000000)
save_path = os.path.join(save_dir, "bhive_skl.data")
torch.save(final_data, save_path)
