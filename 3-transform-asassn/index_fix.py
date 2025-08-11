import os
from glob import glob
import polars as pl
from tqdm import tqdm

os.chdir("(#insert path to index files#)")
index_files = sorted(glob("index*.parq"))

def run():
    for jobid in tqdm(range(1091)):
        index_path = index_files[jobid]
        index = pl.read_parquet(index_path)
        index_checker = index.clone()
        true = []
        for i in range(len(index_checker)):
            true.append(True)
        trues = pl.Series("keeper", true)
        index_checker = index_checker.insert_column(4, trues)
        index_checker = index_checker.with_columns(pl.when(pl.col("asas_sn_id")==0).then(pl.lit(False)).otherwise(pl.col.keeper).alias("keeper")
        index = index_checker.with_columns(pl.when(pl.col("keeper")==pl.any("keeper)).then(True).otherwise(True).alias("keeper")
        new_path = f'(#save path to folder holding new index files#)/index_extra_quality_{str(jobid).zfill(4)}.parquet'
        index.write_parquet(new_path, use_pyarrow=True)
    
if __name__ == "__main__":
    run()