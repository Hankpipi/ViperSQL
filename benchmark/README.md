# Benchmark

## Datasets

We evaluate ViperSQL using two datasets: the **Star Schema Benchmark (SSB)** and a **Point of Interest (POI)** dataset.

### Star Schema Benchmark (SSB)

The SSB is a widely adopted benchmark for data warehouse systems. It features a star schema consisting of one large fact table and four dimension tables, supporting a variety of queries involving joins, aggregations, and selections. To generate the SSB dataset, follow these steps:

```bash
# Prepare the environment for data generation
python3 -m venv data_venv
source data_venv/bin/activate
pip install pandas pyarrow fastparquet

# Data generation
mkdir data/storage
./data/generate_csv.py --sf 16
deactivate
```

### Point of Interest (POI) Dataset

The POI dataset is collected from Yelp and contains 7 million entries, each with rich textual descriptions and associated metadata. This dataset is used to evaluate LLM-driven workloads in ViperSQL.

The POI dataset is available for download [here](https://drive.google.com/file/d/1Th5xxogxCinAjZeJ1Pn0ptV9jmp4JuHo/view?usp=drive_link).



## System Installation

Please refer to the official documentation of each baseline system for installation instructions:

1. [HeavyDB](https://docs.heavy.ai/)
2. [MySQL](https://dev.mysql.com/doc/relnotes/mysql/8.0/en/)
3. [DuckDB](https://duckdb.org/docs/stable/)
4. [Lotus](https://lotus-ai.readthedocs.io/en/latest/)



## Test Queries

The SQL queries used to evaluate the system are provided in the [`./sql`](./sql) directory.
