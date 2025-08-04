#! /usr/bin/env python3

import os
import argparse
import pandas as pd
import subprocess as subp

TABLE_COL_DTYPE = {
    "customer": [
        "int32",
        "str",
        "str",
        "str",
        "str",
        "str",
        "str",
        "str",
        "dummy",
    ],
    "date": [
        "int32",
        "str",
        "str",
        "str",
        "int32",
        "int32",
        "str",
        "int32",
        "int32",
        "int32",
        "int32",
        "int32",
        "str",
        "int32",
        "int32",
        "int32",
        "int32",
        "dummy",
    ],
    "lineorder": [
        "int32",
        "int32",
        "int32",
        "int32",
        "int32",
        "int32",
        "str",
        "str",
        "int32",
        "int32",
        "int32",
        "int32",
        "int32",
        "int32",
        "int32",
        "int32",
        "str",
        "dummy",
    ],
    "supplier": [
        "int32",
        "str",
        "str",
        "str",
        "str",
        "str",
        "str",
        "str",
        "dummy",
    ],
    "part": [
        "int32",
        "str",
        "str",
        "str",
        "str",
        "str",
        "str",
        "int32",
        "str",
        "dummy",
    ],
}


TABLE_COL_NAME = {
    "customer": [
        "c_custkey",
        "c_name",
        "c_address",
        "c_city",
        "c_nation",
        "c_region",
        "c_phone",
        "c_mktsegment",
        "dummy",
    ],
    "date": [
        "d_datekey",
        "d_date",
        "d_dayofweek",
        "d_month",
        "d_year",
        "d_yearmonthnum",
        "d_yearmonth",
        "d_daynuminweek",
        "d_daynuminmonth",
        "d_daynuminyear",
        "d_monthnuminyear",
        "d_weeknuminyear",
        "d_sellingseasin",
        "d_lastdayinweekfl",
        "d_lastdayinmonthfl",
        "d_holidayfl",
        "d_weekdayfl",
        "dummy",
    ],
    "lineorder": [
        "lo_orderkey",
        "lo_linenumber",
        "lo_custkey",
        "lo_partkey",
        "lo_suppkey",
        "lo_orderdate",
        "lo_orderpriority",
        "lo_shippriority",
        "lo_quantity",
        "lo_extendedprice",
        "lo_ordtotalprice",
        "lo_discount",
        "lo_revenue",
        "lo_supplycost",
        "lo_tax",
        "lo_commitdate",
        "lo_shopmode",
        "dummy",
    ],
    "supplier": [
        "s_suppkey",
        "s_name",
        "s_address",
        "s_city",
        "s_nation",
        "s_region",
        "s_phone",
        "dummy",
    ],
    "part": [
        "p_partkey",
        "p_name",
        "p_mfgr",
        "p_category",
        "p_brand1",
        "p_color",
        "p_type",
        "p_size",
        "p_container",
        "dummy",
    ],
}
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sf", type=int, default=1, help="Scale factor.")
    args = parser.parse_args()

    sf = args.sf

    os.chdir("./data/ssbm/ssbm_src")

    # Clean previous generated tables
    for path in os.listdir("."):
        if ".tbl" in path or ".tbl.p" in path or ".parquet" in path or ".csv" in path:
            os.remove(path)

    # Build and generate database
    subp.Popen(["make", "clean"], stdin=subp.PIPE, stdout=subp.PIPE, stderr=subp.STDOUT).wait()
    subp.Popen(["make"], stdin=subp.PIPE, stdout=subp.PIPE, stderr=subp.STDOUT).wait()
    subp.Popen(["./dbgen", "-s", str(sf), "-T", "a", "-f"], stdin=subp.PIPE, stdout=subp.PIPE, stderr=subp.STDOUT).wait()

    # Convert to CSV and move to storage
    for tb_path in os.listdir("."):
        if "tbl" not in tb_path:
            continue
        name = tb_path.split(".")[0]
        output_path = f"sf{sf}_{name}.csv"

        # Prepare dtypes
        dtype_dict = {}
        for i in range(len(TABLE_COL_NAME[name])):
            col_name = TABLE_COL_NAME[name][i]
            if col_name == "dummy":
                continue
            dtype_dict[col_name] = TABLE_COL_DTYPE[name][i]

        df = pd.read_csv(
            tb_path,
            delimiter="|",
            names=TABLE_COL_NAME[name],
            dtype=dtype_dict,
        )
        # Remove the "dummy" column if it exists
        if "dummy" in df.columns:
            df = df.drop(columns=["dummy"])

        df.to_csv(output_path, index=False)

        # Move to storage directory
        os.chdir("../../../")
        subp.Popen(
            ["cp", os.path.join("./data/ssbm/ssbm_src", output_path), "./data/storage/"],
            stdin=subp.PIPE, stdout=subp.PIPE, stderr=subp.STDOUT,
        ).wait()
        # Optionally copy original tbl for reference
        subp.Popen(
            [
                "cp",
                os.path.join("./data/ssbm/ssbm_src", tb_path),
                os.path.join("./data/storage/", output_path.replace("csv", "txt")),
            ],
            stdin=subp.PIPE, stdout=subp.PIPE, stderr=subp.STDOUT,
        ).wait()
        os.chdir("./data/ssbm/ssbm_src")

if __name__ == "__main__":
    main()
