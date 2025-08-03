# ViperSQL: A Unified Database with LLM-assisted and Hardware-accelerated Data Analysis

**ViperSQL** is a next-generation data platform built upon MySQL 8.0.32. The system integrates Large Language Models (LLMs) and GPU acceleration into the core query execution engine, providing unified, efficient, and intelligent data analysis capabilities.

---

## Installation

1. **Clone the Repository**

   Clone the ViperSQL repository along with its submodules:
   ```bash
   git clone https://github.com/Hankpipi/ViperSQL.git
   cd ViperSQL
   git submodule update --init --recursive
   ```

2. **Build and Install**

   Configure and compile the project with the recommended settings:
   ```bash
   cmake . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DWITH_SSL=system -DWITH_ZLIB=bundled -DWITH_ZSTD=bundled -DMYSQL_MAINTAINER_MODE=0 -DENABLED_LOCAL_INFILE=1 -DENABLE_DTRACE=0 -DCMAKE_CXX_FLAGS="-march=native -w" -DFORCE_INSOURCE_BUILD=1 -DDOWNLOAD_BOOST=1 -DWITH_BOOST=./boost/ -DWITH_FB_VECTORDB=1 -DCMAKE_INSTALL_PREFIX=./myrocks -DWITH_SEMANTICDB=1

   make -j 8
   ```

3. **Configuration**

   Create a MySQL configuration file (e.g., `my.cnf`) with the following template:
   ```ini
   [mysqld]
   user        = mysql
   pid-file    = /home/zihao/data_test/mysqld.pid
   socket      = /home/zihao/data_test/mysqld.sock
   port        = 3333
   datadir     = /home/zihao/data_test
   tmpdir      = /home/zihao/mysqltmp

   bind-address        = 127.0.0.1
   mysqlx-bind-address = 127.0.0.1

   myisam-recover-options  = BACKUP
   log_error               = /home/zihao/mysql-log/error.log
   ```

4. **Initialize the Data Directory**

   Initialize the MySQL data directory:
   ```bash
   bin/mysqld --defaults-file=./my.cnf --initialize
   ```

5. **Start the Server and Client**

   - Start the MySQL server:
     ```bash
     bin/mysqld --defaults-file=./my.cnf
     ```
   - Connect using the MySQL client (from a separate terminal):
     ```bash
     bin/mysql -u root -p --port=3333 -h 127.0.0.1
     ```


## Evaluation

Detailed instructions on datasets and benchmarking procedures can be found in the [Benchmark Directory](./benchmark/README).

