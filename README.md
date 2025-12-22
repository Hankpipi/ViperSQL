# ViperSQL

**A Unified Database with LLM-assisted and Hardware-accelerated Data Analysis**

**ViperSQL** is a next-generation data platform built upon MySQL 8.0.32. The system integrates Large Language Models (LLMs) and GPU acceleration directly into the core query execution engine, providing unified, efficient, and intelligent data analysis capabilities.

---

## 📋 Prerequisites

Before building ViperSQL, ensure your environment meets the following requirements:

1. **CUDA Driver**: Ensure that the appropriate CUDA driver is installed for your GPU to enable hardware-accelerated query processing.
2. **OpenAI API Key**: Export your OpenAI API key as an environment variable to enable LLM-assisted query processing.

   ```bash
   export OPENAI_API_KEY=sk-or-v1-xxxxxxxx
   ```

## ⚙️ Build Instructions

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
   cmake . -DCMAKE_BUILD_TYPE=RelWithDebInfo -DWITH_SSL=system -DWITH_ZLIB=bundled -DWITH_ZSTD=bundled -DMYSQL_MAINTAINER_MODE=0 -DENABLED_LOCAL_INFILE=1 -DENABLE_DTRACE=0 -DCMAKE_CXX_FLAGS="-march=native -w" -DFORCE_INSOURCE_BUILD=1 -DDOWNLOAD_BOOST=1 -DWITH_BOOST=./boost/ -DWITH_FB_VECTORDB=1 -DCMAKE_INSTALL_PREFIX=./myrocks -DWITH_SEMANTICDB=1 -DCMAKE_EXE_LINKER_FLAGS="-Wl,--no-as-needed -lzmq"   -DCMAKE_SHARED_LINKER_FLAGS="-Wl,--no-as-needed -lzmq"

   make -j 8
   ```

## 🚀 Configuration & Runtime

1. **Configure MySQL (`my.cnf`)**

   Create a MySQL configuration file (e.g., `my.cnf`) with the following template. Note: Replace `/path/to/` with your actual directory paths.
   ```ini
   [mysqld]
   user                = mysql
   pid-file            = /path/to/data_test/mysqld.pid
   socket              = /path/to/data_test/mysqld.sock
   port                = 3333
   datadir             = /path/to/data_test
   tmpdir              = /path/to/mysqltmp

   bind-address        = 127.0.0.1
   mysqlx-bind-address = 127.0.0.1

   myisam-recover-options  = BACKUP
   log_error               = /path/to/mysql-log/error.log
   ```

2. **Initialize the Data Directory**

   Initialize the MySQL data directory:
   ```bash
   bin/mysqld --defaults-file=./my.cnf --initialize
   ```

3. **Start the Server and Client**

   - Start the MySQL server:
     ```bash
     bin/mysqld --defaults-file=./my.cnf
     ```
   - Connect using the MySQL client (from a separate terminal):
     ```bash
     bin/mysql -u root -p --port=3333 -h 127.0.0.1
     ```

## 📊 Example Usage

### Semantic Filter

Filter rows based on natural language logic.

```sql
SELECT id, text 
FROM poi WHERE
SEMANTIC_FILTER_SINGLE_COL('Is {poi.text} a positive comment?', text) = 1;
```

### Semantic Generate

Rewrite or generate content using the LLM engine.

```sql
SELECT text, 
       semantic_generate('rewrite {poi.text} to make it formal.', text) as revised_review 
FROM poi;
```

### Semantic Join

Join tables based on semantic relevance rather than exact matches.

```sql
SELECT content, topic
FROM poi, topic
WHERE SEM_JOIN( 'Is {poi.content} relevant to {topic.topic}?', poi.content, topic.topic);
```

## 📈 Evaluation

Detailed instructions on datasets and benchmarking procedures can be found in the [Benchmark Directory](./benchmark/README.md).

