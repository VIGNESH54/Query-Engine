# Architectural Decisions & Trade-offs
## Vectorized SQL Engine from Scratch

Building a SQL query engine from scratch using Python without depending on built-in engines like SQLite or DuckDB exposes several underlying design decisions. This document highlights the critical considerations and tradeoffs we made during this implementation.

### 1. Vectorized Iterator Model vs Volcano Model
- **Decision:** We used the **Vectorized Iterator Model** (processing data column-by-column via `numpy` arrays inside `Chunk` dictionaries) over the traditional Volcano (row-by-row Iterator) model.
- **Trade-off:** Row-based execution is easier to reason about in Python but devastatingly slow due to object creation overhead and iterator resolution per row. By processing chunks (65,536 values per column), we amortize function call overhead entirely and defer the tight inner loops to NumPy's compiled C code. 
- **Consequence:** This forced operators like `HashJoin` and `GroupBy` to deal explicitly with NumPy array manipulation, which is significantly more complex in Python than naive dictionary accumulation.

### 2. Physical Engine - Hash Join Vectorization
- **Decision:** The Hash Build/Probe phases were heavily vectorized. Rather than using pure Python iteration over the chunk, we employed NumPy sorting/offsets arrays `np.repeat` to duplicate rows matching multiple hits.
- **Trade-off:** Pure dictionary-based Hash Maps are fast for scalar probes, but extremely slow in Python when probing millions of rows. NumPy does not have a native Hash Map function that easily scales without C bindings (like Pandas' `factorize`). We traded memory overhead (creating dense `sort_idx` and `inverse_idx` mappings via `np.unique` and `np.argsort`) for speed. Native linear mapping allows us to achieve ~13 second query times over 10M rows vs an estimated 2-5 minutes in naive Python loops.

### 3. Parse and Logical Plan Simplification
- **Decision:** We implemented a rudimentary Regex-based tokenizer and a handwritten recursive descent `Parser`, enforcing strict AST-to-Logical plan generation assumptions (e.g. equijoins only).
- **Trade-off:** Since tools like `SqlAlchemy` were strictly forbidden, this was necessary to adhere to the spec. Building a production-grade parser would have cost thousands of lines of code. The current implementation handles canonical inner joins and aggregation, but query syntax is somewhat brittle if misconfigured.

### 4. Query Optimizer (Rule-based)
- **Decision:** The `Optimizer` implements static rules: **Column Pruning** and **Predicate Pushdown** via recursive AST traversal.
- **Trade-off:** The engine does not have a catalog summarizing schema definitions, so predicate pushdown strictly limits itself above joins to avoid pushing columns that don't belong to the left/right tables erroneously. True cost-based optimization (CBO) was omitted for simplicity, but column pruning alone achieves dramatic speedups by preventing unnecessary I/O of unused columns from Parquet files into memory.

### 5. Benchmark Performance Findings
The benchmark tested joining a 1M rows `users` table and a 10M rows `transactions` table with aggregations:
- **DuckDB (C++):** ~0.47s | ~0.01M peak tracked trace-memory
- **Pandas (C/Cython):** ~2.0s | ~760MB peak memory
- **Custom Engine (Python/NumPy):** ~13.3s | ~879MB peak memory

**Conclusion:** Custom Python using NumPy is ~5-6x slower than highly optimized Pandas, and 25x slower than DuckDB. However, it successfully executes full analytical operations over 10 million rows in 13 seconds without using a single pre-built database query API, entirely via mathematical array transformations.
