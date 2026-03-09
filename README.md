# zvec-rs

High-performance embedded vector database in pure Rust. A feature-complete reimplementation of [zvec](https://github.com/alibaba/zvec) with zero C/C++ dependencies.

## Features

- **HNSW Index** — Hierarchical Navigable Small World graph with per-node RwLocks for concurrent reads/writes, heuristic neighbor selection (Algorithm 4), and bloom filter optimization for large graphs
- **Dense + Sparse Vectors** — Full support for both dense (`Vec<f32>`) and sparse (`SparseVector`) embeddings with dedicated HNSW indexes for each
- **SIMD Distance Kernels** — L2, Inner Product, Cosine, and MIPS metrics with NEON (aarch64) and AVX2+FMA (x86_64) acceleration
- **Hybrid Search** — SQL-like filter expressions (`=`, `!=`, `IN`, `CONTAINS`, `LIKE`, `IS NULL`, range comparisons) with inverted index for O(1) filter evaluation
- **Quantization** — FP16, INT8, INT4, and Product Quantization (PQ) with asymmetric distance computation
- **IVF Index** — Inverted File index with k-means clustering, multi-probe search, and IVF-PQ variant for compressed storage
- **Multi-Vector Fusion** — Reciprocal Rank Fusion (RRF), weighted sum, intersect, and union across multiple result sets
- **SQL Query Engine** — `SELECT`, `COUNT(*)`, `GROUP BY` with `WHERE`, `ORDER BY`, and `LIMIT`
- **Persistence** — redb-based storage with schema persistence, batch writes, and graph topology export/import
- **Pluggable Storage** — `StorageBackend` trait with redb (persistent) and in-memory backends
- **Clustering** — Standalone k-means, mini-batch k-means, and elbow method
- **Collection API** — High-level API with upsert, search, delete-by-filter, group-by, merge, parallel search, and diagnostics

## Quick Start

```rust
use zvec_rs::{Collection, CollectionConfig, MetricType, FieldSchema, FieldType, HnswParams};

// Create a collection with schema
let schema = FieldSchema::new(vec![
    ("category".into(), FieldType::Filtered),
    ("tags".into(), FieldType::Tags),
    ("content".into(), FieldType::String),
]);

let config = CollectionConfig {
    dims: 128,
    metric: MetricType::Cosine,
    hnsw_params: HnswParams::new(16, 200),
    schema,
};

let collection = Collection::new(config);

// Insert documents
let mut fields = std::collections::HashMap::new();
fields.insert("category".into(), "science".into());
fields.insert("tags".into(), "physics,quantum".into());
fields.insert("content".into(), "Quantum computing fundamentals".into());

collection.upsert("doc-1", &embedding_vector, fields);

// Search with filter
let results = collection.search(
    &query_vector,
    10,                                    // top-k
    Some("category = 'science'"),          // filter
    None,                                  // output fields
);

// SQL-like queries
use zvec_rs::Query;
let query = Query::parse("SELECT content, category WHERE tags CONTAINS 'physics' LIMIT 5")?;
let result = collection.execute_query(&query, Some(&query_vector));
```

## Persistence

```rust
// Persistent collection (uses redb)
let collection = Collection::open("/path/to/data", config)?;
collection.upsert("doc-1", &vector, fields);
collection.flush()?;

// Reopen later — schema, vectors, metadata, and graph topology are all restored
let collection = Collection::open("/path/to/data", config)?;
```

## Benchmarks

Measured on Apple M4 Max with NEON SIMD, compiled with `RUSTFLAGS="-C target-cpu=native"`.

### Distance Kernels

| Operation | dim=128 | dim=768 | dim=1536 |
|-----------|---------|---------|----------|
| L2 Squared | 9.7 ns | 108 ns | 273 ns |
| Inner Product | 9.3 ns | 106 ns | 250 ns |
| Cosine | 13.4 ns | 123 ns | 295 ns |

### HNSW Index (dim=128, M=16, ef_construction=100)

| Operation | Value |
|-----------|-------|
| Build 1K vectors | 197 ms |
| Build 10K vectors | 2.73 s |
| Search top-10 (10K index) | 40 us |
| Search top-50 (10K index) | 40 us |
| Search top-100 (10K index) | 81 us |

*Search with ef_search=50, M=32 on 10K pre-built index.*

## Architecture

```
zvec-rs/
  src/
    collection.rs    — High-level Collection API (upsert, search, filter, merge, diagnostics)
    distance/        — SIMD-accelerated L2, IP, Cosine, MIPS (NEON + AVX2)
    hnsw/            — HNSW graph (dense + sparse), concurrent with per-node RwLocks
    ivf.rs           — IVF index with k-means + IVF-PQ variant
    filter/          — SQL-like filter parser and evaluator
    query.rs         — SQL query engine (SELECT, COUNT, GROUP BY)
    quantize.rs      — FP16, INT8, INT4, Product Quantization
    multi_vector.rs  — Result fusion (RRF, weighted sum, intersect, union)
    cluster.rs       — k-means, mini-batch k-means, elbow method
    schema.rs        — Field schema (String, Filtered, Tags)
    sparse.rs        — Sparse vector type with dot/L2/cosine
    storage/         — StorageBackend trait, redb + in-memory backends
```

## Building

```bash
# Build
cargo build --release

# Run tests (241 tests)
cargo test

# Run benchmarks with native SIMD
RUSTFLAGS="-C target-cpu=native" cargo bench
```

For best performance, always compile with native CPU features:

```bash
# Enables AVX2+FMA on x86_64, NEON on aarch64
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Comparison with C++ zvec

zvec-rs is a pure Rust reimplementation with full feature parity:

| | C++ zvec | zvec-rs |
|---|---------|---------|
| Language | C++ | Pure Rust |
| Dependencies | RocksDB, glog, protobuf, Arrow, etc. | redb, parking_lot, rayon |
| Storage | RocksDB | redb (pure Rust) |
| SIMD | Manual intrinsics | Auto-vectorized + manual NEON/AVX2 |
| Concurrency | Thread pools | Per-node RwLocks + rayon |
| Build time | ~15 min (with deps) | ~30s |
| Binary size | ~50 MB (static) | ~5 MB |
| `dlopen` safe | No (static destructor crashes) | Yes |

## License

MIT
