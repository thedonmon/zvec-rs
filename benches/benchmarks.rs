use std::collections::HashMap;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::Rng;
use zvec_rs::{
    Collection, CollectionConfig, FieldSchema, FieldType, Fp16Vec, HnswIndex, HnswParams, Int8Vec,
    IvfIndex, IvfParams, MetricType, PqCodebook,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn random_vectors(n: usize, dims: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|_| (0..dims).map(|_| rng.gen::<f32>() - 0.5).collect())
        .collect()
}

fn random_vector(dims: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dims).map(|_| rng.gen::<f32>() - 0.5).collect()
}

// ===========================================================================
// 1. Distance Kernels
// ===========================================================================

fn bench_distance_kernels(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance");

    for &dims in &[128, 384, 768, 1536] {
        let a = random_vector(dims);
        let b = random_vector(dims);

        group.throughput(Throughput::Elements(dims as u64));

        group.bench_with_input(BenchmarkId::new("l2", dims), &dims, |bench, _| {
            bench.iter(|| zvec_rs::distance::l2_squared(&a, &b));
        });

        group.bench_with_input(BenchmarkId::new("ip", dims), &dims, |bench, _| {
            bench.iter(|| zvec_rs::distance::inner_product(&a, &b));
        });

        group.bench_with_input(BenchmarkId::new("cosine", dims), &dims, |bench, _| {
            bench.iter(|| zvec_rs::distance::cosine_similarity(&a, &b));
        });
    }

    group.finish();
}

// ===========================================================================
// 2. HNSW Operations
// ===========================================================================

fn bench_hnsw_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_insert");
    let dims = 128;

    for &n in &[1_000, 10_000] {
        let vectors = random_vectors(n, dims);

        group.throughput(Throughput::Elements(n as u64));
        group.sample_size(10);

        group.bench_with_input(BenchmarkId::new("vectors", n), &n, |b, _| {
            b.iter(|| {
                let index = HnswIndex::new(dims, MetricType::L2, HnswParams::new(16, 100));
                for (i, v) in vectors.iter().enumerate() {
                    index.insert(i as u64, v);
                }
            });
        });
    }

    group.finish();
}

fn bench_hnsw_insert_100k(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_insert_100k");
    let dims = 128;
    let n = 100_000;
    let vectors = random_vectors(n, dims);

    group.throughput(Throughput::Elements(n as u64));
    group.sample_size(10);
    group.measurement_time(std::time::Duration::from_secs(30));

    group.bench_function("vectors_100k", |b| {
        b.iter(|| {
            let index = HnswIndex::new(dims, MetricType::L2, HnswParams::new(16, 100));
            for (i, v) in vectors.iter().enumerate() {
                index.insert(i as u64, v);
            }
        });
    });

    group.finish();
}

fn bench_hnsw_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");
    let dims = 128;
    let n = 10_000;

    let index = HnswIndex::new(
        dims,
        MetricType::L2,
        HnswParams::new(32, 200).with_ef_search(100),
    );
    let vectors = random_vectors(n, dims);
    for (i, v) in vectors.iter().enumerate() {
        index.insert(i as u64, v);
    }

    let query = random_vector(dims);

    for &k in &[10, 100] {
        group.bench_with_input(BenchmarkId::new("top_k", k), &k, |b, &k| {
            b.iter(|| index.search(&query, k));
        });
    }

    group.finish();
}

fn bench_hnsw_concurrent_search(c: &mut Criterion) {
    use std::sync::Arc;

    let mut group = c.benchmark_group("hnsw_concurrent_search");
    let dims = 128;
    let n = 10_000;

    let index = Arc::new(HnswIndex::new(
        dims,
        MetricType::L2,
        HnswParams::new(32, 200).with_ef_search(100),
    ));
    let vectors = random_vectors(n, dims);
    for (i, v) in vectors.iter().enumerate() {
        index.insert(i as u64, v);
    }

    let queries: Vec<Vec<f32>> = random_vectors(8, dims);

    for &n_threads in &[2, 4, 8] {
        group.bench_with_input(
            BenchmarkId::new("threads", n_threads),
            &n_threads,
            |b, &n_threads| {
                b.iter(|| {
                    std::thread::scope(|s| {
                        let handles: Vec<_> = (0..n_threads)
                            .map(|t| {
                                let idx = &index;
                                let q = &queries[t % queries.len()];
                                s.spawn(move || idx.search(q, 10))
                            })
                            .collect();
                        for h in handles {
                            let _ = h.join().unwrap();
                        }
                    });
                });
            },
        );
    }

    group.finish();
}

// ===========================================================================
// 3. Collection Operations
// ===========================================================================

fn build_collection(n: usize, dims: usize) -> (Collection, Vec<Vec<f32>>) {
    let schema = FieldSchema::new(vec![
        ("category".into(), FieldType::Filtered),
        ("tags".into(), FieldType::Tags),
        ("content".into(), FieldType::String),
    ]);
    let config = CollectionConfig::new(dims)
        .with_metric(MetricType::IP)
        .with_hnsw_params(HnswParams::new(16, 100))
        .with_schema(schema);
    let collection = Collection::new(config);

    let mut rng = rand::thread_rng();
    let categories = ["science", "tech", "art", "music", "sports"];
    let all_tags = ["ml", "rust", "gpu", "cloud", "open-source", "research"];

    let vectors = random_vectors(n, dims);
    for (i, v) in vectors.iter().enumerate() {
        let mut fields = HashMap::new();
        fields.insert(
            "category".to_string(),
            categories[rng.gen_range(0..categories.len())].to_string(),
        );
        let n_tags = rng.gen_range(1..=3);
        let tags: Vec<&str> = (0..n_tags)
            .map(|_| all_tags[rng.gen_range(0..all_tags.len())])
            .collect();
        fields.insert("tags".to_string(), tags.join(","));
        fields.insert("content".to_string(), format!("document {}", i));

        collection.upsert(&format!("doc-{}", i), v, fields);
    }

    (collection, vectors)
}

fn bench_collection_upsert(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_upsert");
    let dims = 128;
    let n = 1_000;

    let schema = FieldSchema::new(vec![
        ("category".into(), FieldType::Filtered),
        ("tags".into(), FieldType::Tags),
    ]);

    let vectors = random_vectors(n, dims);
    let categories = ["science", "tech", "art", "music", "sports"];

    group.throughput(Throughput::Elements(n as u64));
    group.sample_size(10);

    group.bench_function("with_metadata", |b| {
        let mut rng = rand::thread_rng();
        b.iter(|| {
            let config = CollectionConfig::new(dims)
                .with_metric(MetricType::IP)
                .with_hnsw_params(HnswParams::new(16, 100))
                .with_schema(schema.clone());
            let col = Collection::new(config);
            for (i, v) in vectors.iter().enumerate() {
                let mut fields = HashMap::new();
                fields.insert(
                    "category".to_string(),
                    categories[rng.gen_range(0..categories.len())].to_string(),
                );
                fields.insert("tags".to_string(), "ml,rust".to_string());
                col.upsert(&format!("doc-{}", i), v, fields);
            }
        });
    });

    group.finish();
}

fn bench_collection_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_search");
    let dims = 128;
    let n = 5_000;

    let (collection, _) = build_collection(n, dims);
    let query = random_vector(dims);

    group.bench_function("no_filter", |b| {
        b.iter(|| collection.search(&query, 10, None));
    });

    group.bench_function("with_filter", |b| {
        b.iter(|| collection.search(&query, 10, Some("category = 'tech'")));
    });

    group.finish();
}

fn bench_collection_group_by(c: &mut Criterion) {
    let mut group = c.benchmark_group("collection_group_by");
    let dims = 128;
    let n = 5_000;

    let (collection, _) = build_collection(n, dims);
    let query = random_vector(dims);

    group.bench_function("group_by_category", |b| {
        b.iter(|| collection.group_by_search(&query, "category", 5, 3, None));
    });

    group.bench_function("group_by_filtered", |b| {
        b.iter(|| {
            collection.group_by_search(&query, "category", 5, 3, Some("tags CONTAINS 'ml'"))
        });
    });

    group.finish();
}

// ===========================================================================
// 4. Quantization
// ===========================================================================

fn bench_fp16(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_fp16");
    let dims = 768;
    let data = random_vector(dims);

    group.bench_function("encode", |b| {
        b.iter(|| Fp16Vec::from_f32(&data));
    });

    let encoded = Fp16Vec::from_f32(&data);
    group.bench_function("decode", |b| {
        b.iter(|| encoded.to_f32());
    });

    group.finish();
}

fn bench_int8(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_int8");
    let dims = 768;
    let data = random_vector(dims);

    group.bench_function("encode", |b| {
        b.iter(|| Int8Vec::from_f32(&data));
    });

    let encoded = Int8Vec::from_f32(&data);
    group.bench_function("decode", |b| {
        b.iter(|| encoded.to_f32());
    });

    let a = Int8Vec::from_f32(&random_vector(dims));
    let b_vec = Int8Vec::from_f32(&random_vector(dims));
    group.bench_function("dot_product", |b| {
        b.iter(|| Int8Vec::dot(&a, &b_vec));
    });

    group.finish();
}

fn bench_pq(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantize_pq");
    let dims = 128;
    let m = 8;
    let k = 256;
    let n_train = 1_000;

    let train_data = random_vectors(n_train, dims);
    let refs: Vec<&[f32]> = train_data.iter().map(|v| v.as_slice()).collect();

    group.sample_size(10);

    group.bench_function("train", |b| {
        b.iter(|| PqCodebook::train(&refs, dims, m, k, 10));
    });

    let codebook = PqCodebook::train(&refs, dims, m, k, 10);
    let test_vec = random_vector(dims);

    group.bench_function("encode", |b| {
        b.iter(|| codebook.encode(&test_vec));
    });

    let code = codebook.encode(&test_vec);
    group.bench_function("decode", |b| {
        b.iter(|| codebook.decode(&code));
    });

    let query = random_vector(dims);
    group.bench_function("adc_distance", |b| {
        b.iter(|| codebook.asymmetric_distance_l2(&query, &code));
    });

    group.bench_function("build_distance_table", |b| {
        b.iter(|| codebook.build_distance_table(&query));
    });

    let table = codebook.build_distance_table(&query);
    group.bench_function("distance_with_table", |b| {
        b.iter(|| codebook.distance_with_table(&table, &code));
    });

    group.finish();
}

// ===========================================================================
// 5. IVF
// ===========================================================================

fn bench_ivf(c: &mut Criterion) {
    let mut group = c.benchmark_group("ivf");
    let dims = 128;
    let n = 10_000;

    let vectors = random_vectors(n, dims);

    group.sample_size(10);

    group.bench_function("train", |b| {
        b.iter(|| {
            let mut index = IvfIndex::new(
                dims,
                MetricType::L2,
                IvfParams {
                    n_list: 100,
                    n_probe: 10,
                    n_iters: 10,
                    ..Default::default()
                },
            );
            index.train(&vectors);
        });
    });

    // Build a trained+populated index for search benchmarks.
    let mut index = IvfIndex::new(
        dims,
        MetricType::L2,
        IvfParams {
            n_list: 100,
            n_probe: 10,
            n_iters: 10,
            ..Default::default()
        },
    );
    index.train(&vectors);
    for (i, v) in vectors.iter().enumerate() {
        index.insert(i as u64, v.clone());
    }

    group.bench_function("insert_100", |b| {
        b.iter(|| {
            let mut idx = IvfIndex::new(
                dims,
                MetricType::L2,
                IvfParams {
                    n_list: 10,
                    n_probe: 10,
                    n_iters: 5,
                    ..Default::default()
                },
            );
            idx.train(&vectors[..200]);
            for i in 0..100 {
                idx.insert(i as u64, random_vector(dims));
            }
        });
    });

    let query = random_vector(dims);

    for &n_probe in &[1, 5, 10, 50] {
        group.bench_with_input(
            BenchmarkId::new("search_n_probe", n_probe),
            &n_probe,
            |b, &n_probe| {
                b.iter(|| index.search(&query, 10, n_probe));
            },
        );
    }

    group.finish();
}

// ===========================================================================
// 6. Batch Operations
// ===========================================================================

fn bench_batch_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_distances");
    let dims = 128;

    let query = random_vector(dims);
    let corpus = random_vectors(10_000, dims);

    group.throughput(Throughput::Elements(corpus.len() as u64));

    group.bench_function("l2_10k", |b| {
        b.iter(|| {
            let _dists: Vec<f32> = corpus
                .iter()
                .map(|v| zvec_rs::distance::l2_squared(&query, v))
                .collect();
        });
    });

    group.bench_function("ip_10k", |b| {
        b.iter(|| {
            let _dists: Vec<f32> = corpus
                .iter()
                .map(|v| zvec_rs::distance::inner_product(&query, v))
                .collect();
        });
    });

    group.finish();
}

fn bench_batch_top_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_top_k");
    let dims = 128;

    let query = random_vector(dims);
    let corpus = random_vectors(10_000, dims);

    group.bench_function("brute_force_top10", |b| {
        b.iter(|| {
            let mut dists: Vec<(usize, f32)> = corpus
                .iter()
                .enumerate()
                .map(|(i, v)| (i, zvec_rs::distance::l2_squared(&query, v)))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.truncate(10);
            dists
        });
    });

    group.bench_function("brute_force_top100", |b| {
        b.iter(|| {
            let mut dists: Vec<(usize, f32)> = corpus
                .iter()
                .enumerate()
                .map(|(i, v)| (i, zvec_rs::distance::l2_squared(&query, v)))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.truncate(100);
            dists
        });
    });

    group.finish();
}

fn bench_distance_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_matrix");

    for &n in &[100, 500] {
        let dims = 128;
        let vectors = random_vectors(n, dims);

        group.throughput(Throughput::Elements((n * n) as u64));
        group.sample_size(10);

        group.bench_with_input(BenchmarkId::new("l2_nxn", n), &n, |b, &n| {
            b.iter(|| {
                let mut matrix = vec![0.0f32; n * n];
                for i in 0..n {
                    for j in i + 1..n {
                        let d = zvec_rs::distance::l2_squared(&vectors[i], &vectors[j]);
                        matrix[i * n + j] = d;
                        matrix[j * n + i] = d;
                    }
                }
                matrix
            });
        });
    }

    group.finish();
}

// ===========================================================================
// Group and main
// ===========================================================================

criterion_group!(
    distance_benches,
    bench_distance_kernels,
);

criterion_group!(
    hnsw_benches,
    bench_hnsw_insert,
    bench_hnsw_search,
    bench_hnsw_concurrent_search,
);

criterion_group! {
    name = hnsw_heavy;
    config = Criterion::default().sample_size(10).measurement_time(std::time::Duration::from_secs(30));
    targets = bench_hnsw_insert_100k
}

criterion_group!(
    collection_benches,
    bench_collection_upsert,
    bench_collection_search,
    bench_collection_group_by,
);

criterion_group!(
    quantize_benches,
    bench_fp16,
    bench_int8,
    bench_pq,
);

criterion_group!(
    ivf_benches,
    bench_ivf,
);

criterion_group!(
    batch_benches,
    bench_batch_distances,
    bench_batch_top_k,
    bench_distance_matrix,
);

criterion_main!(
    distance_benches,
    hnsw_benches,
    hnsw_heavy,
    collection_benches,
    quantize_benches,
    ivf_benches,
    batch_benches,
);
