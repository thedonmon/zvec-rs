use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::Rng;
use zvec_rs::{HnswIndex, HnswParams, MetricType};

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("insert");
    let dims = 128;

    for n in [1000, 10_000] {
        group.bench_with_input(BenchmarkId::new("vectors", n), &n, |b, &n| {
            let mut rng = rand::thread_rng();
            let vectors: Vec<Vec<f32>> = (0..n)
                .map(|_| (0..dims).map(|_| rng.gen()).collect())
                .collect();

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

fn bench_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("search");
    let dims = 128;
    let n = 10_000;

    let mut rng = rand::thread_rng();
    let index = HnswIndex::new(
        dims,
        MetricType::L2,
        HnswParams::new(32, 200).with_ef_search(50),
    );
    for i in 0..n {
        let v: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
        index.insert(i, &v);
    }

    for k in [10, 50, 100] {
        group.bench_with_input(BenchmarkId::new("top_k", k), &k, |b, &k| {
            let query: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
            b.iter(|| index.search(&query, k));
        });
    }
    group.finish();
}

fn bench_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance");
    let mut rng = rand::thread_rng();

    for dims in [128, 768, 1536] {
        let a: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();
        let b: Vec<f32> = (0..dims).map(|_| rng.gen()).collect();

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

criterion_group!(benches, bench_distance, bench_insert, bench_search);
criterion_main!(benches);
