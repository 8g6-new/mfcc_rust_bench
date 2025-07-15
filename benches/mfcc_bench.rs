use criterion::{black_box, criterion_group, criterion_main, Criterion};
use mfcc::Transform;
use rayon::prelude::*; // <- Rayon import

pub fn calculate_mfcc_2d_rayon(c: &mut Criterion) {
    let sample_rate = 16000;
    let duration_secs = 1;
    let total_samples = sample_rate * duration_secs;

    let signal: Vec<i16> = (0..total_samples)
        .map(|x| {
            let freq = 440.0;
            let t = x as f32 / sample_rate as f32;
            (32000.0 * (2.0 * std::f32::consts::PI * freq * t).sin()) as i16
        })
        .collect();

    let frame_len = 1024;
    let hop = 512;
    let num_frames = (total_samples - frame_len) / hop;

    let num_coeffs = 20;
    let mfcc_template = Transform::new(sample_rate, frame_len)
        .nfilters(num_coeffs, 40)
        .normlength(10);

    let mut frames: Vec<&[i16]> = (0..num_frames)
        .map(|i| &signal[i * hop..i * hop + frame_len])
        .collect();

    // Preallocate output buffer
    let mut outputs: Vec<Vec<f64>> = vec![vec![0.0; num_coeffs * 3]; num_frames];

    c.bench_function("transform_2d_rayon", |b| {
        b.iter(|| {
            // Clone a transform per thread (not thread-safe otherwise)
            frames
            .par_iter()
            .zip(outputs.par_iter_mut())
            .for_each(|(frame, out)| {
                // Build a fresh Transform for this thread
                let mut mfcc = Transform::new(sample_rate, frame_len)
                    .nfilters(num_coeffs, 40)
                    .normlength(10);
        
                mfcc.transform(frame, out);
            });
        })
    });
}

criterion_group!(benches, calculate_mfcc_2d_rayon);
criterion_main!(benches);
