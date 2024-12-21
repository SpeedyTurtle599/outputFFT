use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rustfft::{FftPlanner, num_complex::Complex};
use std::{sync::{Arc, Mutex}, thread, time::Duration};

const FFT_SIZE: usize = 1024; // Reduced to match typical audio buffer size
const SAMPLE_RATE: u32 = 44100;

fn apply_hanning_window(buffer: &mut [f32]) {
    for i in 0..buffer.len() {
        let multiplier = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 
            / (buffer.len() - 1) as f32).cos());
        buffer[i] *= multiplier;
    }
}

fn main() {
    let host = cpal::default_host();
    let device = host.default_input_device().expect("no input device");
    
    let config = cpal::StreamConfig {
        channels: 1,
        sample_rate: cpal::SampleRate(SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Fixed(FFT_SIZE as u32),
    };

    let samples = Arc::new(Mutex::new(vec![0.0f32; FFT_SIZE]));
    let samples_clone = samples.clone();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(FFT_SIZE);

    let stream = device.build_input_stream(
        &config,
        move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut samples = samples_clone.lock().unwrap();
            if data.len() == FFT_SIZE {
                samples.copy_from_slice(data);
            }
        },
        |err| eprintln!("Error: {}", err),
        None
    ).expect("failed to build stream");

    stream.play().expect("failed to play stream");

    // Process and display FFT
    loop {
        let mut buffer: Vec<Complex<f32>> = {
            let mut samples = samples.lock().unwrap();
            apply_hanning_window(&mut samples);
            samples.iter().map(|&s| Complex::new(s, 0.0)).collect()
        };

        fft.process(&mut buffer);

        println!("\x1B[2J\x1B[H"); // Clear screen and move cursor to top
        
        // Show frequency bands with logarithmic scaling
        for i in 0..FFT_SIZE/2 {
            let magnitude = buffer[i].norm() * 2.0 / FFT_SIZE as f32;
            let freq = i as f32 * SAMPLE_RATE as f32 / FFT_SIZE as f32;
            if freq < 2000.0 {
                let db = 20.0 * magnitude.log10();
                let normalized_db = ((db + 60.0) / 60.0).max(0.0); // Normalize to 0-1 range
                let bars = (normalized_db * 50.0) as usize;
                if i % 4 == 0 { // Show fewer frequency labels
                    println!("{:4.0}Hz: {}", freq, "#".repeat(bars));
                }
            }
        }
        
        thread::sleep(Duration::from_millis(50));
    }
}