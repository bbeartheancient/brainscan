use crate::types::QAMSignal;
use std::f64::consts::PI;
use tokio::time::{interval, Duration};
use tracing::info;
use std::sync::atomic::{AtomicBool, Ordering, AtomicU64};
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use chrono::Utc;

/// Simulated QAM16 signal generator for network-based spatialization
/// Generates 16 constellation points mapped to 4 FOA quadrants
pub struct SimulatedQAM {
    sample_rate: u32,
    running: Arc<AtomicBool>,
    frame_count: Arc<AtomicU64>,
    // 16 base frequencies for constellation points (4 per quadrant)
    frequencies: Vec<f64>,
    snr: Arc<RwLock<f64>>, // SNR stored as RwLock since AtomicF64 doesn't exist
}

impl SimulatedQAM {
    pub fn new(sample_rate: u32) -> Self {
        // 16 frequencies for QAM constellation points
        // Organized by FOA quadrant:
        // LFU (0-3): 8, 9, 10, 11 Hz
        // LBD (4-7): 12, 13, 14, 15 Hz  
        // RBU (8-11): 16, 17, 18, 19 Hz
        // RFD (12-15): 20, 21, 22, 23 Hz
        let frequencies = vec![
            8.0, 9.0, 10.0, 11.0,    // LFU quadrant
            12.0, 13.0, 14.0, 15.0,   // LBD quadrant
            16.0, 17.0, 18.0, 19.0,   // RBU quadrant
            20.0, 21.0, 22.0, 23.0,   // RFD quadrant
        ];
        
        Self {
            sample_rate,
            running: Arc::new(AtomicBool::new(false)),
            frame_count: Arc::new(AtomicU64::new(0)),
            frequencies,
            snr: Arc::new(RwLock::new(10.0)),
        }
    }

    /// Start generating QAM signals - returns receiver for frames
    pub async fn start(&self) -> mpsc::Receiver<QAMSignal> {
        if self.running.swap(true, Ordering::SeqCst) {
            // Already running - return a dummy channel that will close immediately
            let (_, rx) = mpsc::channel(1);
            return rx;
        }
        
        info!("Simulated QAM16 started at {} Hz", self.sample_rate);
        
        let frame_interval = 1000 / self.sample_rate as u64;
        let (tx, rx) = mpsc::channel(1000);
        
        let running = self.running.clone();
        let frame_count = self.frame_count.clone();
        let sample_rate = self.sample_rate;
        let frequencies = self.frequencies.clone();
        let snr = self.snr.clone();
        
        // Spawn the generation task
        tokio::spawn(async move {
            let mut ticker = interval(Duration::from_millis(frame_interval));
            
            while running.load(Ordering::SeqCst) {
                ticker.tick().await;
                
                // Generate frame
                let count = frame_count.fetch_add(1, Ordering::SeqCst) + 1;
                let t = count as f64 / sample_rate as f64;
                let mut constellation = Vec::with_capacity(16);
                
                // Generate 16 QAM constellation points
                for (ch, freq) in frequencies.iter().enumerate() {
                    // Base signal
                    let mut val = (2.0 * PI * freq * t).sin();
                    
                    // Add harmonics for QAM complexity
                    val += 0.3 * (2.0 * PI * freq * 2.0 * t).sin();
                    val += 0.1 * (2.0 * PI * freq * 3.0 * t).sin();
                    
                    // Phase offset per quadrant for differentiation
                    let quadrant = ch / 4;
                    let phase_offset = quadrant as f64 * PI / 2.0;
                    val = (val + phase_offset.sin()) / 2.0;
                    
                    // Normalize to 0.0-1.0 range
                    let normalized = (val + 1.0) / 2.0;
                    constellation.push(normalized.clamp(0.0, 1.0));
                }
                
                // Create QAM signal and compute derived values
                let mut signal = QAMSignal::new(count);
                signal.constellation = constellation;
                signal.timestamp = Utc::now();
                signal.snr = *snr.read().await; // Use RwLock read instead of Atomic load
                signal.signal_strength = signal.constellation.iter().sum::<f64>() / 16.0;
                
                // Compute FOA quadrants and ambisonics
                signal.compute_quadrants();
                signal.compute_ambisonic();
                
                if tx.send(signal).await.is_err() {
                    info!("QAM channel closed, stopping simulation");
                    running.store(false, Ordering::SeqCst);
                    break;
                }
                
                // Log periodically
                if count % (sample_rate as u64 * 5) == 0 {
                    info!("Simulated QAM16: {} frames generated", count);
                }
            }
            
            info!("QAM generation task ended");
        });
        
        rx
    }
    
    /// Stop generating data
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        info!("Simulated QAM16 stopped at {} frames", self.frame_count.load(Ordering::SeqCst));
    }
    
    /// Check if running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
    
    /// Set SNR value (async because RwLock)
    pub async fn set_snr(&self, snr_db: f64) {
        let mut snr_guard = self.snr.write().await;
        *snr_guard = snr_db;
    }
    
    /// Get current SNR (async because RwLock)
    pub async fn get_snr(&self) -> f64 {
        *self.snr.read().await
    }
    
    /// Get frame count
    pub fn get_frame_count(&self) -> u64 {
        self.frame_count.load(Ordering::SeqCst)
    }
}

/// Convert QAM signal to legacy SignalFeatures for compatibility
pub fn qam_to_signal_features(qam: &QAMSignal) -> crate::types::SignalFeatures {
    use crate::types::SignalFeatures;
    
    SignalFeatures {
        timestamp: qam.timestamp,
        frame: qam.frame,
        eeg_channels: qam.left_hemisphere.iter().chain(qam.right_hemisphere.iter()).cloned().collect(),
        octonion_output: qam.constellation.clone(),
        ambisonic_w: qam.ambisonic_w,
        ambisonic_x: qam.ambisonic_x,
        ambisonic_y: qam.ambisonic_y,
        ambisonic_z: qam.ambisonic_z,
        coherence: qam.coherence,
        impedance_z: qam.impedance_z,
        diode_voltage: (1.0 - qam.coherence) * 0.6, // Approximate from coherence
        spatial_magnitude: (qam.ambisonic_x.powi(2) + qam.ambisonic_y.powi(2)).sqrt(),
        spatial_phase: qam.ambisonic_y.atan2(qam.ambisonic_x),
        left_hemisphere: qam.left_hemisphere.clone(),
        right_hemisphere: qam.right_hemisphere.clone(),
        coherence_left: qam.left_coherence,
        coherence_right: qam.right_coherence,
    }
}

/// Convert QAM signal to WebSocket message
pub fn qam_to_message(qam: &QAMSignal) -> crate::types::ServerMessage {
    use crate::types::{ServerMessage, AmbisonicMessage, QuadrantValues};
    
    ServerMessage::QAMSignalMessage {
        constellation: qam.constellation.clone(),
        snr: qam.snr,
        signal_strength: qam.signal_strength,
        coherence: qam.coherence,
        impedance_z: qam.impedance_z,
        ambisonic: AmbisonicMessage {
            w: qam.ambisonic_w,
            x: qam.ambisonic_x,
            y: qam.ambisonic_y,
            z: qam.ambisonic_z,
        },
        quadrants: QuadrantValues {
            lfu: qam.lfu_quadrant,
            lbd: qam.lbd_quadrant,
            rbu: qam.rbu_quadrant,
            rfd: qam.rfd_quadrant,
        },
        timestamp: qam.timestamp.timestamp_millis(),
        frame: qam.frame,
    }
}
