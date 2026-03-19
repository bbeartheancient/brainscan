/**
 * Brainscan Fast - Optimized 2D Implementation
 * 
 * Performance Features:
 * - 2D Canvas rendering (no Three.js)
 * - Bidirectional processing (Left→Right + Right→Left)
 * - Query caching with LRU eviction
 * - Parallel processing of hemispheres
 * - Data compression for WebSocket transport
 * - Text-based UI elements
 */

(function() {
    'use strict';

    // ==========================================
    // CONFIGURATION
    // ==========================================
    const CONFIG = {
        WS_URL: 'ws://localhost:8766',
        RECONNECT_DELAY: 5000,
        MAX_RECONNECT: 3,
        CACHE_SIZE: 100,
        CACHE_TTL: 300000, // 5 minutes
        FRAME_SKIP: 2, // Render every 2nd frame for performance
        TARGET_FPS: 30,
        BRAIN_RES: 128, // Low-res brain rendering
    };

    // ==========================================
    // STATE
    // ==========================================
    const state = {
        connected: false,
        socket: null,
        reconnectAttempts: 0,
        reconnectTimer: null,
        eegRunning: false,
        models: { left: null, right: null, available: [] },
        chatHistory: [],
        pendingResponses: { left: null, right: null, query: null },
        currentFrame: 0,
        eegData: null,
        lastFrameTime: 0,
        frameCount: 0,
        fps: 0,
        channelData: new Array(8).fill(0.5),
        ambisonic: new Array(4).fill(0),
    };

    // ==========================================
    // QUERY CACHE (LRU with TTL)
    // ==========================================
    class QueryCache {
        constructor(maxSize = CONFIG.CACHE_SIZE) {
            this.maxSize = maxSize;
            this.cache = new Map();
            this.hits = 0;
            this.misses = 0;
        }

        get(key) {
            const entry = this.cache.get(key);
            if (!entry) {
                this.misses++;
                return null;
            }
            
            if (Date.now() - entry.timestamp > CONFIG.CACHE_TTL) {
                this.cache.delete(key);
                this.misses++;
                return null;
            }
            
            // Move to end (LRU)
            this.cache.delete(key);
            this.cache.set(key, entry);
            this.hits++;
            return entry.value;
        }

        set(key, value) {
            if (this.cache.has(key)) {
                this.cache.delete(key);
            } else if (this.cache.size >= this.maxSize) {
                // Evict oldest (first entry)
                const firstKey = this.cache.keys().next().value;
                this.cache.delete(firstKey);
            }
            
            this.cache.set(key, {
                value,
                timestamp: Date.now()
            });
        }

        clear() {
            this.cache.clear();
            this.hits = 0;
            this.misses = 0;
        }

        getStats() {
            const total = this.hits + this.misses;
            return {
                hits: this.hits,
                misses: this.misses,
                size: this.cache.size,
                hitRate: total > 0 ? (this.hits / total) * 100 : 0
            };
        }
    }

    const queryCache = new QueryCache();

    // ==========================================
    // WEBSOCKET CONNECTION
    // ==========================================
    function connect() {
        if (state.socket && state.socket.readyState === WebSocket.OPEN) return;
        
        if (state.reconnectTimer) {
            clearTimeout(state.reconnectTimer);
            state.reconnectTimer = null;
        }

        console.log('[WS] Connecting...');
        
        try {
            state.socket = new WebSocket(CONFIG.WS_URL);
            
            state.socket.onopen = () => {
                console.log('[WS] Connected');
                state.connected = true;
                state.reconnectAttempts = 0;
                updateStatus('bridge', true);
                
                // Request initial data
                send({ type: 'get_models' });
                send({ type: 'get_pipeline_status' });
                addChatMessage('system', 'Connected to bridge', 'both');
            };
            
            state.socket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    handleMessage(data);
                } catch (e) {
                    console.error('[WS] Parse error:', e);
                }
            };
            
            state.socket.onclose = (event) => {
                state.connected = false;
                state.eegRunning = false;
                updateStatus('bridge', false);
                updateStatus('eeg', false);
                
                if (event.code !== 1000 && state.reconnectAttempts < CONFIG.MAX_RECONNECT) {
                    state.reconnectAttempts++;
                    state.reconnectTimer = setTimeout(() => connect(), CONFIG.RECONNECT_DELAY);
                }
            };
            
            state.socket.onerror = (error) => {
                console.error('[WS] Error:', error);
            };
        } catch (e) {
            console.error('[WS] Connection failed:', e);
        }
    }

    function send(data) {
        if (state.socket && state.socket.readyState === WebSocket.OPEN) {
            state.socket.send(JSON.stringify(data));
        }
    }

    // ==========================================
    // MESSAGE HANDLERS
    // ==========================================
    function handleMessage(data) {
        switch (data.type) {
            case 'inference_result':
                handleInferenceResult(data);
                break;
            case 'models_list':
                updateModelList(data.models);
                break;
            case 'pipeline_status':
                updatePipelineStatus(data);
                break;
            case 'eeg_frame':
                handleEEGFrame(data);
                break;
            case 'chat_response':
                handleChatResponse(data);
                break;
            case 'cache_stats':
                updateCacheStats(data);
                break;
            case 'error':
                console.error('[Server] Error:', data.message);
                addChatMessage('system', `Error: ${data.message}`, 'both');
                break;
        }
    }

    // ==========================================
    // BIDIRECTIONAL PROCESSING
    // ==========================================
    function handleInferenceResult(data) {
        if (data.predicted_class === 'thinking' || data.predicted_class === 'processing') {
            updateHemisphereStatus(data.hemisphere, 'Processing...');
            return;
        }
        
        if (data.predicted_class === 'complete') {
            updateHemisphereStatus(data.hemisphere, 'Ready');
            return;
        }
        
        const content = data.message || '';
        const hemisphere = data.hemisphere;
        
        // Parallel processing: store and check for combination
        if (hemisphere === 'left' || hemisphere === 'both') {
            state.pendingResponses.left = { content, timestamp: Date.now() };
            showHemisphereContent('left', content);
        }
        
        if (hemisphere === 'right' || hemisphere === 'both') {
            state.pendingResponses.right = { content, timestamp: Date.now() };
            showHemisphereContent('right', content);
        }
        
        // Bidirectional synthesis
        checkCombinedResponse();
    }

    function handleChatResponse(data) {
        const hemisphere = data.hemisphere;
        const content = data.message || '';
        
        if (hemisphere === 'left' || hemisphere === 'both') {
            showHemisphereContent('left', content);
        }
        if (hemisphere === 'right' || hemisphere === 'both') {
            showHemisphereContent('right', content);
        }
        
        if (hemisphere === 'both') {
            addChatMessage('model', content, 'both');
        }
    }

    function checkCombinedResponse() {
        const { left, right, query } = state.pendingResponses;
        
        if (left && right) {
            // Both hemispheres have responded - synthesize
            const combined = synthesizeResponses(left.content, right.content);
            
            if (query) {
                // Cache the combined response
                queryCache.set(query, combined);
                updateCacheDisplay();
            }
            
            addChatMessage('model', combined, 'both');
            
            // Clear pending
            state.pendingResponses.left = null;
            state.pendingResponses.right = null;
            state.pendingResponses.query = null;
        } else if (left || right) {
            // Partial response - set timeout for async completion
            setTimeout(() => {
                if (state.pendingResponses.left || state.pendingResponses.right) {
                    // Timeout reached, show what we have
                    const partial = left ? left.content : right.content;
                    addChatMessage('model', partial, 'both');
                    state.pendingResponses.left = null;
                    state.pendingResponses.right = null;
                    state.pendingResponses.query = null;
                }
            }, 3000);
        }
    }

    function synthesizeResponses(left, right) {
        // Bidirectional synthesis: combine analytical and intuitive perspectives
        const leftSummary = left.length > 150 ? left.substring(0, 150) + '...' : left;
        const rightSummary = right.length > 150 ? right.substring(0, 150) + '...' : right;
        
        return `◆ ANALYTICAL [L]:\n${leftSummary}\n\n◇ INTUITIVE [R]:\n${rightSummary}`;
    }

    // ==========================================
    // DATA COMPRESSION
    // ==========================================
    function compressEEGFrame(frame) {
        // Simple delta compression for EEG data
        if (!frame || !frame.channels) return frame;
        
        const compressed = {
            t: frame.timestamp,
            c: frame.channels.map(v => Math.round(v * 100) / 100) // 2 decimal precision
        };
        
        return compressed;
    }

    function decompressEEGFrame(data) {
        if (!data.c) return data;
        
        return {
            timestamp: data.t,
            channels: data.c
        };
    }

    // ==========================================
    // EEG HANDLING
    // ==========================================
    let lastEEGProcess = 0;
    const EEG_INTERVAL = 1000 / 60; // 60Hz max

    function handleEEGFrame(data) {
        const now = performance.now();
        if (now - lastEEGProcess < EEG_INTERVAL) {
            return; // Skip frame
        }
        lastEEGProcess = now;
        
        // Decompress if needed
        const frame = data.c ? decompressEEGFrame(data) : data;
        
        if (frame.channels && frame.channels.length === 8) {
            state.channelData = frame.channels;
            updateChannelBars(frame.channels);
            drawWaveform(frame.channels);
        }
    }

    // ==========================================
    // 2D BRAIN RENDERER
    // ==========================================
    const brainCanvas = document.getElementById('brain-canvas');
    const brainCtx = brainCanvas.getContext('2d', { alpha: false });
    
    // Electrode positions (normalized 0-1)
    const ELECTRODES = [
        { name: 'Fp1', x: 0.3, y: 0.25, basis: '1' },
        { name: 'Fp2', x: 0.7, y: 0.25, basis: 'm' },
        { name: 'F7', x: 0.15, y: 0.4, basis: 'n' },
        { name: 'F8', x: 0.85, y: 0.4, basis: 'o' },
        { name: 'Cz', x: 0.5, y: 0.5, basis: 'l' },
        { name: 'Pz', x: 0.5, y: 0.7, basis: 'i' },
        { name: 'P7', x: 0.2, y: 0.75, basis: 'j' },
        { name: 'P8', x: 0.8, y: 0.75, basis: 'k' },
    ];

    function resizeBrainCanvas() {
        const container = brainCanvas.parentElement;
        brainCanvas.width = container.clientWidth;
        brainCanvas.height = container.clientHeight;
    }

    function drawBrain() {
        const w = brainCanvas.width;
        const h = brainCanvas.height;
        const cx = w / 2;
        const cy = h / 2;
        const scale = Math.min(w, h) * 0.4;
        
        // Clear
        brainCtx.fillStyle = '#0a0a0a';
        brainCtx.fillRect(0, 0, w, h);
        
        // Draw brain outline (top-down view)
        brainCtx.strokeStyle = '#fa0';
        brainCtx.lineWidth = 2;
        brainCtx.beginPath();
        
        // Outer contour (oval with slight indentations)
        for (let a = 0; a < Math.PI * 2; a += 0.1) {
            // Brain shape function
            const r = scale * (0.9 + 0.15 * Math.sin(3 * a) + 0.05 * Math.cos(5 * a));
            const x = cx + r * Math.cos(a);
            const y = cy + r * Math.sin(a) * 0.85; // Slightly flattened
            
            if (a === 0) brainCtx.moveTo(x, y);
            else brainCtx.lineTo(x, y);
        }
        brainCtx.closePath();
        brainCtx.stroke();
        
        // Central sulcus (division line)
        brainCtx.strokeStyle = '#553300';
        brainCtx.lineWidth = 1;
        brainCtx.beginPath();
        brainCtx.moveTo(cx - scale * 0.1, cy - scale * 0.3);
        brainCtx.quadraticCurveTo(cx + scale * 0.1, cy, cx - scale * 0.1, cy + scale * 0.3);
        brainCtx.stroke();
        
        // Lateral sulcus
        brainCtx.beginPath();
        brainCtx.moveTo(cx - scale * 0.6, cy);
        brainCtx.quadraticCurveTo(cx - scale * 0.3, cy + scale * 0.1, cx - scale * 0.5, cy + scale * 0.3);
        brainCtx.moveTo(cx + scale * 0.6, cy);
        brainCtx.quadraticCurveTo(cx + scale * 0.3, cy + scale * 0.1, cx + scale * 0.5, cy + scale * 0.3);
        brainCtx.stroke();
        
        // Draw electrodes
        ELECTRODES.forEach((elec, idx) => {
            const ex = cx + (elec.x - 0.5) * scale * 1.8;
            const ey = cy + (elec.y - 0.5) * scale * 1.6;
            const val = state.channelData[idx] || 0.5;
            const intensity = (val - 0.5) * 2; // -1 to 1
            
            // Glow based on activity
            const radius = 8 + Math.abs(intensity) * 12;
            const alpha = 0.3 + Math.abs(intensity) * 0.7;
            
            // Outer glow
            const gradient = brainCtx.createRadialGradient(ex, ey, 0, ex, ey, radius);
            if (intensity > 0) {
                gradient.addColorStop(0, `rgba(255, 170, 0, ${alpha})`);
                gradient.addColorStop(1, 'rgba(255, 170, 0, 0)');
            } else {
                gradient.addColorStop(0, `rgba(230, 0, 3, ${alpha})`);
                gradient.addColorStop(1, 'rgba(230, 0, 3, 0)');
            }
            
            brainCtx.fillStyle = gradient;
            brainCtx.beginPath();
            brainCtx.arc(ex, ey, radius, 0, Math.PI * 2);
            brainCtx.fill();
            
            // Center dot
            brainCtx.fillStyle = intensity > 0 ? '#fa0' : '#e60003';
            brainCtx.beginPath();
            brainCtx.arc(ex, ey, 3, 0, Math.PI * 2);
            brainCtx.fill();
            
            // Label
            brainCtx.fillStyle = '#553300';
            brainCtx.font = '9px monospace';
            brainCtx.textAlign = 'center';
            brainCtx.fillText(elec.name, ex, ey - 12);
        });
        
        // Draw activity connections
        brainCtx.strokeStyle = 'rgba(255, 170, 0, 0.2)';
        brainCtx.lineWidth = 1;
        brainCtx.beginPath();
        
        for (let i = 0; i < ELECTRODES.length; i++) {
            for (let j = i + 1; j < ELECTRODES.length; j++) {
                const val1 = state.channelData[i];
                const val2 = state.channelData[j];
                const correlation = 1 - Math.abs(val1 - val2);
                
                if (correlation > 0.7) {
                    const x1 = cx + (ELECTRODES[i].x - 0.5) * scale * 1.8;
                    const y1 = cy + (ELECTRODES[i].y - 0.5) * scale * 1.6;
                    const x2 = cx + (ELECTRODES[j].x - 0.5) * scale * 1.8;
                    const y2 = cy + (ELECTRODES[j].y - 0.5) * scale * 1.6;
                    
                    brainCtx.moveTo(x1, y1);
                    brainCtx.lineTo(x2, y2);
                }
            }
        }
        brainCtx.stroke();
    }

    // ==========================================
    // WAVEFORM RENDERER
    // ==========================================
    const waveCanvas = document.getElementById('waveform-canvas');
    const waveCtx = waveCanvas.getContext('2d');
    const waveHistory = [];
    const MAX_HISTORY = 100;

    function resizeWaveformCanvas() {
        const rect = waveCanvas.parentElement.getBoundingClientRect();
        waveCanvas.width = rect.width;
        waveCanvas.height = rect.height;
    }

    function drawWaveform(channels) {
        // Add to history
        waveHistory.push([...channels]);
        if (waveHistory.length > MAX_HISTORY) {
            waveHistory.shift();
        }
        
        const w = waveCanvas.width;
        const h = waveCanvas.height;
        const chHeight = h / 8;
        
        waveCtx.fillStyle = '#000';
        waveCtx.fillRect(0, 0, w, h);
        
        // Draw each channel
        const colors = ['#faa800', '#f79502', '#f18204', '#e96f04', '#e15c04', '#d94904', '#d13604', '#c92304'];
        
        for (let ch = 0; ch < 8; ch++) {
            waveCtx.strokeStyle = colors[ch];
            waveCtx.lineWidth = 1;
            waveCtx.beginPath();
            
            for (let i = 0; i < waveHistory.length; i++) {
                const x = (i / MAX_HISTORY) * w;
                const val = waveHistory[i][ch];
                const y = ch * chHeight + chHeight * 0.5 + (val - 0.5) * chHeight * 0.8;
                
                if (i === 0) waveCtx.moveTo(x, y);
                else waveCtx.lineTo(x, y);
            }
            
            waveCtx.stroke();
        }
    }

    // ==========================================
    // UI UPDATES
    // ==========================================
    function updateStatus(type, active) {
        const dot = document.getElementById(`${type}-status`);
        if (dot) {
            dot.classList.toggle('active', active);
        }
    }

    function updatePipelineStatus(data) {
        state.eegRunning = data.eeg_running;
        updateStatus('eeg', data.eeg_running);
        
        const statusEl = document.getElementById('pipeline-status');
        const sourceEl = document.getElementById('data-source');
        
        if (statusEl) {
            statusEl.textContent = data.eeg_running ? 'Running' : 'Stopped';
        }
        if (sourceEl) {
            sourceEl.textContent = data.eeg_source_type || 'None';
        }
        
        // Update buttons
        document.getElementById('btn-start').disabled = !state.connected || state.eegRunning;
        document.getElementById('btn-stop').disabled = !state.connected || !state.eegRunning;
    }

    function updateModelList(models) {
        if (!models || !models.length) return;
        
        state.models.available = models;
        
        const leftSelect = document.getElementById('left-model');
        const rightSelect = document.getElementById('right-model');
        
        const options = models.map(m => `<option value="${m}">${m}</option>`).join('');
        
        if (leftSelect) {
            leftSelect.innerHTML = '<option value="">Left Hemisphere...</option>' + options;
        }
        if (rightSelect) {
            rightSelect.innerHTML = '<option value="">Right Hemisphere...</option>' + options;
        }
    }

    function updateHemisphereStatus(hemisphere, status) {
        const content = document.getElementById(`${hemisphere}-content`);
        if (content) {
            content.textContent = status;
        }
    }

    function showHemisphereContent(hemisphere, content) {
        const el = document.getElementById(`${hemisphere}-content`);
        if (el) {
            // Truncate for display
            const truncated = content.length > 200 ? content.substring(0, 200) + '...' : content;
            el.textContent = truncated;
        }
    }

    function updateChannelBars(channels) {
        const container = document.getElementById('channel-bars');
        if (!container) return;
        
        if (container.children.length === 0) {
            // Initialize bars
            const names = ['Fp1', 'Fp2', 'F7', 'F8', 'Cz', 'Pz', 'P7', 'P8'];
            names.forEach((name, idx) => {
                const bar = document.createElement('div');
                bar.className = 'channel-bar';
                bar.innerHTML = `
                    <span class="channel-label">${name}</span>
                    <div class="channel-meter">
                        <div class="channel-fill" id="ch-fill-${idx}"></div>
                    </div>
                `;
                container.appendChild(bar);
            });
        }
        
        // Update values
        channels.forEach((val, idx) => {
            const fill = document.getElementById(`ch-fill-${idx}`);
            if (fill) {
                const pct = ((val - 0.5) / 1.0) * 100;
                fill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
            }
        });
    }

    function updateCacheStats(data) {
        const hitRateEl = document.getElementById('cache-hit-rate');
        const sizeEl = document.getElementById('cache-size');
        
        if (hitRateEl) hitRateEl.textContent = `${data.hit_rate.toFixed(1)}%`;
        if (sizeEl) sizeEl.textContent = data.size;
        
        updateStatus('cache', data.hit_rate > 50);
    }

    function updateCacheDisplay() {
        const stats = queryCache.getStats();
        const hitRateEl = document.getElementById('cache-hit-rate');
        const sizeEl = document.getElementById('cache-size');
        
        if (hitRateEl) hitRateEl.textContent = `${stats.hitRate.toFixed(1)}%`;
        if (sizeEl) sizeEl.textContent = stats.size;
    }

    // ==========================================
    // CHAT INTERFACE
    // ==========================================
    function addChatMessage(role, content, hemisphere) {
        const container = document.getElementById('chat-messages');
        if (!container) return;
        
        const msg = document.createElement('div');
        msg.className = `chat-message ${role}`;
        
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        msg.innerHTML = `
            <div class="chat-time">${time} ${hemisphere !== 'both' ? '[' + hemisphere + ']' : ''}</div>
            <div>${content.replace(/\n/g, '<br>')}</div>
        `;
        
        container.appendChild(msg);
        container.scrollTop = container.scrollHeight;
        
        // Limit history
        while (container.children.length > 50) {
            container.removeChild(container.firstChild);
        }
        
        state.chatHistory.push({ role, content, hemisphere, time });
    }

    function sendChatMessage() {
        const input = document.getElementById('chat-input');
        if (!input || !input.value.trim()) return;
        
        const message = input.value.trim();
        input.value = '';
        
        // Check cache first
        const cached = queryCache.get(message);
        if (cached) {
            addChatMessage('user', message, 'both');
            addChatMessage('model', `[CACHED]\n${cached}`, 'both');
            updateCacheDisplay();
            return;
        }
        
        addChatMessage('user', message, 'both');
        
        // Store query for caching later
        state.pendingResponses.query = message;
        
        // Send to server
        send({
            type: 'chat_message',
            message: message,
            hemisphere: 'both'
        });
    }

    // ==========================================
    // EVENT LISTENERS
    // ==========================================
    function setupEventListeners() {
        // Connection
        document.getElementById('btn-connect')?.addEventListener('click', connect);
        
        // EEG Control
        document.getElementById('btn-start')?.addEventListener('click', () => {
            send({
                type: 'start_eeg',
                source_type: 'simulated',
                sample_rate: 256
            });
        });
        
        document.getElementById('btn-stop')?.addEventListener('click', () => {
            send({ type: 'stop_eeg' });
        });
        
        // Model Selection
        document.getElementById('btn-set-models')?.addEventListener('click', () => {
            const left = document.getElementById('left-model')?.value;
            const right = document.getElementById('right-model')?.value;
            
            if (left) {
                send({ type: 'set_model', hemisphere: 'left', model_id: left });
                state.models.left = left;
            }
            if (right) {
                send({ type: 'set_model', hemisphere: 'right', model_id: right });
                state.models.right = right;
            }
        });
        
        // Cache
        document.getElementById('btn-clear-cache')?.addEventListener('click', () => {
            queryCache.clear();
            send({ type: 'clear_cache' });
            updateCacheDisplay();
            addChatMessage('system', 'Cache cleared', 'both');
        });
        
        // Chat
        document.getElementById('btn-send')?.addEventListener('click', sendChatMessage);
        document.getElementById('chat-input')?.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') sendChatMessage();
        });
        
        // Window resize
        window.addEventListener('resize', () => {
            resizeBrainCanvas();
            resizeWaveformCanvas();
        });
    }

    // ==========================================
    // ANIMATION LOOP
    // ==========================================
    let frameSkip = 0;
    
    function animate(timestamp) {
        // Calculate FPS
        if (timestamp - state.lastFrameTime >= 1000) {
            document.getElementById('fps-counter').textContent = state.frameCount;
            state.frameCount = 0;
            state.lastFrameTime = timestamp;
        }
        state.frameCount++;
        
        // Skip frames for performance
        frameSkip++;
        if (frameSkip % CONFIG.FRAME_SKIP === 0) {
            drawBrain();
        }
        
        document.getElementById('frame-counter').textContent = state.currentFrame;
        
        requestAnimationFrame(animate);
    }

    // ==========================================
    // INITIALIZATION
    // ==========================================
    function init() {
        resizeBrainCanvas();
        resizeWaveformCanvas();
        setupEventListeners();
        
        // Initialize channel bars
        updateChannelBars(new Array(8).fill(0.5));
        
        // Start animation
        requestAnimationFrame(animate);
        
        // Auto-connect
        setTimeout(connect, 500);
        
        console.log('[Brainscan Fast] Initialized');
    }

    // Export for debugging
    window.BrainscanFast = {
        connect,
        send,
        cache: queryCache,
        getState: () => ({ ...state }),
    };

    // Start
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
