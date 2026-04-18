/* ===================================================================
   Pain Detection Dashboard — Real-time Frontend
   =================================================================== */

const CLASS_COLORS = { none: '#22c55e', mild: '#facc15', moderate: '#f97316', severe: '#ef4444' };
const CLASS_COLORS_FADED = { none: 'rgba(34,197,94,0.25)', mild: 'rgba(250,204,21,0.25)', moderate: 'rgba(249,115,22,0.25)', severe: 'rgba(239,68,68,0.25)' };
const MAX_TIMELINE = 60;

// ---- DOM refs ----
const classLabel     = document.getElementById('class-label');
const confidenceBar  = document.getElementById('confidence-bar');
const confidenceText = document.getElementById('confidence-text');
const liveDot        = document.getElementById('live-dot');
const videoOverlay   = document.getElementById('video-overlay');

const statTotal    = document.getElementById('stat-total');
const statAvgConf  = document.getElementById('stat-avg-conf');
const statDominant = document.getElementById('stat-dominant');
const statFps      = document.getElementById('stat-fps');

// Session Controls
const patientInput = document.getElementById('patient-name');
const btnStart     = document.getElementById('btn-start');
const btnStop      = document.getElementById('btn-stop');
const sessionForm  = document.getElementById('session-form');
const sessionActiveBar = document.getElementById('session-active-bar');
const sessionPatientLabel = document.getElementById('session-patient-label');
const sessionTimer = document.getElementById('session-timer');

// Navigation
const navDashboard = document.getElementById('nav-dashboard');
const navHistory   = document.getElementById('nav-history');
const viewDashboard = document.getElementById('view-dashboard');
const viewHistory  = document.getElementById('view-history');
const historyList  = document.getElementById('history-list');

// Modal
const modalBackdrop = document.getElementById('modal-backdrop');
const modalClose    = document.getElementById('modal-close');
const modalBody     = document.getElementById('modal-body');
const modalPatientName = document.getElementById('modal-patient-name');

// ---- State ----
let sessionActive   = false;
let sessionStart    = null;
let timerInterval   = null;
let currentSessionId= null;

let totalDetections = 0;
let confidenceSum   = 0;
let classCounts     = { none: 0, mild: 0, moderate: 0, severe: 0 };
let fpsCounter      = 0;

setInterval(() => {
    statFps.textContent = fpsCounter;
    fpsCounter = 0;
}, 1000);


// ===========================================================================
//  Session Management
// ===========================================================================
async function startSession() {
    const name = patientInput.value.trim();
    if (!name) { alert('Please enter a patient name'); return; }

    try {
        const res = await fetch('/api/session/start', {
            method: 'POST',
            body: JSON.stringify({ patient_name: name })
        });
        const data = await res.json();
        
        sessionActive = true;
        currentSessionId = data.session_id;
        sessionStart = Date.now();
        
        // Update UI
        patientInput.value = '';
        sessionForm.classList.add('hidden');
        sessionActiveBar.classList.remove('hidden');
        sessionPatientLabel.textContent = data.patient_name;
        videoOverlay.classList.add('hidden');
        liveDot.classList.add('active');
        
        // Reset stats
        resetStats();
        
        // Start timer
        clearInterval(timerInterval);
        timerInterval = setInterval(updateTimer, 1000);
        updateTimer();
        
    } catch (e) {
        console.error('Failed to start session', e);
        alert('Failed to start session');
    }
}

async function stopSession() {
    try {
        await fetch('/api/session/stop', { method: 'POST' });
        
        sessionActive = false;
        clearInterval(timerInterval);
        
        // Update UI
        sessionActiveBar.classList.add('hidden');
        sessionForm.classList.remove('hidden');
        videoOverlay.classList.remove('hidden');
        liveDot.classList.remove('active');
        
        // Reset charts for next session
        resetStats();
        
    } catch (e) {
        console.error('Failed to stop session', e);
    }
}

function updateTimer() {
    if (!sessionStart) return;
    const seconds = Math.floor((Date.now() - sessionStart) / 1000);
    const m = String(Math.floor(seconds / 60)).padStart(2, '0');
    const s = String(seconds % 60).padStart(2, '0');
    sessionTimer.textContent = `${m}:${s}`;
}

btnStart.addEventListener('click', startSession);
btnStop.addEventListener('click', stopSession);
patientInput.addEventListener('keypress', e => { if (e.key === 'Enter') startSession(); });


// ===========================================================================
//  Navigation
// ===========================================================================
navDashboard.addEventListener('click', () => {
    navDashboard.classList.add('active');
    navHistory.classList.remove('active');
    viewDashboard.classList.add('view-active');
    viewHistory.classList.remove('view-active');
});

navHistory.addEventListener('click', () => {
    navHistory.classList.add('active');
    navDashboard.classList.remove('active');
    viewHistory.classList.add('view-active');
    viewDashboard.classList.remove('view-active');
    loadHistory();
});


// ===========================================================================
//  Chart.js Setup
// ===========================================================================
const timelineCtx = document.getElementById('timeline-chart').getContext('2d');
const timelineChart = new Chart(timelineCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: ['none', 'mild', 'moderate', 'severe'].map(c => ({
            label: c.charAt(0).toUpperCase() + c.slice(1),
            data: [],
            borderColor: CLASS_COLORS[c],
            backgroundColor: CLASS_COLORS_FADED[c],
            fill: false, tension: 0.35, pointRadius: 0, borderWidth: 2,
        }))
    },
    options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 0 },
        interaction: { mode: 'index', intersect: false },
        plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } },
        scales: {
            x: { display: true, ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.04)' } },
            y: { min: 0, max: 100, ticks: { color: '#64748b', callback: v => v + '%' }, grid: { color: 'rgba(255,255,255,0.04)' } }
        }
    }
});
const dsMap = { none: 0, mild: 1, moderate: 2, severe: 3 };

const distCtx = document.getElementById('distribution-chart').getContext('2d');
const distChart = new Chart(distCtx, {
    type: 'doughnut',
    data: {
        labels: ['None', 'Mild', 'Moderate', 'Severe'],
        datasets: [{
            data: [0, 0, 0, 0],
            backgroundColor: [CLASS_COLORS.none, CLASS_COLORS.mild, CLASS_COLORS.moderate, CLASS_COLORS.severe],
            borderColor: 'rgba(0,0,0,0.3)', borderWidth: 2, hoverOffset: 8,
        }]
    },
    options: { responsive: true, maintainAspectRatio: false, cutout: '65%', plugins: { legend: { position: 'bottom', labels: { color: '#94a3b8' } } } }
});

function resetStats() {
    totalDetections = 0; confidenceSum = 0;
    classCounts = { none: 0, mild: 0, moderate: 0, severe: 0 };
    
    statTotal.textContent = 0; statAvgConf.textContent = '0%'; statDominant.textContent = '—';
    
    timelineChart.data.labels = [];
    timelineChart.data.datasets.forEach(ds => ds.data = []);
    timelineChart.update();
    
    distChart.data.datasets[0].data = [0, 0, 0, 0];
    distChart.update();
    
    classLabel.textContent = '—'; classLabel.className = 'class-label';
    confidenceBar.style.width = '0%'; confidenceText.textContent = '0%';
    ['none', 'mild', 'moderate', 'severe'].forEach(c => {
        document.getElementById(`prob-${c}`).style.width = '0%';
        document.getElementById(`prob-${c}-val`).textContent = '0%';
    });
}


// ===========================================================================
//  SSE Update UI
// ===========================================================================
function updateUI(data) {
    if (data.keepalive) return;
    
    // Sync state if dashboard refreshed mid-session
    if (data.session_active && !sessionActive) {
        fetch('/api/session/status').then(r => r.json()).then(st => {
            if (st.active && !sessionActive) {
                sessionActive = true;
                sessionForm.classList.add('hidden');
                sessionActiveBar.classList.remove('hidden');
                sessionPatientLabel.textContent = st.patient_name;
                videoOverlay.classList.add('hidden');
                liveDot.classList.add('active');
                
                // Estimate start time
                sessionStart = Date.now() - (st.detections * (1000/15)); // rough estimate
                clearInterval(timerInterval);
                timerInterval = setInterval(updateTimer, 1000);
            }
        });
    }

    if (!data.session_active) return;
    fpsCounter++;

    const cls  = data.class;
    const conf = data.confidence || 0;

    classLabel.textContent = cls === 'no_face' ? 'No Face' : cls;
    classLabel.className   = 'class-label ' + cls;

    const barColor = CLASS_COLORS[cls] || '#64748b';
    confidenceBar.style.width      = conf + '%';
    confidenceBar.style.background = `linear-gradient(90deg, ${barColor}, ${barColor}cc)`;
    confidenceText.textContent     = conf.toFixed(1) + '%';

    if (data.all_probs) {
        for (const c of ['none', 'mild', 'moderate', 'severe']) {
            const pct = data.all_probs[c] || 0;
            document.getElementById(`prob-${c}`).style.width = pct + '%';
            document.getElementById(`prob-${c}-val`).textContent = pct.toFixed(1) + '%';
        }
    }

    if (data.face_detected) {
        totalDetections++;
        confidenceSum += conf;
        classCounts[cls] = (classCounts[cls] || 0) + 1;
        
        // Update Timeline
        timelineChart.data.labels.push(data.timestamp || '');
        if (timelineChart.data.labels.length > MAX_TIMELINE) timelineChart.data.labels.shift();
        
        for (const c of ['none', 'mild', 'moderate', 'severe']) {
            const ds = timelineChart.data.datasets[dsMap[c]];
            ds.data.push(data.all_probs[c] || 0);
            if (ds.data.length > MAX_TIMELINE) ds.data.shift();
        }
        timelineChart.update('none');
        
        // Update Doughnut
        distChart.data.datasets[0].data = [classCounts.none, classCounts.mild, classCounts.moderate, classCounts.severe];
        distChart.update('none');
    }

    statTotal.textContent = totalDetections;
    statAvgConf.textContent = totalDetections > 0 ? (confidenceSum / totalDetections).toFixed(1) + '%' : '0%';
    
    let dominant = '—', maxCount = 0;
    for (const [c, n] of Object.entries(classCounts)) { if (n > maxCount) { maxCount = n; dominant = c; } }
    statDominant.textContent = dominant;
}

const evtSource = new EventSource('/events');
evtSource.onmessage = e => { try { updateUI(JSON.parse(e.data)); } catch(err){} };


// ===========================================================================
//  History View
// ===========================================================================
async function loadHistory() {
    try {
        const res = await fetch('/api/patients');
        const patients = await res.json();
        
        historyList.innerHTML = '';
        if (patients.length === 0) {
            historyList.innerHTML = '<div class="history-empty"><p>No patient records yet.</p></div>';
            return;
        }
        
        // sort newest first
        patients.sort((a,b) => new Date(b.start_time) - new Date(a.start_time));
        
        patients.forEach(p => {
            const el = document.createElement('div');
            el.className = 'history-card';
            el.innerHTML = `
                <div class="history-card-info">
                    <h3>${p.patient_name}</h3>
                    <div class="history-card-meta">
                        <span><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg> ${p.start_time}</span>
                        <span><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg> ${p.total_detections} frames</span>
                    </div>
                </div>
                <div class="history-card-actions">
                    <span class="history-card-badge ${p.dominant_class}">${p.dominant_class} freq</span>
                    <button class="btn-sm" onclick="openModal('${p.session_id}')">View Details</button>
                    <button class="btn-sm btn-danger" onclick="deleteRecord('${p.session_id}', event)">Delete</button>
                </div>
            `;
            el.addEventListener('click', (e) => {
                if(e.target.tagName !== 'BUTTON') openModal(p.session_id);
            });
            historyList.appendChild(el);
        });
    } catch (e) {
        console.error('Failed to load history', e);
    }
}

async function openModal(sid) {
    try {
        const res = await fetch(`/api/patients/${sid}`);
        const data = await res.json();
        
        modalPatientName.textContent = data.patient_name;
        
        modalBody.innerHTML = `
            <div class="history-card-meta" style="margin-bottom: 20px;">
                <span>Session ID: <code>${data.session_id}</code></span>
                <span>Date: <code>${data.start_time}</code> to <code>${data.end_time}</code></span>
            </div>
            
            <div class="modal-stats">
                <div class="modal-stat"><div class="val">${data.total_detections}</div><div class="lbl">Detections</div></div>
                <div class="modal-stat"><div class="val">${data.avg_confidence}%</div><div class="lbl">Avg Confidence</div></div>
                <div class="modal-stat"><div class="val" style="color: ${CLASS_COLORS[data.dominant_class]}">${data.dominant_class}</div><div class="lbl">Most Frequent</div></div>
            </div>
            
            <h4 class="modal-section-title">Session Class Distribution</h4>
            <div class="modal-chart-container">
                <canvas id="modal-chart"></canvas>
            </div>
        `;
        
        modalBackdrop.classList.remove('hidden');
        
        // Render modal chart
        new Chart(document.getElementById('modal-chart').getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['None', 'Mild', 'Moderate', 'Severe'],
                datasets: [{
                    label: 'Frames Detected',
                    data: [data.class_counts.none||0, data.class_counts.mild||0, data.class_counts.moderate||0, data.class_counts.severe||0],
                    backgroundColor: [CLASS_COLORS.none, CLASS_COLORS.mild, CLASS_COLORS.moderate, CLASS_COLORS.severe],
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    y: { beginAtZero: true, ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
                }
            }
        });
        
    } catch (e) {
        console.error('Failed to load modal data', e);
    }
}

modalClose.addEventListener('click', () => modalBackdrop.classList.add('hidden'));
modalBackdrop.addEventListener('click', e => { if(e.target === modalBackdrop) modalBackdrop.classList.add('hidden'); });

async function deleteRecord(sid, event) {
    event.stopPropagation();
    if (!confirm('Are you sure you want to delete this session?')) return;
    try {
        await fetch(`/api/patients/${sid}`, { method: 'DELETE' });
        loadHistory(); // refresh
    } catch (e) { console.error('Failed to delete', e); }
}

// Initial check for active session on load
fetch('/api/session/status').then(r => r.json()).then(st => {
    if (st.active) {
        sessionActive = true;
        currentSessionId = st.session_id;
        sessionForm.classList.add('hidden');
        sessionActiveBar.classList.remove('hidden');
        sessionPatientLabel.textContent = st.patient_name;
        videoOverlay.classList.add('hidden');
        liveDot.classList.add('active');
    }
});
