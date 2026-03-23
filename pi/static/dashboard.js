/* ===================================================================
   Pain Detection Dashboard — Real-time Frontend
   Connects to SSE /events and updates DOM + Chart.js charts.
   =================================================================== */

// ---- Constants ----
const CLASS_COLORS = {
    none:     'rgba(34, 197, 94,  1)',
    mild:     'rgba(250, 204, 21, 1)',
    moderate: 'rgba(249, 115, 22, 1)',
    severe:   'rgba(239, 68, 68,  1)',
};

const CLASS_COLORS_FADED = {
    none:     'rgba(34, 197, 94,  0.25)',
    mild:     'rgba(250, 204, 21, 0.25)',
    moderate: 'rgba(249, 115, 22, 0.25)',
    severe:   'rgba(239, 68, 68,  0.25)',
};

const MAX_TIMELINE = 60;

// ---- DOM refs ----
const classLabel     = document.getElementById('class-label');
const confidenceBar  = document.getElementById('confidence-bar');
const confidenceText = document.getElementById('confidence-text');
const statusBadge    = document.getElementById('status-badge');
const uptimeEl       = document.getElementById('uptime');

const statTotal   = document.getElementById('stat-total');
const statAvgConf = document.getElementById('stat-avg-conf');
const statDominant = document.getElementById('stat-dominant');
const statFps     = document.getElementById('stat-fps');

// ---- State ----
let totalDetections = 0;
let confidenceSum   = 0;
let classCounts     = { none: 0, mild: 0, moderate: 0, severe: 0 };
let fpsCounter      = 0;
let currentFps      = 0;
let startTime       = Date.now();

// ---- FPS counter ----
setInterval(() => {
    currentFps = fpsCounter;
    fpsCounter = 0;
    statFps.textContent = currentFps;
}, 1000);

// ---- Uptime ticker ----
setInterval(() => {
    const seconds = Math.floor((Date.now() - startTime) / 1000);
    const h = String(Math.floor(seconds / 3600)).padStart(2, '0');
    const m = String(Math.floor((seconds % 3600) / 60)).padStart(2, '0');
    const s = String(seconds % 60).padStart(2, '0');
    uptimeEl.textContent = `${h}:${m}:${s}`;
}, 1000);


// ===========================================================================
//  Chart.js setup
// ===========================================================================

// -- Timeline chart --
const timelineCtx = document.getElementById('timeline-chart').getContext('2d');
const timelineChart = new Chart(timelineCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [
            {
                label: 'None',
                data: [],
                borderColor: CLASS_COLORS.none,
                backgroundColor: CLASS_COLORS_FADED.none,
                fill: false,
                tension: 0.35,
                pointRadius: 0,
                borderWidth: 2,
            },
            {
                label: 'Mild',
                data: [],
                borderColor: CLASS_COLORS.mild,
                backgroundColor: CLASS_COLORS_FADED.mild,
                fill: false,
                tension: 0.35,
                pointRadius: 0,
                borderWidth: 2,
            },
            {
                label: 'Moderate',
                data: [],
                borderColor: CLASS_COLORS.moderate,
                backgroundColor: CLASS_COLORS_FADED.moderate,
                fill: false,
                tension: 0.35,
                pointRadius: 0,
                borderWidth: 2,
            },
            {
                label: 'Severe',
                data: [],
                borderColor: CLASS_COLORS.severe,
                backgroundColor: CLASS_COLORS_FADED.severe,
                fill: false,
                tension: 0.35,
                pointRadius: 0,
                borderWidth: 2,
            },
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 200 },
        interaction: { mode: 'index', intersect: false },
        plugins: {
            legend: {
                labels: { color: '#94a3b8', font: { size: 11, family: 'Inter' }, boxWidth: 12, padding: 14 }
            }
        },
        scales: {
            x: {
                display: true,
                ticks: { color: '#64748b', font: { size: 10 }, maxTicksLimit: 8 },
                grid: { color: 'rgba(255,255,255,0.04)' }
            },
            y: {
                min: 0, max: 100,
                ticks: { color: '#64748b', font: { size: 10 }, callback: v => v + '%' },
                grid: { color: 'rgba(255,255,255,0.04)' }
            }
        }
    }
});

// Dataset index mapping
const dsMap = { none: 0, mild: 1, moderate: 2, severe: 3 };

// -- Distribution doughnut --
const distCtx = document.getElementById('distribution-chart').getContext('2d');
const distChart = new Chart(distCtx, {
    type: 'doughnut',
    data: {
        labels: ['None', 'Mild', 'Moderate', 'Severe'],
        datasets: [{
            data: [0, 0, 0, 0],
            backgroundColor: [CLASS_COLORS.none, CLASS_COLORS.mild, CLASS_COLORS.moderate, CLASS_COLORS.severe],
            borderColor: 'rgba(0,0,0,0.3)',
            borderWidth: 2,
            hoverOffset: 8,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '65%',
        animation: { duration: 300 },
        plugins: {
            legend: {
                position: 'bottom',
                labels: { color: '#94a3b8', font: { size: 11, family: 'Inter' }, padding: 16, boxWidth: 12 }
            }
        }
    }
});


// ===========================================================================
//  Update UI from prediction data
// ===========================================================================
function updateUI(data) {
    if (data.keepalive) return;

    fpsCounter++;

    const cls  = data.class;
    const conf = data.confidence || 0;

    // -- Class label --
    classLabel.textContent = cls === 'no_face' ? 'No Face' : cls;
    classLabel.className   = 'class-label ' + cls;

    // -- Confidence bar --
    const barColor = CLASS_COLORS[cls] || '#64748b';
    confidenceBar.style.width      = conf + '%';
    confidenceBar.style.background = `linear-gradient(90deg, ${barColor}, ${barColor}cc)`;
    confidenceText.textContent     = conf.toFixed(1) + '%';

    // -- Per-class probabilities --
    if (data.all_probs) {
        for (const c of ['none', 'mild', 'moderate', 'severe']) {
            const pct = data.all_probs[c] || 0;
            const bar = document.getElementById('prob-' + c);
            const val = document.getElementById('prob-' + c + '-val');
            if (bar) bar.style.width = pct + '%';
            if (val) val.textContent  = pct.toFixed(1) + '%';
        }
    }

    // -- Stats --
    if (data.face_detected) {
        totalDetections++;
        confidenceSum += conf;
        classCounts[cls] = (classCounts[cls] || 0) + 1;
    }

    statTotal.textContent   = totalDetections;
    statAvgConf.textContent = totalDetections > 0
        ? (confidenceSum / totalDetections).toFixed(1) + '%'
        : '0%';

    // Most frequent class
    let dominant = '—';
    let maxCount = 0;
    for (const [c, n] of Object.entries(classCounts)) {
        if (n > maxCount) { maxCount = n; dominant = c; }
    }
    statDominant.textContent = dominant;

    // -- Timeline chart --
    if (data.face_detected && data.all_probs) {
        const label = data.timestamp || '';
        timelineChart.data.labels.push(label);
        if (timelineChart.data.labels.length > MAX_TIMELINE) {
            timelineChart.data.labels.shift();
        }

        for (const c of ['none', 'mild', 'moderate', 'severe']) {
            const ds = timelineChart.data.datasets[dsMap[c]];
            ds.data.push(data.all_probs[c] || 0);
            if (ds.data.length > MAX_TIMELINE) ds.data.shift();
        }
        timelineChart.update('none');
    }

    // -- Distribution doughnut --
    distChart.data.datasets[0].data = [
        classCounts.none || 0,
        classCounts.mild || 0,
        classCounts.moderate || 0,
        classCounts.severe || 0,
    ];
    distChart.update('none');
}


// ===========================================================================
//  SSE connection
// ===========================================================================
function connectSSE() {
    const evtSource = new EventSource('/events');

    evtSource.onopen = () => {
        statusBadge.textContent = 'Live';
        statusBadge.classList.add('connected');
    };

    evtSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            updateUI(data);
        } catch (e) {
            console.warn('SSE parse error', e);
        }
    };

    evtSource.onerror = () => {
        statusBadge.textContent = 'Reconnecting…';
        statusBadge.classList.remove('connected');
        evtSource.close();
        setTimeout(connectSSE, 3000);
    };
}

// Start
connectSSE();
