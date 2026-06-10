// Exoplanet AI Hunter - Frontend
// NASA Space Apps Challenge 2025

let probabilityChart = null;

document.addEventListener('DOMContentLoaded', () => {
    initSmoothScroll();
    initScrollSpy();
    initIntersectionObserver();
});

// ── Navigation ──────────────────────────────────────────
function initSmoothScroll() {
    document.querySelectorAll('.smooth-scroll').forEach(anchor => {
        anchor.addEventListener('click', e => {
            e.preventDefault();
            const target = document.querySelector(anchor.getAttribute('href'));
            if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });
}

function initScrollSpy() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link.smooth-scroll');
    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(s => {
            if (scrollY >= s.offsetTop - 100) current = s.id;
        });
        navLinks.forEach(link => {
            link.classList.toggle('active', link.getAttribute('href') === `#${current}`);
        });
    });
}

function initIntersectionObserver() {
    const observer = new IntersectionObserver(entries => {
        entries.forEach(entry => {
            if (entry.isIntersecting) entry.target.classList.add('is-visible');
        });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });

    document.querySelectorAll('.feature-card, .tech-stack-section').forEach(el => {
        el.classList.add('fade-in-section');
        observer.observe(el);
    });
}

// ── Form Handling ───────────────────────────────────────
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const formData = new FormData(e.target);
    const inputData = {};
    for (let [key, value] of formData.entries()) {
        if (value !== '') inputData[key] = parseFloat(value);
    }

    // Show loading state
    setLoading(true);

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(inputData)
        });
        const result = await response.json();
        if (result.error) {
            displayError(result.message || result.error);
        } else {
            displayResults(result);
        }
    } catch (error) {
        displayError('Failed to connect to the server. Please try again.');
    } finally {
        setLoading(false);
    }
});

function setLoading(loading) {
    const btn = document.getElementById('submitBtn');
    const icon = btn.querySelector('.btn-icon');
    const text = btn.querySelector('.btn-text');
    const spinner = btn.querySelector('.btn-loading');
    btn.disabled = loading;
    icon.style.display = loading ? 'none' : '';
    text.style.display = loading ? 'none' : '';
    spinner.style.display = loading ? 'inline-flex' : 'none';

    if (loading) {
        document.getElementById('results').innerHTML = `
            <div class="loading-state">
                <div class="loading-spinner"></div>
                <p>Analyzing observation data...</p>
            </div>
        `;
    }
}

// ── Results Display ─────────────────────────────────────
function displayResults(result) {
    const resultsDiv = document.getElementById('results');
    if (!result || !result.disposition) {
        displayError('Invalid response from server');
        return;
    }

    const config = {
        'CONFIRMED':      { cls: 'result-confirmed',      icon: '&#x2714;', label: 'Confirmed Exoplanet' },
        'CANDIDATE':      { cls: 'result-candidate',      icon: '&#x25CB;', label: 'Planet Candidate' },
        'FALSE_POSITIVE': { cls: 'result-false-positive',  icon: '&#x2718;', label: 'False Positive' },
        'NO_PREDICT':     { cls: 'result-no-predict',     icon: '&#x26A0;', label: 'Insufficient Data' }
    };
    const c = config[result.disposition] || config['NO_PREDICT'];

    let html = `
        <div class="result-card ${c.cls}">
            <div class="result-header">
                <span class="result-icon">${c.icon}</span>
                <h3>${c.label}</h3>
            </div>
            <div class="result-confidence">
                <div class="confidence-row">
                    <span>Confidence</span>
                    <strong>${(result.confidence * 100).toFixed(1)}%</strong>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
                </div>
            </div>
            <div class="result-message">
                <p>${result.message || 'Classification complete.'}</p>
            </div>
    `;

    // Matched exoplanet
    if (result.best_match) {
        const match = result.best_match;
        const matchedFeatures = Array.isArray(match.matched_features) ? match.matched_features : [];
        const displayName = formatPlanetName(match.name);

        html += `
            <div class="exoplanet-match">
                <h4>Closest Known Match</h4>
                <div class="match-card">
                    <div class="match-header">
                        <h3>${displayName}</h3>
                        <span class="match-score">${(match.similarity * 100).toFixed(1)}% similar</span>
                    </div>
                    <div class="match-details">
                        <p><strong>Catalog:</strong> ${match.name} &middot; <strong>Source:</strong> ${match.source} &middot; <strong>Status:</strong> ${match.disposition}</p>
                    </div>
                    ${matchedFeatures.length > 0 ? `
                    <div class="match-values">
                        <table class="comparison-table">
                            <thead><tr><th>Feature</th><th>Your Input</th><th>Database</th></tr></thead>
                            <tbody>
                                ${matchedFeatures.map(f => {
                                    const iv = result.input_summary?.[f];
                                    const dv = match.features?.[f];
                                    return `<tr>
                                        <td>${formatFeatureName(f)}</td>
                                        <td>${typeof iv === 'number' ? iv.toFixed(2) : 'N/A'}</td>
                                        <td>${typeof dv === 'number' ? dv.toFixed(2) : 'N/A'}</td>
                                    </tr>`;
                                }).join('')}
                            </tbody>
                        </table>
                    </div>` : ''}
                </div>
            </div>
        `;

        if (result.matched_exoplanets?.length > 1) {
            html += `<div class="other-matches"><p><strong>Other matches:</strong>
                ${result.matched_exoplanets.slice(1).map(m => `${m.name} (${(m.similarity*100).toFixed(0)}%)`).join(', ')}
            </p></div>`;
        }
    }

    html += `</div>`;
    resultsDiv.innerHTML = html;

    // Chart
    if (result.all_probabilities && Object.keys(result.all_probabilities).length > 0) {
        createProbabilityChart(result.all_probabilities);
        document.getElementById('chartContainer').style.display = 'block';
    }
}

function displayError(message) {
    document.getElementById('results').innerHTML = `
        <div class="result-card result-error">
            <div class="result-header"><span class="result-icon">&#x2718;</span><h3>Error</h3></div>
            <div class="result-message"><p>${message}</p></div>
        </div>
    `;
}

// ── Chart ───────────────────────────────────────────────
function createProbabilityChart(probabilities) {
    const ctx = document.getElementById('probabilityChart');
    if (probabilityChart) probabilityChart.destroy();

    const labels = Object.keys(probabilities);
    const data = Object.values(probabilities).map(v => v * 100);
    const colors = labels.map(l => {
        if (l === 'CONFIRMED') return 'rgba(16, 185, 129, 0.85)';
        if (l === 'CANDIDATE') return 'rgba(59, 130, 246, 0.85)';
        return 'rgba(239, 68, 68, 0.85)';
    });

    probabilityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels.map(l => l.replace('_', ' ')),
            datasets: [{ data, backgroundColor: colors, borderWidth: 0, borderRadius: 6 }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, max: 100, title: { display: true, text: 'Probability (%)' },
                     grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#a1a1aa' } },
                x: { grid: { display: false }, ticks: { color: '#a1a1aa' } }
            }
        }
    });
}

// ── Utilities ───────────────────────────────────────────
function formatPlanetName(id) {
    if (!id) return 'Unknown';
    if (id.startsWith('K0')) return `Kepler Object ${id}`;
    if (id.startsWith('EPIC')) return `K2 Object ${id}`;
    if (id.startsWith('TOI')) return `TESS Object ${id}`;
    return id;
}

function formatFeatureName(name) {
    return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

function resetForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('results').innerHTML = `
        <div class="empty-state">
            <svg class="empty-icon" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M12 6v6l4 2"></path>
            </svg>
            <h3>Awaiting Classification</h3>
            <p>Enter observation parameters and submit to classify</p>
        </div>
    `;
    document.getElementById('chartContainer').style.display = 'none';
    if (probabilityChart) { probabilityChart.destroy(); probabilityChart = null; }
}

function fillExample() {
    // Kepler-227 b (confirmed exoplanet)
    const example = {
        orbital_period: 9.49,
        transit_duration: 2.96,
        planetary_radius: 2.26,
        equilibrium_temperature: 793,
        stellar_radius: 0.93,
        transit_depth: 615.8,
        impact_parameter: 0.15,
        insolation_flux: 105.0,
        stellar_surface_gravity: 4.47
    };
    for (const [key, val] of Object.entries(example)) {
        const el = document.getElementById(key);
        if (el) el.value = val;
    }
}
