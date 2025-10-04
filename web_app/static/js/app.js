// Exoplanet AI Hunter - Frontend JavaScript
// NASA Space Apps Challenge 2025

let probabilityChart = null;
let featureImportanceChart = null;

// Load model information on page load
document.addEventListener('DOMContentLoaded', function() {
    loadModelInfo();
    loadFeatureImportance();
    initSmoothScroll();
    initScrollSpy();
    initIntersectionObserver();
});

// ============================================
// Smooth Scrolling Navigation
// ============================================
function initSmoothScroll() {
    document.querySelectorAll('.smooth-scroll').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId.startsWith('#')) {
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });
}

// ============================================
// Navigation Active State (Scroll Spy)
// ============================================
function initScrollSpy() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-link.smooth-scroll');
    
    window.addEventListener('scroll', () => {
        let current = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (scrollY >= (sectionTop - 100)) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    });
}

// ============================================
// Intersection Observer for Fade-in Animations
// ============================================
function initIntersectionObserver() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('is-visible');
            }
        });
    }, observerOptions);
    
    // Observe feature cards
    document.querySelectorAll('.feature-card').forEach(card => {
        card.classList.add('fade-in-section');
        observer.observe(card);
    });
}

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch('/api/model-info');
        const data = await response.json();
        
        const modelInfoEl = document.getElementById('modelInfo');
        
        if (data.error) {
            if (modelInfoEl) {
                modelInfoEl.innerHTML = `
                    <div class="model-info-content">
                        <span class="info-icon">⚠️</span>
                        <span>${data.error}</span>
                    </div>
                `;
            }
            return;
        }
        
        // Update hero stats
        const accuracyEl = document.getElementById('modelAccuracy');
        if (data.accuracy && accuracyEl) {
            accuracyEl.textContent = `${(data.accuracy * 100).toFixed(1)}%`;
        }
        
        // Update model info banner
        if (modelInfoEl) {
            modelInfoEl.innerHTML = `
                <div class="model-info-content">
                    <span class="info-icon">ℹ️</span>
                    <span><strong>Model:</strong> ${data.model_type.toUpperCase()} | 
                    <strong>Accuracy:</strong> ${(data.accuracy * 100).toFixed(2)}% | 
                    <strong>Training Date:</strong> ${new Date(data.training_date).toLocaleDateString()} | 
                    <strong>Classes:</strong> ${data.classes.join(', ')}</span>
                </div>
            `;
        }
        
    } catch (error) {
        console.error('Error loading model info:', error);
        const modelInfoEl = document.getElementById('modelInfo');
        if (modelInfoEl) {
            modelInfoEl.innerHTML = `
                <div class="model-info-content">
                    <span class="info-icon">⚠️</span>
                    <span>Failed to load model information</span>
                </div>
            `;
        }
    }
}

// Load feature importance
async function loadFeatureImportance() {
    try {
        const response = await fetch('/api/feature-importance');
        const data = await response.json();
        
        if (data.feature_importance) {
            createFeatureImportanceChart(data.feature_importance);
            document.getElementById('featureImportanceContainer').style.display = 'block';
        }
    } catch (error) {
        console.error('Error loading feature importance:', error);
    }
}

// Handle form submission
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Collect form data
    const formData = new FormData(e.target);
    const inputData = {};
    let featureCount = 0;
    
    for (let [key, value] of formData.entries()) {
        if (value !== '' && value !== null) {
            inputData[key] = parseFloat(value);
            featureCount++;
        }
    }
    
    // Validate minimum features
    if (featureCount < 3) {
        displayResults({
            disposition: 'NO_PREDICT',
            confidence: 0,
            message: 'Please provide at least 3 features for prediction.',
            all_probabilities: {},
            recommendation: 'Enter more feature values to get a classification.'
        });
        return;
    }
    
    // Show loading state
    document.getElementById('results').innerHTML = `
        <div class="loading">
            <p>🔄 Analyzing data...</p>
        </div>
    `;
    
    try {
        // Make prediction request
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(inputData)
        });
        
        const result = await response.json();
        
        if (result.error) {
            displayError(result.message || result.error);
        } else {
            displayResults(result);
        }
    } catch (error) {
        console.error('Prediction error:', error);
        displayError('Failed to get prediction. Please try again.');
    }
});

// Display prediction results
function displayResults(result) {
    try {
        const resultsDiv = document.getElementById('results');
        
        if (!result || !result.disposition) {
            throw new Error('Invalid result data');
        }
        
        // Determine result styling
        let dispositionClass = '';
        let dispositionIcon = '';
        
        switch(result.disposition) {
            case 'CONFIRMED':
                dispositionClass = 'result-confirmed';
                dispositionIcon = '✅';
                break;
            case 'CANDIDATE':
                dispositionClass = 'result-candidate';
                dispositionIcon = '🔍';
                break;
            case 'FALSE_POSITIVE':
                dispositionClass = 'result-false-positive';
                dispositionIcon = '❌';
                break;
            case 'NO_PREDICT':
                dispositionClass = 'result-no-predict';
                dispositionIcon = '⚠️';
                break;
            default:
                dispositionClass = 'result-no-predict';
                dispositionIcon = '❓';
        }
        
        // Build results HTML
        let resultsHTML = `
            <div class="result-card ${dispositionClass}">
                <div class="result-header">
                    <span class="result-icon">${dispositionIcon}</span>
                    <h3>${result.disposition.replace('_', ' ')}</h3>
                </div>
                <div class="result-confidence">
                    <p>ML Model Confidence: <strong>${(result.confidence * 100).toFixed(2)}%</strong></p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
                    </div>
                    <p class="confidence-explanation">This is how confident the AI model is in its classification</p>
                </div>
                <div class="result-message">
                    <p>${result.message || 'Classification complete.'}</p>
                </div>
        `;
        
        // Add matched exoplanet information if available
        if (result.best_match) {
            const match = result.best_match;
            
            // Defensive check for matched_features
            const matchedFeatures = Array.isArray(match.matched_features) ? match.matched_features : [];
            const featuresProvided = Array.isArray(result.features_provided) ? result.features_provided : [];
            
            // Get planet name (convert KOI to Kepler name if possible)
            const displayName = formatPlanetName(match.name);
            
            resultsHTML += `
                <div class="exoplanet-match">
                    <h4>🎯 Known Exoplanet Match Found!</h4>
                    <p class="match-explanation">Your input closely matches a known exoplanet in our database. The <strong>similarity score</strong> below indicates how well your measurements align with this confirmed planet's characteristics.</p>
                    <div class="match-card">
                        <div class="match-header">
                            <h3>${displayName}</h3>
                            <span class="match-score">${(match.similarity * 100).toFixed(1)}% Similarity</span>
                        </div>
                        <div class="match-details">
                            <p><strong>Catalog ID:</strong> ${match.name || 'Unknown'}</p>
                            <p><strong>Source:</strong> ${match.source || 'Unknown'} Mission</p>
                            ${match.discovery_year ? `<p><strong>Discovery Year:</strong> ${match.discovery_year}</p>` : ''}
                            <p><strong>Features Compared:</strong> ${match.num_matched || matchedFeatures.length}/${match.features_provided || featuresProvided.length}</p>
                            ${match.features_skipped > 0 ? `<p class="features-skipped"><em>Note: ${match.features_skipped} feature(s) not available in database for this object</em></p>` : ''}
                        </div>
                        <details class="match-values">
                            <summary>View Comparison</summary>
                            <table class="comparison-table">
                                <thead>
                                    <tr>
                                        <th>Feature</th>
                                        <th>Your Input</th>
                                        <th>Database Value</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${matchedFeatures.map(feature => {
                                        const inputVal = result.input_summary ? result.input_summary[feature] : 'N/A';
                                        const dbVal = match.features && match.features[feature] ? match.features[feature] : null;
                                        return `
                                            <tr>
                                                <td>${formatFeatureName(feature)}</td>
                                                <td>${typeof inputVal === 'number' ? inputVal.toFixed(2) : inputVal}</td>
                                                <td>${typeof dbVal === 'number' ? dbVal.toFixed(2) : (dbVal || 'N/A')}</td>
                                            </tr>
                                        `;
                                    }).join('')}
                                </tbody>
                            </table>
                        </details>
                    </div>
                </div>
            `;
            
            // Show additional matches if available
            if (result.matched_exoplanets && Array.isArray(result.matched_exoplanets) && result.matched_exoplanets.length > 1) {
                resultsHTML += `
                    <div class="other-matches">
                        <p><strong>Other Possible Matches:</strong></p>
                        <ul>
                            ${result.matched_exoplanets.slice(1).map(m => 
                                `<li>${m.name || 'Unknown'} (${(m.similarity * 100).toFixed(1)}% match)</li>`
                            ).join('')}
                        </ul>
                    </div>
                `;
            }
        }
        
        // Add quality score if available
        if (result.quality_score !== undefined) {
            resultsHTML += `
                <div class="quality-info">
                    <p><strong>Data Quality:</strong> ${result.quality_message || 'N/A'}</p>
                    <p><strong>Features Provided:</strong> ${result.features_provided ? result.features_provided.length : 0}/9</p>
                </div>
            `;
        }
        
        // Add recommendation
        if (result.recommendation) {
            resultsHTML += `
                <div class="recommendation">
                    <p><strong>💡 Recommendation:</strong> ${result.recommendation}</p>
                </div>
            `;
        }
        
        resultsHTML += `</div>`;
        
        resultsDiv.innerHTML = resultsHTML;
        
        // Update probability chart
        if (result.all_probabilities && Object.keys(result.all_probabilities).length > 0) {
            createProbabilityChart(result.all_probabilities);
            document.getElementById('chartContainer').style.display = 'block';
        }
        
    } catch (error) {
        console.error('Error in displayResults:', error);
        throw error;
    }
}

// Display error message
function displayError(message) {
    document.getElementById('results').innerHTML = `
        <div class="result-card result-error">
            <div class="result-header">
                <span class="result-icon">❌</span>
                <h3>Error</h3>
            </div>
            <div class="result-message">
                <p>${message}</p>
            </div>
        </div>
    `;
}

// Create probability chart
function createProbabilityChart(probabilities) {
    const ctx = document.getElementById('probabilityChart');
    
    // Destroy existing chart
    if (probabilityChart) {
        probabilityChart.destroy();
    }
    
    const labels = Object.keys(probabilities);
    const data = Object.values(probabilities).map(v => v * 100);
    
    // Color scheme
    const colors = labels.map(label => {
        switch(label) {
            case 'CONFIRMED': return 'rgba(76, 175, 80, 0.8)';
            case 'CANDIDATE': return 'rgba(33, 150, 243, 0.8)';
            case 'FALSE_POSITIVE': return 'rgba(244, 67, 54, 0.8)';
            default: return 'rgba(158, 158, 158, 0.8)';
        }
    });
    
    probabilityChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels.map(l => l.replace('_', ' ')),
            datasets: [{
                label: 'Probability (%)',
                data: data,
                backgroundColor: colors,
                borderColor: colors.map(c => c.replace('0.8', '1')),
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Probability (%)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Classification Probabilities'
                }
            }
        }
    });
}

// Create feature importance chart
function createFeatureImportanceChart(featureImportance) {
    const ctx = document.getElementById('featureImportanceChart');
    
    // Destroy existing chart
    if (featureImportanceChart) {
        featureImportanceChart.destroy();
    }
    
    // Sort by importance and take top 9
    const sortedFeatures = featureImportance.slice(0, 9);
    const labels = sortedFeatures.map(f => formatFeatureName(f.feature));
    const data = sortedFeatures.map(f => f.importance * 100);
    
    featureImportanceChart = new Chart(ctx, {
        type: 'bar',  // Changed from 'horizontalBar' to 'bar' for Chart.js v3+
        data: {
            labels: labels,
            datasets: [{
                label: 'Importance (%)',
                data: data,
                backgroundColor: 'rgba(156, 39, 176, 0.6)',
                borderColor: 'rgba(156, 39, 176, 1)',
                borderWidth: 2
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Importance (%)'
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Top Feature Importances'
                }
            }
        }
    });
}

// Format planet name for display (KOI to Kepler name mapping)
function formatPlanetName(catalogId) {
    // Known KOI to Kepler name mappings (partial list - expand as needed)
    const keplernameMap = {
        'K00752.01': 'Kepler-227 b',
        'K00752.02': 'Kepler-227 c',
        'K00082.01': 'Kepler-10 b',
        'K00082.02': 'Kepler-10 c',
        'K00266.01': 'Kepler-68 b',
        'K00266.02': 'Kepler-68 c',
        'K00266.03': 'Kepler-68 d',
        'K01593.01': 'Kepler-62 e',
        'K01593.02': 'Kepler-62 f',
        // Add more mappings as needed
    };
    
    // Check if we have a known planet name
    if (keplernameMap[catalogId]) {
        return keplernameMap[catalogId];
    }
    
    // Otherwise, return catalog ID with formatted label
    if (catalogId && catalogId.startsWith('K')) {
        return `Kepler Object ${catalogId}`;
    } else if (catalogId && catalogId.startsWith('EPIC')) {
        return `K2 Object ${catalogId}`;
    } else if (catalogId && catalogId.startsWith('TOI')) {
        return `TESS Object ${catalogId}`;
    }
    
    return catalogId || 'Unknown Exoplanet';
}

// Format feature name for display
function formatFeatureName(name) {
    return name
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
}

// Reset form
function resetForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('results').innerHTML = `
        <p>👆 Enter features and click "Classify Exoplanet" to see results</p>
    `;
    document.getElementById('chartContainer').style.display = 'none';
    
    if (probabilityChart) {
        probabilityChart.destroy();
        probabilityChart = null;
    }
}
