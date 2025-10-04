// Global state
let exoplanets = [];
let currentFilters = {
    size: 1,
    temperature: 'All Temperatures',
    starType: 'All'
};

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', () => {
    fetchExoplanets();
    initializeFilters();
    initializeVisualizations();
    setupSearch();
    initializeAnimations();
});

// Fetch exoplanet data from backend
async function fetchExoplanets() {
    try {
        const response = await fetch('/api/exoplanets');
        exoplanets = await response.json();
        displayExoplanets(exoplanets);
        populateComparisonDropdowns(exoplanets);
    } catch (error) {
        console.error('Error fetching exoplanets:', error);
        showError('Failed to load exoplanet data');
    }
}

// Display exoplanets in the grid
function displayExoplanets(planets) {
    const grid = document.getElementById('exoplanetGrid');
    grid.innerHTML = '';

    if (planets.length === 0) {
        grid.innerHTML = `
            <div class="col-12 text-center">
                <h3>No planets found matching your criteria</h3>
                <p>Try adjusting your filters</p>
            </div>
        `;
        return;
    }

    planets.forEach(planet => {
        const card = createExoplanetCard(planet);
        grid.appendChild(card);
    });
}

// Create a card for an individual exoplanet
function createExoplanetCard(planet) {
    const div = document.createElement('div');
    div.className = 'col-md-4 mb-4';
    
    const habitabilityScore = calculateHabitabilityScore(planet);
    const habitabilityClass = getHabitabilityClass(habitabilityScore);
    
    div.innerHTML = `
        <div class="exoplanet-card ${habitabilityClass}" onclick="showPlanetDetails('${planet.id}')">
            <h3 class="planet-name">${planet.name}</h3>
            <div class="parameter-badges">
                <span class="parameter-badge">
                    <i class="fas fa-ruler"></i> ${formatNumber(planet.radius)} R⊕
                </span>
                <span class="parameter-badge">
                    <i class="fas fa-thermometer-half"></i> ${formatNumber(planet.temperature)}K
                </span>
                <span class="parameter-badge">
                    <i class="fas fa-clock"></i> ${formatNumber(planet.orbital_period)}d
                </span>
            </div>
            <div class="planet-preview">
                <canvas id="preview-${planet.id}" width="150" height="150"></canvas>
            </div>
            <div class="habitability-score mt-2">
                Habitability Score: ${habitabilityScore}%
            </div>
        </div>
    `;
    
    // After the card is added to DOM, initialize the preview
    setTimeout(() => renderPlanetPreview(planet), 0);
    return div;
}

// Initialize filter controls
function initializeFilters() {
    // Size range slider
    const sizeRange = document.getElementById('sizeRange');
    const sizeValue = document.getElementById('sizeValue');
    
    sizeRange.addEventListener('input', (e) => {
        currentFilters.size = e.target.value;
        sizeValue.textContent = e.target.value;
        applyFilters();
    });

    // Temperature select
    document.querySelector('.cosmic-select').addEventListener('change', (e) => {
        currentFilters.temperature = e.target.value;
        applyFilters();
    });

    // Star type buttons
    document.querySelectorAll('.btn-group .btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.querySelectorAll('.btn-group .btn').forEach(b => b.classList.remove('active'));
            e.target.classList.add('active');
            currentFilters.starType = e.target.textContent;
            applyFilters();
        });
    });
}

// Apply filters to exoplanet list
function applyFilters() {
    const filteredPlanets = exoplanets.filter(planet => {
        const sizeMatch = planet.radius <= currentFilters.size;
        const tempMatch = currentFilters.temperature === 'All Temperatures' || 
                         matchTemperatureRange(planet.temperature, currentFilters.temperature);
        const starMatch = currentFilters.starType === 'All' ||
                         planet.star_type === currentFilters.starType;
        
        return sizeMatch && tempMatch && starMatch;
    });
    
    displayExoplanets(filteredPlanets);
    updateVisualization(filteredPlanets);
}

// Initialize visualizations
function initializeVisualizations() {
    document.querySelector('[data-viz="scatter"]').addEventListener('click', () => {
        const trace = {
            x: exoplanets.map(p => p.mass),
            y: exoplanets.map(p => p.radius),
            mode: 'markers',
            type: 'scatter',
            marker: {
                color: exoplanets.map(p => p.temperature),
                colorscale: 'Viridis',
                showscale: true,
                size: 10
            },
            text: exoplanets.map(p => p.name),
            hovertemplate: 
                '<b>%{text}</b><br>' +
                'Mass: %{x}M⊕<br>' +
                'Radius: %{y}R⊕<br>' +
                'Temperature: %{marker.color}K<br>' +
                '<extra></extra>'
        };

        const layout = {
            title: 'Exoplanet Mass vs Radius',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#fff' },
            xaxis: { 
                title: 'Mass (Earth Masses)',
                gridcolor: '#333'
            },
            yaxis: { 
                title: 'Radius (Earth Radii)',
                gridcolor: '#333'
            }
        };

        Plotly.newPlot('vizContainer', [trace], layout);
    });

    // Initialize with scatter plot
    document.querySelector('[data-viz="scatter"]').click();
}

// Show detailed planet information
function showPlanetDetails(planetId) {
    const planet = exoplanets.find(p => p.id === planetId);
    if (!planet) return;

    const habitabilityScore = calculateHabitabilityScore(planet);
    const modal = new bootstrap.Modal(document.getElementById('planetModal'));
    const detailsContainer = document.querySelector('.planet-details');
    
    detailsContainer.innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <div class="planet-visualization">
                    <canvas id="detail-${planetId}" width="300" height="300"></canvas>
                </div>
            </div>
            <div class="col-md-6">
                <h2>${planet.name}</h2>
                <div class="planet-parameters">
                    <p><strong>Mass:</strong> ${formatNumber(planet.mass)}M⊕</p>
                    <p><strong>Radius:</strong> ${formatNumber(planet.radius)}R⊕</p>
                    <p><strong>Temperature:</strong> ${formatNumber(planet.temperature)}K</p>
                    <p><strong>Orbital Period:</strong> ${formatNumber(planet.orbital_period)} days</p>
                    <p><strong>Star Type:</strong> ${planet.star_type}</p>
                </div>
                <div class="habitability-score">
                    <h4>Habitability Score: ${habitabilityScore}%</h4>
                    <div class="progress">
                        <div class="progress-bar" role="progressbar" 
                             style="width: ${habitabilityScore}%" 
                             aria-valuenow="${habitabilityScore}" 
                             aria-valuemin="0" 
                             aria-valuemax="100">
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    modal.show();
    setTimeout(() => renderPlanetDetail(planet), 0);
}

// Calculate habitability score
function calculateHabitabilityScore(planet) {
    let score = 0;
    
    // Temperature scoring (Earth-like temperatures get higher scores)
    const tempScore = 1 - Math.min(Math.abs(planet.temperature - 288) / 288, 1);
    score += tempScore * 40;
    
    // Size scoring (Earth-like sizes get higher scores)
    const sizeScore = 1 - Math.min(Math.abs(planet.radius - 1), 1);
    score += sizeScore * 30;
    
    // Orbital period scoring (preference for periods similar to Earth's)
    const periodScore = 1 - Math.min(Math.abs(planet.orbital_period - 365) / 365, 1);
    score += periodScore * 30;
    
    return Math.round(score);
}

// Setup search functionality
function setupSearch() {
    const searchInput = document.querySelector('.cosmic-search');
    let debounceTimeout;

    searchInput.addEventListener('input', (e) => {
        clearTimeout(debounceTimeout);
        debounceTimeout = setTimeout(() => {
            const searchTerm = e.target.value.toLowerCase();
            const filteredPlanets = exoplanets.filter(planet => 
                planet.name.toLowerCase().includes(searchTerm) ||
                planet.star_type.toLowerCase().includes(searchTerm)
            );
            displayExoplanets(filteredPlanets);
        }, 300);
    });
}

// Initialize animations
function initializeAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate__animated', 'animate__fadeIn');
            }
        });
    });

    document.querySelectorAll('.exoplanet-card').forEach(card => {
        observer.observe(card);
    });
}

// Helper Functions
function formatNumber(num) {
    return Number.isFinite(num) ? num.toFixed(2) : 'N/A';
}

function getHabitabilityClass(score) {
    if (score >= 80) return 'highly-habitable';
    if (score >= 60) return 'potentially-habitable';
    if (score >= 40) return 'marginally-habitable';
    return 'less-habitable';
}

function matchTemperatureRange(temp, range) {
    switch(range) {
        case 'Habitable Zone (200-300K)':
            return temp >= 200 && temp <= 300;
        case 'Hot (300-700K)':
            return temp > 300 && temp <= 700;
        case 'Very Hot (700K+)':
            return temp > 700;
        default:
            return true;
    }
}

// Error handling
function showError(message) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-danger position-fixed top-0 end-0 m-3 animate__animated animate__fadeIn';
    alert.innerHTML = `
        <i class="fas fa-exclamation-circle"></i>
        ${message}
        <button type="button" class="btn-close" onclick="this.parentElement.remove()"></button>
    `;
    document.body.appendChild(alert);
    setTimeout(() => {
        alert.classList.add('animate__fadeOut');
        setTimeout(() => alert.remove(), 500);
    }, 4500);
}
