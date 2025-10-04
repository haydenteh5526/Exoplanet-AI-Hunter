from flask import Flask, render_template, jsonify, request
import pandas as pd
from pathlib import Path
import numpy as np

app = Flask(__name__)

# Load exoplanet data
DATA_PATH = Path(__file__).parent.parent / 'data' / 'processed' / 'combined.parquet'

def load_exoplanet_data():
    try:
        df = pd.read_parquet(DATA_PATH)
        return process_exoplanet_data(df)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return []

def process_exoplanet_data(df):
    # Convert DataFrame to list of dictionaries with processed values
    exoplanets = []
    
    for _, row in df.iterrows():
        exoplanet = {
            'id': str(row.get('rowid', '')),
            'name': row.get('kepoi_name', 'Unknown'),
            'radius': float(row.get('koi_prad', 0)),  # Planet radius in Earth radii
            'temperature': float(row.get('koi_teq', 0)),  # Equilibrium temperature in Kelvin
            'orbital_period': float(row.get('koi_period', 0)),  # Orbital period in days
            'star_type': determine_star_type(row),
            'mass': estimate_planet_mass(row.get('koi_prad', 0)),  # Estimated mass based on radius
            'status': row.get('koi_disposition', 'UNKNOWN')
        }
        exoplanets.append(exoplanet)
    
    return exoplanets

def determine_star_type(row):
    # Simplified star type determination based on temperature
    temp = row.get('koi_steff', 0)
    
    if temp > 30000:
        return 'O'
    elif temp > 10000:
        return 'B'
    elif temp > 7500:
        return 'A'
    elif temp > 6000:
        return 'F'
    elif temp > 5200:
        return 'G'
    elif temp > 3700:
        return 'K'
    else:
        return 'M'

def estimate_planet_mass(radius):
    # Simple mass estimation based on radius (very approximate)
    # Using mass-radius relationship for rocky planets
    if radius < 1.5:
        # Rocky planet approximation
        return radius ** 3.7
    else:
        # Gas giant approximation
        return radius ** 2.06

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/exoplanets')
def get_exoplanets():
    # Get query parameters
    size_filter = request.args.get('size', type=float)
    temp_filter = request.args.get('temperature')
    star_filter = request.args.get('star_type')
    
    # Load and filter data
    exoplanets = load_exoplanet_data()
    
    if size_filter:
        exoplanets = [p for p in exoplanets if p['radius'] <= size_filter]
        
    if temp_filter:
        if temp_filter == 'habitable':
            exoplanets = [p for p in exoplanets if 200 <= p['temperature'] <= 300]
        elif temp_filter == 'hot':
            exoplanets = [p for p in exoplanets if 300 < p['temperature'] <= 700]
        elif temp_filter == 'very_hot':
            exoplanets = [p for p in exoplanets if p['temperature'] > 700]
            
    if star_filter and star_filter != 'All':
        exoplanets = [p for p in exoplanets if p['star_type'] == star_filter]
    
    return jsonify(exoplanets)

@app.route('/api/exoplanet/<planet_id>')
def get_exoplanet(planet_id):
    exoplanets = load_exoplanet_data()
    planet = next((p for p in exoplanets if p['id'] == planet_id), None)
    
    if planet:
        return jsonify(planet)
    else:
        return jsonify({'error': 'Planet not found'}), 404

@app.route('/api/stats')
def get_stats():
    exoplanets = load_exoplanet_data()
    
    stats = {
        'total_count': len(exoplanets),
        'habitable_count': len([p for p in exoplanets if 200 <= p['temperature'] <= 300]),
        'avg_radius': np.mean([p['radius'] for p in exoplanets]),
        'avg_temperature': np.mean([p['temperature'] for p in exoplanets]),
        'star_type_distribution': {},
        'temperature_ranges': {
            'cold': len([p for p in exoplanets if p['temperature'] < 200]),
            'habitable': len([p for p in exoplanets if 200 <= p['temperature'] <= 300]),
            'hot': len([p for p in exoplanets if 300 < p['temperature'] <= 700]),
            'very_hot': len([p for p in exoplanets if p['temperature'] > 700])
        }
    }
    
    # Calculate star type distribution
    for planet in exoplanets:
        star_type = planet['star_type']
        stats['star_type_distribution'][star_type] = stats['star_type_distribution'].get(star_type, 0) + 1
    
    return jsonify(stats)

if __name__ == '__main__':
    app.run(debug=True)
