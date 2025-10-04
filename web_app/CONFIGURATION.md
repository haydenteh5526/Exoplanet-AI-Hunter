# Web App Configuration Summary

## ✅ All Hardcoded Values Removed

The web application has been updated to eliminate all hardcoded values. All configuration is now controlled through environment variables with sensible defaults.

## 🚀 How to Run the Web App

```cmd
python web_app\app.py
```

The application will start on `http://127.0.0.1:5000` by default.

## ⚙️ Configuration Options

All configuration is done through environment variables. You can:

1. **Set environment variables directly** (Windows CMD):
   ```cmd
   set FLASK_PORT=8080
   set FLASK_DEBUG=False
   python web_app\app.py
   ```

2. **Create a `.env` file** in the `web_app` directory (recommended):
   - Copy `.env.example` to `.env`
   - Modify values as needed
   - Install python-dotenv: `pip install python-dotenv`

## 📋 Available Configuration Variables

### Flask Server Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_PORT` | 5000 | Port number for the web server |
| `FLASK_HOST` | 0.0.0.0 | Host address (0.0.0.0 = all interfaces) |
| `FLASK_DEBUG` | True | Enable Flask debug mode |

### Exoplanet Matching Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `MATCHING_TOP_N` | 3 | Number of top matching exoplanets to return |
| `MATCHING_MIN_FEATURES` | 3 | Minimum features required for matching |
| `MATCHING_SIMILARITY_CONFIRMED` | 0.80 | Similarity threshold for CONFIRMED predictions (0.0-1.0) |
| `MATCHING_SIMILARITY_DEFAULT` | 0.85 | Default similarity threshold for matching (0.0-1.0) |

## 🔧 Changes Made

1. **Removed hardcoded values**:
   - Port number (was 5000)
   - Host address (was 0.0.0.0)
   - Debug mode (was True)
   - Top N matches (was 3)
   - Similarity thresholds (were 0.80 and 0.85)
   - Minimum features for matching (was 3)

2. **Added configuration constants** at the top of `app.py`:
   - All values now read from environment variables
   - Sensible defaults if environment variables not set
   - Uses `os.getenv()` for safe fallback behavior

3. **Created configuration files**:
   - `.env.example`: Template for environment configuration
   - Updated `README.md`: Documentation on how to configure and run

## 📝 Example Configuration

For production deployment on a different port:

```env
FLASK_PORT=8080
FLASK_HOST=0.0.0.0
FLASK_DEBUG=False
MATCHING_TOP_N=5
MATCHING_SIMILARITY_CONFIRMED=0.85
```

For development with more lenient matching:

```env
FLASK_PORT=5000
FLASK_HOST=127.0.0.1
FLASK_DEBUG=True
MATCHING_TOP_N=5
MATCHING_SIMILARITY_CONFIRMED=0.70
MATCHING_SIMILARITY_DEFAULT=0.75
```

## ✨ Benefits

- **No hardcoded values**: All configuration is external
- **Environment-specific settings**: Different configs for dev/prod
- **Easy deployment**: Change settings without modifying code
- **Better maintainability**: Clear separation of code and configuration
- **Flexibility**: Adjust behavior without redeploying code
