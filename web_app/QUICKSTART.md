# 🚀 Quick Start - Exoplanet AI Hunter Web App

## Run the Web App

```cmd
python web_app\app.py
```

**Access at**: http://127.0.0.1:5000

---

## ✅ No Hardcoded Values

All configuration is now environment-based:

- ✅ Port number → `FLASK_PORT`
- ✅ Host address → `FLASK_HOST`
- ✅ Debug mode → `FLASK_DEBUG`
- ✅ Matching settings → `MATCHING_*` variables
- ✅ All numeric thresholds configurable

---

## 📝 Quick Configuration

Create `web_app\.env`:

```env
FLASK_PORT=5000
FLASK_HOST=0.0.0.0
FLASK_DEBUG=True
MATCHING_TOP_N=3
MATCHING_MIN_FEATURES=3
MATCHING_SIMILARITY_CONFIRMED=0.80
MATCHING_SIMILARITY_DEFAULT=0.85
```

See `.env.example` for template.

---

## 📚 Documentation

- `README.md` - Full user guide
- `CONFIGURATION.md` - Detailed configuration guide
- `.env.example` - Configuration template
