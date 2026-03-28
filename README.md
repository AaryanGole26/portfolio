# Portfolio - Aaryan Gole

A full-stack personal portfolio website showcasing my AI/ML projects, skills, and experience with a RAG-powered chatbot backend.

**🌐 Live:** [Visit Portfolio](https://clever-meerkat-91da30.netlify.app)  
**🔌 API:** [Render Backend](https://portfolio-backend.onrender.com)

## Features

- ✨ **Modern Design**: Clean, responsive portfolio interface
- 💬 **AI Chatbot**: RAG-powered chatbot with TF-IDF semantic search
- 📧 **Contact Form**: Email notifications with background threading
- 🗄️ **Database**: MongoDB Atlas for persistent storage
- ⚡ **Production Ready**: Deployed on Netlify (Frontend) + Render (Backend)
- 🧪 **Fully Tested**: 12 unit tests, all passing

## Tech Stack

### Frontend
- HTML5, CSS3 (Responsive Design)
- Plain JavaScript (no frameworks)
- Netlify (Static hosting + CDN)

### Backend
- **Framework**: Flask 2.3.2
- **Server**: Gunicorn 21.2.0 on Render
- **Database**: MongoDB Atlas with SSL/TLS
- **ML/NLP**: scikit-learn (TF-IDF), NumPy
- **AI**: Groq API (LLaMA 3.1 8B) + fallback responses
- **Email**: Gmail SMTP with background threading

### DevOps
- **CI/CD**: GitHub + Netlify + Render auto-deploy
- **Environment**: Python 3.12
- **Testing**: pytest (12 unit tests)

## Live Deployment

### Frontend
**Netlify**: https://clever-meerkat-91da30.netlify.app
- Auto-deployed from main branch
- Global CDN distribution
- HTTPS/SSL auto-managed

### Backend  
**Render**: https://portfolio-backend.onrender.com
- Auto-deployed from main branch
- REST API with 4 endpoints
- MongoDB Atlas database

## Local Development

### Frontend Setup
Simply open `index.html` in your browser.

### Backend Setup
```bash
# 1. Navigate to backend
cd backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Setup environment variables
cp .env.example .env
# Edit .env with your credentials

# 4. Run locally
python app.py
# Server: http://127.0.0.1:5000
```

### Running Tests
```bash
pytest test_api.py -v          # Unit tests
python integration_test.py      # Live server test
python deployment_check.py      # Config validation
```

## Project Structure

```
portfolio/
├── index.html                    # Frontend entry point
├── netlify.toml                  # Netlify config + API redirects
├── assets/
│   ├── css/                      # Responsive stylesheets
│   ├── img/                      # Portfolio images
│   └── videos/                   # Portfolio videos
├── backend/
│   ├── app.py                    # Flask application
│   ├── render.yaml               # Render deployment config
│   ├── requirements.txt          # Python dependencies
│   ├── test_api.py               # Unit tests (12 tests)
│   └── integration_test.py       # Live server tests
├── DEPLOYMENT_GUIDE.md           # Full deployment docs
└── README.md                     # This file
```

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/health` | Health check |
| POST | `/api/chat` | RAG chatbot query |
| POST | `/api/contact` | Submit contact form |
| GET | `/api/messages` | Retrieve submissions (admin) |

## Testing

### Test Coverage
- ✅ 12 unit tests (all passing)
- ✅ Email configuration
- ✅ Contact form validation
- ✅ Chat endpoint
- ✅ RAG search functionality
- ✅ Error handling
- ✅ Integration tests

### Running Tests
```bash
cd backend
pytest test_api.py -v        # Unit tests (12/12 passing)
python integration_test.py    # Live server test
python deployment_check.py    # Config validation
```

## Deployment Info

**Frontend**: Netlify (Auto-deployed on push to main)  
**Backend**: Render (Auto-deployed on push to main)  
**Database**: MongoDB Atlas with automatic backups  
**Status**: ✅ Live & Production-Ready

For detailed deployment instructions, see [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

## Contact

📧 Email: [goleaaryan7@gmail.com](mailto:goleaaryan7@gmail.com)  
🔗 LinkedIn: [linkedin.com/in/aaryan-gole](https://linkedin.com/in/aaryan-gole)  
💻 GitHub: [github.com/AaryanGole26](https://github.com/AaryanGole26)  
📞 Phone: +91 93097 44137

## License

MIT License - Feel free to fork and use!
