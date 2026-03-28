from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_pymongo import PyMongo
from dotenv import load_dotenv
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, UTC
from bson.objectid import ObjectId
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import certifi
import requests

load_dotenv()

app = Flask(__name__)
CORS(app)

# MongoDB Configuration with SSL certificate
mongo_uri = os.getenv('MONGO_URI')
if not mongo_uri:
    raise ValueError("MONGO_URI not found in environment variables. Please set it in your .env file.")

# Add tlsCAFile parameter for SSL certificate
if '?' in mongo_uri:
    app.config['MONGO_URI'] = f"{mongo_uri}&tlsCAFile={certifi.where()}"
else:
    app.config['MONGO_URI'] = f"{mongo_uri}?tlsCAFile={certifi.where()}"

mongo = PyMongo(app)

# Groq API for AI responses (free tier available)
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# RAG Knowledge Base (using TF-IDF for lightweight deployment)
class PortfolioRAG:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        self.knowledge_base = self._create_knowledge_base()
        self.tfidf_matrix = None
        self._generate_embeddings()
    
    def _create_knowledge_base(self):
        """Portfolio information structured for RAG"""
        return [
            {
                "category": "about",
                "content": "I am Aaryan Gole, a passionate AI & Data Science student at VCET with hands-on experience in building intelligent applications and scalable solutions. I have expertise in machine learning, backend development, and RAG chatbots."
            },
            {
                "category": "skills_ml",
                "content": "My AI/ML skills include: TensorFlow, RNN & Transformers, RAG Models, NLP, and Computer Vision. I work with deep learning frameworks and build production-grade AI systems."
            },
            {
                "category": "skills_backend",
                "content": "Backend development skills: Flask, Django, Python, SQL, and RESTful APIs. I design scalable systems and handle database management with MongoDB and relational databases."
            },
            {
                "category": "skills_frontend",
                "content": "Frontend skills: JavaScript, ReactJS, HTML/CSS, and responsive design. I build interactive user interfaces and web applications."
            },
            {
                "category": "tools",
                "content": "Tools and technologies: Postman for API testing, Figma for design, Power BI for data visualization, Docker for containerization, and Git for version control."
            },
            {
                "category": "experience_citius",
                "content": "I worked as an Academic Intern at Citius Cloud from June 2024 to July 2024. I implemented RNN, Transformers, and RAG models in TensorFlow to build a chatbot trained on web-scraped data, gaining experience in production-ready AI systems."
            },
            {
                "category": "education",
                "content": "I am pursuing a Bachelor of Engineering in AI & Data Science at VCET, Vasai with a current GPA of 8.07. I have completed AISSCE (CBSE XII) with 74% in Science stream and AISSE (CBSE X) with 94.2%."
            },
            {
                "category": "certifications",
                "content": "My certifications include: Fundamentals of Machine Learning from Microsoft Learn, Data Analytics Program from Godrej Infotech Ltd., and MLOps with Data Version Control from Infosys Springboard."
            },
            {
                "category": "project_lawpal",
                "content": "LawPal is a legal chatbot with voice assistant built using Flask backend with RAG architecture. I implemented semantic legal search and voice-enabled navigation to simplify access to complex legal documents for non-technical users."
            },
            {
                "category": "project_optiresume",
                "content": "OptiResume-AI is a full-stack system for resume optimization and career enhancement. It parses PDF/DOCX resumes (PyMuPDF, spaCy), analyzes them against job descriptions, and generates ATS-friendly CVs with 6 professional templates. Key features: Resume Analysis (skills gap, match scores), 3-layer Job Link Extraction (Indeed/Naukri), AI-powered bullet rewriting, and LaTeX/PDF export. It also includes optional MCP servers for tool integration."
            },
            {
                "category": "project_finslash",
                "content": "FinSlash is an AI-powered loan approval dashboard using machine learning. I implemented exploratory data analysis and logistic regression models with Streamlit to make loan decisions faster, smarter, and more transparent."
            },
            {
                "category": "project_rento",
                "content": "Rento is a Django-based web application for renting engineering tools. It features user authentication, model relationships, e-commerce cart management, and order handling, demonstrating full-stack development skills."
            },
            {
                "category": "project_cinesleuth",
                "content": "CineSLEUTH is an intelligent movie recommendation system combining TF-IDF vectorization, cosine similarity, fuzzy matching, and Apriori-based collaborative filtering for accurate recommendations."
            },
            {
                "category": "project_drivesense",
                "content": "DriveSense is an AI-powered driver wellness system monitoring facial cues and eye movements using deep learning and OpenCV to detect fatigue, stress, or distress with real-time interventions."
            },
            {
                "category": "project_lume",
                "content": "LUME is an AI-powered app that turns your images into viral memes. Upload a photo, add an optional trending topic, and let Lume use BLIP and Groq AI to craft witty, high-quality captions with stylish overlays—ready to download and share instantly. It features real-time caption synthesis, font overlays, and easy sharing options."
            },
            {
                "category": "contact",
                "content": "You can reach me through LinkedIn at linkedin.com/in/aaryan-gole, GitHub at github.com/AaryanGole26, email at goleaaryan7@gmail.com, or phone at +91 93097 44137."
            }
        ]
    
    def _generate_embeddings(self):
        """Generate TF-IDF vectors for all knowledge base chunks"""
        texts = [doc['content'] for doc in self.knowledge_base]
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
    
    def search(self, query, top_k=3):
        """Retrieve relevant portfolio information using TF-IDF similarity"""
        query_lower = query.lower()
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Boost scores for keyword matches in category names
        for idx, doc in enumerate(self.knowledge_base):
            category = doc['category'].lower().replace('_', ' ')
            content_lower = doc['content'].lower()
            # Boost if query word appears in category or content
            if any(word in category or word in content_lower for word in query_lower.split()):
                similarities[idx] = max(similarities[idx], 0.3)  # Ensure minimum score for keyword matches
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [
            {
                "content": self.knowledge_base[idx]['content'],
                "category": self.knowledge_base[idx]['category'],
                "score": float(similarities[idx])
            }
            for idx in top_indices if similarities[idx] > 0.01
        ]
        return results

# Initialize RAG system
rag = PortfolioRAG()

def send_email(to_email, name, message):
    """Send email using SendGrid API (preferred for Render) or SMTP fallback (non-blocking)"""
    import threading
    def _send():
        try:
            sendgrid_api_key = os.getenv('SENDGRID_API_KEY')
            sender_email = os.getenv('EMAIL_ADDRESS', 'goleaaryan7@gmail.com')

            # User email content
            user_subject = "Thank you for reaching out!"
            user_body = f"""Hi {name},

Thank you for your message! I've received your inquiry and will get back to you as soon as possible.

Best regards,
Aaryan Gole"""

            # Admin email content
            admin_subject = f"New Contact Form Submission from {name}"
            admin_body = f"""New message received on your portfolio:

Name: {name}
Email: {to_email}
Message: {message}

---
Reply to: {to_email}"""

            if sendgrid_api_key:
                # Use SendGrid
                print(f"Sending email via SendGrid...")
                headers = {
                    "Authorization": f"Bearer {sendgrid_api_key}",
                    "Content-Type": "application/json"
                }
                
                user_email_data = {
                    "personalizations": [{"to": [{"email": to_email}], "subject": user_subject}],
                    "from": {"email": sender_email, "name": "Aaryan Gole"},
                    "content": [{"type": "text/plain", "value": user_body}]
                }
                
                admin_email_data = {
                    "personalizations": [{"to": [{"email": sender_email}], "subject": admin_subject}],
                    "from": {"email": sender_email, "name": "Portfolio Contact Form"},
                    "content": [{"type": "text/plain", "value": admin_body}]
                }

                # Send to user
                response_user = requests.post("https://api.sendgrid.com/v3/mail/send", headers=headers, json=user_email_data, timeout=10)
                if response_user.status_code == 202:
                    print("Email to user sent successfully via SendGrid.")
                else:
                    print(f"Failed to send email to user via SendGrid: {response_user.status_code} - {response_user.text}")

                # Send to admin
                response_admin = requests.post("https://api.sendgrid.com/v3/mail/send", headers=headers, json=admin_email_data, timeout=10)
                if response_admin.status_code == 202:
                    print("Admin notification sent successfully via SendGrid.")
                else:
                    print(f"Failed to send admin notification via SendGrid: {response_admin.status_code} - {response_admin.text}")
                
                return True
                # Fallback to SMTP
                sender_password = os.getenv('EMAIL_PASSWORD')
                if sender_password:
                    sender_password = sender_password.replace(" ", "") # Strip spaces from Google App Password
                
                smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
                smtp_port = int(os.getenv('SMTP_PORT', 587))
                
                if not sender_email or not sender_password:
                    print("CRITICAL: Email credentials not configured. Skipping email notification.")
                    return True
                    
                print(f"Sending email via SMTP ({smtp_server}:{smtp_port})...")
                server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
                server.starttls()
                server.login(sender_email, sender_password)

                msg_user = MIMEMultipart()
                msg_user['From'] = sender_email
                msg_user['To'] = to_email
                msg_user['Subject'] = user_subject
                msg_user.attach(MIMEText(user_body, 'plain'))
                server.send_message(msg_user)

                msg_admin = MIMEMultipart()
                msg_admin['From'] = sender_email
                msg_admin['To'] = sender_email
                msg_admin['Subject'] = admin_subject
                msg_admin.attach(MIMEText(admin_body, 'plain'))
                server.send_message(msg_admin)

                server.quit()
                print(f"SUCCESS: Emails dispatched successfully to {to_email} and admin via SMTP.")
                return True
        except Exception as e:
            print(f"CRITICAL ERROR in background email thread: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    # Run email sending in a background thread
    thread = threading.Thread(target=_send)
    thread.daemon = True
    thread.start()
    return True

@app.route('/api/contact', methods=['POST'])
def contact():
    """Handle contact form submission"""
    try:
        data = request.get_json()

        # Validate input
        if not data.get('name') or not data.get('email') or not data.get('message'):
            return jsonify({'error': 'All fields are required'}), 400

        name = data.get('name').strip()
        email = data.get('email').strip()
        message = data.get('message').strip()

        # Validate email format
        if '@' not in email or '.' not in email.split('@')[1]:
            return jsonify({'error': 'Invalid email address'}), 400

        # Create message document
        message_doc = {
            'name': name,
            'email': email,
            'message': message,
            'timestamp': datetime.now(UTC),
            'read': False
        }

        # Insert into MongoDB
        result = mongo.db.messages.insert_one(message_doc)
        message_doc['_id'] = str(result.inserted_id)

        # Send emails
        send_email(email, name, message)

        return jsonify({
            'success': True,
            'message': 'Your message has been received! I will get back to you soon.',
            'data': {
                'id': str(result.inserted_id),
                'name': name,
                'email': email,
                'message': message,
                'timestamp': datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')
            }
        }), 201

    except Exception as e:
        print(f"Error in contact endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/messages', methods=['GET'])
def get_messages():
    """Get all messages (admin only)"""
    try:
        # In production, add authentication here
        messages = list(mongo.db.messages.find().sort('timestamp', -1))
        
        # Convert ObjectId to string for JSON serialization
        for msg in messages:
            msg['_id'] = str(msg['_id'])
            msg['timestamp'] = msg['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({
            'success': True,
            'count': len(messages),
            'messages': messages
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """RAG-based chatbot endpoint"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Search knowledge base using RAG
        relevant_docs = rag.search(user_message, top_k=3)
        
        if not relevant_docs or all(doc['score'] < 0.05 for doc in relevant_docs):
            response = "I'm not sure about that. I can help you learn more about my background, skills, projects, experience, or how to contact me. What would you like to know?"
        else:
            # Build response from retrieved documents
            context = " ".join([doc['content'] for doc in relevant_docs])
            response = generate_response(user_message, context, relevant_docs)
        
        return jsonify({
            'success': True,
            'message': response,
            'sources': [doc['category'] for doc in relevant_docs]
        }), 200
    
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

def generate_response(user_query, context, relevant_docs):
    """Generate AI response using Groq API with RAG context"""
    
    # If Groq API key is available, use AI
    if GROQ_API_KEY:
        try:
            system_prompt = """You are a helpful portfolio assistant for Aaryan Gole. 
Answer questions about Aaryan based ONLY on the provided context. 
Be conversational, friendly, and concise (2-3 sentences max).
If the context doesn't contain relevant information, politely say you can help with questions about Aaryan's background, skills, projects, experience, or contact info.
Always refer to Aaryan in third person (he/him) unless the user is clearly talking to Aaryan directly."""

            user_prompt = f"""Context about Aaryan:
{context}

User question: {user_query}

Answer based on the context:"""

            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "max_tokens": 200,
                "temperature": 0.7
            }
            
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Groq API error: {e}")
    
    # Fallback to rule-based responses if no API key or API fails
    query_lower = user_query.lower()
    
    # Greeting responses
    if any(word in query_lower for word in ['hi', 'hello', 'hey', 'greetings']):
        return f"Hi there! I'm Aaryan Gole's portfolio assistant. {context.split('.')[0]}. How can I help you learn more about him?"
    
    # Why hire / strengths questions
    if any(phrase in query_lower for phrase in ['hire', 'why should', 'strengths', 'best', 'good at', 'capable']):
        return f"Aaryan brings a unique combination of AI/ML expertise and full-stack development skills. {context.split('.')[0]}. He has hands-on experience building production-grade AI systems, RAG chatbots, and scalable web applications."
    
    # Project-specific responses
    if 'project' in query_lower or 'built' in query_lower or 'created' in query_lower:
        return f"Here's what I found: {context.split('.')[0]}."
    
    # Default conversational response
    return f"Based on Aaryan's portfolio: {context.split('.')[0]}."

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        # Test MongoDB connection
        mongo.db.messages.find_one()
        return jsonify({
            'status': 'healthy',
            'database': 'MongoDB Atlas'
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)

