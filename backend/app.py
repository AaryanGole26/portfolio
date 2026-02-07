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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

app = Flask(__name__)
CORS(app)

# MongoDB Configuration
app.config['MONGO_URI'] = os.getenv('MONGO_URI')
if not app.config['MONGO_URI']:
    raise ValueError("MONGO_URI not found in environment variables. Please set it in your .env file.")
mongo = PyMongo(app)

# RAG Knowledge Base
class PortfolioRAG:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_base = self._create_knowledge_base()
        self.embeddings = None
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
                "category": "project_elevatr",
                "content": "Elevatr is an AI-driven resume analyzer and ATS-friendly resume builder. It evaluates resumes against ATS criteria using NLP and suggests improvements. Users can generate optimized, ATS-friendly resumes using curated templates."
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
                "category": "contact",
                "content": "You can reach me through LinkedIn at linkedin.com/in/aaryan-gole, GitHub at github.com/AaryanGole26, email at goleaaryan7@gmail.com, or phone at +91 93097 44137."
            }
        ]
    
    def _generate_embeddings(self):
        """Generate embeddings for all knowledge base chunks"""
        texts = [doc['content'] for doc in self.knowledge_base]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)
    
    def search(self, query, top_k=3):
        """Retrieve relevant portfolio information using semantic search"""
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [
            {
                "content": self.knowledge_base[idx]['content'],
                "category": self.knowledge_base[idx]['category'],
                "score": float(similarities[idx])
            }
            for idx in top_indices if similarities[idx] > 0.1
        ]
        return results

# Initialize RAG system
rag = PortfolioRAG()

def send_email(to_email, name, message):
    """Send email to user and yourself"""
    try:
        sender_email = os.getenv('EMAIL_ADDRESS')
        sender_password = os.getenv('EMAIL_PASSWORD')
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', 587))

        print(f"DEBUG: Email config - Server: {smtp_server}, Port: {smtp_port}, Email: {sender_email}")

        if not sender_email or not sender_password:
            print("Email credentials not configured. Skipping email notification.")
            return True

        # Email to user
        user_subject = "Thank you for reaching out!"
        user_body = f"""
        Hi {name},

        Thank you for your message! I've received your inquiry and will get back to you as soon as possible.

        Best regards,
        Aaryan Gole
        """

        # Email to admin
        admin_subject = f"New Contact Form Submission from {name}"
        admin_body = f"""
        New message received on your portfolio:

        Name: {name}
        Email: {to_email}
        Message: {message}

        ---
        Reply to: {to_email}
        """

        # Send emails
        print(f"DEBUG: Connecting to {smtp_server}:{smtp_port}...")
        server = smtplib.SMTP(smtp_server, smtp_port)
        print("DEBUG: Connected, starting TLS...")
        server.starttls()
        print("DEBUG: TLS started, logging in...")
        server.login(sender_email, sender_password)
        print("DEBUG: Login successful")

        # Send to user
        msg_user = MIMEMultipart()
        msg_user['From'] = sender_email
        msg_user['To'] = to_email
        msg_user['Subject'] = user_subject
        msg_user.attach(MIMEText(user_body, 'plain'))
        print(f"DEBUG: Sending email to {to_email}...")
        server.send_message(msg_user)
        print("DEBUG: Email to user sent")

        # Send to admin
        msg_admin = MIMEMultipart()
        msg_admin['From'] = sender_email
        msg_admin['To'] = sender_email
        msg_admin['Subject'] = admin_subject
        msg_admin.attach(MIMEText(admin_body, 'plain'))
        print(f"DEBUG: Sending email to admin {sender_email}...")
        server.send_message(msg_admin)
        print("DEBUG: Email to admin sent")

        server.quit()
        print("DEBUG: Email sending completed successfully")
        return True
    except Exception as e:
        print(f"ERROR sending email: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

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
        
        if not relevant_docs or all(doc['score'] < 0.2 for doc in relevant_docs):
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
    """Generate conversational response from retrieved context"""
    query_lower = user_query.lower()
    
    # Greeting responses
    if any(word in query_lower for word in ['hi', 'hello', 'hey', 'greetings']):
        return f"Hi there! I'm Aaryan Gole's portfolio assistant. {context.split('.')[0]}. How can I help you learn more about me?"
    
    # Project-specific responses
    if 'project' in query_lower or 'built' in query_lower or 'created' in query_lower:
        if relevant_docs:
            doc = relevant_docs[0]
            if 'project_' in doc['category']:
                return f"Great question! {context.split('.')[0]}. This project demonstrates my expertise in {doc['category'].replace('project_', '').upper()}."
        return f"I've worked on several interesting projects: {context}"
    
    # Skills responses
    if any(word in query_lower for word in ['skill', 'expertise', 'know', 'experience', 'technology', 'tech']):
        return f"My key competencies include: {context}"
    
    # Default conversational response
    return f"Based on my portfolio: {context}"

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

