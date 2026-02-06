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
app.config['MONGO_URI'] = os.getenv('MONGO_URI', 'mongodb+srv://username:password@cluster.mongodb.net/portfolio?retryWrites=true&w=majority')
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
        ]
    
    def _generate_embeddings(self):
        """Generate embeddings for all knowledge base items"""
        contents = [item['content'] for item in self.knowledge_base]
        self.embeddings = self.model.encode(contents)
    
    def retrieve(self, query, top_k=3):
        """Retrieve relevant documents based on query"""
        query_embedding = self.model.encode(query)
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [self.knowledge_base[i] for i in top_indices]

# Initialize RAG
rag = PortfolioRAG()

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat endpoint with RAG capabilities"""
    try:
        data = request.json
        user_message = data.get('message')
        
        if not user_message:
            return jsonify({'error': 'Message required'}), 400
        
        # Retrieve relevant context
        retrieved_docs = rag.retrieve(user_message)
        context = '\n'.join([doc['content'] for doc in retrieved_docs])
        
        # Save to database
        message_doc = {
            'user_message': user_message,
            'context': context,
            'timestamp': datetime.now(UTC)
        }
        mongo.db.messages.insert_one(message_doc)
        
        return jsonify({
            'response': f"Based on my portfolio: {context}",
            'context': retrieved_docs
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/contact', methods=['POST'])
def contact():
    """Contact form endpoint"""
    try:
        data = request.json
        name = data.get('name')
        email = data.get('email')
        message = data.get('message')
        
        if not all([name, email, message]):
            return jsonify({'error': 'Name, email, and message required'}), 400
        
        # Save to MongoDB
        contact_doc = {
            'name': name,
            'email': email,
            'message': message,
            'timestamp': datetime.now(UTC)
        }
        mongo.db.contacts.insert_one(contact_doc)
        
        # Send email (optional)
        # You can add email sending logic here
        
        return jsonify({'message': 'Contact form submitted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects', methods=['GET'])
def get_projects():
    """Get all projects"""
    try:
        projects = list(mongo.db.projects.find({}))
        for project in projects:
            project['_id'] = str(project['_id'])
        return jsonify(projects), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/projects', methods=['POST'])
def add_project():
    """Add a new project"""
    try:
        data = request.json
        result = mongo.db.projects.insert_one(data)
        return jsonify({'_id': str(result.inserted_id)}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True)
