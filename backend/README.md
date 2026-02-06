# Portfolio RAG Chatbot Backend

This backend implements a **Retrieval-Augmented Generation (RAG)** chatbot for your portfolio using Flask and semantic search.

## Features

✨ **RAG-Powered Intelligence**
- Uses sentence embeddings to understand user queries semantically
- Retrieves relevant information from your portfolio knowledge base
- Generates contextual responses based on your actual data
- **Zero hallucinations** - only uses information about you

🔍 **Semantic Search**
- Finds relevant sections from your portfolio automatically
- Understands natural language questions
- Suggests related content intelligently

🚀 **Production Ready**
- Clean REST API endpoints
- Error handling and logging
- CORS enabled for frontend integration

## Setup Instructions

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

This will install:
- Flask and CORS for the web server
- sentence-transformers for semantic embeddings (all-MiniLM-L6-v2 model)
- scikit-learn for similarity calculations
- MongoDB support for contact messages

### 2. Environment Variables

Create a `.env` file in the backend directory:

```env
MONGO_URI=mongodb+srv://username:password@cluster.mongodb.net/portfolio?retryWrites=true&w=majority
EMAIL_ADDRESS=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

### 3. Run the Backend

```bash
python app.py
```

The server will start at `http://localhost:5000`

## API Endpoints

### 1. Chat Endpoint (NEW - RAG Powered)

**POST** `/api/chat`

Request:
```json
{
  "message": "Tell me about your projects"
}
```

Response:
```json
{
  "success": true,
  "message": "Based on my portfolio: [AI-generated response from RAG]",
  "sources": ["project_lawpal", "project_elevatr"]
}
```

**What happens:**
1. Your question is converted to semantic embeddings
2. The system searches the knowledge base for relevant sections
3. Most relevant sections are retrieved (top K=3)
4. A contextual response is generated from the retrieved information

### 2. Contact Endpoint (Existing)

**POST** `/api/contact`

### 3. Messages Endpoint (Existing)

**GET** `/api/messages`

### 4. Health Check

**GET** `/api/health`

## Knowledge Base Structure

The RAG system maintains a knowledge base with your portfolio information organized by categories:

- **about** - Your background and introduction
- **skills_ml** - ML/AI expertise
- **skills_backend** - Backend development skills
- **skills_frontend** - Frontend development skills
- **tools** - Tools and technologies
- **experience_citius** - Work experience
- **education** - Educational background
- **certifications** - Certifications and courses
- **project_*** - Individual project descriptions
- **contact** - Contact information

## How RAG Works

```
User Query
    ↓
Convert to embeddings (sentence-transformers)
    ↓
Search knowledge base (cosine similarity)
    ↓
Retrieve top 3 relevant sections
    ↓
Generate response using retrieved context
    ↓
Response sent to frontend
```

## Technology Stack

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
  - Lightweight (~33MB)
  - Fast inference (CPU compatible)
  - Great for semantic search tasks
  
- **Similarity**: scikit-learn (cosine similarity)
  - Fast computation
  - No additional data structures needed
  
- **Web Framework**: Flask
  - Lightweight and flexible
  - Easy to extend

## Customization

### Adding New Information

Edit the `_create_knowledge_base()` method in the `PortfolioRAG` class to add:
- New projects
- Skills
- Experience
- or any other information

Example:
```python
{
    "category": "new_info",
    "content": "Your new content here"
}
```

### Adjusting Search Results

Modify the `search()` method parameters:
- `top_k`: Number of results to return (default: 3)
- `similarity_threshold`: Minimum score to accept (default: 0.1)

### Response Generation

Edit the `generate_response()` function to customize how responses are created based on query intent.

## Troubleshooting

**Q: "Error loading sentence-transformers model"**
A: The model (~433MB) downloads on first run. Ensure you have internet connectivity and ~1GB free disk space.

**Q: "Connection to localhost:5000 refused"**
A: Make sure the Flask server is running and CORS is enabled for your frontend URL.

**Q: "Low quality responses"**
A: This might indicate:
- No matching content in knowledge base
- Query similarity score too low (adjust threshold)
- Need to add more detailed information to knowledge base

## Performance Notes

- First request takes ~1-2 seconds (model loading)
- Subsequent requests take ~100-200ms
- Search is performed in real-time (no pre-indexing needed)
- Works offline once the model is cached

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Set up MongoDB connection in `.env`
3. Start the server: `python app.py`
4. Test the `/api/chat` endpoint
5. The frontend chatbot will automatically use the RAG backend
