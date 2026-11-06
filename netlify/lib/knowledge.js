// Single source of truth for the portfolio chatbot. Keep in sync with index.html.
const KNOWLEDGE = [
  { core: true, topic: 'about background who summary', text: 'Aaryan Gole is a backend-focused AI engineer from Mumbai, India, who builds scalable backend systems with Flask, Django and FastAPI, and AI features such as RAG pipelines and NLP.' },
  { topic: 'experience work job current nexus ai engineer', text: 'Aaryan has worked as an AI Engineer (Contractual) at Nexus AI since August 2026, developing and validating benchmark tasks, oracle solutions and automated test suites that evaluate LLMs and AI coding agents on real software engineering work, using Python, Git, Docker and CLI tools.' },
  { topic: 'experience internship intern citius cloud generative', text: 'Aaryan was a Generative AI Intern at Citius Cloud Services LLP from June to July 2024. He built a prototype academic chatbot using TensorFlow-based RNN, Transformer and RAG architectures, with a web-scraping and knowledge-base pipeline indexed in Pinecone.' },
  { topic: 'education college degree cgpa gpa university vcet school marks', text: "Aaryan completed a Bachelor of Engineering in Artificial Intelligence and Data Science at Vidyavardhini's College of Engineering and Technology (VCET), Vasai, in May 2026 with a CGPA of 8.27/10. He scored 94.20% in CBSE Class X, with a school rank of 2 and a CBSE Certificate of Merit." },
  { topic: 'publications research paper arxiv ieee citations', text: "Aaryan's publications are \"LawPal: A RAG-Based System for Enhanced Legal Accessibility in India\" (arXiv:2502.16573, 2025, 14 citations) and \"Towards Explainable Agentic Intelligence: A LangGraph-MCP Hybrid Framework for Scalable, Fair, and Self-Improving Multi-Agent Systems\" (IEEE IC3ET, 2026)." },
  { topic: 'skills languages programming python javascript sql', text: 'Aaryan programs in Python, JavaScript, SQL, C and HTML/CSS.' },
  { topic: 'skills frameworks backend flask django fastapi react api', text: 'His frameworks and tools include Flask, Django, FastAPI, ReactJS, Postman, Figma and Power BI. He builds REST APIs with authentication and uses Docker.' },
  { topic: 'skills ai ml machine learning tensorflow nlp rag embeddings', text: 'His AI and ML skills include TensorFlow, Scikit-Learn, NumPy, Pandas, RAG pipelines, NLP, prompt engineering, LLM applications, embeddings, vector search and computer vision.' },
  { topic: 'skills tools platforms devops supabase pinecone gemini groq git', text: 'He works with Pinecone, Supabase, the Gemini and Groq APIs, Docker, Git, GitHub, CI/CD, Linux and basic AWS.' },
  { topic: 'certifications certificates courses', text: 'His certifications include the Machine Learning Specialization from DeepLearning.AI, Fundamentals of Machine Learning from Microsoft Learn, the Data Analytics Program from Godrej Infotech and Centum Foundation, MLOps with Data Version Control from Infosys Springboard, and GenAI Academy from Google Cloud.' },
  { topic: 'activities leadership mlsc microsoft club outreach hackathon', text: "Aaryan progressed from Outreach Associate to Outreach Lead at the Microsoft Learn Students Club, VCET (July 2024 to June 2026), leading outreach for events including Coherence '26, a national hackathon with 774 registrations." },
  { core: true, topic: 'projects portfolio built list', text: 'His projects include LawPal, OptiResume-AI, RasoIQ, Rento, FinSlash, CineSLEUTH, DriveSense, ArcadePy, Collatz Conjecture, CGPA Calculator and LUME.' },
  { topic: 'project lawpal legal law chatbot voice', text: 'LawPal is a Flask-based RAG legal assistant with semantic search over 7,000+ document chunks in Pinecone across 3 legal domains. It uses MiniLM transformers, supports text and voice queries in all 22 scheduled Indian languages, and uses Supabase authentication. Code: github.com/AaryanGole26/LawPal.' },
  { topic: 'project optiresume opti resume cv ats', text: 'OptiResume-AI is a resume optimization system that parses resumes, scores them against job descriptions, rewrites bullets with AI, and exports ATS-friendly CVs from 6 templates in PDF and LaTeX. Code: github.com/AaryanGole26/OptiResume-AI.' },
  { topic: 'project rasoiq raso recommendation', text: 'RasoIQ is a smart recipe recommender built with Flask and vanilla JavaScript for a Recommendation Systems course. It ranks recipes by weighted ingredient overlap with penalties for missing items, assumes kitchen staples, and filters by diet and cuisine. Live: rasoiq.netlify.app. Code: github.com/AaryanGole26/RasoIQ.' },
  { topic: 'project rento rental tools django booking', text: 'Rento is a Django web app for renting engineering tools, with authentication, catalog browsing, a cart, and order and return tracking. Live: rento-srcr.onrender.com. Code: github.com/AaryanGole26/Rento.' },
  { topic: 'project finslash loan finance dashboard streamlit', text: 'FinSlash is a loan approval dashboard built with Streamlit that uses EDA and regression models for explainable risk scoring. Live: finslash-3jt.streamlit.app. Code: github.com/AaryanGole26/FinSlash.' },
  { topic: 'project cinesleuth movie film recommendation', text: 'CineSLEUTH is a movie recommendation system that combines TF-IDF, cosine similarity, fuzzy matching and Apriori-based collaborative filtering. Live: huggingface.co/spaces/ARYN26/CineSLEUTH. Code: github.com/AaryanGole26/CineSLEUTH.' },
  { topic: 'project drivesense driver fatigue drowsiness computer vision opencv', text: 'DriveSense is a driver wellness system that uses OpenCV, deep learning and TensorFlow to track eye and face landmarks, detect fatigue or distress, and play gentle audio alerts. Code: github.com/AaryanGole26/DriveSense.' },
  { topic: 'project arcadepy games pygame arcade', text: 'ArcadePy is a Python game suite that launches Slither, Pong, Hovering UFO and Hangman from one menu, built with PyGame and Tkinter. Code: github.com/AaryanGole26/ArcadePy.' },
  { topic: 'project collatz conjecture math visualizer', text: 'Collatz Conjecture is a Python visualizer that computes 3x + 1 sequences for any natural number and displays them as interactive graphs. Code: github.com/AaryanGole26/Collatz_Conjecture.' },
  { topic: 'project cgpa calculator mumbai university', text: 'CGPA Calculator computes Mumbai University CGPI and percentage, shows semester trends, and predicts the SGPA needed for a target. Live: aaryangole26.github.io/CGPA-Calculator. Code: github.com/AaryanGole26/CGPA-Calculator.' },
  { topic: 'project lume meme ai image blip groq', text: 'LUME is an AI meme generator that captions uploaded images using BLIP and Groq and exports them ready to share. Live: lume-ai-l9ji.onrender.com. Code: github.com/AaryanGole26/LUME.' },
  { core: true, topic: 'freelance services website websites web app apps build client project hire price pricing cost charge charges fee fees rate rates quote available availability', text: 'Aaryan is open to full-time AI and backend roles and to freelance web work, including websites, web apps, backend APIs, and AI features such as chatbots and RAG search. For pricing and availability, contact him directly at goleaaryan7@gmail.com.' },
  { core: true, topic: 'good candidate hire hiring hireable why strengths strong fit recruit recommend', text: 'Aaryan combines backend engineering with applied AI. He works as an AI Engineer at Nexus AI evaluating LLMs and coding agents, has published LawPal on arXiv (14 citations) and a paper at IEEE IC3ET, and has built and deployed live projects, including RasoIQ, Rento and CineSLEUTH.' },
  { core: true, topic: 'contact email phone linkedin github reach', text: 'You can reach Aaryan at goleaaryan7@gmail.com or +91 93097 44137, on LinkedIn at linkedin.com/in/aaryan-gole, or on GitHub at github.com/AaryanGole26.' }
];

const STOPWORDS = new Set('a an and are about can could did do does for from has have he her his how i in is it me my of on or she tell that the their them they this to was what when where which who why will with would you your aaryan'.split(' '));

function tokenize(text) {
  return String(text).toLowerCase().match(/[a-z0-9+#]+/g) || [];
}

const MAX_EXTRA_FACTS = 4;

function rank(query) {
  const terms = tokenize(query).filter((term) => !STOPWORDS.has(term));
  return KNOWLEDGE
    .map((entry) => {
      const topic = new Set(tokenize(entry.topic));
      const body = new Set(tokenize(entry.text));
      return {
        entry,
        topicScore: terms.filter((term) => topic.has(term)).length,
        bodyScore: terms.filter((term) => body.has(term)).length
      };
    })
    .sort((a, b) => b.topicScore - a.topicScore || b.bodyScore - a.bodyScore);
}

// The free Groq tier allows few tokens per minute, so each question sends only what it needs.
function factsFor(query) {
  const extras = rank(query)
    .filter(({ entry, topicScore, bodyScore }) => !entry.core && topicScore + bodyScore > 0)
    .slice(0, MAX_EXTRA_FACTS)
    .map(({ entry }) => entry);
  return [...KNOWLEDGE.filter((entry) => entry.core), ...extras].map((entry) => entry.text);
}

// Answer used when the AI provider is unavailable. It only ever returns one real fact.
function fallbackAnswer(query) {
  const [best] = rank(query);
  return best && best.topicScore > 0 ? best.entry.text : null;
}

module.exports = { factsFor, fallbackAnswer };
