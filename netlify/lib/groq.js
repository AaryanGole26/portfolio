const GROQ_API_BASE = 'https://api.groq.com/openai/v1';
const DEFAULT_MODEL = 'openai/gpt-oss-120b';

function groqModel() {
  return process.env.GROQ_MODEL || DEFAULT_MODEL;
}

module.exports = { GROQ_API_BASE, groqModel };
