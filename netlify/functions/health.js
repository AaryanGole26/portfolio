// Open /api/health to see which integrations work. Reveals status only, never values.
const { GROQ_API_BASE, groqModel } = require('../lib/groq');
const { createMailer } = require('../lib/mail');
const { database, isConfigured } = require('../lib/mongo');

const CACHE_MS = 5 * 60 * 1000;
const CHECK_TIMEOUT_MS = 6000;
let cached = null;

async function checkChat() {
  const key = process.env.GROQ_API_KEY;
  if (!key) return 'not configured: chat answers from built-in portfolio facts';
  try {
    const response = await fetch(`${GROQ_API_BASE}/models`, {
      headers: { authorization: `Bearer ${key}` },
      signal: AbortSignal.timeout(CHECK_TIMEOUT_MS)
    });
    if (response.status === 401) return 'invalid GROQ_API_KEY';
    if (!response.ok) return `Groq returned ${response.status}`;
    const { data } = await response.json();
    return data.some((model) => model.id === groqModel()) ? 'ok' : `model ${groqModel()} unavailable: set GROQ_MODEL`;
  } catch (error) {
    return `unreachable (${error.name})`;
  }
}

async function checkEmail() {
  const mailer = createMailer();
  if (!mailer) return 'not configured: no emails are sent';
  try {
    await mailer.verify();
    return 'ok';
  } catch (error) {
    return error.code === 'EAUTH' ? 'Gmail rejected EMAIL_ADDRESS or EMAIL_PASSWORD: create a new App Password' : `unreachable (${error.code || error.name})`;
  } finally {
    mailer.close();
  }
}

async function checkDatabase() {
  if (!isConfigured()) return 'not configured: messages stay in Netlify Forms';
  try {
    const db = await database();
    await db.command({ ping: 1 });
    return 'ok';
  } catch (error) {
    return `unreachable (${error.name}): is the Atlas cluster paused, or is this IP not allowed?`;
  }
}

exports.handler = async () => {
  if (!cached || Date.now() - cached.at > CACHE_MS) {
    const [chat, email, databaseStatus] = await Promise.all([checkChat(), checkEmail(), checkDatabase()]);
    cached = { at: Date.now(), body: { chat, email, database: databaseStatus } };
  }
  return {
    statusCode: 200,
    headers: { 'content-type': 'application/json', 'cache-control': 'no-store' },
    body: JSON.stringify(cached.body, null, 2)
  };
};
