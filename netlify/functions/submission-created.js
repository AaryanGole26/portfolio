// Netlify runs this automatically after a verified Netlify Forms submission.
// The message is already stored by Netlify, so failures here never lose it.
const { createMailer } = require('../lib/mail');
const { database, isConfigured } = require('../lib/mongo');

const MAX_NAME_LENGTH = 80;
const VALID_EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

// Keeps plain words only, so emails cannot be used to deliver links or addresses.
function safeName(value) {
  return String(value || '')
    .split(/\s+/)
    .filter((word) => !/[@:/.]/.test(word))
    .join(' ')
    .slice(0, MAX_NAME_LENGTH);
}

async function saveMessage({ name, email, message }) {
  if (!isConfigured()) return 'skipped';

  const db = await database();
  await db.collection('messages').insertOne({ name, email, message, timestamp: new Date(), read: false });
  return 'saved';
}

async function notifyOwner({ name, email, message }) {
  const mailer = createMailer();
  if (!mailer) return 'skipped';

  const owner = process.env.EMAIL_ADDRESS;
  await mailer.sendMail({
    from: `"Portfolio Contact Form" <${owner}>`,
    to: owner,
    replyTo: VALID_EMAIL.test(email) ? email : undefined,
    subject: `New portfolio message from ${safeName(name) || 'a visitor'}`,
    text: `Name: ${name}\nEmail: ${email}\n\nMessage:\n${message}`
  });
  return 'sent';
}

async function sendAutoReply({ name, email }) {
  const mailer = createMailer();
  if (!mailer) return 'skipped';
  if (!VALID_EMAIL.test(email)) return 'invalid-address';

  await mailer.sendMail({
    from: `"Aaryan Gole" <${process.env.EMAIL_ADDRESS}>`,
    to: email,
    subject: 'Thanks for reaching out!',
    text: `Hi ${safeName(name) || 'there'},\n\nThanks for your message! I've received it and will get back to you as soon as possible.\n\nBest regards,\nAaryan Gole`
  });
  return 'sent';
}

exports.handler = async (event) => {
  let submission;
  try {
    submission = JSON.parse(event.body).payload;
  } catch {
    return { statusCode: 400, body: 'Invalid payload' };
  }

  const data = submission?.data || {};
  const fields = {
    name: String(data.name || '').trim(),
    email: String(data.email || '').trim(),
    message: String(data.message || '').trim()
  };

  const [saved, notified, replied] = await Promise.allSettled([
    saveMessage(fields),
    notifyOwner(fields),
    sendAutoReply(fields)
  ]);
  const describe = (result) => (result.status === 'fulfilled' ? result.value : `failed: ${result.reason?.message}`);
  console.log(`Contact follow-up. Database: ${describe(saved)}. Owner email: ${describe(notified)}. Auto-reply: ${describe(replied)}.`);

  return { statusCode: 200, body: 'ok' };
};
