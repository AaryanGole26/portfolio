const nodemailer = require('nodemailer');

const SMTP_TIMEOUT_MS = 8000;

function createMailer() {
  const user = process.env.EMAIL_ADDRESS;
  const pass = (process.env.EMAIL_PASSWORD || '').replace(/\s/g, '');
  if (!user || !pass) return null;

  const port = Number(process.env.SMTP_PORT) || 587;
  return nodemailer.createTransport({
    host: process.env.SMTP_SERVER || 'smtp.gmail.com',
    port,
    secure: port === 465,
    auth: { user, pass },
    connectionTimeout: SMTP_TIMEOUT_MS,
    greetingTimeout: SMTP_TIMEOUT_MS,
    socketTimeout: SMTP_TIMEOUT_MS
  });
}

module.exports = { createMailer };
