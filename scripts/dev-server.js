// Runs the site locally the way Netlify serves it. Start with: npm run dev
// Netlify Forms does not exist locally, so a form POST goes straight to the follow-up
// function. That function saves to the database in MONGO_URI and sends real email.
const fs = require('fs');
const http = require('http');
const path = require('path');

const chat = require('../netlify/functions/chat');
const health = require('../netlify/functions/health');
const submissionCreated = require('../netlify/functions/submission-created');

const ROOT = path.resolve(__dirname, '..');
const PORT = Number(process.env.DEV_PORT) || 8888;
const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css',
  '.js': 'text/javascript',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.mp4': 'video/mp4',
  '.pdf': 'application/pdf'
};
const FUNCTIONS = { '/api/chat': chat, '/api/health': health };

async function readBody(request) {
  let body = '';
  for await (const chunk of request) body += chunk;
  return body;
}

function sendFunctionResult(response, result) {
  response.writeHead(result.statusCode, result.headers);
  response.end(result.body);
}

async function handleForm(body, response) {
  const fields = Object.fromEntries(new URLSearchParams(body));
  if (fields['form-name'] !== 'contact') {
    response.writeHead(404);
    response.end('Unknown form');
    return;
  }
  if (!fields['bot-field']) {
    await submissionCreated.handler({ body: JSON.stringify({ payload: { data: fields } }) });
  }
  response.writeHead(200);
  response.end('ok');
}

function serveFile(url, request, response) {
  const relative = url.pathname === '/' ? 'index.html' : decodeURIComponent(url.pathname).slice(1);
  const file = path.resolve(ROOT, relative);
  const isPublished = relative === 'index.html' || file.startsWith(path.join(ROOT, 'assets') + path.sep);
  if (!isPublished || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
    response.writeHead(404);
    response.end('Not found');
    return;
  }

  const size = fs.statSync(file).size;
  const type = MIME_TYPES[path.extname(file).toLowerCase()] || 'application/octet-stream';
  const range = /bytes=(\d+)-(\d*)/.exec(request.headers.range || '');
  if (range) {
    const start = Number(range[1]);
    const end = range[2] ? Number(range[2]) : size - 1;
    response.writeHead(206, {
      'Content-Type': type,
      'Accept-Ranges': 'bytes',
      'Content-Range': `bytes ${start}-${end}/${size}`,
      'Content-Length': end - start + 1
    });
    fs.createReadStream(file, { start, end }).pipe(response);
    return;
  }
  response.writeHead(200, { 'Content-Type': type, 'Accept-Ranges': 'bytes', 'Content-Length': size });
  fs.createReadStream(file).pipe(response);
}

const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, `http://${request.headers.host}`);
    const body = await readBody(request);
    const fn = FUNCTIONS[url.pathname];

    if (fn) {
      const event = { httpMethod: request.method, body, headers: request.headers };
      sendFunctionResult(response, await fn.handler(event));
    } else if (request.method === 'POST' && url.pathname === '/') {
      await handleForm(body, response);
    } else {
      serveFile(url, request, response);
    }
  } catch (error) {
    console.error(error);
    response.writeHead(500);
    response.end('Server error');
  }
});

server.on('error', (error) => {
  console.error(`Could not use port ${PORT} (${error.code}). Try another: set DEV_PORT=9000 && npm run dev`);
  process.exit(1);
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`Portfolio running at http://localhost:${PORT}`);
  console.log(`Integration status at http://localhost:${PORT}/api/health`);
});
