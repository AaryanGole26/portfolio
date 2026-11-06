const { MongoClient } = require('mongodb');

const TIMEOUT_MS = 5000;
let clientPromise = null;

// The connection is reused while a function instance stays warm.
function connect() {
  if (!clientPromise) {
    const client = new MongoClient(process.env.MONGO_URI, {
      serverSelectionTimeoutMS: TIMEOUT_MS,
      connectTimeoutMS: TIMEOUT_MS,
      maxPoolSize: 1
    });
    clientPromise = client.connect().catch((error) => {
      clientPromise = null;
      throw error;
    });
  }
  return clientPromise;
}

function isConfigured() {
  return Boolean(process.env.MONGO_URI);
}

async function database() {
  return (await connect()).db();
}

module.exports = { database, isConfigured };
