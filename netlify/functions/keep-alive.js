// Runs weekly (see netlify.toml). Atlas pauses free clusters after 60 days without a connection.
const { database, isConfigured } = require('../lib/mongo');

exports.handler = async () => {
  if (!isConfigured()) return { statusCode: 200, body: 'MONGO_URI is not set' };

  try {
    const db = await database();
    await db.command({ ping: 1 });
    console.log('Database keep-alive succeeded');
  } catch (error) {
    console.error(`Database keep-alive failed: ${error.name}`);
  }
  return { statusCode: 200, body: 'done' };
};
