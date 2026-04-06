exports.handler = async function (event) {
  const backendUrl = process.env.BACKEND_URL;

  if (!backendUrl) {
    return {
      statusCode: 500,
      headers: {
        "content-type": "application/json"
      },
      body: JSON.stringify({
        error: "BACKEND_URL is not configured"
      })
    };
  }

  try {
    const fnPrefix = "/.netlify/functions/api-proxy/";
    const relativePath = event.path.startsWith(fnPrefix)
      ? event.path.slice(fnPrefix.length)
      : "";

    const trimmedBase = backendUrl.replace(/\/+$/, "");
    const query = event.rawQuery ? `?${event.rawQuery}` : "";
    const targetUrl = `${trimmedBase}/api/${relativePath}${query}`;

    const incomingHeaders = event.headers || {};
    const forwardHeaders = {};

    Object.keys(incomingHeaders).forEach((key) => {
      const normalized = key.toLowerCase();
      if (["host", "content-length", "x-forwarded-for", "x-forwarded-proto"].includes(normalized)) {
        return;
      }
      forwardHeaders[key] = incomingHeaders[key];
    });

    const response = await fetch(targetUrl, {
      method: event.httpMethod,
      headers: forwardHeaders,
      body: ["GET", "HEAD"].includes(event.httpMethod)
        ? undefined
        : event.isBase64Encoded
          ? Buffer.from(event.body || "", "base64")
          : event.body
    });

    const responseBuffer = Buffer.from(await response.arrayBuffer());
    const contentType = response.headers.get("content-type") || "application/json";

    return {
      statusCode: response.status,
      headers: {
        "content-type": contentType,
        "cache-control": "no-store"
      },
      body: responseBuffer.toString("base64"),
      isBase64Encoded: true
    };
  } catch (error) {
    return {
      statusCode: 502,
      headers: {
        "content-type": "application/json"
      },
      body: JSON.stringify({
        error: "Proxy request failed",
        details: error.message
      })
    };
  }
};
