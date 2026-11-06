# Portfolio - Aaryan Gole

Personal portfolio showcasing my backend and AI projects, experience, and publications.

**Live:** [aaryan-gole.netlify.app](https://aaryan-gole.netlify.app)

## How it works

Everything runs on Netlify, so there is no server to wake up.

| Feature | Handled by | If it fails |
| --- | --- | --- |
| Site | Static files published from `dist/` | - |
| Contact form | [Netlify Forms](https://docs.netlify.com/forms/setup/) | Visitor is offered a pre-filled email link and keeps their text |
| Follow-up (MongoDB copy, emails) | `netlify/functions/submission-created.js` | The message is still saved in Netlify Forms |
| Chatbot | `netlify/functions/chat.js` using the Groq API | Answers from the facts in `netlify/lib/knowledge.js` |
| Database keep-alive | `netlify/functions/keep-alive.js`, weekly | Atlas may pause the cluster after 60 idle days |
| Status check | `/api/health` | - |

## Run locally

Opening `index.html` directly will not work for chat or the contact form, because they need a server.

```
npm install
npm run dev
```

Then open http://localhost:8888. This needs Node.js 22.9 or newer and reads settings from `backend/.env`.

Local form submissions go straight to the follow-up function, so they save to the real database and send real emails.

## Deploy setup

In the Netlify dashboard:

1. **Forms:** enable form detection, then redeploy.
2. **Environment variables:**

| Variable | Purpose |
| --- | --- |
| `GROQ_API_KEY` | AI chatbot answers |
| `GROQ_MODEL` | Overrides the default model, `openai/gpt-oss-120b` |
| `MONGO_URI` | Saves a copy of each message to the `messages` collection |
| `EMAIL_ADDRESS`, `EMAIL_PASSWORD` | Message alerts and visitor auto-replies through Gmail, using an App Password |

In MongoDB Atlas, Network Access must allow `0.0.0.0/0`, because Netlify Functions have no fixed IP address.

After deploying, open `/api/health` to see which of these work.

## Updating content

- Add a project in `index.html`: one card in `.projects-list` and one entry in `.projects-nav`.
- Update `netlify/lib/knowledge.js` so the chatbot knows about it.
- Encode demo videos as H.264 with `-movflags +faststart` so they start quickly.

## Tech

HTML, CSS, and vanilla JavaScript, with Netlify Functions on Node.js and MongoDB Atlas. `backend/` holds the previous Flask service, which the site no longer uses.

## License

MIT License - Feel free to fork and use!
