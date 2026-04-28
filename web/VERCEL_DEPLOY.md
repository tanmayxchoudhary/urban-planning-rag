# Vercel Deployment Guide

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI**: Install with `npm install -g vercel`
3. **API Gateway Deployed**: The FastAPI backend must be deployed before the web app

## Deployment Steps

### 1. Connect GitHub Repository to Vercel

```bash
cd web
vercel login
vercel link
```

Or via Vercel dashboard:
1. Go to [vercel.com/dashboard](https://vercel.com/dashboard)
2. Click "Add New..." → "Project"
3. Import `tanmayxchoudhary/urban-planning-rag`
4. Select the `web/` directory as the project root
5. Configure build command: `npm run build`
6. Deploy

### 2. Set Environment Variables

In Vercel dashboard → Project → Settings → Environment Variables:

| Variable | Value | Note |
|----------|-------|------|
| `NEXT_PUBLIC_API_URL` | `https://your-api-gateway-url.vercel.app` | Your deployed API gateway URL |
| `NEXT_PUBLIC_POSTHOG_KEY` | (optional) | PostHog project API key |
| `NEXT_PUBLIC_POSTHOG_HOST` | (optional) | PostHog host (self-hosted or cloud) |

### 3. Deploy

```bash
cd web
vercel --prod
```

Or push to `main` branch — Vercel auto-deploys on merge.

## Domain & TLS

- Vercel automatically provides TLS certificates for all deployments
- Custom domains can be configured in Project → Settings → Domains
- Cloudflare can be used in front for CDN and additional security (see PLAN.md §11.2)

## API Gateway Configuration

The web app connects to the API via `NEXT_PUBLIC_API_URL`. In production:

1. Deploy the FastAPI gateway to Lightning AI (see `infra/lightning/`)
2. Set `NEXT_PUBLIC_API_URL` to the gateway's public URL
3. The web app will proxy all `/v1/*` requests to that URL

## Analytics

Analytics events are tracked via PostHog:

- `query_submitted` — when user submits a question
- `answer_received` — when generation completes  
- `citation_clicked` — when user clicks a citation chip
- `feedback_submitted` — when user submits feedback

Configure PostHog via `NEXT_PUBLIC_POSTHOG_KEY` and `NEXT_PUBLIC_POSTHOG_HOST` environment variables.

## Troubleshooting

### Build fails
- Ensure `npm install` runs successfully
- Check TypeScript errors: `npm run typecheck` locally

### API requests fail
- Verify `NEXT_PUBLIC_API_URL` is set correctly
- Ensure the API gateway is deployed and accessible
- Check browser console for CORS errors

### Analytics not working
- PostHog is optional — events log to console in dev mode
- Verify `NEXT_PUBLIC_POSTHOG_KEY` is set in production
