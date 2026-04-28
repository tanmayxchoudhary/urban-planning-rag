/** @type {import('next').NextConfig} */
const nextConfig = {
  // API URL for production - must be set via VERCEL_URL env var or overrides below
  // In production on Vercel, VERCEL_URL is auto-set to the deployment URL
  env: {
    NEXT_PUBLIC_API_URL: process.env.VERCEL_URL
      ? `https://${process.env.VERCEL_URL}`
      : process.env.NEXT_PUBLIC_API_URL || "https://api.urban-rag.example.com",
  },
  async rewrites() {
    // Local dev: proxy /v1/* to API gateway on port 3100
    // Production: use NEXT_PUBLIC_API_URL directly (no rewrite needed)
    return [];
  },
};

export default nextConfig;
