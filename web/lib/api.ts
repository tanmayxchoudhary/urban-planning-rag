/**
 * API client — constructs absolute URLs for production deployment.
 * NEXT_PUBLIC_API_URL must be set to the API gateway base URL.
 * In dev, defaults to relative paths (same origin).
 */

export function getApiBaseUrl(): string {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  // Dev fallback: relative URL (same origin, served by next.config rewrites)
  return "";
}

export function apiUrl(path: string): string {
  const base = getApiBaseUrl();
  if (base) {
    return `${base}${path}`;
  }
  // Dev: use relative path (Next.js dev server rewrites to localhost:3100)
  return path;
}
