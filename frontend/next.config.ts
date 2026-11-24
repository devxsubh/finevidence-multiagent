import type { NextConfig } from "next";

// The FastAPI backend (`uv run finagents serve`). Requests to /api/* are proxied to it,
// so the browser talks to one origin.
const API_URL = process.env.FINAGENTS_API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_URL}/api/:path*` }];
  },
};

export default nextConfig;
