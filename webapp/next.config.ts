import type { NextConfig } from "next";

// When deploying to GitHub Pages as a project site, the app lives at
// /<repo-name>. The workflow injects NEXT_PUBLIC_BASE_PATH; locally it's empty
// so `next dev` and `next build` keep working unchanged.
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  output: "export",
  basePath: basePath || undefined,
  assetPrefix: basePath || undefined,
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
