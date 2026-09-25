// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import rehypeFigure from './src/plugins/rehype-figure.mjs';

// User site (https://indianspeedster.github.io) deploys from the root,
// so no `base` path is needed.
export default defineConfig({
  site: 'https://indianspeedster.github.io',
  markdown: {
    rehypePlugins: [rehypeFigure],
    // Light code theme to match the paper-coloured page.
    shikiConfig: { theme: 'github-light' },
  },
  vite: {
    plugins: [tailwindcss()],
  },
});
