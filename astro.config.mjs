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
  },
  vite: {
    plugins: [tailwindcss()],
  },
});
