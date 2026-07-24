// @ts-check
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';

// User site (https://indianspeedster.github.io) deploys from the root,
// so no `base` path is needed.
export default defineConfig({
  site: 'https://indianspeedster.github.io',
  vite: {
    plugins: [tailwindcss()],
  },
});
