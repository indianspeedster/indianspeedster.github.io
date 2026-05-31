# indianspeedster.github.io

My personal website — a static [Astro](https://astro.build) site with a dark
theme, an experience timeline, project cards, and a Markdown blog.

## Develop

```bash
npm install        # once
npm run dev        # local preview at http://localhost:4321
npm run build      # production build -> dist/
npm run preview    # serve the built site locally
```

## Editing content

Almost everything lives in two places:

| What | Where |
|------|-------|
| Name, tagline, about, socials, resume, email | `src/data/site.ts` |
| Experience timeline | `experience[]` in `src/data/site.ts` |
| Projects | `projects[]` in `src/data/site.ts` |
| Publications | `publications[]` in `src/data/site.ts` |
| Blog posts | Markdown files in `src/content/blog/` |
| Colors & styling | `src/styles/global.css` (CSS variables at the top) |
| Profile photo | `public/shekhar.jpg` |

### Adding a blog post

Create a new file like `src/content/blog/my-post.md`:

```markdown
---
title: "My post title"
description: "One-line summary shown in lists."
date: 2026-06-01
tags: ["kernels", "triton"]
draft: false      # set true to hide from the site
---

Write the post in Markdown here.
```

The URL is derived from the filename: `my-post.md` → `/blog/my-post/`.

## Deploying

This repo deploys automatically to GitHub Pages via
`.github/workflows/deploy.yml` on every push to `main`.

**One-time setup:** in the repo on GitHub, go to
**Settings → Pages → Build and deployment → Source** and select
**GitHub Actions**.
