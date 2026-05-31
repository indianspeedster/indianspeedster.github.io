---
title: "Hello, world"
description: "A fresh start — what this site is and what I plan to write about."
date: 2026-05-30
tags: ["meta"]
draft: false
---

This is the first post on the rebuilt site. I tore down the old page and started
from scratch with a dark, minimal layout focused on the work I actually do now:
**ML systems, GPU kernels, and large-model training.**

## What I'll write about

- Kernel work — Triton / ROCm / CUDA, quantization (MXFP8 and friends), grouped GEMMs
- Notes from reading model code, layer by layer
- Things I learn while training and tuning inference paths

## How this site is built

It's a static [Astro](https://astro.build) site — plain Markdown for posts, a
small data file for the timeline and projects, and a GitHub Action that builds and
deploys to GitHub Pages on every push.

```bash
npm run dev      # local preview at http://localhost:4321
npm run build    # output to dist/
```

To add a post, drop a new `.md` file in `src/content/blog/`. That's it.

> More soon.
