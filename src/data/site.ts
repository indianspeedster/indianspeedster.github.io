// ---------------------------------------------------------------------------
// Edit this file to update the site content. Everything below feeds the pages.
// ---------------------------------------------------------------------------

export const profile = {
  name: 'Shekhar Pandey',
  // short, punchy role line shown under the name
  tagline: 'ML Systems · GPU Kernels · LLM Training',
  // one or two sentences for the About section
  about:
    "I work at the intersection of machine learning and systems — writing GPU kernels, " +
    "tuning inference paths, and training large models. Recently focused on quantization, " +
    "ROCm/CUDA kernels, and reinforcement-learning post-training for LLMs and VLMs.",
  location: 'San Jose, CA',
  photo: '/shekhar.jpg',
  resume: 'https://www.overleaf.com/read/ygwkpfgcysym#cf0b59',
  email: '', // optional: set to show a mailto link, e.g. 'you@example.com'
};

export const socials = [
  { label: 'GitHub', href: 'https://github.com/indianspeedster' },
  { label: 'LinkedIn', href: 'https://www.linkedin.com/in/shekhar-p-aa90249a/' },
  { label: 'LeetCode', href: 'https://leetcode.com/indianspeedster' },
  // { label: 'Twitter', href: 'https://twitter.com/your_handle' },
];

export type Experience = {
  role: string;
  org: string;
  orgUrl?: string;
  period: string;
  blurb: string;
};

export const work: Experience[] = [
  {
    role: 'Sr. Software Development Engineer',
    org: 'AMD',
    orgUrl: 'https://www.amd.com',
    period: 'Jan 2025 – Present · San Jose, CA',
    blurb:
      'GPU performance & ML systems. Optimized large-scale MoE pre-training on MI325X clusters — FP8 grouped-GEMM kernels and Expert Parallelism hitting 96% scaling efficiency at 1K GPUs for DeepSeek-V3-671B. Co-authored TorchTitan/Primus-Turbo results showing a 2.77× end-to-end training speedup, shipped FP8/MXFP8 kernels to TorchAO (25–27% kernel speedup), and enabled Day-0 support for gpt-oss-120B/20B on ROCm via vLLM and PyTorch.',
  },
  {
    role: 'Machine Learning Intern',
    org: 'Bytez',
    orgUrl: 'https://bytez.com',
    period: 'Feb 2024 – May 2024 · San Francisco, CA',
    blurb:
      'Fine-tuned CodeLlama-13B into a text-to-Cypher model behind an interactive chat feature, and built semantic search with a Neo4j vector store over ~3M research papers.',
  },
  {
    role: 'Software Development Engineer',
    org: 'Bosch Global Software Technologies',
    orgUrl: 'https://www.bosch-softwaretechnologies.com/en/',
    period: 'Jan 2021 – Jul 2022 · India',
    blurb:
      'Built end-to-end automation testing scripts with 12+ peer groups across geographies, cutting basic functional-test time by 80% by simulating failure cases in Python.',
  },
];

export const education: Experience[] = [
  {
    role: 'M.S. in Computer Engineering',
    org: 'New York University',
    orgUrl: 'https://www.nyu.edu',
    period: 'Aug 2022 – May 2024 · GPA 3.9/4.0',
    blurb:
      'Coursework across Machine Learning, Deep Learning, Cloud Computing, Big Data, Internet Architecture & Protocols, and Computing Systems & Architecture.',
  },
];

export type Project = {
  name: string;
  href: string;
  blurb: string;
  tags: string[];
};

export const projects: Project[] = [
  {
    name: 'unet.cu',
    href: 'https://github.com/indianspeedster/unet.cu',
    blurb:
      'A UNet diffusion-model training framework in C++/CUDA with HIP support for unconditional diffusion training and inference on NVIDIA and AMD GPUs — reaching ~40% of PyTorch (torch.compile) end-to-end training speed.',
    tags: ['C++', 'CUDA', 'HIP', 'Diffusion'],
  },
  {
    name: 'SummarizeNow',
    href: 'https://github.com/indianspeedster/SummarizeNow',
    blurb:
      'Fine-tuned T5 for news-article summarization (ROUGE-L 0.42), packaged in a Docker container and served via a Flask web app.',
    tags: ['T5', 'NLP', 'Docker', 'Flask'],
  },
  {
    name: 'llm.c (open source)',
    href: 'https://github.com/indianspeedster/llm.c',
    blurb:
      "Contributed to Andrej Karpathy's llm.c, making the CUDA kernels portable to HIP to add support for AMD devices.",
    tags: ['CUDA', 'HIP', 'Open Source'],
  },
];

export type Publication = {
  title: string;
  href?: string;
  venue: string;
  authors: string;
};

export const publications: Publication[] = [
  {
    title:
      '[Re] Exploring the Role of Grammar and Word Choice in Bias Toward African American English (AAE) in Hate Speech Classification',
    href: 'https://rescience.github.io/',
    venue: 'ReScience C, Vol. 9, Issue 2, Article 35',
    authors: 'Priyanka Bose*, Chandra Shekhar Pandey*, Fraida Fund',
  },
];
