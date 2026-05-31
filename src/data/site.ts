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
  location: 'New York, USA',
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

export const experience: Experience[] = [
  {
    role: 'Machine Learning Intern',
    org: 'Bytez',
    period: 'Feb 2024 – Present',
    blurb:
      'Upgraded an LLM chatbot with retrieval-augmented generation and semantic search over a vector DB — improved answer accuracy and cut token misuse by rejecting out-of-scope queries.',
  },
  {
    role: 'ML Reproducibility Fellow',
    org: 'UC Santa Cruz — Summer of Reproducibility',
    orgUrl: 'https://ucsc-ospo.github.io/sor23/',
    period: 'May 2023 – Aug 2023',
    blurb:
      'Studied reproducibility in ML education using few-shot intent classification with BERT, and built educational resources on how incomplete methodology reporting harms result validation.',
  },
  {
    role: 'Graduate Student / Research & Teaching Assistant',
    org: 'New York University',
    orgUrl: 'https://www.nyu.edu',
    period: 'Aug 2022 – Present',
    blurb:
      'M.S. coursework across ML, Deep Learning, Big Data, Cloud Computing, and Systems. Built teaching materials for deploying ML models on NSF cloud testbeds with Kubernetes; TA for Machine Learning.',
  },
  {
    role: 'Associate Software Engineer',
    org: 'Bosch India',
    orgUrl: 'https://www.bosch-softwaretechnologies.com/en/',
    period: 'Jan 2021 – Jul 2022',
    blurb:
      'Built end-to-end automation test scripts for the Smart Automation Test project, collaborating with 12+ peer groups across geographies.',
  },
  {
    role: 'Machine Learning Intern',
    org: 'Magic Finserv',
    orgUrl: 'https://www.magicfinserv.com/',
    period: 'Jan 2020 – Jun 2020',
    blurb:
      'Built a deep-learning model with FastText embeddings to detect and highlight financial risk statements in documents.',
  },
  {
    role: 'B.Tech, Information Technology',
    org: 'G.L. Bajaj Institute of Technology and Management',
    period: '2016 – 2020',
    blurb: 'Where it all started — Python programming and my first machine-learning classes.',
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
    name: 'pytorch/ao',
    href: 'https://github.com/indianspeedster/ao',
    blurb: 'Contributions to PyTorch-native quantization and sparsity for training and inference.',
    tags: ['PyTorch', 'Quantization'],
  },
  {
    name: 'deepseek-v4-mi35x-kernels',
    href: 'https://github.com/indianspeedster/deepseek-v4-mi35x-kernels',
    blurb: 'Per-kernel benchmarks and accuracy tests for AMD/ROCm kernels on the DeepSeek V4 inference path (MI35x / MI300x).',
    tags: ['ROCm', 'Kernels', 'Benchmarks'],
  },
  {
    name: 'grouped-gemms',
    href: 'https://github.com/indianspeedster/grouped-gemms',
    blurb: 'Standalone ROCm MXFP8 grouped-GEMM Triton kernel for AMD MI350+ (gfx950 / CDNA4).',
    tags: ['Triton', 'MXFP8', 'AMD'],
  },
  {
    name: 'hyperion',
    href: 'https://github.com/indianspeedster/hyperion',
    blurb: 'A layer-by-layer study of DeepSeek-V4 — every class in model.py unpacked into runnable, paper-grounded folders.',
    tags: ['LLM', 'Study'],
  },
  {
    name: 'QueryNavigator',
    href: 'https://github.com/indianspeedster/QueryNavigator',
    blurb: 'A RAG chatbot over a PDF-derived knowledge base using vector databases and the OpenAI API.',
    tags: ['RAG', 'LLM'],
  },
  {
    name: 'autoresearch',
    href: 'https://github.com/indianspeedster/autoresearch',
    blurb: 'AI agents that autonomously run research on single-GPU nanochat training.',
    tags: ['Agents', 'Training'],
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
