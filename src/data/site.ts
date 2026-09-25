// ---------------------------------------------------------------------------
// Edit this file to update the site content. Everything below feeds the pages.
// ---------------------------------------------------------------------------

export const profile = {
  name: 'Shekhar Pandey',
  // short, punchy role line shown under the name
  tagline: 'ML Systems · GPU Kernels · LLM Training',
  // one or two sentences for the About section
  about:
    "I'm an ML systems engineer at AMD, working on large-scale MoE training, LLM inference, " +
    "and low-precision GPU kernels. I've delivered MXFP8 training and serving optimizations " +
    "across AMD CDNA3/CDNA4 systems, scaled DeepSeek-V3 to 1,024 GPUs, and contributed " +
    "performance improvements to PyTorch, TorchAO, and vLLM.",
  location: 'San Jose, CA',
  photo: '/shekhar.jpg',
  resume: '/Shekhar_Pandey_Resume.pdf',
  email: 'shekharptx@gmail.com',
};

export const socials = [
  { label: 'GitHub', href: 'https://github.com/indianspeedster' },
  { label: 'LinkedIn', href: 'https://www.linkedin.com/in/indianspeedster/' },
  { label: 'X', href: 'https://x.com/indianspeedster' },
  { label: 'LeetCode', href: 'https://leetcode.com/indianspeedster' },
];

export type Experience = {
  role: string;
  org: string;
  orgUrl?: string;
  period: string;
  blurb: string;
  // optional highlights, shown as a bulleted list under the blurb
  points?: string[];
};

export const work: Experience[] = [
  {
    role: 'Member of Technical Staff',
    org: 'AMD',
    orgUrl: 'https://www.amd.com',
    period: 'Jan 2025 – Present · San Jose, CA',
    blurb:
      'Promoted from Senior Software Development Engineer. Distributed training, LLM inference, and GPU kernel performance.',
    points: [
      'MXFP8 training for DeepSeek-V3: delivered the full MoE training path in TorchTitan/TorchAO with a FlyDSL backend on MI355X, now at 1.2× BF16 end-to-end. Designed a ragged MXFP8 wgrad grouped GEMM reaching 2.27 PFLOP/s (45% of peak), 13× the prior Triton kernel and up to 3× BF16 hipBLASLt; being upstreamed to PyTorch Inductor.',
      'Large-scale MoE pre-training: scaled DeepSeek-V3 671B to 1,024 MI325X GPUs at 96% scaling efficiency with Expert Parallelism and FP8 grouped GEMM kernels; worked with the AMD and Meta PyTorch teams on TorchTitan/Primus-Turbo and co-authored published results showing a 2.77× end-to-end training speedup.',
      'MXFP8/MXFP4 kernels: built CDNA4 forward and dgrad grouped GEMMs with XOR-swizzled LDS layouts, ping-pong software pipelining, XCD swizzle for L2 locality, fused quantization epilogues, and Split-K for K-heavy shapes, gated by bit-exact operator parity tests.',
      'RL post-training on ROCm: led end-to-end enablement of vime on MI355X across Megatron training and vLLM rollout, validated the GRPO pipeline, upstreamed ROCm fixes, shipped a prebuilt container, and published the work on the vLLM blog.',
      'Inference: delivered launch-day ROCm support for OpenAI gpt-oss-120B/20B on MI300X/MI355X, and built TTFT, throughput, concurrency, and Kineto/Perfetto profiling workflows plus MoE prefill/decode roofline studies that guided serving and kernel decisions.',
      "Recognition: Next 5% Award presented by AMD's CEO, plus 2 Executive Spotlight and 6 Spotlight awards in my first year.",
    ],
  },
  {
    role: 'Machine Learning Intern',
    org: 'Bytez',
    orgUrl: 'https://bytez.com',
    period: 'Feb 2024 – May 2024 · San Francisco, CA',
    blurb:
      'Fine-tuned Code Llama 13B into a text-to-Cypher model for natural-language graph querying, and built semantic search over a Neo4j graph of about 3 million research papers.',
  },
  {
    role: 'Graduate Teaching Assistant — ECE-GY 6143 Machine Learning',
    org: 'New York University',
    orgUrl: 'https://www.nyu.edu',
    period: 'Sep 2023 – May 2024 · New York, NY',
    blurb:
      'TA for ECE-GY 6143 Machine Learning — answered student questions, guided assignments, and ran regular office hours and review sessions.',
  },
  {
    role: 'Graduate Research Assistant',
    org: 'New York University',
    orgUrl: 'https://www.nyu.edu',
    period: 'Sep 2022 – Sep 2023 · New York, NY',
    blurb:
      'Built educational materials for ML system deployment on NSF-funded cloud testbeds, covering load balancing and scaling with Kubernetes. Assisted Prof. Fraida Fund on the "Fount" project.',
  },
  {
    role: 'Summer Research Intern — ML Reproducibility Fellow',
    org: 'University of California, Santa Cruz',
    orgUrl: 'https://ucsc-ospo.github.io/sor23/',
    period: 'May 2023 – Aug 2023 · Remote',
    blurb:
      'Implemented few-shot intent classification with BERT to demonstrate the impact of synonym-based text augmentation, and built educational materials on the role of complete methodology reporting in reproducibility — incorporated into the UCSC curriculum.',
  },
  {
    role: 'Software Development Engineer',
    org: 'Bosch Global Software Technologies',
    orgUrl: 'https://www.bosch-softwaretechnologies.com/en/',
    period: 'Jan 2021 – Jul 2022 · Coimbatore, India',
    blurb:
      'Built Python automation frameworks for end-to-end testing across 12+ peer groups, cutting functional test time by 80% through scripted failure-case simulation. Also built a pre-check build tool that cut missing-system-constant failure identification from 1.5 hours to 30 seconds.',
  },
  {
    role: 'Machine Learning Intern',
    org: 'Magic FinServ',
    orgUrl: 'https://www.magicfinserv.com/',
    period: 'Jan 2020 – Jun 2020 · Noida, India',
    blurb:
      'Built a deep-learning model with FastText embeddings to predict financial risk in textual statements, highlighting potential risk passages in documents.',
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
  {
    role: 'B.Tech in Information Technology',
    org: 'G.L. Bajaj Institute of Technology',
    period: '2016 – 2020',
    blurb:
      'Bachelor of Technology, Information Technology — where I first picked up Python programming and machine learning.',
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
  authors?: string;
};

export const publications: Publication[] = [
  {
    title: 'AMD ROCm Blogs: six posts on MI355X occupancy, gpt-oss day-0 enablement, DeepSeek-V3 profiling, Llama 4 inference, AITER, and Gemma 3 deployment',
    href: 'https://rocm.blogs.amd.com/authors/shekhar-pandey.html',
    venue: 'rocm.blogs.amd.com',
  },
  {
    title: 'vime + ROCm: End-to-End RL Post-Training on AMD Instinct GPUs',
    href: 'https://vllm.ai/blog/2026-07-10-vime-rocm',
    venue: 'vLLM Blog, July 2026',
  },
  {
    title:
      '[Re] Exploring the Role of Grammar and Word Choice in Bias Toward African American English (AAE) in Hate Speech Classification',
    href: 'https://rescience.github.io/',
    venue: 'ReScience C, Vol. 9, Issue 2, Article 35 · poster at NeurIPS 2023',
    authors: 'Priyanka Bose*, Chandra Shekhar Pandey*, Fraida Fund',
  },
];
