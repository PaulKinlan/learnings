/**
 * JevImageBench v0.1.5 Benchmark Dataset
 * Source: https://benchmarkheaven.com/image-jev-bench
 * Transcribed from source HTML (fetch date: 2026-10-03)
 *
 * Tracks 50 vision and multimodal decision models evaluated on bounded System One
 * image decision tasks (Choice, Score, Noul) across UI automation, document verification,
 * security credential detection, and robot hazard navigation.
 *
 * Range: Latency 0.26 s to 4.83 s; Cost $0.0074 to $2.06 per 1,000 decisions.
 * Breakdown: 38 Jev-class bounded decision models + 12 general multimodal systems outside 2x budget limits.
 */

export const BENCHMARK_META = {
  name: 'JevImageBench',
  version: '0.1.5',
  url: 'https://benchmarkheaven.com/image-jev-bench',
  date: 'October 2026',
  totalSystems: 50,
  jevClassCount: 38,
  outsideLimitsCount: 12,
  metrics: {
    capabilityScore: 'Composite decision capability index (0–100) averaging intelligence and calibration',
    intelligence: 'Perceptual reasoning and subtle visual discrimination score (0–100)',
    calibration: 'Empirical calibration accuracy (ECE-weighted score, 0–100%)',
    speed: 'Median decision latency per image query in seconds',
    cost: 'Estimated operational cost in USD per 1,000 decisions'
  }
};

export const JEV_IMAGE_BENCH_DATA = [
  {
    "rank": 1,
    "name": "Wity-1",
    "capability": 85.7,
    "capabilityScore": 85.7,
    "intelligence": 83.7,
    "calibration": 87.7,
    "latency": 0.71,
    "speed": 0.71,
    "cost": 0.0074,
    "baseModel": "undisclosed",
    "access": "Closed API",
    "category": "Jev-Class Vision",
    "link": ""
  },
  {
    "rank": 2,
    "name": "JPT-9B",
    "capability": 83.6,
    "capabilityScore": 83.6,
    "intelligence": 75.7,
    "calibration": 91.5,
    "latency": 0.46,
    "speed": 0.46,
    "cost": 0.053,
    "baseModel": "Qwen3.5-9B",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://github.com/fstandhartinger/jevbench/issues/78"
  },
  {
    "rank": 3,
    "name": "Imajev-4B",
    "capability": 82.1,
    "capabilityScore": 82.1,
    "intelligence": 73.8,
    "calibration": 90.5,
    "latency": 0.35,
    "speed": 0.35,
    "cost": 0.02,
    "baseModel": "Qwen3.5-4B",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://github.com/fstandhartinger/jevbench/issues/80"
  },
  {
    "rank": 4,
    "name": "shisa-de-1",
    "capability": 82.1,
    "capabilityScore": 82.1,
    "intelligence": 72.4,
    "calibration": 91.8,
    "latency": 0.48,
    "speed": 0.48,
    "cost": 0.056,
    "baseModel": "Gemma 4 26B-A4B",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://github.com/fstandhartinger/jevbench/issues/55"
  },
  {
    "rank": 5,
    "name": "NeoHorse Jev 4B",
    "capability": 82.0,
    "capabilityScore": 82.0,
    "intelligence": 72.9,
    "calibration": 91.2,
    "latency": 0.43,
    "speed": 0.43,
    "cost": 0.041,
    "baseModel": "NeoHorse-1-4B",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://github.com/fstandhartinger/jevbench/issues/131"
  },
  {
    "rank": 6,
    "name": "imajev 9B",
    "capability": 82.0,
    "capabilityScore": 82.0,
    "intelligence": 74.7,
    "calibration": 89.3,
    "latency": 0.53,
    "speed": 0.53,
    "cost": 0.063,
    "baseModel": "Qwen3.5-9B",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://github.com/fstandhartinger/jevbench/issues/80"
  },
  {
    "rank": 7,
    "name": "Surogate Rune 26B-A4B v3",
    "capability": 81.3,
    "capabilityScore": 81.3,
    "intelligence": 74.9,
    "calibration": 87.8,
    "latency": 0.48,
    "speed": 0.48,
    "cost": 0.055,
    "baseModel": "gemma-4-26B-A4B-it",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://huggingface.co/api/models/surogate/rune-26b-a4b-GGUF"
  },
  {
    "rank": 8,
    "name": "JevAny-27B RLCR",
    "capability": 80.1,
    "capabilityScore": 80.1,
    "intelligence": 70.2,
    "calibration": 90.0,
    "latency": 0.47,
    "speed": 0.47,
    "cost": 0.055,
    "baseModel": "undisclosed",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": ""
  },
  {
    "rank": 9,
    "name": "Jevify Gemma 4 26B-A4B",
    "capability": 79.8,
    "capabilityScore": 79.8,
    "intelligence": 71.6,
    "calibration": 88.0,
    "latency": 0.46,
    "speed": 0.46,
    "cost": 0.053,
    "baseModel": "google/gemma-4-26B-A4B-it",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://huggingface.co/kushalpatil/jevify-gemma4-26b-a4b/blob/d4c0d1d455892957d274f68ec64e9fc0881011c8/README.md"
  },
  {
    "rank": 10,
    "name": "JevAny-27B SFT",
    "capability": 79.8,
    "capabilityScore": 79.8,
    "intelligence": 69.6,
    "calibration": 90.0,
    "latency": 0.47,
    "speed": 0.47,
    "cost": 0.054,
    "baseModel": "undisclosed",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": ""
  },
  {
    "rank": 11,
    "name": "JPT-4B",
    "capability": 77.1,
    "capabilityScore": 77.1,
    "intelligence": 66.9,
    "calibration": 87.3,
    "latency": 0.4,
    "speed": 0.4,
    "cost": 0.043,
    "baseModel": "Qwen3.5-4B",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://github.com/fstandhartinger/jevbench/issues/78"
  },
  {
    "rank": 12,
    "name": "Jev-Omni",
    "capability": 76.9,
    "capabilityScore": 76.9,
    "intelligence": 63.9,
    "calibration": 89.9,
    "latency": 0.3,
    "speed": 0.3,
    "cost": 0.022,
    "baseModel": "Gemma 4 12B IT",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://huggingface.co/akhilaaa3/Jev-Omni"
  },
  {
    "rank": 13,
    "name": "AutoJev-27B",
    "capability": 75.9,
    "capabilityScore": 75.9,
    "intelligence": 69.3,
    "calibration": 82.4,
    "latency": 0.43,
    "speed": 0.43,
    "cost": 0.049,
    "baseModel": "Qwen/Qwen3.8-27B",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://huggingface.co/denis-pplx/autojev-27b/blob/6f5b557e037f5edb25c7dc92dbc6553e5a19c015/README.md"
  },
  {
    "rank": 14,
    "name": "Visual-Jev 4B Answer-SFT",
    "capability": 74.8,
    "capabilityScore": 74.8,
    "intelligence": 65.4,
    "calibration": 84.1,
    "latency": 0.35,
    "speed": 0.35,
    "cost": 0.036,
    "baseModel": "Qwen3-VL-4B-Instruct",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://huggingface.co/guanxuyu/visual-jev-4b-answer-sft/blob/7a3f1bb0d8f7bfbbd8514d968ab7de3518849c8d/README.md"
  },
  {
    "rank": 15,
    "name": "CUA-S1 4B",
    "capability": 74.4,
    "capabilityScore": 74.4,
    "intelligence": 60.2,
    "calibration": 88.6,
    "latency": 0.45,
    "speed": 0.45,
    "cost": 0.052,
    "baseModel": "Qwen/Qwen3.5-4B",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://huggingface.co/cua-ai/cua-s1-4b-0.2/blob/16818868b0cc7813808aae4e87b417657046ab79/README.md"
  },
  {
    "rank": 16,
    "name": "Visual-Jev generic",
    "capability": 74.2,
    "capabilityScore": 74.2,
    "intelligence": 60.2,
    "calibration": 88.1,
    "latency": 0.51,
    "speed": 0.51,
    "cost": 0.059,
    "baseModel": "Qwen3.5-4B baseline",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": ""
  },
  {
    "rank": 17,
    "name": "imajev 2B",
    "capability": 74.1,
    "capabilityScore": 74.1,
    "intelligence": 57.7,
    "calibration": 90.5,
    "latency": 0.34,
    "speed": 0.34,
    "cost": 0.034,
    "baseModel": "Qwen3.5-2B",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://github.com/fstandhartinger/jevbench/issues/80"
  },
  {
    "rank": 18,
    "name": "Reflex 4B",
    "capability": 73.5,
    "capabilityScore": 73.5,
    "intelligence": 63.3,
    "calibration": 83.7,
    "latency": 0.46,
    "speed": 0.46,
    "cost": 0.049,
    "baseModel": "Qwen/Qwen3.5-4B",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://huggingface.co/kshetrajna12/reflex-qwen3.5-4b-lora/blob/9df1cbbbc1837ba494530a59d3d9d345688ed250/README.md"
  },
  {
    "rank": 19,
    "name": "Standard One 8B",
    "capability": 72.4,
    "capabilityScore": 72.4,
    "intelligence": 52.8,
    "calibration": 92.1,
    "latency": 0.43,
    "speed": 0.43,
    "cost": 0.053,
    "baseModel": "mistralai/Ministral-3-8B-Instruct-2512-BF16",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://github.com/fstandhartinger/jevbench/issues/74"
  },
  {
    "rank": 20,
    "name": "OmniJev 4B",
    "capability": 69.7,
    "capabilityScore": 69.7,
    "intelligence": 56.9,
    "calibration": 82.6,
    "latency": 0.41,
    "speed": 0.41,
    "cost": 0.044,
    "baseModel": "Qwen/Qwen3.5-4B",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://huggingface.co/tinnel123/OmniJev/blob/91248107b7f5230bd5715db2e62e83c9a72fcd81/README.md"
  },
  {
    "rank": 21,
    "name": "Mapika decider-2b-vision BF16",
    "capability": 67.1,
    "capabilityScore": 67.1,
    "intelligence": 52.8,
    "calibration": 81.5,
    "latency": 0.33,
    "speed": 0.33,
    "cost": 0.026,
    "baseModel": "Qwen3.5-2B-Base",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://huggingface.co/Mapika/decider-2b-vision/blob/863e290863655f1d6b69324d77d09ac972d21609/README.md"
  },
  {
    "rank": 22,
    "name": "Jevify Qwen3-VL-2B T2",
    "capability": 67.1,
    "capabilityScore": 67.1,
    "intelligence": 47.3,
    "calibration": 86.8,
    "latency": 0.27,
    "speed": 0.27,
    "cost": 0.023,
    "baseModel": "Qwen3-VL-2B-Instruct",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://huggingface.co/Praveenrajus/jevify-qwen3-vl-2b-t2/blob/1593ab894637b342bb20700b5af0ee17ce28d5d7/README.md"
  },
  {
    "rank": 23,
    "name": "djev-spark NVFP4",
    "capability": 66.7,
    "capabilityScore": 66.7,
    "intelligence": 49.1,
    "calibration": 84.3,
    "latency": 0.5,
    "speed": 0.5,
    "cost": 0.063,
    "baseModel": "undisclosed",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://raw.githubusercontent.com/Davipar/djev-dev/HEAD/README.md"
  },
  {
    "rank": 24,
    "name": "Glance",
    "capability": 66.2,
    "capabilityScore": 66.2,
    "intelligence": 59.4,
    "calibration": 73.0,
    "latency": 0.33,
    "speed": 0.33,
    "cost": 0.03,
    "baseModel": "Qwen3-VL-4B",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://raw.githubusercontent.com/yoheinakajima/glance/master/README.md"
  },
  {
    "rank": 25,
    "name": "OmniJev 2B",
    "capability": 65.0,
    "capabilityScore": 65.0,
    "intelligence": 49.5,
    "calibration": 80.5,
    "latency": 0.34,
    "speed": 0.34,
    "cost": 0.033,
    "baseModel": "Qwen/Qwen3.5-2B",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://huggingface.co/tinnel123/OmniJev-2B/blob/37659c000df592edaf201590a622326ce60484d4/README.md"
  },
  {
    "rank": 26,
    "name": "Jev-Vision 8B",
    "capability": 63.7,
    "capabilityScore": 63.7,
    "intelligence": 64.2,
    "calibration": 63.1,
    "latency": 0.45,
    "speed": 0.45,
    "cost": 0.046,
    "baseModel": "Qwen/Qwen3-VL-8B-Instruct",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://huggingface.co/SeanLiu/Jev-Vision/blob/dc2d645b9d019e66b8f63def5c8a8be842448f4c/README.md"
  },
  {
    "rank": 27,
    "name": "vjev-vision",
    "capability": 63.5,
    "capabilityScore": 63.5,
    "intelligence": 43.2,
    "calibration": 83.8,
    "latency": 0.27,
    "speed": 0.27,
    "cost": 0.02,
    "baseModel": "Qwen/Qwen3.5-4B",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://huggingface.co/yah01/vjev-vision/blob/2fa8b58e40e5bc351a7d6dd39b953469a8f3ded2/README.md"
  },
  {
    "rank": 28,
    "name": "jev-spatial",
    "capability": 62.5,
    "capabilityScore": 62.5,
    "intelligence": 52.0,
    "calibration": 73.0,
    "latency": 0.27,
    "speed": 0.27,
    "cost": 0.022,
    "baseModel": "Molmo2-ER",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://huggingface.co/Fr0zencr4nE/jev-spatial/blob/5727eade6e1af2750ff852b1b1e90bb1f41a0602/README.md"
  },
  {
    "rank": 29,
    "name": "OmniJev-Qwen3.5-9B-v4",
    "capability": 60.9,
    "capabilityScore": 60.9,
    "intelligence": 52.2,
    "calibration": 69.6,
    "latency": 0.27,
    "speed": 0.27,
    "cost": 0.022,
    "baseModel": "undisclosed",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": ""
  },
  {
    "rank": 30,
    "name": "Standard One 3B",
    "capability": 59.8,
    "capabilityScore": 59.8,
    "intelligence": 35.8,
    "calibration": 83.9,
    "latency": 0.35,
    "speed": 0.35,
    "cost": 0.039,
    "baseModel": "mistralai/Ministral-3-3B-Instruct-2512-BF16",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://github.com/fstandhartinger/jevbench/issues/74"
  },
  {
    "rank": 31,
    "name": "JPT-0.8B",
    "capability": 57.2,
    "capabilityScore": 57.2,
    "intelligence": 36.3,
    "calibration": 78.1,
    "latency": 0.3,
    "speed": 0.3,
    "cost": 0.026,
    "baseModel": "Qwen3.5-0.8B",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://github.com/fstandhartinger/jevbench/issues/78"
  },
  {
    "rank": 32,
    "name": "Jevify Gemma 4 E4B",
    "capability": 56.2,
    "capabilityScore": 56.2,
    "intelligence": 35.7,
    "calibration": 76.7,
    "latency": 0.26,
    "speed": 0.26,
    "cost": 0.02,
    "baseModel": "google/gemma-4-E4B-it",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://huggingface.co/kushalpatil/jevify-gemma4-e4b-lora/blob/cca1f55e19f59dd493928459c4551131d10fdff9/README.md"
  },
  {
    "rank": 33,
    "name": "OmniJev 0.8B",
    "capability": 55.5,
    "capabilityScore": 55.5,
    "intelligence": 34.6,
    "calibration": 76.5,
    "latency": 0.33,
    "speed": 0.33,
    "cost": 0.031,
    "baseModel": "Qwen/Qwen3.5-0.8B",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://huggingface.co/tinnel123/OmniJev-SFT-0.8B/blob/4d0752d353d21eac324213180752317a7e027737/README.md"
  },
  {
    "rank": 34,
    "name": "Gevva E2B multimodal",
    "capability": 54.0,
    "capabilityScore": 54.0,
    "intelligence": 32.1,
    "calibration": 76.0,
    "latency": 0.41,
    "speed": 0.41,
    "cost": 0.043,
    "baseModel": "Gemma 4 E2B",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://github.com/fstandhartinger/jevbench/issues/94"
  },
  {
    "rank": 35,
    "name": "Gevva E4B",
    "capability": 52.9,
    "capabilityScore": 52.9,
    "intelligence": 35.9,
    "calibration": 69.9,
    "latency": 0.47,
    "speed": 0.47,
    "cost": 0.05,
    "baseModel": "Gemma 4 E4B",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://github.com/fstandhartinger/jevbench/issues/94"
  },
  {
    "rank": 36,
    "name": "Qevi-2B",
    "capability": 42.6,
    "capabilityScore": 42.6,
    "intelligence": 29.5,
    "calibration": 55.7,
    "latency": 0.29,
    "speed": 0.29,
    "cost": 0.024,
    "baseModel": "Qwen/Qwen3-VL-2B-Instruct",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://huggingface.co/MeerDevelopment/Qevi-2B/blob/9a0f5ec8654598ed0c5a819daf6b4ecce749a190/README.md"
  },
  {
    "rank": 37,
    "name": "JEVision",
    "capability": 39.9,
    "capabilityScore": 39.9,
    "intelligence": 35.4,
    "calibration": 44.4,
    "latency": 0.49,
    "speed": 0.49,
    "cost": 0.057,
    "baseModel": "Qwen/Qwen3.5-0.8B-Base",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://huggingface.co/divyanshx11/JEVision/blob/e1cfde19bfae6548a46812c64a9a6f1a67459c95/README.md"
  },
  {
    "rank": 38,
    "name": "PlayJev 0.8B",
    "capability": 24.4,
    "capabilityScore": 24.4,
    "intelligence": 4.4,
    "calibration": 59.4,
    "latency": 0.49,
    "speed": 0.49,
    "cost": 0.023,
    "baseModel": "Qwen/Qwen3.5-0.8B-Base",
    "access": "Open Weights",
    "category": "Jev-Class Vision",
    "link": "https://huggingface.co/OmniJev/PlayJev-0.8B/blob/a7348002b1e159add7037d6d50812cd4db2d96e9/README.md"
  },
  {
    "rank": 39,
    "name": "GPT-5.6 Luna",
    "capability": 93.8,
    "capabilityScore": 93.8,
    "intelligence": 91.8,
    "calibration": 95.8,
    "latency": 2.88,
    "speed": 2.88,
    "cost": 0.35,
    "baseModel": "undisclosed",
    "access": "Closed API",
    "category": "Baseline Multimodal",
    "link": ""
  },
  {
    "rank": 40,
    "name": "Gemini 3.1 Flash Lite",
    "capability": 87.9,
    "capabilityScore": 87.9,
    "intelligence": 84.3,
    "calibration": 91.4,
    "latency": 1.8,
    "speed": 1.8,
    "cost": 0.39,
    "baseModel": "undisclosed",
    "access": "Closed API",
    "category": "Baseline Multimodal",
    "link": ""
  },
  {
    "rank": 41,
    "name": "GPT-6 Luna (low)",
    "capability": 84.7,
    "capabilityScore": 84.7,
    "intelligence": 81.1,
    "calibration": 88.3,
    "latency": 2.91,
    "speed": 2.91,
    "cost": 0.22,
    "baseModel": "undisclosed",
    "access": "Closed API",
    "category": "Baseline Multimodal",
    "link": ""
  },
  {
    "rank": 42,
    "name": "Bonsai-2-27B v2 PQ2_0",
    "capability": 82.8,
    "capabilityScore": 82.8,
    "intelligence": 74.9,
    "calibration": 90.7,
    "latency": 1.75,
    "speed": 1.75,
    "cost": 0.23,
    "baseModel": "undisclosed",
    "access": "Open Weights",
    "category": "Baseline Multimodal",
    "link": ""
  },
  {
    "rank": 43,
    "name": "Autoloops - Gemma 4 31B IT",
    "capability": 80.8,
    "capabilityScore": 80.8,
    "intelligence": 79.3,
    "calibration": 82.4,
    "latency": 1.42,
    "speed": 1.42,
    "cost": 0.082,
    "baseModel": "undisclosed",
    "access": "Closed API",
    "category": "Baseline Multimodal",
    "link": ""
  },
  {
    "rank": 44,
    "name": "Gemini 3.8 Flash",
    "capability": 80.7,
    "capabilityScore": 80.7,
    "intelligence": 88.5,
    "calibration": 73.0,
    "latency": 4.83,
    "speed": 4.83,
    "cost": 2.06,
    "baseModel": "undisclosed",
    "access": "Closed API",
    "category": "Baseline Multimodal",
    "link": ""
  },
  {
    "rank": 45,
    "name": "Winnow-12B",
    "capability": 76.8,
    "capabilityScore": 76.8,
    "intelligence": 67.4,
    "calibration": 86.2,
    "latency": 0.69,
    "speed": 0.69,
    "cost": 0.09,
    "baseModel": "google/gemma-4-12B-it",
    "access": "Open Weights",
    "category": "Baseline Multimodal",
    "link": "https://huggingface.co/EldanRing/Winnow-12B"
  },
  {
    "rank": 46,
    "name": "djev-distill-v4",
    "capability": 68.7,
    "capabilityScore": 68.7,
    "intelligence": 52.5,
    "calibration": 84.8,
    "latency": 0.54,
    "speed": 0.54,
    "cost": 0.068,
    "baseModel": "undisclosed",
    "access": "Open Weights",
    "category": "Baseline Multimodal",
    "link": ""
  },
  {
    "rank": 47,
    "name": "diffusiongemma-26b djev v10",
    "capability": 65.7,
    "capabilityScore": 65.7,
    "intelligence": 57.2,
    "calibration": 74.3,
    "latency": 0.56,
    "speed": 0.56,
    "cost": 0.07,
    "baseModel": "undisclosed",
    "access": "Open Weights",
    "category": "Baseline Multimodal",
    "link": ""
  },
  {
    "rank": 48,
    "name": "djev-dev BF16",
    "capability": 63.8,
    "capabilityScore": 63.8,
    "intelligence": 49.6,
    "calibration": 78.0,
    "latency": 0.54,
    "speed": 0.54,
    "cost": 0.07,
    "baseModel": "google/diffusiongemma-26B-A4B-it",
    "access": "Open Weights",
    "category": "Baseline Multimodal",
    "link": "https://github.com/Davipar/djev-dev/blob/HEAD/README.md"
  },
  {
    "rank": 49,
    "name": "OpenJev 4B NLI v5",
    "capability": 33.6,
    "capabilityScore": 33.6,
    "intelligence": 67.2,
    "calibration": 0.0,
    "latency": 0.79,
    "speed": 0.79,
    "cost": 0.092,
    "baseModel": "Qwen/Qwen3.5-4B",
    "access": "Open Weights",
    "category": "Baseline Multimodal",
    "link": "https://huggingface.co/AlexWortega/openjev/blob/a20448012c213128955ca0c693e7c943865cab77/README.md"
  },
  {
    "rank": 50,
    "name": "OpenJev 4B NLI v2",
    "capability": 31.8,
    "capabilityScore": 31.8,
    "intelligence": 63.6,
    "calibration": 0.0,
    "latency": 0.78,
    "speed": 0.78,
    "cost": 0.093,
    "baseModel": "Qwen/Qwen3.5-4B",
    "access": "Open Weights",
    "category": "Baseline Multimodal",
    "link": "https://huggingface.co/AlexWortega/openjev/blob/a20448012c213128955ca0c693e7c943865cab77/README.md"
  }
];

export function getTopSystems(limit = 10) {
  return JEV_IMAGE_BENCH_DATA
    .filter((s) => s.category === 'Jev-Class Vision')
    .slice(0, limit);
}

export function filterImageBench(list, { access = 'all', query = '', category = 'all' } = {}) {
  let result = [...list];

  if (access === 'open') {
    result = result.filter(m => m.access === 'Open Weights');
  } else if (access === 'closed') {
    result = result.filter(m => m.access === 'Closed API');
  }

  if (query && query.trim()) {
    const q = query.toLowerCase().trim();
    result = result.filter(m => m.name.toLowerCase().includes(q) || m.baseModel.toLowerCase().includes(q));
  }

  if (category === 'jev-class') {
    result = result.filter(m => m.category === 'Jev-Class Vision');
  } else if (category === 'baseline') {
    result = result.filter(m => m.category === 'Baseline Multimodal');
  }

  return result;
}

export function sortImageBench(list, sortKey = 'rank', ascending = true) {
  const sorted = [...list];
  sorted.sort((a, b) => {
    let va = a[sortKey];
    let vb = b[sortKey];
    if (sortKey === 'capability') {
      va = a.capabilityScore;
      vb = b.capabilityScore;
    }
    if (va < vb) return ascending ? -1 : 1;
    if (va > vb) return ascending ? 1 : -1;
    return 0;
  });
  return sorted;
}
