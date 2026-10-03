// site/neural-networks/chapters.js
// The single list of chapters. The hub, every chapter's sidebar and every prev/next link are
// built from it, and tests/nn-chapters-links.test.js checks that every chapter that is not
// marked `planned` exists on disk and that every local link in every page resolves.

export const PARTS = [
  { id: "foundations", title: "Part 1 · One neuron" },
  { id: "learning", title: "Part 2 · How networks learn" },
  { id: "architectures", title: "Part 3 · Architectures" },
  { id: "frontier", title: "Part 4 · The modern stack" },
  { id: "backends", title: "Part 5 · How it runs" },
];

export const CHAPTERS = [
  {
    slug: "neuron-and-perceptron.html",
    part: "foundations",
    title: "The neuron and the perceptron",
    years: "1943–1969",
    summary:
      "A weighted sum and a threshold. How McCulloch and Pitts turned neurons into logic, how Rosenblatt made them learn, and why one layer can never learn XOR.",
    buildsOn: [],
  },
  {
    slug: "gradient-descent.html",
    part: "learning",
    title: "Gradient descent",
    years: "1847–2017",
    summary:
      "Learning as walking downhill. From Cauchy's 1847 method to momentum, Adam and AdamW, with a loss landscape you can roll a ball across.",
    buildsOn: ["neuron-and-perceptron.html"],
  },
  {
    slug: "backpropagation.html",
    part: "learning",
    title: "Backpropagation",
    years: "1960–1986",
    summary:
      "The chain rule, run backwards. Why a gradient for every weight costs about as much as one forward pass, and who worked that out.",
    buildsOn: ["gradient-descent.html"],
  },
  {
    slug: "activations.html",
    part: "learning",
    title: "Activation functions",
    years: "1943–2020",
    summary:
      "Why the nonlinearity matters, why deep sigmoid networks stopped learning, and how ReLU, GELU and SwiGLU fixed it.",
    buildsOn: ["backpropagation.html"],
  },
  {
    slug: "loss-and-calibration.html",
    part: "learning",
    title: "Loss, softmax and calibration",
    years: "1805–2017",
    summary:
      "What a network actually minimises, how softmax turns scores into probabilities, and why a confident model can be wrong about its own confidence.",
    buildsOn: ["backpropagation.html"],
  },
  {
    slug: "architectures.html",
    part: "architectures",
    title: "Architectures: MLPs, CNNs and RNNs",
    years: "1959–2015",
    summary:
      "Stacking neurons into layers, building in assumptions about images and sequences, and the skip connection that made very deep networks trainable.",
    buildsOn: ["activations.html", "loss-and-calibration.html"],
  },
  {
    slug: "cnn.html",
    part: "architectures",
    title: "Convolution, hands on",
    years: "1980–2015",
    summary:
      "Slide a 3×3 kernel across an image one step at a time, change stride and padding, and build a CNN forward pass you can run in the console.",
    buildsOn: ["architectures.html"],
  },
  {
    slug: "transformer.html",
    part: "architectures",
    title: "The Transformer",
    years: "2014–2023",
    summary:
      "Queries, keys and values; why attention divides by √d_k; and multi-head, multi-query and grouped-query attention.",
    buildsOn: ["architectures.html"],
  },
  {
    slug: "training-recipes.html",
    part: "frontier",
    title: "Making deep networks train",
    years: "2010–2022",
    summary:
      "The unglamorous fixes that made depth work: careful initialisation, batch and layer normalisation, dropout, mixed precision, and the scaling laws that set model size.",
    buildsOn: ["activations.html", "architectures.html"],
  },
  {
    slug: "modern-advancements.html",
    part: "frontier",
    title: "The modern stack",
    years: "2016–2025",
    summary:
      "RMSNorm, RoPE, FlashAttention, mixture-of-experts, state space models and KV-cache tricks: what changed between the 2017 Transformer and today's models.",
    buildsOn: ["transformer.html"],
  },
  {
    slug: "backends/index.html",
    part: "backends",
    title: "How it runs: five backends",
    years: "",
    summary:
      "The same matrix multiply on five runtimes: what each one does with memory, threads and precision, and when to pick which.",
    buildsOn: ["architectures.html"],
  },
  {
    slug: "backends/javascript.html",
    part: "backends",
    title: "JavaScript and typed arrays",
    years: "",
    summary: "Float32Array, loop order and the JIT: how fast plain JavaScript can multiply matrices.",
    buildsOn: ["backends/index.html"],
  },
  {
    slug: "backends/webassembly.html",
    part: "backends",
    title: "WebAssembly and SIMD",
    years: "",
    summary: "Linear memory, a GEMM kernel built byte by byte, and four multiplies per instruction with 128-bit SIMD.",
    buildsOn: ["backends/javascript.html"],
  },
  {
    slug: "backends/webgpu.html",
    part: "backends",
    title: "WebGPU compute shaders",
    years: "",
    summary: "WGSL, workgroups, storage buffers and the asynchronous readback: matrix multiply on the GPU.",
    buildsOn: ["backends/javascript.html"],
  },
  {
    slug: "backends/litert.html",
    part: "backends",
    title: "LiteRT on phones and the edge",
    years: "",
    summary: "FlatBuffer models, delegates such as XNNPACK, and int8 quantisation you can watch happen.",
    buildsOn: ["backends/index.html"],
  },
  {
    slug: "backends/pytorch.html",
    part: "backends",
    title: "PyTorch and autograd",
    years: "",
    summary: "Tensors, strides, the autograd tape, hooks and torch.compile, mirrored by a tape you can step through here.",
    buildsOn: ["backpropagation.html"],
  },
];

/** Anchors that lived on the hub URL before the single-page workbench moved to lab.html. */
export const LAB_ANCHORS = [
  "mlp-backprop", "network-stepper", "cnn-lab", "rnn-lab", "deep-resnet-lab", "transformer-lab",
  "diffusion-lab", "decision-lab", "block-builder", "first-principles-guide", "arch-matrix",
  "backend-comparison", "wasm-gemm-code", "backprop-code", "plugin-api-code",
];

/** Where an old deep link to the hub should go now, or null when it is not an old workbench anchor. */
export function labRedirectFor(hash) {
  const id = String(hash ?? "").replace(/^#/, "");
  return LAB_ANCHORS.includes(id) ? `./lab.html#${id}` : null;
}

/** Chapters that exist, in reading order. */
export function readyChapters() {
  return CHAPTERS.filter((c) => !c.planned);
}

function link(root, chapter, text = chapter.title, doc = document) {
  const a = doc.createElement("a");
  a.href = root + chapter.slug;
  a.textContent = text;
  return a;
}

/**
 * Fill every [data-chapter-nav] with the chapter list (current page marked) and every
 * [data-prev-next] with links to the neighbouring chapters. Pages declare their position with
 * <body data-nn-root="./" data-chapter="gradient-descent.html">.
 */
export function mountChapterNav(doc = document) {
  const root = doc.body.dataset.nnRoot ?? "./";
  const here = doc.body.dataset.chapter ?? "";

  for (const nav of doc.querySelectorAll("[data-chapter-nav]")) {
    const outer = doc.createElement("ol");
    for (const part of PARTS) {
      const items = CHAPTERS.filter((c) => c.part === part.id);
      if (!items.length) continue;
      const li = doc.createElement("li");
      const heading = doc.createElement("span");
      heading.className = "toc-part";
      heading.textContent = part.title;
      const inner = doc.createElement("ol");
      for (const c of items) {
        const item = doc.createElement("li");
        if (c.planned) {
          const span = doc.createElement("span");
          span.className = "planned";
          span.textContent = `${c.title} (planned)`;
          item.append(span);
        } else {
          const a = link(root, c, c.title, doc);
          if (c.slug === here) a.setAttribute("aria-current", "page");
          item.append(a);
        }
        inner.append(item);
      }
      li.append(heading, inner);
      outer.append(li);
    }
    nav.replaceChildren(outer);
  }

  const ready = readyChapters();
  const i = ready.findIndex((c) => c.slug === here);
  for (const box of doc.querySelectorAll("[data-prev-next]")) {
    const parts = [];
    if (i > 0) {
      const a = link(root, ready[i - 1], `← ${ready[i - 1].title}`, doc);
      a.rel = "prev";
      parts.push(a);
    }
    const hub = doc.createElement("a");
    hub.href = root;
    hub.textContent = "All chapters";
    parts.push(hub);
    if (i >= 0 && i < ready.length - 1) {
      const a = link(root, ready[i + 1], `${ready[i + 1].title} →`, doc);
      a.rel = "next";
      parts.push(a);
    }
    box.replaceChildren(...parts);
  }
}

if (typeof document !== "undefined" && document.body?.dataset.chapter) mountChapterNav();
