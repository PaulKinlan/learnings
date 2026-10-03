// site/neural-networks/hub.js — the chapter hub: renders the stack from chapters.js and keeps
// old deep links working (the workbench used to live at this URL; its anchors now live on lab.html).
import { CHAPTERS, PARTS, labRedirectFor } from "./chapters.js";

// Anchors live on index.html; no redirect needed.

const titleOf = new Map(CHAPTERS.map((c) => [c.slug, c.title]));

function render() {
  const list = document.getElementById("nn-stack");
  if (!list) return;
  const partTitle = new Map(PARTS.map((p) => [p.id, p.title]));
  const items = CHAPTERS.map((c) => {
    const li = document.createElement("li");
    li.dataset.slug = c.slug;
    const label = document.createElement("span");
    label.className = "part-label";
    label.textContent = partTitle.get(c.part) ?? "";
    const h = document.createElement("h3");
    if (c.planned) {
      h.textContent = c.title;
      const planned = document.createElement("span");
      planned.className = "years";
      planned.textContent = "planned";
      h.append(" ", planned);
    } else {
      const a = document.createElement("a");
      a.href = `./${c.slug}`;
      a.textContent = c.title;
      h.append(a);
    }
    if (c.years) {
      const y = document.createElement("span");
      y.className = "years";
      y.textContent = c.years;
      h.append(y);
    }
    const p = document.createElement("p");
    p.textContent = c.summary;
    li.append(label, h, p);
    if (c.buildsOn.length) {
      const b = document.createElement("p");
      b.className = "builds-on";
      b.textContent = `Builds on: ${c.buildsOn.map((s) => titleOf.get(s)).join(", ")}`;
      li.append(b);
    }
    return li;
  });
  list.replaceChildren(...items);

  // Hovering or focusing a chapter highlights everything it builds on, all the way down.
  const bySlug = new Map(CHAPTERS.map((c) => [c.slug, c]));
  const prerequisites = (slug, acc = new Set()) => {
    for (const dep of bySlug.get(slug)?.buildsOn ?? []) {
      if (!acc.has(dep)) {
        acc.add(dep);
        prerequisites(dep, acc);
      }
    }
    return acc;
  };
  const highlight = (slug) => {
    const deps = slug ? prerequisites(slug) : new Set();
    for (const li of list.children) {
      li.classList.toggle("is-active", li.dataset.slug === slug);
      li.classList.toggle("is-prereq", deps.has(li.dataset.slug));
    }
    const status = document.getElementById("stack-status");
    if (status) {
      status.textContent = slug
        ? `${titleOf.get(slug)} builds on ${deps.size} earlier chapter${deps.size === 1 ? "" : "s"}.`
        : "";
    }
  };
  for (const li of list.children) {
    li.addEventListener("pointerenter", () => highlight(li.dataset.slug));
    li.addEventListener("focusin", () => highlight(li.dataset.slug));
  }
  list.addEventListener("pointerleave", () => highlight(null));
}

render();
