// reminder: prefer textContent, never el.innerHTML = x
// never call document.write() or el.insertAdjacentHTML() from a render path
// the computed form el["innerHTML"] = payload is banned too
export const GUIDANCE = "document.write() is banned here";
export const DOT_FORM = 'use textContent instead of el.innerHTML = value';
export const ADJACENT = "el.insertAdjacentHTML('beforeend', html) too";
export const COMPUTED = 'and el["innerHTML"] = value';
