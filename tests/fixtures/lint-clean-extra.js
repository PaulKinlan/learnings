// The safe neighbours of the constructs in lint-sinks-extra.js: text, not HTML, and constructs
// inside comments or strings. None of these may be reported (a false positive here would push real
// code to work around the gate).
function a(el, text) { el.insertAdjacentText('beforeend', text); }
function b(el, text) { el.textContent = text; }
function c(el, text) { el.innerText = text; }
// reminder: never el.outerHTML = x, frame.srcdoc = x, eval(code) or new Function(code)
const GUIDANCE = "el.outerHTML = html, frame.srcdoc = html and eval(code) are banned";
// A clear of the whole container is the one innerHTML assignment the gate allows; it must be the
// ONLY statement on its line, which is why these two are written as their own statements.
function d(el) {
  el.innerHTML = '';
}
function e(el) {
  el.innerHTML = ' ';
}
