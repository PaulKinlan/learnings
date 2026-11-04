// One line per DOM-injection construct the scanner must catch beyond innerHTML (learnings-blv).
function a(el, html) { el.outerHTML = html; }
function b(el, html) { el.insertAdjacentHTML('afterbegin', html); }
function c(html) { document.writeln(html); }
function d(range, html) { return range.createContextualFragment(html); }
function e(frame, html) { frame.srcdoc = html; }
function f(code) { return eval(code); }
function g(code) { return new Function(code)(); }
function h(el, value) { el.setAttribute('onclick', value); }
const i = (el, value) => { el['outerHTML'] = value; };
// learnings-wltf: the adjacent spelling of an already-listed construct is the same injection. One
// line per bypass the reviewer proved against the widened gate.
function j(frame, html) { frame['srcdoc'] = html; }
function k(frame, html) { frame.srcdoc += html; }
function l(code) { return Function(code)(); }
function m(el, value) { el.setAttributeNS(null, 'onclick', value); }
function n(code) { return globalThis['eval'](code); }
