function testA(el) { el.innerHTML += "<p>x</p>"; }
function testB(el) { el["innerHTML"] = "<b>x</b>"; }
function testC(el) { el.insertAdjacentHTML("beforeend", "<p>x</p>"); }
function testD() { document.write("<p>x</p>"); }
