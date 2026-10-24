const nav = document.querySelector("#lesson-nav");
const title = document.querySelector("#lesson-title");
const stepLabel = document.querySelector("#step-label");
const idea = document.querySelector("#lesson-idea");
const watch = document.querySelector("#watch");
const fields = document.querySelector("#param-fields");
const form = document.querySelector("#controls");
const runButton = document.querySelector("#run");
const prevButton = document.querySelector("#prev");
const nextButton = document.querySelector("#next");
const status = document.querySelector("#status");
const callBlock = document.querySelector("#call");
const linesBlock = document.querySelector("#lines");
const images = document.querySelector("#images");
const notice = document.querySelector("#notice");
const elapsed = document.querySelector("#elapsed");

let lessons = [];
let currentIndex = 0;
let requestSerial = 0;

function currentLesson() {
  return lessons[currentIndex];
}

function renderNav() {
  nav.replaceChildren();
  lessons.forEach((lesson, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "lesson-link" + (index === currentIndex ? " current" : "");
    button.innerHTML = `<span>${lesson.step}</span>${lesson.title}`;
    button.addEventListener("click", () => select(index));
    nav.append(button);
  });
}

function renderFields(lesson) {
  fields.replaceChildren();
  if (lesson.params.length === 0) {
    const note = document.createElement("p");
    note.className = "empty-note";
    note.textContent = "This check takes no arguments.";
    fields.append(note);
    return;
  }
  for (const param of lesson.params) {
    const wrapper = document.createElement("label");
    wrapper.className = "field";
    const heading = document.createElement("span");
    heading.className = "field-label";
    const name = document.createElement("span");
    name.textContent = param.label;
    heading.append(name);
    const input = document.createElement("input");
    input.dataset.name = param.name;
    if (param.kind === "numbers") {
      input.type = "text";
      input.spellcheck = false;
      input.value = param.default;
      input.addEventListener("change", () => runLesson());
    } else {
      input.type = "range";
      input.min = String(param.min);
      input.max = String(param.max);
      input.step = String(param.step);
      input.value = String(param.default);
      const value = document.createElement("b");
      value.id = `val-${param.name}`;
      value.textContent = formatValue(param, input.value);
      heading.append(value);
      input.addEventListener("input", () => {
        value.textContent = formatValue(param, input.value);
      });
      input.addEventListener("change", () => runLesson());
    }
    const help = document.createElement("p");
    help.className = "help";
    help.textContent = param.help;
    wrapper.append(heading, input, help);
    fields.append(wrapper);
  }
}

function formatValue(param, value) {
  const number = Number(value);
  if (param.kind === "int") return String(Math.round(number));
  const step = String(param.step);
  const places = step.includes(".") ? step.split(".")[1].length : 0;
  return number.toFixed(places);
}

function readParams() {
  const params = {};
  for (const input of fields.querySelectorAll("input")) {
    const lessonParam = currentLesson().params.find((item) => item.name === input.dataset.name);
    if (lessonParam.kind === "numbers") params[input.dataset.name] = input.value;
    else if (lessonParam.kind === "int") params[input.dataset.name] = Math.round(Number(input.value));
    else params[input.dataset.name] = Number(input.value);
  }
  return params;
}

function select(index, shouldRun = true) {
  requestSerial += 1;
  runButton.disabled = false;
  runButton.textContent = "Run this function";
  currentIndex = index;
  const lesson = currentLesson();
  stepLabel.textContent = `Lesson ${lesson.step}`;
  title.textContent = lesson.title;
  idea.textContent = lesson.idea;
  watch.textContent = lesson.watch;
  renderNav();
  renderFields(lesson);
  clearOutput();
  prevButton.disabled = index === 0;
  nextButton.disabled = index === lessons.length - 1;
  if (shouldRun && lesson.auto) runLesson();
  else if (!lesson.auto) {
    status.hidden = false;
    status.textContent = "Press Run this function. The first detector run loads a network.";
  }
}

function clearOutput() {
  status.hidden = false;
  status.className = "status";
  status.textContent = "Running the function…";
  callBlock.hidden = true;
  linesBlock.replaceChildren();
  images.replaceChildren();
  notice.hidden = true;
  elapsed.textContent = "";
}

function showError(message) {
  status.hidden = false;
  status.className = "status error";
  status.textContent = message;
  callBlock.hidden = true;
  linesBlock.replaceChildren();
  images.replaceChildren();
  notice.hidden = true;
}

function showResult(result) {
  status.hidden = true;
  callBlock.hidden = false;
  callBlock.textContent = result.call;
  linesBlock.textContent = result.lines.join("\n");
  images.replaceChildren();
  for (const image of result.images) {
    const figure = document.createElement("figure");
    const img = document.createElement("img");
    img.src = `data:image/png;base64,${image.png}`;
    img.alt = image.title;
    const caption = document.createElement("figcaption");
    caption.textContent = image.title;
    figure.append(img, caption);
    images.append(figure);
  }
  notice.hidden = false;
  notice.textContent = result.notice;
  elapsed.textContent = `${result.elapsed_ms.toFixed(0)} ms`;
  applyParams(result.params);
}

function applyParams(params) {
  const lesson = currentLesson();
  for (const [name, value] of Object.entries(params)) {
    const input = fields.querySelector(`[data-name="${name}"]`);
    if (!input) continue;
    input.value = value;
    const spec = lesson.params.find((item) => item.name === name);
    const readout = document.querySelector(`#val-${name}`);
    if (readout && spec) readout.textContent = formatValue(spec, value);
  }
}

async function runLesson() {
  const lesson = currentLesson();
  const serial = ++requestSerial;
  runButton.disabled = true;
  runButton.textContent = "Running…";
  status.hidden = false;
  status.className = "status";
  status.textContent = "Running the function…";
  try {
    const response = await fetch(`/api/lessons/${lesson.id}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ params: readParams() }),
    });
    const payload = await response.json();
    if (serial !== requestSerial) return;
    if (!response.ok) showError(payload.error || "The function could not run.");
    else showResult(payload);
  } catch (error) {
    if (serial !== requestSerial) return;
    showError("The lab server did not answer. Start it with python -m physical_ai_lab.server.");
  } finally {
    if (serial === requestSerial) {
      runButton.disabled = false;
      runButton.textContent = "Run this function";
    }
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  runLesson();
});

prevButton.addEventListener("click", () => {
  if (currentIndex > 0) select(currentIndex - 1);
});

nextButton.addEventListener("click", () => {
  if (currentIndex < lessons.length - 1) select(currentIndex + 1);
});

async function boot() {
  const response = await fetch("/api/lessons");
  lessons = await response.json();
  select(0);
}

boot();
