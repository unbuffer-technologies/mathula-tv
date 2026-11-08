"""The console's page: one self-contained HTML document (no external fonts, scripts or requests; the server sends a strict content security policy with a nonce)."""

PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pronunciation Console</title>
<style nonce="__NONCE__">
:root {
  color-scheme: light dark;
  --bg: #f4f2ee; --panel: #fffdf9; --ink: #1e1b16; --muted: #6b645a; --line: #dcd5c8; --accent: #1d5c4b; --accent-ink: #ffffff;
  --good: #1d6b3a; --good-bg: #e2f0e6; --warn: #8a4b08; --warn-bg: #f7e9d3; --new: #3b4a6b; --new-bg: #e5e9f2; --bad: #9b2c2c;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #15171a; --panel: #1d2024; --ink: #ece8e1; --muted: #9a958b; --line: #31353b; --accent: #5fb89c; --accent-ink: #0d1f19;
    --good: #7fd09a; --good-bg: #1c2f24; --warn: #e8b36a; --warn-bg: #33281a; --new: #a9b9dd; --new-bg: #232a3a; --bad: #e58a8a;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.5 system-ui, "Segoe UI", Roboto, sans-serif; }
header { display: flex; flex-wrap: wrap; align-items: center; gap: 16px; padding: 12px 20px; border-bottom: 1px solid var(--line); background: var(--panel); position: sticky; top: 0; z-index: 2; }
header h1 { font-size: 16px; margin: 0; letter-spacing: .01em; }
nav { display: flex; gap: 4px; }
nav button { background: none; border: 0; padding: 8px 14px; border-radius: 8px; color: var(--muted); font: inherit; cursor: pointer; }
nav button[aria-current="page"] { background: var(--bg); color: var(--ink); font-weight: 600; }
.spacer { flex: 1; }
main { max-width: 1240px; margin: 0 auto; padding: 20px; }
button, select, input, textarea { font: inherit; color: inherit; }
.btn { border: 1px solid var(--line); background: var(--panel); border-radius: 8px; padding: 8px 14px; cursor: pointer; }
.btn:hover { border-color: var(--accent); }
.btn.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); font-weight: 600; }
.btn.danger { color: var(--bad); }
.btn:disabled { opacity: .5; cursor: default; }
.play::after { content: " \25CB"; color: var(--muted); } .play.ready::after { content: " \25CF"; color: var(--good); }
input[type=text], input[type=password], select, textarea { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 8px 10px; width: 100%; }
.bar { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 16px; }
.bar select { width: auto; max-width: 100%; }
.cols { display: grid; grid-template-columns: minmax(260px, 340px) 1fr; gap: 20px; align-items: start; }
@media (max-width: 800px) { .cols { grid-template-columns: 1fr; } }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; }
.list { display: flex; flex-direction: column; max-height: 72vh; overflow: auto; padding: 6px; }
.row { display: grid; grid-template-columns: 1fr auto; gap: 2px 10px; text-align: left; background: none; border: 0; border-radius: 8px; padding: 9px 10px; cursor: pointer; }
.row:hover, .row[aria-current="true"] { background: var(--bg); }
.row .w { font-weight: 600; } .row .s { color: var(--muted); font-family: ui-monospace, Consolas, monospace; font-size: 13px; grid-column: 1; }
.badge { font-size: 12px; border-radius: 99px; padding: 1px 9px; align-self: center; }
.badge.unreviewed { background: var(--new-bg); color: var(--new); } .badge.approved { background: var(--good-bg); color: var(--good); } .badge.custom { background: var(--warn-bg); color: var(--warn); }
.mono { font-family: ui-monospace, Consolas, monospace; }
.big { font-size: 30px; font-weight: 600; letter-spacing: .01em; margin: 4px 0 8px; }
h2 { margin: 0 0 2px; font-size: 20px; } h3 { margin: 18px 0 6px; font-size: 13px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.muted { color: var(--muted); } .small { font-size: 13px; }
ul.notes { margin: 6px 0 0; padding-left: 18px; color: var(--muted); font-size: 13px; }
.plays { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }
.sentence { font-size: 13px; color: var(--muted); margin: 2px 0 8px; }
.decide { display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0 4px; }
.alt { display: grid; grid-template-columns: 1fr auto; gap: 6px 12px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 10px; margin: 8px 0; }
.alt[data-picked="true"] { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent) inset; }
.alt .sp { font-size: 18px; font-weight: 600; }
.tag { display: inline-block; font-size: 12px; background: var(--bg); border-radius: 6px; padding: 1px 7px; margin: 2px 4px 0 0; color: var(--muted); }
table { width: 100%; border-collapse: collapse; } th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; } th { color: var(--muted); font-weight: 500; font-size: 13px; }
.msg { margin: 10px 0; padding: 8px 12px; border-radius: 8px; background: var(--warn-bg); color: var(--warn); } .msg.ok { background: var(--good-bg); color: var(--good); }
.login { max-width: 360px; margin: 14vh auto; } .login h1 { margin-top: 0; } .field { margin: 10px 0; }
label.opt { display: flex; gap: 8px; align-items: flex-start; margin: 6px 0; } label.opt input { margin-top: 4px; }
.empty { color: var(--muted); padding: 24px; text-align: center; }
.smith { max-width: 820px; margin: 0 auto; }
.voices { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; margin: 10px 0; }
.voices audio { width: 100%; max-width: 100%; }
.voices .plays { flex-direction: column; align-items: stretch; }
.alt.finding { display: flex; flex-direction: column; gap: 4px; }
.src { display: grid; grid-template-columns: auto 1fr; gap: 6px 12px; align-items: center; padding: 8px 0; border-bottom: 1px solid var(--line); }
.src:last-child { border-bottom: 0; }
.src mark { background: var(--good-bg); color: var(--good); border-radius: 4px; padding: 0 4px; font-weight: 600; }
.src .heard { grid-column: 1 / -1; }
.src .btns { display: flex; gap: 6px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 2px 6px; text-align: left; }
.qs { max-width: 900px; margin: 0 auto; display: flex; flex-direction: column; gap: 14px; }
.q h2 { font-size: 18px; }
.qline { display: grid; grid-template-columns: 150px 1fr; gap: 8px; padding: 5px 0; border-bottom: 1px solid var(--line); }
.qline:last-of-type { border-bottom: 0; }
.qline.win { background: var(--good-bg); }
.qlist { display: flex; flex-direction: column; margin: 6px 0 10px; }
.q textarea { margin: 6px 0; }
.count { display: inline-block; min-width: 18px; margin-left: 6px; padding: 0 6px; border-radius: 99px; background: var(--warn-bg); color: var(--warn); font-size: 12px; text-align: center; }
.chips.gap { margin: 10px 0 4px; }
@media (max-width: 600px) { .qline { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div id="app"></div>
<audio id="player" preload="none"></audio>
<script nonce="__NONCE__">
"use strict";
const app = document.getElementById("app"), player = document.getElementById("player");
const state = { tab: "review", jobs: [], job: null, run: null, filter: "all", words: [], word: null, detail: null, picked: null, custom: "", note: "", source: null, showAlts: false, voices: [], message: null, busy: false };

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (k === "class") el.className = v; else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v === true) el.setAttribute(k, ""); else if (v !== false && v != null) el.setAttribute(k, v);
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return el;
}
async function api(path, method, body) {
  const res = await fetch(path, { method: method || "GET", headers: { "X-Requested-With": "console", "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined, credentials: "same-origin" });
  const data = await res.json().catch(() => ({}));
  if (res.status === 401 && path !== "/api/login") { state.signedIn = false; render(); throw new Error("sign in"); }
  if (!res.ok) throw new Error(data.error || ("error " + res.status));
  return data;
}
function say(text, ok) { state.message = text ? { text, ok } : null; render(); }
const short = (s) => (s || "").slice(0, 40);
const plain = (s) => String(s).replace(/\s*\([^)]*\)/g, "");
const READINGS = { researched: "web research of this name", word: "the dictionary", letters: "the names of its letters", guessed: "a guess from its spelling (no dictionary entry)", number: "the English number", committed: "your committed spelling" };
const voiceName = (v) => v.replace("zu-ZA-", "").replace("Neural", "");

async function start() {
  try { const s = await api("/api/session"); state.signedIn = true; state.voices = s.voices; await loadJobs(); } catch (e) { state.signedIn = false; }
  render();
}
async function loadJobs() {
  state.jobs = await api("/api/jobs");
  if (!state.job && state.jobs.length) { state.job = state.jobs[0].job; state.run = state.jobs[0].runs[0].run; await loadWords(); }
}
async function loadWords() {
  state.words = []; state.word = null; state.detail = null; state.showAlts = false; state.picked = null; state.custom = ""; state.note = ""; state.questions = null;
  if (!state.job) return;
  try { state.words = await api(`/api/words?job=${state.job}&run=${encodeURIComponent(state.run)}`); } catch (e) { say(e.message); }
  loadQuestions().then(render);
}
async function loadQuestions() {
  state.questions = null;
  if (!state.job) return;
  try { state.questions = await api(`/api/questions?job=${state.job}&run=${encodeURIComponent(state.run)}`); } catch (e) { state.questions = { words: [], answered: [], resolved: [], stretches: [], open: 0, languages: [] }; }
}
async function openWord(word) {
  state.word = word; state.detail = null; state.showAlts = false; state.picked = null; state.custom = ""; state.note = ""; state.message = null; render();
  state.source = null;
  api(`/api/source?job=${state.job}&word=${encodeURIComponent(word)}`).then((r) => { if (state.word === word) { state.source = r.occurrences; render(); } }).catch(() => { if (state.word === word) { state.source = []; render(); } });
  try { state.detail = await api(`/api/word?job=${state.job}&run=${encodeURIComponent(state.run)}&word=${encodeURIComponent(word)}`); } catch (e) { say(e.message); return; }
  render();
  prefetchAhead(false); watchReady();
}
const AHEAD = 2;                       // the clips of this many following words are made before they are asked for
let buttons = [], watching = 0;
function playBtn(spelling, voice) {
  const el = h("button", { class: "btn play", onclick: () => play(spelling, voice) }, "▶ " + voiceName(voice));
  buttons.push({ el, spelling, voice });
  return el;
}
function watchReady() {                // polls which clips on screen are made, so a button shows ● when it will play at once
  const token = ++watching;
  const loop = async (round) => {
    if (token !== watching || !state.word || !state.signedIn || !buttons.length) return;
    try {
      const r = await api("/api/ready", "POST", { job: state.job, run: state.run, word: state.word, items: buttons.map((b) => ({ spelling: b.spelling, voice: b.voice })) });
      buttons.forEach((b, i) => b.el.classList.toggle("ready", !!r.ready[i]));
      if (r.ready.some((x) => !x) && round < 80) setTimeout(() => loop(round + 1), 1500);
    } catch (e) {}
  };
  loop(0);
}
async function prefetchAhead(alternatives) {
  const shown = state.words.filter((w) => state.filter === "all" || w.status === state.filter);
  const at = shown.findIndex((w) => w.word === state.word);
  const words = alternatives ? [state.word] : [state.word, ...shown.slice(at + 1, at + 1 + AHEAD).map((w) => w.word)];
  try { await api("/api/prefetch", "POST", { job: state.job, run: state.run, words, alternatives: !!alternatives }); } catch (e) {}
}
async function play(spelling, voice, sample) {
  say("Making the clip…", true);
  try {
    const r = await api("/api/audio", "POST", { job: state.job, run: state.run, word: state.word, spelling, voice });
    player.src = r.url; await player.play(); say(null);
  } catch (e) { say(e.message); }
}
async function refreshAfterDecision(text) {
  const word = state.word; await loadWords(); state.word = word; say(text, true);
  try { state.detail = await api(`/api/word?job=${state.job}&run=${encodeURIComponent(state.run)}&word=${encodeURIComponent(word)}`); } catch (e) {}
  render();
}

async function playSource(o, mode, btn) {
  if (btn) btn.disabled = true;
  try { const r = await api("/api/source-audio", "POST", { job: o.job, index: o.index, mode }); await playUrl(r.url); } catch (e) { say(e.message); }
  if (btn) btn.disabled = false;
}
async function hearSource(o, box, btn) {                  // a recogniser's phones and its reading of the clip, shown for the ear to compare; nothing uses them
  btn.disabled = true; box.replaceChildren(h("span", { class: "muted small" }, "A recogniser is listening (a few seconds)…"));
  try {
    const rows = [];
    for (const mode of ["word", "phrase"]) rows.push(await api("/api/source-hear", "POST", { job: o.job, index: o.index, mode }));
    box.replaceChildren(...rows.map((r) => h("div", { class: "small" }, h("b", {}, r.mode + ": "), h("span", { class: "mono" }, r.phones || "(no phones)"), h("span", { class: "muted" }, "  ·  reads as “" + (r.reads_as || "") + "”"))),
      h("div", { class: "muted small" }, "A recogniser’s guess, here only to compare with your ear: nothing in the dub uses it."));
  } catch (e) { box.replaceChildren(h("span", { class: "muted small" }, e.message)); }
  btn.disabled = false;
}
function sourcePanel(list, showJob) {
  const box = h("div", {}, h("h3", {}, "Heard in the source audio"));
  if (list == null) { box.append(h("div", { class: "muted small" }, "Looking for the word in the source audio…")); return box; }
  if (!list.length) { box.append(h("div", { class: "muted small" }, "The word is not in the source transcript, so there is no source audio of it here.")); return box; }
  list.forEach((o) => {
    const word = h("button", { class: "btn", onclick: (e) => playSource(o, "word", e.currentTarget) }, "▶ word");
    const phrase = h("button", { class: "btn", onclick: (e) => playSource(o, "phrase", e.currentTarget) }, "▶ phrase");
    const heard = h("div", { class: "heard" });
    const ear = h("button", { class: "btn", onclick: (e) => hearSource(o, heard, e.currentTarget) }, "what a recogniser hears");
    box.append(h("div", { class: "src" }, h("div", { class: "btns" }, word, phrase, ear),
      h("div", { class: "small" }, o.pre + " ", h("mark", {}, o.word), " " + o.post, h("div", { class: "muted" }, `${showJob ? o.job_name + " · " : ""}${o.at}${o.speaker ? " · " + o.speaker.replace("AZURE_", "speaker ") : ""}`)), heard));
  });
  return box;
}

function login() {
  const pw = h("input", { type: "password", autocomplete: "current-password", id: "pw", autofocus: true });
  const go = async () => { try { await api("/api/login", "POST", { password: pw.value }); await start(); } catch (e) { say(e.message); } };
  pw.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
  return h("div", { class: "login card" }, h("h1", {}, "Pronunciation Console"), h("p", { class: "muted" }, "Sign in to listen to the pronunciations and decide them."),
    h("div", { class: "field" }, pw), h("button", { class: "btn primary", onclick: go }, "Sign in"), state.message && h("div", { class: "msg" }, state.message.text));
}

function header() {
  const tab = (id, label) => h("button", { "aria-current": state.tab === id ? "page" : null, onclick: () => { state.tab = id; state.message = null; loadTab(); } }, label);
  const open = state.questions ? state.questions.open : 0;
  return h("header", {}, h("h1", {}, "Pronunciation Console"), h("nav", {}, tab("review", "Review"), tab("questions", h("span", {}, "Questions", open ? h("span", { class: "count" }, open) : null)), tab("smith", "Wordsmith"), tab("book", "Book"), tab("rules", "Rules")), h("div", { class: "spacer" }),
    h("button", { class: "btn", onclick: async () => { await api("/api/logout", "POST"); state.signedIn = false; render(); } }, "Sign out"));
}
async function loadTab() {
  try { if (state.tab === "book") state.book = await api("/api/book"); if (state.tab === "rules") state.rules = await api("/api/rules"); if (state.tab === "questions") await loadQuestions(); } catch (e) { say(e.message); }
  render();
}

const LANGS = { english: "English", zulu: "Zulu", sepedi: "Sepedi", setswana: "Setswana", sesotho: "Sesotho", afrikaans: "Afrikaans", other: "Another language" };
const typed = {};                       // what is typed for each stretch, kept while the page redraws
async function sendAnswer(body, text) {
  try {
    state.questions = await api("/api/answer", "POST", { job: state.job, run: state.run, ...body });
    state.words = await api(`/api/words?job=${state.job}&run=${encodeURIComponent(state.run)}`);
    say(text, true);
  } catch (e) { say(e.message); }
}
const answerWord = (word, language) => sendAnswer({ kind: "word", word, language }, `Kept: “${word}” is ${LANGS[language]}. It is settled in every job from now on` + (language === "english" ? ", and the rules spell it for the voice." : ", and left as it is written."));
async function revokeWord(word) {
  try { state.questions = await api("/api/answer-revoke", "POST", { job: state.job, run: state.run, word }); say(`Taken back: “${word}” will be asked about again.`, true); } catch (e) { say(e.message); }
}
async function hearStretch(q, btn) {
  btn.disabled = true;
  try { const r = await api("/api/segment-audio", "POST", { job: state.job, id: q.id }); await playUrl(r.url); } catch (e) { say(e.message); }
  btn.disabled = false;
}
const clock = (ms) => `${Math.floor(ms / 60000)}:${String(Math.floor(ms / 1000) % 60).padStart(2, "0")}`;
function line(label, text, win) { return h("div", { class: "qline" + (win ? " win" : "") }, h("span", { class: "muted small" }, label), h("span", {}, text || "—")); }
function wordCard(q) {
  const cur = q.answer && q.answer.language;
  const status = !q.answer ? "Nobody said this word, but the English dictionary has it, so it may be a word from another language." : q.answer.by_itself
    ? `Settled by itself from your earlier answers: ${LANGS[cur]}. Change it if that is wrong.` : `Your answer: ${LANGS[cur]}.`;
  return h("div", { class: "card q" }, h("h2", {}, q.word), h("div", { class: "small muted" }, status),
    h("div", { class: "qlist" }, line("In the Zulu dub", q.dub), line("English at that moment", q.source)),
    h("div", { class: "muted small" }, "What language is this word?"),
    h("div", { class: "chips gap" }, state.questions.languages.map((l) => h("button", { class: "btn" + (cur === l ? " primary" : ""), onclick: () => answerWord(q.word, l) }, LANGS[l])),
      q.answer && !q.answer.by_itself ? h("button", { class: "btn danger", onclick: () => revokeWord(q.word) }, "Take back") : null));
}
function stretchCard(q) {
  const a = q.answer, box = h("textarea", { rows: 3, placeholder: "What was really said", oninput: (e) => { typed[q.id] = e.target.value; } });
  box.value = typed[q.id] != null ? typed[q.id] : (a && a.choice === "typed" ? a.text : q.model);
  const hear = h("button", { class: "btn", onclick: (e) => hearStretch(q, e.currentTarget) }, "▶ Hear it");
  const pick = (choice, text) => sendAnswer({ kind: "stretch", id: q.id, choice, text }, "Kept. It is applied to this video’s text the next time its dub is made, and the model sees it as an example from now on.");
  return h("div", { class: "card q" }, h("h2", {}, `${clock(q.start_ms)} · ${q.speaker.replace("AZURE_", "speaker ") || "speaker"}`),
    h("div", { class: "small muted" }, a ? `Your answer: ${{ model: "the model’s text", transcript: "the transcript", typed: "your own words" }[a.choice]}.` : `The restored text and YouTube’s captions (${q.track}) differ: ${Math.round(q.share * 100)}% of the words are not in the caption.`),
    h("div", { class: "qlist" }, line("Transcript (recogniser)", q.transcript, a && a.choice === "transcript"), line("Model’s restored text", q.model, a && a.choice === "model"),
      line(`Caption (${q.track})`, q.caption), a && a.choice === "typed" ? line("Your words", a.text, true) : null),
    h("div", { class: "chips gap" }, hear, h("button", { class: "btn", onclick: () => pick("model") }, "The model’s text is right"), h("button", { class: "btn", onclick: () => pick("transcript") }, "The transcript is right")),
    box, h("button", { class: "btn primary", onclick: () => pick("typed", box.value) }, "Use my words"));
}
function questionsTab() {
  const jobSel = h("select", { onchange: async (e) => { state.job = e.target.value; state.run = state.jobs.find((j) => j.job === state.job).runs[0].run; await loadWords(); render(); } },
    state.jobs.map((j) => h("option", { value: j.job, selected: j.job === state.job }, `${short(j.name)} · ${j.created || j.clip}`)));
  const q = state.questions;
  const parts = [h("div", { class: "bar" }, jobSel, h("span", { class: "muted small" }, q ? (q.open ? `${q.open} to answer` : "nothing to answer") : "")), state.message && h("div", { class: "msg" + (state.message.ok ? " ok" : "") }, state.message.text)];
  if (!q) return h("div", { class: "qs" }, ...parts, h("div", { class: "card empty" }, "Looking for what the pipeline could not settle…"));
  parts.push(h("p", { class: "muted" }, "Where the model and YouTube’s captions disagree, or a word of the Zulu dub was never said in the source, the pipeline asks instead of guessing. Every answer is kept as a lesson: the same word is settled in every later video, words that look like ones you answered are settled the same way, and the model is shown your corrections as examples."));
  if (q.words.length) parts.push(h("h3", {}, "Words nobody said"), ...q.words.map(wordCard));
  if (q.stretches.length) parts.push(h("h3", {}, "Where the readings differ"), ...q.stretches.map(stretchCard));
  if (q.resolved.length) parts.push(h("h3", {}, "Settled by itself from your earlier answers"), ...q.resolved.map(wordCard));
  if (q.answered.length) parts.push(h("h3", {}, "Words you answered"), ...q.answered.map(wordCard));
  if (!q.words.length && !q.stretches.length && !q.resolved.length && !q.answered.length) parts.push(h("div", { class: "card empty" }, "Nothing to ask about in this dub."));
  return h("div", { class: "qs" }, ...parts);
}

function reviewTab() {
  const jobSel = h("select", { onchange: async (e) => { state.job = e.target.value; state.run = state.jobs.find((j) => j.job === state.job).runs[0].run; await loadWords(); render(); } },
    state.jobs.map((j) => h("option", { value: j.job, selected: j.job === state.job }, `${short(j.name)} · ${j.created || j.clip}`)));
  const job = state.jobs.find((j) => j.job === state.job);
  const runSel = h("select", { onchange: async (e) => { state.run = e.target.value; await loadWords(); render(); } }, (job ? job.runs : []).map((r) => h("option", { value: r.run, selected: r.run === state.run }, `${r.run} · ${r.at}`)));
  const filterSel = h("select", { onchange: (e) => { state.filter = e.target.value; render(); } },
    [["all", "All words"], ["unreviewed", "Still to hear"], ["approved", "Approved"], ["custom", "Custom spellings"]].map(([v, l]) => h("option", { value: v, selected: v === state.filter }, l)));
  const shown = state.words.filter((w) => state.filter === "all" || w.status === state.filter);
  const left = h("div", { class: "card list" }, shown.length ? shown.map((w) => h("button", { class: "row", "aria-current": w.word === state.word ? "true" : null, onclick: () => openWord(w.word) },
      h("span", { class: "w" }, w.word), h("span", { class: "badge " + w.status }, w.status === "unreviewed" ? "to hear" : w.status), h("span", { class: "s" }, w.spelling), h("span", { class: "small muted" }, w.times + "×")))
    : h("div", { class: "empty" }, state.jobs.length ? "No English words in this dub." : "No dubs to review yet."));
  const done = state.words.filter((w) => w.status !== "unreviewed").length;
  return h("div", {}, h("div", { class: "bar" }, jobSel, runSel, filterSel, h("span", { class: "muted small" }, state.words.length ? `${done} of ${state.words.length} decided` : "")),
    state.message && h("div", { class: "msg" + (state.message.ok ? " ok" : "") }, state.message.text),
    h("div", { class: "cols" }, left, h("div", {}, detailCard())));
}

function detailCard() {
  const d = state.detail;
  if (!state.word) return h("div", { class: "card empty" }, "Choose a word to hear what the rules make of it.");
  if (!d) return h("div", { class: "card empty" }, "Loading…");
  const spelling = d.committed && d.committed.spelling ? Object.values(d.committed.spelling)[0] : d.rules;
  const card = h("div", { class: "card" }, h("h2", {}, d.word),
    h("div", { class: "muted small" }, `sounds from ${READINGS[d.reading] || d.reading} · English sounds: ${d.english || "—"}`),
    sourcePanel(state.source, false),
    h("h3", {}, "The rules say"), h("div", { class: "big mono" }, d.rules),
    d.derivation.length ? h("ul", { class: "notes" }, d.derivation.map((n) => h("li", {}, n))) : null);
  if (d.committed) {
    card.append(h("h3", {}, "Committed"), h("div", {}, h("span", { class: "badge " + d.committed.kind }, d.committed.kind), " ", h("span", { class: "mono" }, Object.entries(d.committed.spelling).map(([v, s]) => (v === "*" ? "" : voiceName(v) + ": ") + s).join("   ")),
      d.committed.note ? h("div", { class: "muted small" }, d.committed.note) : null,
      h("div", { style: "margin-top:8px" }, h("button", { class: "btn danger", onclick: async () => { await api("/api/book/revoke", "POST", { word: d.word }); await refreshAfterDecision("Taken back: the rules decide this word again."); } }, "Take back"))));
  }
  card.append(h("h3", {}, "Hear it in its sentence"));
  card.append(h("div", { class: "plays" }, d.voices.map((v) => playBtn(spelling, v))));
  d.samples.slice(0, 2).forEach((s) => card.append(h("div", { class: "sentence" }, `${voiceName(s.voice)} · ${s.unit}: ${s.text}`)));
  card.append(h("div", { class: "decide" },
    h("button", { class: "btn primary", onclick: async () => { try { await api("/api/correct", "POST", { job: state.job, run: state.run, word: d.word, spelling }); await refreshAfterDecision(`Committed: ${d.word} is said as “${spelling}” in every job from now on.`); } catch (e) { say(e.message); } } }, "✔ Correct — commit it"),
    h("button", { class: "btn", onclick: () => { state.showAlts = true; render(); prefetchAhead(true).then(watchReady); } }, "✘ Not correct — hear alternatives")));
  if (state.showAlts) card.append(alternatives(d));
  return card;
}

function alternatives(d) {
  const box = h("div", {}, h("h3", {}, "Alternatives from the same rules"));
  const note = h("textarea", { rows: 2, placeholder: "Why? What did you hear wrong? (optional: it helps the rules learn)", oninput: (e) => { state.note = e.target.value; } });
  note.value = state.note;
  const custom = h("input", { type: "text", placeholder: "or type your own spelling", value: state.custom, oninput: (e) => { state.custom = e.target.value; } });
  const decision = h("div", {});
  if (!state.picked) decision.append(h("div", { class: "muted small" }, "Choose the spelling that sounds right: one of these, or your own."));
  else {
    decision.append(h("div", {}, "The rules wrote ", h("span", { class: "mono" }, d.rules), " — you chose ", h("span", { class: "mono big" }, state.picked)),
      h("p", { class: "muted small" }, "It is committed for this word in every job. It is also kept as a lesson for the rules: the sounds the dictionary heard, how the rules built theirs, and what you wrote instead. The Rules page shows what your spellings have in common."),
      note, h("div", { style: "margin-top:10px" }, h("button", { class: "btn primary", onclick: async () => {
        try { await api("/api/correct", "POST", { job: state.job, run: state.run, word: d.word, spelling: state.picked, note: state.note });
          await refreshAfterDecision(`Committed: ${d.word} is said as “${state.picked}” in every job from now on, and the rules have the lesson.`); } catch (e) { say(e.message); } } }, "Commit this spelling")));
  }
  d.alternatives.forEach((a, i) => {
    const row = h("div", { class: "alt", "data-picked": state.picked === a.spelling ? "true" : "false" },
      h("div", {}, h("span", { class: "sp mono" }, a.spelling), i === 0 ? h("span", { class: "tag" }, "the rules’ choice") : null, h("div", {}, a.differences.map((x) => h("span", { class: "tag" }, `${x.label}: ${plain(x.other_means)}`)))),
      h("div", { class: "plays" }, d.voices.map((v) => playBtn(a.spelling, v)),
        h("button", { class: "btn", onclick: () => { state.picked = a.spelling; render(); } }, "This one")));
    box.append(row);
  });
  const customRow = h("div", { class: "alt" }, h("div", {}, custom), h("div", { class: "plays" },
    d.voices.map((v) => h("button", { class: "btn", onclick: () => state.custom.trim() && play(state.custom.trim(), v, 0) }, "▶ " + voiceName(v))),
    h("button", { class: "btn", onclick: () => { if (state.custom.trim()) { state.picked = state.custom.trim(); render(); } } }, "This one")));
  box.append(customRow, h("h3", {}, "Your decision"), decision);
  return box;
}

const smith = { source: null, text: "", asTyped: false, result: null, spelling: {}, urls: {}, loading: {}, error: null, busy: false, run: 0 };
function playUrl(url) {
  return new Promise((resolve) => { player.onended = resolve; player.onerror = resolve; player.src = url; player.play().catch(resolve); });
}
async function smithPlayBoth() { for (const v of state.voices) if (smith.urls[v]) await playUrl(smith.urls[v]); }
async function smithHear(voices) {                          // makes the clips one voice after the other, so the first can be heard while the second is made
  const run = ++smith.run;
  voices.forEach((v) => { smith.loading[v] = true; delete smith.urls[v]; });
  render();
  for (const v of voices) {
    try { smith.urls[v] = (await api("/api/smith-audio", "POST", { spelling: smith.spelling[v], voice: v })).url; smith.error = null; }
    catch (e) { smith.error = e.message; }
    smith.loading[v] = false;
    if (run !== smith.run) return;
    render();
  }
  if (!smith.error) smithPlayBoth();
}
async function smithRefresh() {                              // the result again (the book changed), without touching the boxes or the clips
  try { smith.result = await api("/api/smith", "POST", { text: smith.result.text, as_typed: smith.asTyped }); smith.error = null; } catch (e) { smith.error = e.message; }
  render();
}
async function smithRun() {
  const text = smith.text.trim();
  if (!text) return;
  smith.busy = true; smith.error = null; smith.note = null; smith.urls = {}; render();
  try {
    smith.result = await api("/api/smith", "POST", { text, as_typed: smith.asTyped });
    smith.spelling = { ...smith.result.spelling };
  } catch (e) { smith.error = e.message; smith.result = null; smith.busy = false; render(); return; }
  smith.busy = false;
  smith.source = null;
  const one = smith.result.pieces.length === 1 && smith.result.pieces[0].word;
  if (one) api(`/api/source?word=${encodeURIComponent(one)}`).then((r) => { if (smith.result && smith.result.text === text) { smith.source = r.occurrences; render(); } }).catch(() => { smith.source = []; render(); });
  smithHear(state.voices);
}
const SOURCES = { book: "from your pronunciation book", rules: "spelled by the rules from the dictionary’s sounds", "as typed": "said as typed (no English reading, or “say as typed” is on)", mixed: "several words, each spelled on its own" };
function smithTab() {
  const input = h("input", { type: "text", id: "smith-text", placeholder: "an English word, a name, an initialism or a short phrase", maxlength: "80", value: smith.text, autofocus: true,
    oninput: (e) => { smith.text = e.target.value; } });
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") smithRun(); });
  const r = smith.result;
  const out = h("div", {});
  if (smith.error) out.append(h("div", { class: "msg" }, smith.error));
  if (smith.note && !smith.error) out.append(h("div", { class: "msg ok" }, smith.note));
  if (r) {
    out.append(h("div", { class: "muted small", style: "margin:12px 0 4px" }, `“${r.text}” — ${SOURCES[r.source] || r.source}`));
    const cards = h("div", { class: "voices" }, state.voices.map((v) => {
      const field = h("input", { type: "text", class: "mono", value: smith.spelling[v] || "", "aria-label": voiceName(v) + " spelling", oninput: (e) => { smith.spelling[v] = e.target.value; } });
      field.addEventListener("keydown", (e) => { if (e.key === "Enter") smithHear([v]).then(() => {}); });
      return h("div", { class: "card" }, h("h3", { style: "margin-top:0" }, voiceName(v)), field,
        h("div", { class: "plays" }, h("button", { class: "btn", disabled: !!smith.loading[v], onclick: () => smithHear([v]) }, "Say this spelling"),
          smith.urls[v] ? h("audio", { controls: true, src: smith.urls[v], preload: "auto" }) : (smith.loading[v] ? h("span", { class: "muted small" }, "Making the clip…") : null)));
    }));
    out.append(cards, h("div", { class: "plays" }, h("button", { class: "btn", disabled: !state.voices.every((v) => smith.urls[v]), onclick: smithPlayBoth }, "▶ Play both")));
    if (r.pieces.length === 1 && r.pieces[0].word) out.append(sourcePanel(smith.source, true));
    if (r.committable) {
      const word = r.pieces[0].word;
      out.append(h("div", { class: "decide" },
        h("button", { class: "btn primary", onclick: async () => {
          try {
            const e = await api("/api/smith-commit", "POST", { word, spelling: smith.spelling });
            smith.note = `Committed: ${word} is said as “${[...new Set(Object.values(e.spelling))].join("” / “")}” in every job from now on${e.kind === "custom" ? ", and the rules have the lesson" : ""}.`;
            await smithRefresh();
          } catch (err) { smith.error = err.message; render(); }
        } }, "✔ Correct — commit it"),
        r.source === "book" ? h("button", { class: "btn danger", onclick: async () => {
          try { await api("/api/book/revoke", "POST", { word }); smith.note = "Taken back: the rules decide this word again."; await smithRefresh(); } catch (err) { smith.error = err.message; render(); }
        } }, "Take back") : null),
        h("div", { class: "muted small" }, r.source === "book" ? "This word is committed in your pronunciation book." : "Commits what is in the boxes above, for each voice."));
    } else {
      out.append(h("div", { class: "muted small", style: "margin-top:8px" }, "Only a single bare word can be committed (no phrase, Zulu prefix or punctuation)."));
    }
    if (r.pieces.length > 1) {
      out.append(h("h3", {}, "Word by word"), h("table", {}, h("tbody", {}, r.pieces.map((p) => h("tr", {}, h("td", {}, p.token), h("td", { class: "mono" }, [...new Set(state.voices.map((v) => p.by_voice[v]))].join(" / ")), h("td", { class: "muted small" }, p.source))))));
    } else if (r.pieces[0].derivation.length) {
      out.append(h("h3", {}, "How the rules made it"), h("div", { class: "muted small" }, `${READINGS[r.pieces[0].reading] || r.pieces[0].reading}${r.pieces[0].english ? " · English sounds: " + r.pieces[0].english : ""}`),
        h("ul", { class: "notes" }, r.pieces[0].derivation.map((n) => h("li", {}, n))));
    }
    if (r.alternatives.length > 1) {
      out.append(h("h3", {}, "Other spellings from the same rules"), h("div", { class: "chips" }, r.alternatives.map((a, i) =>
        h("button", { class: "btn chip", title: a.derivation.join("; "), onclick: () => { state.voices.forEach((v) => { smith.spelling[v] = a.spelling; }); smithHear(state.voices); } },
          h("span", { class: "mono" }, a.spelling), i === 0 ? h("span", { class: "tag" }, "the rules’ choice") : null, a.differences.map((x) => h("span", { class: "tag" }, `${x.label}: ${plain(x.other_means)}`))))));
    }
  }
  return h("div", { class: "smith" }, h("div", { class: "card" }, h("h2", {}, "Wordsmith"),
    h("p", { class: "muted small" }, "Type a word and hear how Thando and Themba say it. The rules (or your pronunciation book) make the spelling; change the spelling in a box and say it again to try your own."),
    h("div", { class: "bar", style: "margin-bottom:0" }, h("div", { style: "flex:1;min-width:220px" }, input), h("button", { class: "btn primary", disabled: smith.busy, onclick: smithRun }, smith.busy ? "Working…" : "Hear it")),
    h("label", { class: "opt" }, h("input", { type: "checkbox", checked: smith.asTyped, onchange: (e) => { smith.asTyped = e.target.checked; } }), h("span", {}, "Say it as typed (for a Zulu word the rules should leave alone)")), out));
}

function bookTab() {
  const b = state.book; if (!b) return h("div", { class: "empty" }, "Loading…");
  const rows = b.entries.map((e) => h("tr", {}, h("td", {}, h("b", {}, e.word)), h("td", { class: "mono" }, Object.entries(e.spelling).map(([v, s]) => (v === "*" ? "" : voiceName(v) + ": ") + s).join(" · ")),
    h("td", {}, h("span", { class: "badge " + e.kind }, e.kind)), h("td", { class: "mono muted" }, e.rule_spelling || "—"), h("td", { class: "muted small" }, (e.decided_at || "").slice(0, 16).replace("T", " "), e.note ? h("div", {}, e.note) : null),
    h("td", {}, h("button", { class: "btn danger", onclick: async () => { await api("/api/book/revoke", "POST", { word: e.word }); state.book = await api("/api/book"); render(); } }, "Take back"))));
  return h("div", {}, h("div", { class: "card" }, h("h2", {}, "Committed pronunciations"),
    h("p", { class: "muted small" }, `${b.tally.approved_as_ruled} approved as the rules wrote them · ${b.tally.custom} custom spellings. Every job uses these before the rules; every custom spelling is also a lesson for them.`),
    b.entries.length ? h("table", {}, h("thead", {}, h("tr", {}, ["Word", "Said as", "Kind", "Rules said", "Decided", ""].map((t) => h("th", {}, t)))), h("tbody", {}, rows)) : h("div", { class: "empty" }, "Nothing committed yet.")),
    h("div", { class: "card", style: "margin-top:16px" }, h("h2", {}, "Lessons for the rules: every custom spelling"), b.feedback.length ? h("table", {}, h("thead", {}, h("tr", {}, ["Word", "Rules said", "You chose", "What changed", "Your note"].map((t) => h("th", {}, t)))),
      h("tbody", {}, b.feedback.map((f) => h("tr", {}, h("td", {}, f.word), h("td", { class: "mono" }, f.rule_spelling), h("td", { class: "mono" }, f.chosen),
        h("td", { class: "small" }, (f.diff || []).map((x) => h("div", {}, `${x.from} → ${x.to}  (${x.shape}, ${x.where})`)), f.differences.map((d) => h("div", {}, d.label + " → " + plain(d.other_means)))), h("td", { class: "small muted" }, f.note || ""))))) : h("div", { class: "empty" }, "No custom spellings yet.")));
}

function patternsCard(r) {
  const a = r.analysis, res = a && a.result;
  const ask = h("button", { class: "btn primary", disabled: !r.lessons || state.asking, onclick: async () => {
    state.asking = true; render();
    try { await api("/api/learn", "POST", {}); state.rules = await api("/api/rules"); say("The model has looked at your spellings.", true); } catch (e) { say(e.message); }
    state.asking = false; render();
  } }, state.asking ? "The model is reading your spellings…" : "Ask the model why");
  return h("div", { class: "card", style: "margin-bottom:16px" }, h("h2", {}, "What your spellings are teaching the rules"),
    h("p", { class: "muted small" }, r.lessons ? `${r.lessons} custom spelling(s) kept. Each one carries the sounds the dictionary heard, how the rules built their spelling, and what you wrote instead.` : "No custom spelling yet. When you commit a spelling that is not the rules’, it is kept here as a lesson."),
    r.patterns.length ? h("table", {}, h("thead", {}, h("tr", {}, ["Change (syllable shape, where)", "Words", "Examples"].map((t) => h("th", {}, t)))), h("tbody", {}, r.patterns.map((p) =>
      h("tr", {}, h("td", { class: "mono" }, p.pattern), h("td", {}, String(p.words.length)), h("td", { class: "small muted" }, p.examples.slice(0, 4).map((x) => h("div", {}, x))))))) : null,
    h("div", { style: "margin:12px 0" }, ask, " ", h("span", { class: "muted small" }, "One request to the model; it changes nothing by itself.")),
    res ? h("div", {}, h("h3", {}, `The model’s reading · ${a.at.slice(0, 16).replace("T", " ")} · ${a.lessons} spelling(s)`), h("p", {}, res.summary),
      (res.findings || []).map((f) => h("div", { class: "alt finding" }, h("div", {}, h("b", {}, f.pattern), h("span", { class: "tag" }, "confidence: " + f.confidence), f.open_choice ? h("span", { class: "tag" }, "open choice: " + f.open_choice) : null),
        h("div", {}, h("span", { class: "muted" }, "Why: "), f.why), h("div", {}, h("span", { class: "muted" }, "Change: "), f.proposed_change), h("div", { class: "muted small" }, "Words: " + (f.words || []).join(", "))))) : null);
}

function rulesTab() {
  const r = state.rules; if (!r) return h("div", { class: "empty" }, "Loading…");
  const cards = r.rules.map((rule) => {
    const current = rule.default == null ? rule.as_written : rule.default;
    let picked = current;
    const wantedKeys = Object.keys(rule.wanted);
    return h("div", { class: "card", style: "margin-bottom:12px" }, h("b", {}, rule.label),
      rule.options.map((o) => h("label", { class: "opt" }, h("input", { type: "radio", name: "r-" + rule.rule, checked: o.value === current, onchange: () => { picked = o.value; } }),
        h("span", {}, o.means, o.value === rule.as_written ? h("span", { class: "tag" }, "as written") : null, rule.default != null && o.value === rule.default ? h("span", { class: "tag" }, "your default") : null))),
      wantedKeys.length ? h("div", { class: "small", style: "margin:6px 0" }, wantedKeys.map((k) => { const opt = rule.options.find((o) => String(o.value) === k); return h("div", {}, `You wanted “${opt ? opt.means : k}” for ${rule.wanted[k]} word(s): ${(rule.words[k] || []).join(", ")}`); })) : h("div", { class: "muted small" }, "No word has asked for a change here."),
      h("button", { class: "btn primary", onclick: async () => { const pick = picked === rule.as_written ? null : picked; state.rules = await api("/api/rules", "POST", { rule: rule.rule, pick, note: "set in the console" }); say(`Saved. Spellings made from now on use it${pick == null ? " as the rules were written" : ""}; committed words do not change.`, true); } }, "Make this the default"));
  });
  const learned = patternsCard(r);
  return h("div", {}, state.message && h("div", { class: "msg ok" }, state.message.text), learned, h("p", { class: "muted" }, "The rules have open choices. Changing a default changes the spelling the rules make for every word that has not been committed. Committed words never change."), cards,
    h("div", { class: "card" }, h("h2", {}, "History"), r.history.length ? h("table", {}, h("tbody", {}, r.history.map((x) => h("tr", {}, h("td", { class: "muted small" }, x.at.slice(0, 16).replace("T", " ")), h("td", {}, x.rule), h("td", { class: "mono" }, String(x.from) + " → " + String(x.to)), h("td", { class: "muted small" }, x.note))))) : h("div", { class: "empty" }, "No changes yet.")));
}

function render() {
  buttons = [];
  app.replaceChildren();
  if (!state.signedIn) { app.append(login()); return; }
  app.append(header(), h("main", {}, state.tab === "review" ? reviewTab() : state.tab === "questions" ? questionsTab() : state.tab === "smith" ? smithTab() : state.tab === "book" ? bookTab() : rulesTab()));
  if (state.tab === "review" && state.detail) watchReady();
}
start();
</script>
</body>
</html>
"""
