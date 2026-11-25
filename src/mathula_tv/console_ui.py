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
.commentbox { margin-top: 12px; padding-top: 10px; border-top: 1px dashed var(--line); }
.commentbox textarea { margin: 4px 0 6px; }
@media (max-width: 600px) { .qline { grid-template-columns: 1fr; } }
nav { overflow-x: auto; max-width: 100%; scrollbar-width: none; } nav::-webkit-scrollbar { display: none; } nav button { white-space: nowrap; }
.cols > *, .alt > *, .src > * { min-width: 0; } .card, .row { overflow-wrap: anywhere; }
.tablewrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
.plays .btn, .decide .btn, .chips .btn { touch-action: manipulation; }
@media (max-width: 700px) {                       /* a phone: the menu gets its own row and slides sideways, inputs are 16px so iOS does not zoom in, buttons are finger-sized */
  header { padding: 8px 12px; gap: 6px 10px; }
  header h1 { flex: 1; font-size: 15px; } .spacer { display: none; }
  header > .btn { order: 2; padding: 6px 12px; } header nav { order: 3; flex: 1 0 100%; }
  nav button { padding: 9px 14px; }
  main { padding: 12px; }
  input, select, textarea, input[type=text], input[type=password] { font-size: 16px; }
  .btn { min-height: 44px; padding: 10px 14px; }
  .card { padding: 14px; }
  .bar select { width: 100%; }
  .list { max-height: 46vh; }
  .alt { grid-template-columns: 1fr; }
  .big { font-size: 24px; }
  .plays, .decide { gap: 8px; } .decide .btn, .plays .btn { flex: 1 1 auto; }
  .login { margin: 8vh 16px; max-width: none; }
  th, td { padding: 6px 8px; }
  table { display: block; overflow-x: auto; -webkit-overflow-scrolling: touch; }
  .src { grid-template-columns: 1fr; } .src .btns { flex-wrap: wrap; } .src .btns .btn { flex: 1 1 auto; }
}
.studio { max-width: 1000px; margin: 0 auto; display: flex; flex-direction: column; gap: 12px; }
.studio .bar { position: sticky; top: 0; z-index: 3; background: var(--bg); padding: 8px 0; display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.studio audio { flex: 1 1 320px; min-width: 0; }
.srow { display: grid; grid-template-columns: 54px 1fr; gap: 4px 10px; padding: 8px 6px; border-bottom: 1px solid var(--line); }
.srow .t { color: var(--muted); font-size: 12px; font-family: ui-monospace, Consolas, monospace; padding-top: 6px; }
.srow.now { background: var(--panel); }
.srow.picked { box-shadow: 0 0 0 1px var(--accent) inset; border-radius: 8px; }
.word { padding: 3px 7px; margin: 2px 1px; border: 1px solid transparent; border-radius: 6px; background: none; color: inherit; font: inherit; cursor: pointer; line-height: 1.35; }
.word:hover { border-color: var(--line); }
.word.now { background: var(--accent); color: var(--bg); }
.word.sel { outline: 2px solid var(--accent); }
.srow .tag.done { background: var(--good-bg); color: var(--good); }
.drawpanel { position: sticky; bottom: 0; z-index: 3; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; box-shadow: 0 -4px 14px rgba(0,0,0,.12); }
.unitline { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); margin: 14px 6px 0; }
@media (max-width: 700px) { .srow { grid-template-columns: 1fr; } .word { padding: 8px 9px; } }
.player { position: sticky; top: 0; z-index: 4; background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; display: flex; flex-direction: column; gap: 8px; }
.player .row1 { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.player .time { font-family: ui-monospace, Consolas, monospace; font-size: 15px; min-width: 120px; }
.seek { width: 100%; height: 44px; margin: 0; accent-color: var(--accent); cursor: pointer; }
.stripbox { position: relative; }
.stripbox::after { content: ""; position: absolute; left: 50%; top: 6px; bottom: 6px; width: 2px; margin-left: -1px; background: var(--accent); opacity: .55; pointer-events: none; border-radius: 2px; }
.strip.grabbing { cursor: grabbing; user-select: none; }
.strip { cursor: grab; scrollbar-width: thin; display: flex; align-items: center; gap: 2px; overflow-x: auto; white-space: nowrap; padding: 14px 10px; border: 1px solid var(--line); border-radius: 12px; background: var(--panel); -webkit-overflow-scrolling: touch; }
.strip .word { flex: 0 0 auto; font-size: 18px; padding: 8px 11px; }
.strip .unitchip { flex: 0 0 auto; font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); padding: 0 10px 0 16px; border-left: 2px solid var(--line); margin-left: 8px; }
.strip .stop { flex: 0 0 auto; width: 10px; }
.strip .pad { flex: 0 0 auto; width: 50vw; }
.drawpanel2 { border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; background: var(--panel); }
.word.flagged { text-decoration: underline wavy var(--warn); text-underline-offset: 4px; }
.repairbox { margin-top: 10px; padding: 10px 12px; border: 1px dashed var(--line); border-radius: 10px; display: flex; flex-direction: column; gap: 6px; }
.unittext { line-height: 1.75; font-size: 17px; max-height: 8.5em; overflow-y: auto; margin: 6px 0 8px; padding-right: 6px; position: relative; }
.unittext .uw { padding: 1px 0; }
.unittext .inspan { background: var(--good-bg); color: var(--good); border-radius: 3px; box-shadow: 4px 0 0 var(--good-bg), -1px 0 0 var(--good-bg); }
.unittext .clicked { font-weight: 700; text-decoration: underline; text-underline-offset: 3px; }
.drawpanel2 mark { background: var(--good-bg); color: var(--good); border-radius: 4px; padding: 0 4px; font-weight: 600; }
.draws3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 12px; }
.draws3 .d { border: 1px solid var(--line); border-radius: 10px; padding: 10px; display: flex; flex-direction: column; gap: 8px; }
.draws3 .d[data-live="true"] { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent) inset; }
.draws3 .btn { width: 100%; }
@media (max-width: 700px) { .draws3 { grid-template-columns: 1fr; } .strip .word { padding: 12px 12px; } }
.seek { -webkit-appearance: none; appearance: none; background: transparent; height: 44px; }
.seek::-webkit-slider-runnable-track { height: 16px; border-radius: 8px; background: var(--line); }
.seek::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 30px; height: 30px; margin-top: -7px; border-radius: 50%; background: var(--accent); border: 3px solid var(--panel); box-shadow: 0 1px 4px rgba(0,0,0,.4); }
.seek::-moz-range-track { height: 16px; border-radius: 8px; background: var(--line); }
.seek::-moz-range-thumb { width: 26px; height: 26px; border-radius: 50%; background: var(--accent); border: 3px solid var(--panel); }
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
    say(null); playUrl(r.url);
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
  return h("header", {}, h("h1", {}, "Pronunciation Console"), h("nav", {}, tab("review", "Review"), tab("questions", h("span", {}, "Questions", open ? h("span", { class: "count" }, open) : null)), tab("studio", "Studio"), tab("draws", "Draws"), tab("smith", "Wordsmith"), tab("book", "Book"), tab("rules", "Rules")), h("div", { class: "spacer" }),
    h("button", { class: "btn", onclick: async () => { await api("/api/logout", "POST"); state.signedIn = false; render(); } }, "Sign out"));
}
async function loadTab() {
  try { if (state.tab === "book") state.book = await api("/api/book"); if (state.tab === "rules") state.rules = await api("/api/rules"); if (state.tab === "questions") await loadQuestions(); if (state.tab === "draws") await loadDraws(); if (state.tab === "studio" && !studio.data && !studio.loading) loadStudio(); } catch (e) { say(e.message); }
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
async function playApi(path, body, btn) {
  btn.disabled = true; const label = btn.textContent; btn.textContent = "…";
  try { const r = await api(path, "POST", body); await playUrl(r.url); } catch (e) { say(e.message); }
  btn.textContent = label; btn.disabled = false;
}
const clock = (ms) => `${Math.floor(ms / 60000)}:${String(Math.floor(ms / 1000) % 60).padStart(2, "0")}`;
const cardId = (q) => q.kind + ":" + q.id;
const comments = {};                    // what is written in each comment box, kept while the page redraws
function hearRow(q) {                   // the source audio of the stretch, and the dub as its voice reads it with nothing applied to it
  const b = [];
  if (q.kind === "stretch" || q.start_ms != null) b.push(h("button", { class: "btn", onclick: (e) => playApi("/api/segment-audio", q.kind === "stretch" ? { job: state.job, id: q.id } : { job: state.job, start_ms: q.start_ms, end_ms: q.end_ms }, e.currentTarget) }, "▶ Source audio"));
  if (q.item != null) b.push(h("button", { class: "btn", onclick: (e) => playApi("/api/raw-audio", { job: state.job, run: state.run, item: q.item }, e.currentTarget) }, `▶ Dub, raw TTS (${voiceName(q.voice)})`));
  return b.length ? h("div", { class: "chips gap" }, b) : null;
}
function line(label, text, win) { return h("div", { class: "qline" + (win ? " win" : "") }, h("span", { class: "muted small" }, label), h("span", {}, text || "—")); }
function commentBox(q) {                // the listener's own words to the model about this question: kept, and shown to the model the next time it reads this video
  const box = h("textarea", { rows: 2, placeholder: "Comment for the model: what you know about this (optional). It is kept and shown to the model.", oninput: (e) => { comments[cardId(q)] = e.target.value; } });
  box.value = comments[cardId(q)] != null ? comments[cardId(q)] : (q.comment || "");
  const save = h("button", { class: "btn", onclick: () => sendAnswer({ kind: q.kind, ...(q.kind === "word" ? { word: q.word } : { id: q.id }), comment: box.value }, "Comment kept. The model reads it the next time it works on this video, and on later ones too.") }, "Save comment");
  return h("div", { class: "commentbox" }, h("div", { class: "muted small" }, "Comment for the model"), box, save);
}
const withComment = (q) => (comments[cardId(q)] != null ? { comment: comments[cardId(q)] } : {});          // an answer carries what is written in the box
function wordCard(q) {
  const cur = q.answer && q.answer.language;
  const status = !q.answer ? "Nobody said this word, but the English dictionary has it, so it may be a word from another language." : q.answer.by_itself
    ? `Settled by itself from your earlier answers: ${LANGS[cur]}. Change it if that is wrong.` : `Your answer: ${LANGS[cur]}.`;
  return h("div", { class: "card q" }, h("h2", {}, q.word), h("div", { class: "small muted" }, status),
    h("div", { class: "qlist" }, line("In the Zulu dub", q.dub), line("English at that moment", q.source), q.caption ? line(`YouTube caption (${q.track})`, q.caption) : null),
    hearRow(q),
    h("div", { class: "muted small" }, `What language is “${q.word}”?`),
    h("div", { class: "chips gap" }, state.questions.languages.map((l) => h("button", { class: "btn" + (cur === l ? " primary" : ""), onclick: () => sendAnswer({ kind: "word", word: q.word, language: l, ...withComment(q) }, `Kept: “${q.word}” is ${LANGS[l]}. It is settled in every job from now on` + (l === "english" ? ", and the rules spell it for the voice." : ", and left as it is written.")) }, LANGS[l])),
      q.answer && !q.answer.by_itself ? h("button", { class: "btn danger", onclick: () => revokeWord(q.word) }, "Take back") : null),
    commentBox(q));
}
function stretchCard(q) {
  const a = q.answer, box = h("textarea", { rows: 3, placeholder: "What was really said", oninput: (e) => { typed[q.id] = e.target.value; } });
  box.value = typed[q.id] != null ? typed[q.id] : (a && a.choice === "typed" ? a.text : q.model);
  const pick = (choice, text) => sendAnswer({ kind: "stretch", id: q.id, choice, text, ...withComment(q) }, "Kept. It is applied to this video’s text the next time its dub is made, and the model sees it as an example from now on.");
  return h("div", { class: "card q" }, h("h2", {}, `${clock(q.start_ms)} · ${q.speaker.replace("AZURE_", "speaker ") || "speaker"}`),
    h("div", { class: "small muted" }, a ? `Your answer: ${{ model: "the model’s text", transcript: "the transcript", typed: "your own words" }[a.choice]}.` : `The restored text and YouTube’s captions (${q.track}) differ: ${Math.round(q.share * 100)}% of the words are not in the caption.`),
    h("div", { class: "qlist" }, line("Transcript (recogniser)", q.transcript, a && a.choice === "transcript"), line("Model’s restored text", q.model, a && a.choice === "model"),
      line(`YouTube caption (${q.track})`, q.caption), q.dub ? line("The dub (Zulu)", q.dub) : null, a && a.choice === "typed" ? line("Your words", a.text, true) : null),
    hearRow(q),
    h("div", { class: "chips gap" }, h("button", { class: "btn", onclick: () => pick("model") }, "The model’s text is right"), h("button", { class: "btn", onclick: () => pick("transcript") }, "The transcript is right")),
    box, h("button", { class: "btn primary", onclick: () => pick("typed", box.value) }, "Use my words"), commentBox(q));
}
const saidText = {};                    // what is typed as "what was said" for each of the agent's questions, kept while the page redraws
function evidenceCard(q) {              // something the agent said it needed to know and was not given: the source audio, what the recogniser heard there, and your answer
  const a = q.answer, box = h("textarea", { rows: 3, placeholder: "What was really said (or what you know about it)", oninput: (e) => { saidText[q.id] = e.target.value; } });
  box.value = saidText[q.id] != null ? saidText[q.id] : (a ? a.text : "");
  const save = () => sendAnswer({ kind: "evidence", id: q.id, text: box.value, ...withComment(q) }, "Kept. The agent is shown your answer the next time it works on this video, and it is recorded against its question.");
  return h("div", { class: "card q" }, h("h2", {}, q.start_ms != null ? `${clock(q.start_ms)} · ${q.unit}` : q.unit),
    h("div", { class: "small muted" }, a ? "Your answer is kept." : "The agent asked for this and had nothing to go on."),
    h("div", { class: "qlist" }, line("The agent wanted", q.wanted, true), line("Why", q.why), q.transcript ? line("Recogniser heard", q.transcript) : null, q.caption ? line(`YouTube caption (${q.track})`, q.caption) : null,
      q.start_ms != null && q.located === false ? line("Note", "The words it named were not found in the source words: this is the start of its unit.") : null),
    hearRow(q), box, h("button", { class: "btn primary", onclick: save }, "Save my answer"), commentBox(q));
}
function questionsTab() {
  const jobSel = h("select", { onchange: async (e) => { state.job = e.target.value; state.run = state.jobs.find((j) => j.job === state.job).runs[0].run; await loadWords(); render(); } },
    state.jobs.map((j) => h("option", { value: j.job, selected: j.job === state.job }, `${short(j.name)} · ${j.created || j.clip}`)));
  const q = state.questions;
  const parts = [h("div", { class: "bar" }, jobSel, h("span", { class: "muted small" }, q ? (q.open ? `${q.open} to answer` : "nothing to answer") : "")), state.message && h("div", { class: "msg" + (state.message.ok ? " ok" : "") }, state.message.text)];
  if (!q) return h("div", { class: "qs" }, ...parts, h("div", { class: "card empty" }, "Looking for what the pipeline could not settle…"));
  parts.push(h("p", { class: "muted small" }, "Each question has the source audio, YouTube’s caption and the dub read raw by its voice (no respelling). Use the comment box to tell the model anything you know: it is kept and shown to the model."));
  parts.push(h("p", { class: "muted" }, "Where the model and YouTube’s captions disagree, or a word of the Zulu dub was never said in the source, the pipeline asks instead of guessing. Every answer is kept as a lesson: the same word is settled in every later video, words that look like ones you answered are settled the same way, and the model is shown your corrections as examples."));
  if (q.words.length) parts.push(h("h3", {}, "Words nobody said"), ...q.words.map(wordCard));
  if (q.stretches.length) parts.push(h("h3", {}, "Where the readings differ"), ...q.stretches.map(stretchCard));
  if ((q.evidence || []).length) parts.push(h("h3", {}, "What the agent could not find out"), ...q.evidence.map(evidenceCard));
  if (q.resolved.length) parts.push(h("h3", {}, "Settled by itself from your earlier answers"), ...q.resolved.map(wordCard));
  if (q.answered.length) parts.push(h("h3", {}, "Words you answered"), ...q.answered.map(wordCard));
  if (!q.words.length && !q.stretches.length && !(q.evidence || []).length && !q.resolved.length && !q.answered.length) parts.push(h("div", { class: "card empty" }, "Nothing to ask about in this dub."));
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
// A phone lets a page start sound only inside a tap, and every clip here is started after a wait for the server, so the first tap on the page
// "unlocks" the one shared player with a moment of silence; after that the clips can follow the tap by themselves (iOS Safari and Chrome on Android).
const SILENCE = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAIA+AAACABAAZGF0YQAAAAA=";
let unlocked = false;
function unlockPlayer() {
  if (unlocked) return;
  unlocked = true;
  try { player.src = SILENCE; const p = player.play(); if (p && p.catch) p.catch(() => { unlocked = false; }); } catch (e) { unlocked = false; }
}
for (const ev of ["touchend", "pointerdown", "click"]) document.addEventListener(ev, unlockPlayer, { capture: true, passive: true });
function playUrl(url, throwOnBlock) {
  return new Promise((resolve, reject) => {
    player.onended = resolve; player.onerror = () => resolve();
    player.src = url;
    const p = player.play();
    if (p && p.catch) p.catch((e) => { if (e && e.name === "NotAllowedError") { const m = "The phone blocked the sound. Tap the button once more to play it."; throwOnBlock ? reject(new Error(m)) : (say(m), resolve()); } else resolve(); });
  });
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


const drawPick = {};                    // the draw number chosen for each sentence, kept while the page redraws
async function loadDraws() {
  state.draws = null;
  if (!state.job) return;
  try { state.draws = await api(`/api/draws?job=${state.job}&run=${encodeURIComponent(state.run)}`); } catch (e) { say(e.message); }
}
async function drawHear(u, s, btn) {
  const n = drawPick[u.index + ":" + s.index]; btn.disabled = true; const label = btn.textContent; btn.textContent = "Making it… (up to a minute)";
  try {
    for (let i = 0; i < 60; i++) {
      const r = await api("/api/draw-hear", "POST", { job: state.job, run: state.run, unit: u.index, span: s.index, retake: n });
      if (r.url) { btn.textContent = label; btn.disabled = false; await playUrl(r.url); return; }
      await new Promise((ok) => setTimeout(ok, 3000));
    }
    say("The voice took too long. Try again in a moment.");
  } catch (e) { say(e.message); }
  btn.textContent = label; btn.disabled = false;
}
async function drawCommit(u, s) {
  const n = drawPick[u.index + ":" + s.index];
  try { await api("/api/draw-commit", "POST", { job: state.job, run: state.run, unit: u.index, span: s.index, retake: n }); await loadDraws(); say(`Committed: draw ${n} of this sentence. The next render of this video uses it.`, true); } catch (e) { say(e.message); }
}
async function drawRevoke(u, s) {
  try { await api("/api/draw-revoke", "POST", { job: state.job, run: state.run, unit: u.index, span: s.index }); drawPick[u.index + ":" + s.index] = 0; await loadDraws(); say("Taken back: this sentence is read with its plain draw again.", true); } catch (e) { say(e.message); }
}
function drawRow(u, s) {
  const id = u.index + ":" + s.index;
  if (drawPick[id] == null) drawPick[id] = s.retake;
  const pick = h("select", { onchange: (e) => { drawPick[id] = Number(e.target.value); } }, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((n) => h("option", { value: n, selected: n === drawPick[id] }, "Draw " + n + (s.marked && n === s.retake ? " (committed)" : ""))));
  const status = s.marked ? `Committed: draw ${s.retake}.` : s.stale ? "A draw was committed with another reference clip, so it no longer applies." : "Not touched: the plain draw (0).";
  return h("div", { class: "alt" }, h("div", {}, h("div", { class: "sp" }, s.text), h("div", { class: "muted small" }, status)),
    h("div", { class: "plays" }, pick, h("button", { class: "btn", onclick: (e) => drawHear(u, s, e.currentTarget) }, "▶ Hear"), h("button", { class: "btn primary", onclick: () => drawCommit(u, s) }, "Commit this draw"),
      s.marked ? h("button", { class: "btn", onclick: () => drawRevoke(u, s) }, "Take back") : null));
}
function drawsTab() {
  const d = state.draws;
  if (!d) return h("div", { class: "empty" }, "Loading the sentences… (the first time takes a moment)");
  return h("div", { class: "qs" }, state.message && h("div", { class: "msg" + (state.message.ok ? " ok" : "") }, state.message.text),
    h("p", { class: "muted" }, "The voice reads each sentence with one fixed draw, and a word it says right in most draws can be wrong in that one. Pick a draw, hear it, and commit the one that sounds right. Only that sentence changes, and nothing is scored. A committed draw belongs to the exact sentence and the reference clip it was heard with."),
    d.units.map((u) => h("div", { class: "card" }, h("h2", {}, `${u.id} · from ${clock(u.start_ms)}`), u.spans.map((s) => drawRow(u, s)))));
}


const studio = { data: null, loading: false, progress: null, sel: null, follow: true, dirty: false, words: [], last: -1, strip: null, seek: null, time: null, playBtn: null, dragging: false, drawing: null };
const studioAudio = document.createElement("audio"); studioAudio.preload = "auto";          // the main player has no browser controls: the page has its own, with a big seekbar
const drawAudio = document.createElement("audio"); drawAudio.preload = "auto";
const centreOf = (el) => { const strip = studio.strip; return el.getBoundingClientRect().left - strip.getBoundingClientRect().left + strip.scrollLeft - strip.clientWidth / 2 + el.offsetWidth / 2; };          // where the strip must be so that this word is in its middle
function studioGrab(strip) {          // the strip moves with the mouse: press on it and drag sideways (a button press without a drag is still a click)
  let down = false, moved = false, startX = 0, startLeft = 0;
  strip.addEventListener("pointerdown", (e) => { if (e.pointerType === "touch") return; down = true; moved = false; startX = e.clientX; startLeft = strip.scrollLeft; });
  window.addEventListener("pointermove", (e) => {
    if (!down) return;
    const dx = e.clientX - startX;
    if (!moved && Math.abs(dx) > 5) { moved = true; studio.grabbed = true; strip.classList.add("grabbing"); strip.style.scrollBehavior = "auto"; }
    if (moved) strip.scrollLeft = startLeft - dx;
  });
  window.addEventListener("pointerup", () => { if (!down) return; down = false; strip.classList.remove("grabbing"); strip.style.scrollBehavior = ""; if (moved) setTimeout(() => { studio.grabbed = false; }, 1500); });
  strip.addEventListener("click", (e) => { if (moved) { e.stopPropagation(); e.preventDefault(); moved = false; } }, true);       // the click that ends a drag is not a press on a word
  strip.addEventListener("wheel", (e) => { if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) { strip.scrollLeft += e.deltaY; e.preventDefault(); studio.grabbed = true; setTimeout(() => { studio.grabbed = false; }, 1500); } }, { passive: false });
}
const stamp = (t) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, "0")}`;
async function loadStudio() {
  studio.loading = true; studio.progress = null; render();
  try {
    for (let i = 0; i < 600; i++) {
      const r = await api("/api/studio-run", "POST", { job: state.job, run: state.run });
      if (r.ready) {
        const at = studioAudio.currentTime || 0, playing = !studioAudio.paused;
        studio.data = r; studio.loading = false; studio.progress = null; studio.dirty = false; studio.last = -1;
        studioAudio.src = r.url;
        studioAudio.addEventListener("loadedmetadata", () => { studioAudio.currentTime = at; if (playing) studioAudio.play().catch(() => {}); }, { once: true });
        render(); return;
      }
      studio.progress = r; render();
      await new Promise((ok) => setTimeout(ok, 4000));
    }
    say("The voice took too long. Try again.");
  } catch (e) { say(e.message); }
  studio.loading = false; render();
}
function studioTick() {
  const t = studioAudio.currentTime;
  if (studio.seek && !studio.dragging) studio.seek.value = t;
  if (studio.time) studio.time.textContent = `${stamp(t)} / ${stamp(studioAudio.duration || (studio.data ? studio.data.seconds : 0))}`;
  let at = -1;
  for (let i = 0; i < studio.words.length; i++) { if (studio.words[i].start <= t) at = i; else break; }
  if (at !== studio.last) {
    if (studio.strip) studio.strip.querySelectorAll(".word.now").forEach((e) => e.classList.remove("now"));          // exactly one word is lit, however the position moved (a seek back used to leave a trail)
    studio.last = at;
    if (at >= 0) studio.words[at].el.classList.add("now");
  }
  if (studio.follow && studio.strip && !studio.grabbed && at >= 0) {          // the words glide under the centre line: between two words the strip is part of the way from the one to the next
    const a = studio.words[at], b = studio.words[at + 1];
    const from = centreOf(a.el), to = b ? centreOf(b.el) : from;
    const f = b && b.start > a.start ? Math.min(1, Math.max(0, (t - a.start) / (b.start - a.start))) : 0;
    studio.strip.scrollLeft = from + f * (to - from);
  }
}
let gliding = false;
function studioGlide() { if (studioAudio.paused) { gliding = false; return; } studioTick(); requestAnimationFrame(studioGlide); }
studioAudio.addEventListener("timeupdate", studioTick);
studioAudio.addEventListener("play", () => { if (studio.playBtn) studio.playBtn.textContent = "❚❚ Pause"; if (!gliding) { gliding = true; requestAnimationFrame(studioGlide); } });
studioAudio.addEventListener("pause", () => { if (studio.playBtn) studio.playBtn.textContent = "▶ Play"; });
function studioPick(s, w) {            // a word is pressed: the main player stops where the word begins, and the panel opens with the three draws of its sentence
  studioAudio.pause(); drawAudio.pause(); studioAudio.currentTime = w.start; studio.last = -1;
  studio.sel = { unit: s.unit, index: s.index, text: s.text, word: w.text, draws: s.draws, primed: s.primed || [], picked: s.picked, committed: s.committed, start: s.start, end: s.end, words: s.words.length, at: s.words.findIndex((x) => x === w) + 1, row: s, wordObj: w };
  studio.drawing = null; render();
}
function playPrimed(t, k) {          // the primed sentence a draw's names were cut from, exactly as the voice made it
  const s = studio.sel; const item = s && s.primed && s.primed[t] && s.primed[t][k]; if (!item) return;
  studioAudio.pause(); drawAudio.pause(); drawAudio.src = item.url; drawAudio.currentTime = 0; drawAudio.play().catch(() => {});
  studio.drawing = null; render();
}
function playDraw(t) {
  const s = studio.sel; if (!s || !s.draws[t]) return;
  studioAudio.pause(); drawAudio.pause(); drawAudio.src = s.draws[t]; drawAudio.currentTime = 0; drawAudio.play().catch(() => {});
  studio.drawing = t; render();
}
drawAudio.addEventListener("ended", () => { studio.drawing = null; render(); });
async function studioCommit(t) {
  const s = studio.sel; if (!s) return;
  try {
    await api("/api/studio-commit", "POST", { job: state.job, run: state.run, unit: s.unit, span: s.index, take: t, note: s.word });
    const row = studio.data.sentences.find((x) => x.unit === s.unit && x.index === s.index);
    if (row) { row.committed = t; row.marked = true; }
    s.committed = t; studio.dirty = true;
    say(`Committed: draw ${t + 1} of this sentence. The seed is saved. When you have committed what you want, press “Regenerate the full audio”.`, true);
  } catch (e) { say(e.message); }
}
async function studioFlag(on) {          // what the ear says: this word is rushed or clipped (or take the flag back). Saved exactly as heard.
  const s = studio.sel; if (!s) return;
  try {
    const r = await api("/api/studio-flag", "POST", { job: state.job, run: state.run, unit: s.unit, span: s.index, word: s.word, flagged: on });
    const row = studio.data.sentences.find((x) => x.unit === s.unit && x.index === s.index);
    if (row) row.flagged = on ? [...new Set([...(row.flagged || []), s.word])] : (row.flagged || []).filter((w) => w !== s.word);
    say(on ? `Flagged “${s.word}” as rushed or clipped (${r.flags_in_this_run} flagged in this run).` : `Took the flag off “${s.word}”.`, true);
  } catch (e) { say(e.message); }
}
async function studioRepair() {         // Repair: the word and its span, with the next one, are sent to be investigated: why is it rushed
  const s = studio.sel; if (!s) return;
  studio.repair = { busy: true, key: s.unit + ":" + s.index + ":" + s.word }; render();
  try {
    const r = await api("/api/studio-investigate", "POST", { job: state.job, run: state.run, unit: s.unit, span: s.index, word: s.word });
    studio.repair = { key: s.unit + ":" + s.index + ":" + s.word, result: r };
  } catch (e) { studio.repair = { key: s.unit + ":" + s.index + ":" + s.word, error: e.message }; }
  render();
}
function repairBox(s) {
  const r = studio.repair;
  if (!r || r.key !== s.unit + ":" + s.index + ":" + s.word) return null;
  if (r.busy) return h("div", { class: "small muted" }, "Investigating this span and the next… (a few seconds)");
  if (r.error) return h("div", { class: "msg" }, r.error);
  const f = r.result, unclear = !f.rule && !f.rewrite;
  return h("div", { class: "repairbox" },
    f.word_clip_url ? h("div", { class: "plays" }, h("button", { class: "btn", onclick: () => { studioAudio.pause(); drawAudio.pause(); drawAudio.src = f.word_clip_url; drawAudio.currentTime = 0; drawAudio.play().catch(() => {}); } }, `▶ The word “${f.word}” alone`)) : null,
    h("div", {}, h("b", {}, "Why: "), f.cause || "—"),
    f.rule ? h("div", {}, h("b", {}, "Rule: "), f.rule) : null,
    f.rewrite ? h("div", {}, h("b", {}, "A rewrite to try: "), h("span", { class: "sp" }, f.rewrite)) : null,
    f.next_note ? h("div", {}, h("b", {}, "The next sentence: "), f.next_note) : null,
    h("div", { class: "small muted" }, unclear ? "The model could not find the cause in the text." : "An opinion, not a fix: hear the rewrite before you use it."));
}
async function studioRevoke() {
  const s = studio.sel; if (!s) return;
  try {
    await api("/api/draw-revoke", "POST", { job: state.job, run: state.run, unit: s.unit, span: s.index });
    const row = studio.data.sentences.find((x) => x.unit === s.unit && x.index === s.index);
    if (row) { row.committed = null; row.marked = false; }
    s.committed = null; studio.dirty = true; say("Taken back: the render chooses this sentence's draw again.", true);
  } catch (e) { say(e.message); }
}
function studioSwitchRun(job, run) {          // a different dub: stop the audio, forget this one's strip and let the listener press Load (a run that is not made yet is made only then)
  studioAudio.pause(); drawAudio.pause();
  Object.assign(studio, { data: null, sel: null, loading: false, progress: null, dirty: false, last: -1, words: [] });
  state.job = job; state.run = run; render();
}
function studioPicker() {
  const job = state.jobs.find((j) => j.job === state.job);
  const jobSel = h("select", { "aria-label": "Job", onchange: (e) => { const j = state.jobs.find((x) => x.job === e.target.value); studioSwitchRun(j.job, j.runs[0].run); } },
    state.jobs.map((j) => h("option", { value: j.job, selected: j.job === state.job }, `${short(j.name)} · ${j.created || j.clip}`)));
  const runSel = h("select", { "aria-label": "Run", onchange: (e) => studioSwitchRun(state.job, e.target.value) },
    (job ? job.runs : []).map((r) => h("option", { value: r.run, selected: r.run === state.run }, `${r.run} · ${r.at}`)));
  return h("div", { class: "bar" }, jobSel, runSel);
}
function studioTab() {
  const d = studio.data;
  if (!d) {
    const p = studio.progress;
    return h("div", { class: "studio" }, studioPicker(), state.message && h("div", { class: "msg" }, state.message.text),
      studio.loading ? h("div", { class: "card" }, h("h2", {}, "Making the whole dub…"), h("p", { class: "muted" }, p ? `${p.made} of ${p.total} units are made. The first time takes a few minutes, with the three draws of every sentence; after that only a unit you changed is made again.` : "Starting…"))
        : h("div", { class: "card" }, h("h2", {}, "Studio"), h("p", { class: "muted" }, "Play the whole dub from start to finish. Under the player every word is a button that follows the audio. Press a word to hear the three draws of its sentence, and commit the one that sounds right."), h("button", { class: "btn primary", onclick: loadStudio }, "Load the whole dub")));
  }
  studio.words = [];
  const strip = h("div", { class: "strip" }, h("span", { class: "pad" }));
  let unitSeen = -1;
  for (const s of d.sentences) {
    if (s.unit !== unitSeen) { unitSeen = s.unit; const u = d.units[s.unit]; strip.append(h("span", { class: "unitchip" }, `${u.id} · ${clock(u.start_s * 1000)}`)); }
    s.words.forEach((w) => {
      const picked = studio.sel && studio.sel.unit === s.unit && studio.sel.index === s.index && studio.sel.word === w.text;
      const el = h("button", { class: "word" + (picked ? " sel" : "") + ((s.flagged || []).includes(w.text) ? " flagged" : ""), title: s.marked ? `draw ${s.committed + 1} committed` : "", onclick: () => studioPick(s, w) }, w.text + (s.marked && w === s.words[0] ? " ●" : ""));
      studio.words.push({ start: w.start, end: w.end, el }); strip.append(el);
    });
    strip.append(h("span", { class: "stop" }));
  }
  strip.append(h("span", { class: "pad" }));
  studio.strip = strip; studioGrab(strip);
  setTimeout(() => { const w = studio.words[Math.max(0, studio.last)]; if (w && studio.strip) studio.strip.scrollLeft = centreOf(w.el); }, 0);          // a redraw keeps the current word in the middle
  const seek = h("input", { type: "range", class: "seek", min: 0, max: d.seconds, step: 0.05, value: studioAudio.currentTime || 0, "aria-label": "Position in the dub",
    oninput: (e) => { studioAudio.currentTime = Number(e.target.value); studio.last = -1; studioTick(); }, onpointerdown: () => { studio.dragging = true; }, onpointerup: () => { studio.dragging = false; } });
  studio.seek = seek;
  studio.time = h("span", { class: "time" }, `${stamp(studioAudio.currentTime || 0)} / ${stamp(d.seconds)}`);
  studio.playBtn = h("button", { class: "btn primary", onclick: () => { drawAudio.pause(); studioAudio.paused ? studioAudio.play().catch(() => {}) : studioAudio.pause(); } }, studioAudio.paused ? "▶ Play" : "❚❚ Pause");
  const player = h("div", { class: "player" },
    h("div", { class: "row1" }, studio.playBtn, studio.time, h("label", { class: "opt" }, h("input", { type: "checkbox", checked: studio.follow, onchange: (e) => { studio.follow = e.target.checked; } }), "Follow the audio"),
      h("button", { class: "btn" + (studio.dirty ? " primary" : ""), disabled: studio.loading, onclick: loadStudio }, studio.dirty ? "Regenerate the full audio with the commits" : "Reload the audio")),
    seek, studio.loading ? h("div", { class: "small muted" }, studio.progress ? `Making the audio: ${studio.progress.made} of ${studio.progress.total} units…` : "Making the audio…") : null);
  const s = studio.sel;
  let panel = h("div", { class: "drawpanel2 muted" }, "Press a word to open its sentence here, with three draws to hear.");
  if (s) {
    const inAudio = s.committed != null ? s.committed : s.picked;
    const unitText = d.sentences.filter((x) => x.unit === s.unit).map((x) => x.words.map((w) => h("span", { class: "uw" + (x.index === s.index ? " inspan" : "") + (w === s.wordObj ? " clicked" : "") }, w.text + " "))).flat();
    const inUnit = d.sentences.filter((x) => x.unit === s.unit).length, secs = Math.max(0.01, s.end - s.start);
    const info = `Span ${s.index + 1} of ${inUnit} in ${d.units[s.unit].id} · ${stamp(s.start)}–${stamp(s.end)} (${secs.toFixed(1)} s) · ${s.text.length} characters, ${(s.text.length / secs).toFixed(1)} per second · ${s.words} words, “${s.word}” is word ${s.at}`
      + ` · in the audio: draw ${inAudio + 1}` + (s.committed != null ? ` · committed: draw ${s.committed + 1}` : "");
    panel = h("div", { class: "drawpanel2" }, h("div", { class: "small muted" }, `${d.units[s.unit].id} · “${s.word}”`), h("div", { class: "unittext" }, unitText), h("div", { class: "small muted" }, info),
      h("div", { class: "draws3" }, [0, 1, 2].map((t) => h("div", { class: "d", "data-live": studio.drawing === t ? "true" : "false" },
        h("button", { class: "btn" + (studio.drawing === t ? " primary" : ""), disabled: !s.draws[t], onclick: () => playDraw(t) }, studio.drawing === t ? `❚❚ Draw ${t + 1}` : `▶ Draw ${t + 1}`),
        ...((s.primed && s.primed[t]) || []).map((item, k) => h("button", { class: "btn", title: item.text, onclick: () => playPrimed(t, k) },
          `▶ Primed sentence${s.primed[t].length > 1 ? " " + (k + 1) : ""} (raw)` + ((item.names || []).length ? " · " + item.names.map((n) => `${Number(n.s).toFixed(2)} s` + (n.pace ? ` (${n.pace}× pace)` : "")).join(" + ") : "") + (item.stretched ? " ⚠ set aside: stretched" : ""))),
        h("div", { class: "small muted" }, s.committed === t ? "Committed" : s.committed == null && inAudio === t ? "In the audio now" : t === inAudio ? "In the audio now" : " "),
        h("button", { class: "btn", disabled: !s.draws[t] || s.committed === t, onclick: () => studioCommit(t) }, `Commit draw ${t + 1}`)))),
      h("div", { class: "decide" },
        ((studio.data.sentences.find((x) => x.unit === s.unit && x.index === s.index) || {}).flagged || []).includes(s.word)
          ? h("button", { class: "btn", onclick: () => studioFlag(false) }, "Flagged as rushed or clipped (take the flag back)")
          : h("button", { class: "btn primary", onclick: () => studioFlag(true) }, "I hear this word rushed or clipped"),
        h("button", { class: "btn", disabled: !!(studio.repair && studio.repair.busy), onclick: studioRepair }, "Ask why (an opinion)"),
        s.committed != null ? h("button", { class: "btn", onclick: studioRevoke }, "Take back the committed draw") : null),
      repairBox(s));
  }
  setTimeout(() => { studio.last = -1; studioTick(); const first = document.querySelector(".unittext .inspan"); if (first) { const box = first.parentElement; box.scrollTop = Math.max(0, first.offsetTop - 8); } }, 0);          // a redraw makes new buttons: light the current word again, and show the span inside the unit text
  return h("div", { class: "studio" }, studioPicker(), state.message && h("div", { class: "msg" + (state.message.ok ? " ok" : "") }, state.message.text), player, h("div", { class: "stripbox" }, strip), panel);
}

function render() {
  buttons = [];
  app.replaceChildren();
  if (!state.signedIn) { app.append(login()); return; }
  app.append(header(), h("main", {}, state.tab === "review" ? reviewTab() : state.tab === "questions" ? questionsTab() : state.tab === "studio" ? studioTab() : state.tab === "draws" ? drawsTab() : state.tab === "smith" ? smithTab() : state.tab === "book" ? bookTab() : rulesTab()));
  if (state.tab === "review" && state.detail) watchReady();
}
start();
</script>
</body>
</html>
"""
