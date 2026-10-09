#!/usr/bin/env node
/**
 * Token compliance audit (spec Phase 13): raw colour and pixel literals may only appear in
 * `src/design/` (tokens). Everything else must read a CSS variable.
 *
 * Allowed outside tokens (each is a deliberate, narrow exception, not a loophole):
 *  - hairline/ring widths: 1px, 2px, 3px on border/outline/stroke properties;
 *  - media-query breakpoints (a CSS variable cannot be used in @media);
 *  - `0px`; px inside SVG/canvas drawing code is unitless numbers, not CSS, and is not scanned;
 *  - white `#fff` / `#ffffff` for text on accent/dark fills and the robot-disc outline.
 * Test files, fixtures and generated files are skipped.
 * Usage: node scripts/check_tokens.mjs [--report]
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";

const SRC = join(import.meta.dirname, "..", "src");
const SKIP_DIR = new Set(["design", "__fixtures__", "mocks"]);
const FILE_RE = /\.(css|tsx?)$/;
const SKIP_FILE = /(\.test\.tsx?|\.gen\.ts|fixtures\.ts)$/;

const COLOUR = /#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(/g;
const PX = /(?<![\w.-])(-?\d*\.?\d+)px\b/g;
const ALLOWED_WHITE = new Set(["#fff", "#ffffff"]);
const RING_PROP = /\b(border[\w-]*|outline[\w-]*|stroke-width)\s*:/;

function walk(dir, out = []) {
  for (const e of readdirSync(dir)) {
    const p = join(dir, e);
    if (statSync(p).isDirectory()) {
      if (!SKIP_DIR.has(e)) walk(p, out);
    } else if (FILE_RE.test(e) && !SKIP_FILE.test(e)) out.push(p);
  }
  return out;
}

const findings = [];
for (const file of walk(SRC)) {
  const rel = relative(SRC, file);
  const isCss = file.endsWith(".css");
  readFileSync(file, "utf-8").split("\n").forEach((line, i) => {
    const code = line.replace(/\/\*.*?\*\//g, "").replace(/\/\/.*$/, "");
    if (/^\s*(\*|\/\*)/.test(line)) return;
    for (const m of code.matchAll(COLOUR)) {
      if (ALLOWED_WHITE.has(m[0].toLowerCase())) continue;
      if (/&#\d/.test(code)) continue;
      findings.push({ rel, line: i + 1, kind: "colour", text: m[0], src: line.trim() });
    }
    if (!isCss && !/(style=|style:|"\s*\d+px|\d+px")/.test(code)) return; // TSX: only inline-style strings
    if (/@media/.test(code)) return;
    for (const m of code.matchAll(PX)) {
      const v = Number(m[1]);
      if (v === 0) continue;
      if (v <= 3 && RING_PROP.test(code)) continue;
      findings.push({ rel, line: i + 1, kind: "px", text: m[0], src: line.trim() });
    }
  });
}

if (findings.length) {
  const by = {};
  for (const f of findings) (by[f.rel] ??= []).push(f);
  console.error(`check_tokens: ${findings.length} raw literal(s) outside src/design/:`);
  for (const [rel, fs] of Object.entries(by)) {
    console.error(`  ${rel}: ${fs.length}`);
    if (process.argv.includes("--report")) for (const f of fs) console.error(`    ${f.line}: [${f.kind}] ${f.text}   ${f.src.slice(0, 90)}`);
  }
  process.exit(1);
}
console.log("check_tokens: OK (no raw colour or px literals outside src/design/)");
