#!/usr/bin/env node
/**
 * Real-data rule enforcement (spec §29.4). Run in `npm run build` and CI.
 * Fails if:
 *   1. Any non-mock source file imports from `src/mocks/`.
 *   2. VITE_USE_MOCKS=true is set for a production build.
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";

const SRC_DIR = join(import.meta.dirname, "..", "src");
const MOCKS_PREFIX = "mocks/";

if (process.env.VITE_USE_MOCKS === "true" && process.env.NODE_ENV === "production") {
  console.error("check_no_mocks: VITE_USE_MOCKS=true is set for a production build.");
  process.exit(1);
}

/** @param {string} dir */
function walk(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    const st = statSync(full);
    if (st.isDirectory()) {
      out.push(...walk(full));
    } else if (/\.(tsx?|jsx?)$/.test(entry)) {
      out.push(full);
    }
  }
  return out;
}

const violations = [];
for (const file of walk(SRC_DIR)) {
  const rel = relative(SRC_DIR, file);
  if (rel.startsWith("mocks/")) continue; // mock files may reference each other
  const content = readFileSync(file, "utf-8");
  const importRegex = /from\s+["']([^"']+)["']/g;
  let match;
  while ((match = importRegex.exec(content))) {
    const spec = match[1];
    if (spec.includes(MOCKS_PREFIX) || spec.endsWith("/mocks") || spec === "./mocks" || spec === "../mocks") {
      violations.push(`${rel}: imports mock module '${spec}'`);
    }
  }
}

if (violations.length > 0) {
  console.error("check_no_mocks: found forbidden mock imports outside src/mocks/:");
  for (const v of violations) console.error(`  - ${v}`);
  process.exit(1);
}

console.log("check_no_mocks: OK (no mock imports found outside src/mocks/)");
