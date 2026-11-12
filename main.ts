// pumpfun-top5-forecast-01: fresh forecast entry (momentum + Gaussian rank model)
// Reads MAKEFUN_INPUT_DIR (task.json, history.ndjson); writes MAKEFUN_OUTPUT_DIR/forecast.json
// with exactly three keys: probabilities, top5, sizes.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const IDIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const ODIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const DAY_MS = 8.64e7;
const LAMPORTS_PER_SOL = 1e9;
const CHART = 100;              // coins each answer read lists
const SIZE_SD = 0.40;           // log10(SOL) std dev for the size bins
const NEWCOMER_RESERVE = 0.35;  // share left unnamed per place for movers + newcomers
const RANK_SIGMA = 1.6;         // Gaussian width over ranks
const P_MIN = 1e-6;             // drop probabilities that round to 0

type RawRow = { mint: string; symbol?: string; rank: number; market_cap_lamports: string | number };
type RawMark = { at: string; ok: boolean; rows: RawRow[] };

const task = JSON.parse(readFileSync(join(IDIR, 'task.json'), 'utf8')) as any;
const places: number = task.places ?? 5;
const outName: string = task.output ?? 'forecast.json';
const edges: number[] = task.size?.edges ?? [];
const t0: number = Date.parse(task.t0 ?? task.history_until);

// The window may be given as {from,until} or as {deadline,answer_at}; accept both.
const winUntil: number = task.window?.until
  ? Date.parse(task.window.until)
  : Date.parse(task.window?.answer_at ?? task.window?.deadline ?? task.history_until);

const raw = readFileSync(join(IDIR, 'history.ndjson'), 'utf8').split('\n').filter(Boolean);
const marks: RawMark[] = raw
  .map((l) => JSON.parse(l) as RawMark)
  .filter((m) => m.ok && Array.isArray(m.rows) && m.rows.length > 0);

type Coin = {
  mint: string; symbol: string;
  times: number[]; ranks: number[]; caps: number[];
  lastRank: number; lastCap: number; momentum: number; vol: number; adjRank: number;
};
const coinMap = new Map<string, Coin>();
let tLast = t0;
for (const m of marks) {
  const t = Date.parse(m.at);
  if (t > tLast) tLast = t;
  for (const r of m.rows) {
    const cap = Number(r.market_cap_lamports);
    let c = coinMap.get(r.mint);
    if (!c) {
      c = { mint: r.mint, symbol: r.symbol ?? '', times: [], ranks: [], caps: [], lastRank: 9999, lastCap: 0, momentum: 0, vol: 0, adjRank: 9999 };
      coinMap.set(r.mint, c);
    }
    c.times.push(t); c.ranks.push(r.rank); c.caps.push(cap);
    c.lastRank = r.rank; c.lastCap = cap;
  }
}

// Per-coin momentum: recency-weighted rank drift per day, plus its volatility.
for (const c of coinMap.values()) {
  const n = c.ranks.length;
  if (n >= 2) {
    let sw = 0, sd = 0, svar = 0;
    for (let i = 1; i < n; i++) {
      const dt = (c.times[i] - c.times[i - 1]) / DAY_MS;
      if (dt <= 0) continue;
      const dr = (c.ranks[i] - c.ranks[i - 1]) / dt;
      const w = Math.exp(-(c.times[n - 1] - c.times[i]) / DAY_MS / 5);
      sw += w; sd += w * dr; svar += w * dr * dr;
    }
    if (sw > 0) {
      c.momentum = sd / sw;
      c.vol = Math.sqrt(Math.max(0, svar / sw - c.momentum * c.momentum));
    }
  }
}

// Numeric guard: a non-finite or absurd horizon (bad/unset window) must not
// poison every adjRank — clamp to a sane [0, 7]-day extrapolation.
const rawExtrap = (winUntil - tLast) / DAY_MS;
const extrapDays = Number.isFinite(rawExtrap) ? Math.max(0, Math.min(rawExtrap, 7)) : 0;

// Extrapolate each coin's rank to the answer window with a volatility-shrunk
// drift (computed field): a coin whose rank has been swinging wildly reverts
// toward its last rank, so one noisy jump cannot fling it into place 1.
const coins = [...coinMap.values()];
for (const c of coins) {
  const shrink = 1 / (1 + Math.min(c.vol, 4));
  c.adjRank = c.lastRank + c.momentum * shrink * extrapDays;
}
coins.sort((a, b) => a.adjRank - b.adjRank);
const top = coins.slice(0, Math.min(CHART, coins.length));

// Order probabilities: a Gaussian over adjRank centred on each place.
const sigma = Math.max(RANK_SIGMA, 0.5 + extrapDays);
const namedBudget = 1 - NEWCOMER_RESERVE;
const probabilities: Array<{ mint: string; place: number; p: number }> = [];
for (let place = 1; place <= places; place++) {
  const ws = top.map((c) => Math.exp(-((c.adjRank - place) ** 2) / (2 * sigma * sigma)));
  const wSum = ws.reduce((a, b) => a + b, 0) || 1;
  let count = 0;
  for (let i = 0; i < top.length && count < CHART; i++) {
    const p = (namedBudget * ws[i]) / wSum;
    if (p < P_MIN) continue;
    probabilities.push({ mint: top[i].mint, place, p: Math.floor(p * 1e6) / 1e6 });
    count++;
  }
}

const top5: string[] = top.slice(0, 5).map((c) => c.mint);

// Size bins: a Gaussian in log10(SOL) around the cap of the coin predicted at each place.
const binCenters: number[] = [];
for (let b = 0; b < 82; b++) {
  if (b === 0) binCenters.push(edges.length ? edges[0] - 0.025 : 1.975);
  else if (b <= 80 && edges.length >= b) binCenters.push((edges[b - 1] + edges[b]) / 2);
  else binCenters.push(edges.length ? edges[edges.length - 1] + 0.025 : 6.025);
}
const sizes: Array<{ place: number; bin: number; p: number }> = [];
for (let place = 1; place <= places; place++) {
  const pred = top[place - 1];
  const capSol = pred && pred.lastCap > 0 ? pred.lastCap / LAMPORTS_PER_SOL : 1000;
  const logCap = Math.log10(capSol);
  const bw = binCenters.map((x) => Math.exp(-((x - logCap) ** 2) / (2 * SIZE_SD * SIZE_SD)));
  const bSum = bw.reduce((a, b) => a + b, 0) || 1;
  for (let b = 0; b < 82; b++) {
    const p = bw[b] / bSum;
    if (p < P_MIN) continue;
    sizes.push({ place, bin: b, p: Math.floor(p * 1e6) / 1e6 });
  }
}

mkdirSync(ODIR, { recursive: true });
writeFileSync(join(ODIR, outName), JSON.stringify({ probabilities, top5, sizes }) + '\n');
process.stderr.write(
  `forecast fresh: ${probabilities.length} prob rows, ${sizes.length} size rows, ` +
    `${coins.length} coins, extrap=${extrapDays.toFixed(4)}d sigma=${sigma.toFixed(2)}\n`,
);
