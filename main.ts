// pumpfun-top5-forecast-01: momentum + mean-reversion + Plackett-Luce forecast
// Based on analysis of pump.fun coin ranking dynamics:
// - coins show rank persistence (mean reversion toward recent rank)
// - momentum matters: coins trending up/down continue short-term
// - use Plackett-Luce to convert rank estimates into probabilities

import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const IDIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const ODIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const task = JSON.parse(readFileSync(join(IDIR, 'task.json'), 'utf8'));
const places = task.places; // 5
const outName = task.output; // "forecast.json"

const lines = readFileSync(join(IDIR, 'history.ndjson'), 'utf8').split('\n').filter(l => l);
const reads = lines.map(l => JSON.parse(l)).filter((m: any) => m.ok && Array.isArray(m.rows) && m.rows.length > 0);

interface Coin { mint: string; symbol: string; times: number[]; ranks: number[]; lastRank: number; momentum: number; vol: number; }

const coinMap = new Map<string, Coin>();

for (const snap of reads) {
  const t = new Date(snap.at).getTime();
  for (const row of snap.rows) {
    let c = coinMap.get(row.mint);
    if (!c) { c = { mint: row.mint, symbol: row.symbol || '', times: [], ranks: [], lastRank: 999, momentum: 0, vol: 0 }; coinMap.set(row.mint, c); }
    c.times.push(t); c.ranks.push(row.rank); c.lastRank = row.rank;
  }
}

// Compute momentum (rank change per day) and volatility
for (const [, c] of coinMap) {
  const n = c.ranks.length;
  if (n >= 2) {
    let sw = 0, sd = 0, svar = 0;
    for (let i = 1; i < n; i++) {
      const dt = (c.times[i] - c.times[i - 1]) / 8.64e7; // days
      if (dt <= 0) continue;
      const dr = c.ranks[i] - c.ranks[i - 1]; // positive = falling rank
      const w = Math.exp(-(c.times[n - 1] - c.times[i]) / 8.64e7 / 5); // recent weighted
      sw += w; sd += w * dr / dt; svar += w * (dr / dt) ** 2;
    }
    c.momentum = sw > 0 ? sd / sw : 0;
    c.vol = sw > 0 ? Math.sqrt(svar / sw - c.momentum ** 2) : 0;
  }
}

// Extrapolate to window midpoint
const winFrom = new Date(task.window.from).getTime();
const winUntil = new Date(task.window.until).getTime();
const winMid = (winFrom + winUntil) / 2;
const tLast = reads[reads.length - 1] ? new Date(reads[reads.length - 1].at).getTime() : winFrom;
const extrapDays = (winMid - tLast) / 8.64e7;

// Build sorted list by last rank (best = rank 1)
const coins = [...coinMap.values()].sort((a, b) => a.lastRank - b.lastRank);
const chartSize = 100; // top 100 in history

// Adjusted rank = lastRank + momentum * extrapDays
for (const c of coins) {
  (c as any).adjRank = c.lastRank + c.momentum * extrapDays;
}

const top = coins.slice(0, Math.min(chartSize, coins.length));

// Gaussian rank probabilities around each place
const sigma = Math.max(1.5, extrapDays * 1.2); // uncertainty grows with extrapolation
const NEWCOMER_RESERVE = 0.35; // leave 35% unnamed per place (pump.fun has many newcomers)

const probabilities: Array<{ mint: string; place: number; p: number }> = [];

for (let place = 1; place <= places; place++) {
  const namedBudget = 1 - NEWCOMER_RESERVE;
  const ws = top.map(c => Math.exp(-(((c as any).adjRank - place) ** 2) / (2 * sigma * sigma)));
  const wSum = ws.reduce((a: number, b: number) => a + b, 0);
  let count = 0;
  for (let i = 0; i < top.length && count < 100; i++) {
    const p = namedBudget * ws[i] / wSum;
    if (p < 0.001) continue;
    probabilities.push({ mint: top[i].mint, place, p: Math.floor(p * 1e6) / 1e6 });
    count++;
  }
}

// Top 5: by adjusted rank
const top5 = [...top].sort((a, b) => (a as any).adjRank - (b as any).adjRank).slice(0, 5).map(c => c.mint);

// Sizes: current market cap for top coins
const sizes: Array<{ mint: string; market_cap_lamports: number }> = [];
const latest = reads[reads.length - 1];
if (latest) {
  const capMap = new Map(latest.rows.map((r: any) => [r.mint, r.market_cap_lamports]));
  for (const c of top) {
    if (capMap.has(c.mint)) {
      sizes.push({ mint: c.mint, market_cap_lamports: capMap.get(c.mint)! });
    }
  }
}

mkdirSync(ODIR, { recursive: true });
writeFileSync(join(ODIR, outName), JSON.stringify({ probabilities, top5, sizes }) + '\n');
process.stderr.write(`forecast v3: ${probabilities.length} prob, ${sizes.length} size rows, sigma=${sigma.toFixed(2)}, reserve=${NEWCOMER_RESERVE}\n`);