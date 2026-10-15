// Session receipt: run the same deterministic sim the app uses across a
// q-sweep, averaged over multiple seeds, and write receipts/session.json.
// The README quotes this file.
//
//   node app/session_receipt.js
//
'use strict';
const fs = require('fs');
const path = require('path');

// load sim.js (browser-style global, works in node)
const sim = fs.readFileSync(path.join(__dirname, 'sim.js'), 'utf8');
const sandbox = {};
new Function('exports', sim + '\nObject.assign(exports, {runLane});')(sandbox);

const CFG = {
  vocab: 24,
  c: 0.1,
  budget: 60,
  price_base: 2.0, // $ per 1M tokens at naive-equivalent pricing (naive = 1 unit/token)
  prompt: [1, 2, 3],
};
const SEEDS = Array.from({ length: 50 }, (_, i) => 100 + i * 7);
const DRIFT_SEEDS = Array.from({ length: 9 }, (_, i) => 500 + i * 11);

const mean = xs => xs.reduce((a, b) => a + b, 0) / xs.length;
const sd = xs => {
  const m = mean(xs);
  return Math.sqrt(mean(xs.map(x => (x - m) * (x - m))));
};
const r2 = x => Math.round(x * 100) / 100;
const r4 = x => Math.round(x * 10000) / 10000;

const out = {
  config: Object.assign({}, CFG, { seeds: SEEDS.length }),
  generated_by: 'app/session_receipt.js (node) — same sim as app/session.html',
  workload_note:
    'synthetic i.i.d. Bernoulli(q) verification (prefix-independent target); ' +
    'the closed form is predictive in this regime by construction. ' +
    'Numbers are means over SEEDS sessions; steady state = last 20 rounds of ' +
    'each 60-round session (controller warm-up flushed).',
  pricing_note: 'usd_per_mtok = tok_per_unit * price_base, naive-equivalent pricing',
  scenarios: [],
};

const scenarios = [0.30, 0.50, 0.70, 0.85, 0.95];
for (const q of scenarios) {
  const per = { naive: [], fixed_k8: [], adaptive: [] };
  for (const seed of SEEDS) {
    const cfg = Object.assign({}, CFG, { q, seed });
    for (const lane of ['naive', 'k8', 'adaptive']) {
      const res = sandbox.runLane(lane, cfg);
      per[lane === 'k8' ? 'fixed_k8' : lane].push({
        tok_per_unit: res.steady_state.tok_per_unit,
        usd_per_mtok: res.steady_state.usd_per_mtok,
        tokens: res.steady_state.tokens,
        measured_p: res.measured_p,
        settled_k: res.settled_k,
      });
    }
  }
  const agg = arr => ({
    tok_per_unit: r4(mean(arr.map(x => x.tok_per_unit))),
    tok_per_unit_sd: r4(sd(arr.map(x => x.tok_per_unit))),
    usd_per_mtok: r4(mean(arr.map(x => x.usd_per_mtok))),
    tokens: Math.round(mean(arr.map(x => x.tokens))),
    measured_p: r4(mean(arr.filter(x => x.measured_p !== null).map(x => x.measured_p))),
    settled_k: r4(mean(arr.map(x => x.settled_k))),
  });
  out.scenarios.push({
    q,
    naive: agg(per.naive),
    fixed_k8: agg(per.fixed_k8),
    adaptive: agg(per.adaptive),
  });
}

// ---- assertions on multi-seed means (the demonstrable claims) ---------------
for (const s of out.scenarios) {
  const a = s.adaptive.tok_per_unit, n = s.naive.tok_per_unit, k = s.fixed_k8.tok_per_unit;
  if (!(a > n * 1.02)) {
    throw new Error(`adaptive must beat naive by >2% at q=${s.q}: ${a} vs ${n}`);
  }
  if (s.q <= 0.5 && !(a > k * 1.10)) {
    throw new Error(`adaptive must beat k8 by >10% at q=${s.q}: ${a} vs ${k}`);
  }
  if (s.q > 0.5 && s.q <= 0.7 && !(a > k * 1.03)) {
    throw new Error(`adaptive must beat k8 by >3% at q=${s.q}: ${a} vs ${k}`);
  }
  if (s.q > 0.7 && Math.abs(a - k) / k >= 0.06) {
    throw new Error(`adaptive must be within 6% of k8 at q=${s.q}: ${a} vs ${k} (sd adaptive=${s.adaptive.tok_per_unit_sd})`);
  }
}
// the headline failure mode: the default K=8 must lose to not speculating
const lowQ = out.scenarios[0];
if (!(lowQ.fixed_k8.tok_per_unit < lowQ.naive.tok_per_unit)) {
  throw new Error('expected fixed K=8 below naive at q=0.3 (the collapse case)');
}

// ---- regime drift: controller must track a mid-session acceptance step -------
const beforeAll = [], afterAll = [], settledAll = [], driftAdaptive = [], driftK8 = [];
for (const seed of DRIFT_SEEDS) {
  const cfg = Object.assign({}, CFG, {
    seed,
    q: { before: 0.30, after: 0.90, at: 30 },
  });
  const ad = sandbox.runLane('adaptive', cfg);
  const k8 = sandbox.runLane('k8', cfg);
  beforeAll.push(mean(ad.rounds.slice(0, 30).map(r => r.k)));
  afterAll.push(mean(ad.rounds.slice(30).map(r => r.k)));
  settledAll.push(ad.settled_k);
  driftAdaptive.push(ad.steady_state.tok_per_unit);
  driftK8.push(k8.steady_state.tok_per_unit);
}
out.regime_drift = {
  config: Object.assign({}, CFG, {
    q: { before: 0.30, after: 0.90, at: 30 },
    budget: 60,
    seeds: DRIFT_SEEDS.length,
  }),
  note: 'q steps 0.30 -> 0.90 at round 30; adaptive K must follow, then win on cost',
  mean_k_before: r2(mean(beforeAll)),
  mean_k_after: r2(mean(afterAll)),
  settled_k: r2(mean(settledAll)),
  steady_tok_per_unit_adaptive: r4(mean(driftAdaptive)),
  steady_tok_per_unit_k8: r4(mean(driftK8)),
};
if (!(out.regime_drift.mean_k_after > out.regime_drift.mean_k_before + 2)) {
  throw new Error(`controller did not raise K after the drift: ${out.regime_drift.mean_k_before} -> ${out.regime_drift.mean_k_after}`);
}
if (!(out.regime_drift.settled_k >= 6)) {
  throw new Error(`settled K=${out.regime_drift.settled_k} — controller missed the high-q regime`);
}
// in the post-drift steady state adaptive must beat k8 (which spent rounds
// 0..29 dumping 8-draft rounds at q=0.30)
if (!(out.regime_drift.steady_tok_per_unit_adaptive > out.regime_drift.steady_tok_per_unit_k8)) {
  throw new Error('adaptive should beat k8 in post-drift steady state');
}

const dest = path.join(__dirname, '..', 'receipts', 'session.json');
fs.writeFileSync(dest, JSON.stringify(out, null, 2) + '\n');
console.log('wrote', dest);
console.log('steady-state tok/unit cost (mean of ' + SEEDS.length + ' sessions):');
for (const s of out.scenarios) {
  console.log(
    `  q=${s.q}  naive ${s.naive.tok_per_unit}  k8 ${s.fixed_k8.tok_per_unit}  ` +
    `adaptive ${s.adaptive.tok_per_unit} (p=${s.adaptive.measured_p}, K*=${s.adaptive.settled_k})`
  );
}
const rd = out.regime_drift;
console.log(
  `drift 0.30->0.90 @round 30: mean K ${rd.mean_k_before} -> ${rd.mean_k_after} ` +
  `(settled ${rd.settled_k}); steady adaptive ${rd.steady_tok_per_unit_adaptive} ` +
  `vs k8 ${rd.steady_tok_per_unit_k8}`
);
