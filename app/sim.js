'use strict';
/*
 * sdecode session simulation — deterministic port of sdecode.core semantics.
 *
 * Workload model (labeled synthetic, by design):
 *   The target model's next-token preference is prefix-independent, so the
 *   verification of drafted position i is an i.i.d. Bernoulli(q) draw —
 *   homogeneous acceptance, exactly the regime the closed form models:
 *       E[accepted run] = q(1-q^K)/(1-q)
 *   Real LMs have heterogeneous acceptance; this workload isolates the cost
 *   accounting so the session ledger demonstrates the controller rather than
 *   sampling noise. Optional regime drift (q steps mid-session) shows the
 *   controller re-tuning live.
 *
 * Faithfulness notes:
 *   - specRound mirrors core.spec_step + greedy_verify exactly, including the
 *     full-accept case (returns the K drafted tokens, no bonus token).
 *   - AdaptiveController mirrors AdaptiveSpeculator (window=8, K=2 warm start).
 *   - The closed form mirrors expected_tokens / optimal_draft_len.
 *   - The PRNG (mulberry32) is fixed so this file, the committed receipt run
 *     (receipts/session.json) and the browser app are the same session.
 */

function mulberry32(a) {
  return function () {
    a |= 0;
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---- closed form (identical to sdecode.core) ---------------------------- */
function expectedTokens(p, k) {
  return p >= 1 ? k + 1 : (1 - Math.pow(p, k + 1)) / (1 - p);
}
function optimalDraftLen(p, c, kMax) {
  if (kMax === undefined) kMax = 8;
  let bestK = 1, bestG = 0;
  for (let k = 1; k <= kMax; k++) {
    const g = expectedTokens(p, k) / (1 + c * k);
    if (g > bestG) { bestG = g; bestK = k; }
  }
  return bestK;
}

/* ---- synthetic target: one preferred token, prefix-independent ------------ */
class ToyTarget {
  constructor(vocab, rng) {
    this.vocab = vocab;
    this.preferred = Math.floor(rng() * vocab);
  }
  forward() {
    const lg = new Array(this.vocab).fill(-1);
    lg[this.preferred] = 1;
    return lg;
  }
}
function argmax(a) {
  let bi = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i;
  return bi;
}

/* ---- drafter: proposes the target's preferred token w.p. q, else random ----
 * The flip is a fresh draw per call, so proposed-position correctness is
 * i.i.d. Bernoulli(q) regardless of prefix. */
function makeDrafter(target, qOf, rng) {
  return function () {
    if (rng() < qOf()) return target.preferred;
    let r = Math.floor(rng() * target.vocab);
    if (r === target.preferred) r = (r + 1) % target.vocab;
    return r;
  };
}

/* ---- one speculative round (port of core.spec_step + greedy_verify) ------- */
function specRound(seq, target, drafter, K) {
  const draft = [];
  for (let i = 0; i < K; i++) draft.push(drafter());
  const tgt = [];
  for (let i = 0; i <= K; i++) tgt.push(argmax(target.forward()));
  let acc = 0;
  while (acc < K && draft[acc] === tgt[acc]) acc++;
  // accepted + 1 tokens in EVERY case: drafts up to the first mismatch plus
  // exactly one target token (correction on failure, bonus on full accept).
  const emitted = draft.slice(0, acc).concat([tgt[acc]]);
  return { emitted: emitted, accepted: acc };
}

/* ---- controller (port of AdaptiveSpeculator) ------------------------------ */
function AdaptiveController(c, kMax, window) {
  if (c === undefined) c = 0.1;
  if (kMax === undefined) kMax = 8;
  if (window === undefined) window = 8;
  this.c = c;
  this.kMax = kMax;
  this.window = window;
  this.rounds = []; // [accepted, drafted] per round
}
AdaptiveController.prototype.estimateP = function () {
  if (!this.rounds.length) return 0;
  let acceptedFail = 0, draftedFull = 0, failures = 0;
  for (const r of this.rounds) {
    if (r[0] === r[1]) draftedFull += r[1];
    else { acceptedFail += r[0]; failures += 1; }
  }
  const num = acceptedFail + draftedFull + 0.5;
  const den = acceptedFail + draftedFull + failures + 1.0;
  return Math.min(1, Math.max(0, num / den));
};
AdaptiveController.prototype.nextK = function () {
  if (this.rounds.length < 2) return 2;
  return optimalDraftLen(this.estimateP(), this.c, this.kMax);
};
AdaptiveController.prototype.observe = function (accepted, drafted) {
  this.rounds.push([accepted, drafted]);
  if (this.rounds.length > this.window) this.rounds.shift();
};

/* censored-geometric MLE over a lane's rounds — mirrors
 * AdaptiveSpeculator.estimate_p; this is what "measured p" reports. */
function estimatePFromRounds(rounds) {
  if (!rounds.length) return null;
  let acceptedFail = 0, draftedFull = 0, failures = 0;
  for (const r of rounds) {
    if (r.k === 0) continue;
    if (r.accepted === r.k) draftedFull += r.k;
    else { acceptedFail += r.accepted; failures += 1; }
  }
  if (acceptedFail + draftedFull + failures === 0) return null;
  const num = acceptedFail + draftedFull + 0.5;
  const den = acceptedFail + draftedFull + failures + 1.0;
  return Math.round(Math.min(1, Math.max(0, num / den)) * 1e4) / 1e4;
}

/* ---- one decode lane under a fixed target-call budget ---------------------
 * kind: 'naive' | 'k8' | 'adaptive'
 * Every round costs exactly ONE target call, so all lanes spend the same
 * target budget and the honest comparison is tokens-per-target-call.
 * cfg.q may be a number, or {before, after, at} for a mid-session regime step.
 * Steady state = the last 30 rounds (past the controller's 8-round window),
 * which flushes the controller's warm-up so the ledger compares settled
 * policy rather than startup luck.
 */
function runLane(kind, cfg) {
  const vocab = cfg.vocab, c = cfg.c, budget = cfg.budget;
  const rng = mulberry32(cfg.seed + (kind === 'naive' ? 0 : kind === 'k8' ? 1 : 2));
  const target = new ToyTarget(vocab, rng);
  const qSpec = typeof cfg.q === 'object' ? cfg.q : { before: cfg.q, after: cfg.q, at: budget };
  const qOf = function (round) { return round < qSpec.at ? qSpec.before : qSpec.after; };
  const drafter = makeDrafter(target, function () { return qOf(rounds.length); }, rng);
  const ctrl = new AdaptiveController(c);
  let seq = cfg.prompt.slice();
  let targetCalls = 0, drafterCalls = 0, tokens = 0, acceptedSum = 0, draftedSum = 0;
  const rounds = [];
  while (targetCalls < budget) {
    if (kind === 'naive') {
      seq.push(target.preferred);
      tokens += 1;
      targetCalls += 1;
      rounds.push({ k: 0, accepted: 1, emitted_len: 1, q: qOf(rounds.length) });
    } else {
      const K = kind === 'k8' ? 8 : ctrl.nextK();
      const r = specRound(seq, target, drafter, K);
      seq = seq.concat(r.emitted);
      tokens += r.emitted.length;
      drafterCalls += K;
      targetCalls += 1;
      acceptedSum += r.accepted;
      draftedSum += K;
      rounds.push({
        k: K, accepted: r.accepted, emitted_len: r.emitted.length, q: qOf(rounds.length),
      });
      if (kind === 'adaptive') ctrl.observe(r.accepted, K);
    }
  }
  const units = targetCalls + c * drafterCalls;
  // steady state: last 30 rounds (past the controller's 8-round window)
  const steadyFrom = Math.max(0, budget - 30);
  const steadyRounds = rounds.slice(steadyFrom);
  let sTokens = 0, sDrafter = 0;
  for (const r of steadyRounds) { sTokens += r.emitted_len; sDrafter += r.k; }
  const sTarget = steadyRounds.length;
  const sUnits = sTarget + c * sDrafter;
  return {
    kind: kind,
    tokens: tokens,
    target_calls: targetCalls,
    drafter_calls: drafterCalls,
    units: Math.round(units * 1e6) / 1e6,
    units_per_token: Math.round((units / tokens) * 1e6) / 1e6,
    tokens_per_target_call: Math.round((tokens / targetCalls) * 1e4) / 1e4,
    usd_per_mtok: Math.round((units / tokens) * cfg.price_base * 1e4) / 1e4,
    measured_p: estimatePFromRounds(rounds),
    settled_k: rounds.length ? rounds[rounds.length - 1].k : null,
    steady_state: {
      rounds: sTarget,
      tokens: sTokens,
      tok_per_unit: Math.round((sTokens / sUnits) * 1e4) / 1e4,
      usd_per_mtok: Math.round((sTokens / sUnits) * cfg.price_base * 1e4) / 1e4,
      vs_naive: Math.round((sTokens / sUnits) * 100) / 100,
    },
    rounds: rounds,
  };
}
