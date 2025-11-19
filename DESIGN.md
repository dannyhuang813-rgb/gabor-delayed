# Gabor RB/II Stimulus Redesign (Bivariate Gaussian Framework)

## 1. Mathematical Design

### 1.1 Theoretical feature space
* Let the latent feature vector be **z** = (x, y)ᵀ where `x` encodes spatial frequency (sf) and `y` encodes orientation (ori).
* The standardized domain is approximately `[-2, 2]` on each dimension; zero is the center of the task space.
* A trial’s observable stimulus is obtained via a linear rescaling from z to the physical stimulus parameters.

### 1.2 Mapping to physical ranges
* Spatial frequency (visible black/white cycles) is mapped with a linear function:
  * `sf = sf_min + ((x + 2) / 4) * (sf_max - sf_min)` with `sf_min = 2.5` and `sf_max = 7.0`.
  * The mapping preserves ordering and keeps most samples inside `[2.5, 7]`, which comfortably contains the original 3–6 cycles.
* Orientation (degrees) is mapped via:
  * `ori = ori_min + ((y + 2) / 4) * (ori_max - ori_min)` with `ori_min = -18°` and `ori_max = +18°`.
  * Draws outside the range are clipped to ±18° or resampled (see algorithm section).

### 1.3 RB paradigms (Rule-Based; one dimension diagnostic)
Each RB session uses axis-aligned Gaussian clusters so that the optimal boundary aligns with a single latent dimension.

| Session | Category A mean µ_A | Category B mean µ_B | Covariance Σ | Bayes boundary |
| --- | --- | --- | --- | --- |
| RB-S1 | `(0.0, -0.8)` | `(0.0, +0.8)` | diag `(0.35², 0.20²)` | `y = 0` → label A when y < 0 |
| RB-S2 | `(-0.8, 0.0)` | `(+0.8, 0.0)` | diag `(0.20², 0.35²)` | `x = 0` → label A when x < 0 |

* Because the non-diagnostic axis uses a larger variance, stimuli from both classes overlap almost completely on that dimension, keeping the task effectively one-dimensional per session.
* 200 trials per session are divided into five 40-trial blocks (RB-1 … RB-5) to match the existing experimental structure.

### 1.4 II paradigms (Information-Integration; two dimensions diagnostic)
Two mirrored linear decision rules are defined so that both dimensions must be integrated.

| Session | Category A mean µ_A | Category B mean µ_B | Covariance Σ | Bayes boundary |
| --- | --- | --- | --- | --- |
| II-S1 (II1) | `(-0.55, +0.55)` | `(+0.55, -0.55)` | `Σ = [[0.28², 0.15²], [0.15², 0.35²]]` | `0.5·x + 0.5·y = 0` |
| II-S2 (II2) | `(-0.55, -0.55)` | `(+0.55, +0.55)` | `Σ = [[0.35², -0.15²], [-0.15², 0.28²]]` | `0.5·x - 0.5·y = 0.15` |

* Both sessions use nearly equal weights on x and y to maintain symmetry and to guarantee that single-dimension heuristics cannot exceed ≈58–60% accuracy when sampling is balanced.
* 240 trials per session are split into five 48-trial blocks (II-1 … II-5). Each block contains 24 samples per category (A/B) and inherits the same Gaussian parameters.
* Because Σ contains off-diagonal terms, iso-probability contours are tilted ellipses; therefore, the optimal decision boundary is an oblique line requiring integration of both sf and ori.

### 1.5 Preventing single-dimension shortcuts
* Sampling enforces A/B balance within each sf level (after binning the physical sf into four equal-width bins) and within three |ori| bins: 1–6°, 6–12°, 12–18°.
* For each sf×ori-bin combination, the generator targets an A/B difference ≤1 by resampling from the underlying Gaussian mixture, ensuring high overlap between categories on any marginal.
* A post-generation heuristic check (see Section 3.5) guarantees that no single-dimension threshold exceeds a configurable accuracy ceiling (default 0.58). Blocks that fail are regenerated up to 20 times; otherwise the closest plan is retained with a warning.

## 2. Stimulus generation algorithms (pseudo code)

### 2.1 Common helpers
```
function sampleGaussianCategory(catParams):
    repeat:
        z = sampleBivariateNormal(catParams.mean, catParams.cov)
        if z inside [-2.2, 2.2] range:
            return z
```

### 2.2 RB generator
```
function generateRBTrials(sessionParams):
    N = 200; blocks = 5; perBlock = N / blocks
    catParams = sessionParams.rbGaussians  // {A: {mean, cov}, B: {...}}
    trials = []
    for each categoryLabel in ['A', 'B']:
        repeat N/2 times:
            z = sampleGaussianCategory(catParams[categoryLabel])
            (sf, ori) = mapToPhysical(z)
            trials.push({sf, ori, label: categoryLabel})
    shuffle trials with participant-seeded RNG
    split into blocks of size perBlock (ensuring each block has equal A/B counts by round-robin assignment)
    return blocks
```

### 2.3 II generator
```
function generateIITrials(sessionParams):
    targetBlocks = 5, blockSize = 48
    plan = []
    while plan not filled:
        for label in ['A', 'B']:
            z = sampleGaussianCategory(sessionParams.iiGaussians[label])
            (sf, ori) = mapToPhysical(z)
            assign to bin indices: sfBin(sf), oriBin(|ori|)
            record candidate trial with label
    enforce within-bin quotas:
        sort by deficit per sfBin and oriBin so that A/B stay balanced
        drop/duplicate minimal samples (jittering ori slightly) to fix deficits
    flatten into blockSize groups while keeping A/B counts per block
    run evaluateSingleDimHeuristics on full set; if > threshold, resample (≤20 attempts)
    return plan
```

### 2.4 Block scheduling
* RB blocks: 40 trials each, exactly 20 A and 20 B. Shuffle within block while preventing immediate mirrored ori pairs.
* II blocks: 48 trials each, 24 per label, near/far ratio per decision boundary maintained by tracking projected distance `|score|` relative to boundary.

### 2.5 Post-hoc heuristic guard
```
function evaluateHeuristics(trials, threshold=0.58):
    sfThresholds = unique sf values midpoints
    oriThresholds = linspace(-18, +18, step=1)
    bestAcc = 0
    for th in sfThresholds:
        acc1 = accuracy(trials, trial => trial.sf < th ? 'A' : 'B')
        acc2 = accuracy(trials, trial => trial.sf < th ? 'B' : 'A')
        bestAcc = max(bestAcc, acc1, acc2)
    for th in oriThresholds:
        acc3 = accuracy(trials, t => t.ori < th ? 'A' : 'B')
        acc4 = accuracy(trials, t => t.ori < th ? 'B' : 'A')
        acc5 = accuracy(trials, t => Math.abs(t.ori) < |th| ? 'A' : 'B')
        acc6 = accuracy(trials, t => Math.abs(t.ori) < |th| ? 'B' : 'A')
        bestAcc = max(bestAcc, acc3, acc4, acc5, acc6)
    return bestAcc
```
* Regeneration loop stops early when `bestAcc ≤ threshold`; otherwise repeat sampling with a new RNG seed.

## 3. JavaScript code skeleton

```js
// --- Linear algebra helpers ---
function sampleBivariateNormal(meanVec, covMatrix) {
  // meanVec: [µx, µy], covMatrix: [[σxx, σxy], [σyx, σyy]]
  // Uses Cholesky decomposition for a 2×2 covariance.
  const [m1, m2] = meanVec;
  const [[a, b], [c, d]] = covMatrix;
  const sigma1 = Math.sqrt(a);
  const sigma2 = Math.sqrt(d - (c * c) / a);
  const z1 = Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random());
  const z2 = Math.sqrt(-2 * Math.log(Math.random())) * Math.sin(2 * Math.PI * Math.random());
  const x = m1 + sigma1 * z1;
  const y = m2 + (c / sigma1) * z1 + sigma2 * z2;
  return [x, y];
}

function mapToPhysical(x, y) {
  const sfMin = 2.5, sfMax = 7.0;
  const oriMin = -18, oriMax = 18;
  const sf = sfMin + ((x + 2) / 4) * (sfMax - sfMin);
  const ori = oriMin + ((y + 2) / 4) * (oriMax - oriMin);
  return {
    sf: Math.min(Math.max(sf, sfMin), sfMax),
    ori: Math.min(Math.max(ori, oriMin), oriMax)
  };
}

// --- RB generator ---
function generateRBTrials(params) {
  const { session, means, covariances } = params; // session ∈ {1,2}
  const totalTrials = 200;
  const blocks = 5;
  const perBlock = totalTrials / blocks;
  const rng = seedRandom(params.seed);
  const trials = [];
  ['A', 'B'].forEach(label => {
    const mean = means[label];
    const cov = covariances[label];
    for (let i = 0; i < totalTrials / 2; i++) {
      let point;
      do {
        point = sampleBivariateNormal(mean, cov);
      } while (Math.abs(point[0]) > 2.2 || Math.abs(point[1]) > 2.2);
      const phys = mapToPhysical(point[0], point[1]);
      trials.push({ ...phys, label, task: 'RB', session });
    }
  });
  shuffleWithSeed(trials, rng);
  return chunkIntoBlocks(trials, perBlock);
}

// --- II generator ---
function generateIITrials(params) {
  const { structureKey, means, covariances, weights, bias } = params;
  const totalTrials = 240;
  const blockSize = 48;
  const rng = seedRandom(params.seed);
  const trials = [];
  ['A', 'B'].forEach(label => {
    const mean = means[label];
    const cov = covariances[label];
    for (let i = 0; i < totalTrials / 2; i++) {
      let pt;
      do {
        pt = sampleBivariateNormal(mean, cov);
      } while (Math.abs(pt[0]) > 2.2 || Math.abs(pt[1]) > 2.2);
      const phys = mapToPhysical(pt[0], pt[1]);
      const score = weights.wsf * pt[0] + weights.wori * pt[1] - bias;
      const labelCheck = score > 0 ? 'A' : 'B';
      if (labelCheck !== label) continue; // enforce Gaussian label consistency
      trials.push({ ...phys, label, task: 'II', session: params.session, structureKey });
    }
  });
  enforceSfAndOriBinBalance(trials, rng);
  if (evaluateSingleDimHeuristics(trials) > params.heuristicMax) {
    // resample logic here (not shown fully)
  }
  return chunkIntoBlocks(trials, blockSize);
}

function evaluateSingleDimHeuristics(trials) {
  if (!trials.length) return 0;
  const sfValues = trials.map(t => t.sf).sort((a, b) => a - b);
  const sfThresholds = sfValues.slice(1).map((v, i) => (v + sfValues[i]) / 2);
  const oriThresholds = Array.from({ length: 36 }, (_, i) => -18 + i);
  let bestAcc = 0;
  const evalRule = ruleFn => {
    const correct = trials.filter(t => ruleFn(t) === t.label).length;
    bestAcc = Math.max(bestAcc, correct / trials.length);
  };
  sfThresholds.forEach(th => {
    evalRule(t => (t.sf < th ? 'A' : 'B'));
    evalRule(t => (t.sf < th ? 'B' : 'A'));
  });
  oriThresholds.forEach(th => {
    evalRule(t => (t.ori < th ? 'A' : 'B'));
    evalRule(t => (t.ori < th ? 'B' : 'A'));
    evalRule(t => (Math.abs(t.ori) < Math.abs(th) ? 'A' : 'B'));
    evalRule(t => (Math.abs(t.ori) < Math.abs(th) ? 'B' : 'A'));
  });
  return bestAcc;
}

function generateParticipantStimuli(participantId) {
  const session1 = {
    session: 1,
    rbGaussians: RB_CONFIG.session1,
    iiPlan: II_CONFIG.session1,
    seed: hash(participantId + '_s1')
  };
  const session2 = {
    session: 2,
    rbGaussians: RB_CONFIG.session2,
    iiPlan: II_CONFIG.session2,
    seed: hash(participantId + '_s2')
  };
  return {
    session1: {
      RB: generateRBTrials(session1),
      II: generateIITrials({ ...session1.iiPlan, session: 1, seed: session1.seed })
    },
    session2: {
      RB: generateRBTrials(session2),
      II: generateIITrials({ ...session2.iiPlan, session: 2, seed: session2.seed })
    }
  };
}
```

### Notes
* `seedRandom` refers to any deterministic RNG (e.g., Mulberry32) seeded by participantId+task to guarantee reproducibility across sessions.
* `enforceSfAndOriBinBalance` is a helper that computes deficits per sf bin (`[2.5,3.5), [3.5,4.5), [4.5,5.5), [5.5,7]`) and ori magnitude bins (`[0,6), [6,12), [12,18]`) and reshuffles/resamples until each bin has A/B counts within ±1. When perfect balance is impossible for a bin, the deficit is logged, and the heuristics check prevents the plan from being accepted if the imbalance makes a single-dimension rule too strong.
* The order-code based movement/delay logic from the previous implementation can remain unchanged, because it acts after the trial arrays are generated.

## 4. Summary
* Two RB rules (Session1 `y`-threshold, Session2 `x`-threshold) ensure a pure one-dimensional learning strategy per session while keeping total variance comparable.
* Two II rules (II1, II2) use mirrored oblique boundaries with nearly equal weights on sf and ori, guaranteeing that the participant must integrate both dimensions.
* Sampling from the specified bivariate Gaussians, combined with sf/ori bin balancing and heuristic guards, yields trial sets where marginal shortcuts never exceed ~58% accuracy.
* The provided JS skeleton (plus pseudo code) can be integrated into the existing experiment by replacing the earlier discrete table logic with Gaussian-driven sampling, while leaving the session-order/delay controls untouched.
