// Copyright 2026 The LoongForge Authors.
// SPDX-License-Identifier: Apache-2.0

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const { validatePullRequestTitle } = require('../.github/scripts/pr_title');
const { filterModelsWithBaselines, missingModels } = require('../.github/scripts/ci_model_validation');
const {
  findActionRequiredCheck,
  gpuConclusion,
  hasGpuValidation,
} = require('../.github/scripts/gpu_checks');

const repoRoot = path.resolve(__dirname, '..');

function readWorkflow(name) {
  return fs.readFileSync(path.join(repoRoot, '.github/workflows', name), 'utf8');
}

test('accepts a valid multi-module title', () => {
  const result = validatePullRequestTitle('[llm, ckpt] feat: add converter');
  assert.equal(result.ok, true);
  assert.deepEqual(result.modules, ['llm', 'ckpt']);
});

test('accepts Conventional Commits titles with a scope', () => {
  const result = validatePullRequestTitle('feat(fastwam): add structured FlashAttention backend');
  assert.equal(result.ok, true);
  assert.equal(result.type, 'feat');
  assert.equal(result.scope, 'fastwam');
  assert.deepEqual(result.modules, []);
});

test('accepts unscoped and breaking Conventional Commits titles', () => {
  assert.equal(validatePullRequestTitle('fix: handle empty input').ok, true);
  assert.equal(validatePullRequestTitle('feat(fastwam)!: change checkpoint format').ok, true);
  assert.equal(validatePullRequestTitle('feat!: change checkpoint format').breaking, true);
});

test('rejects unsafe Conventional Commits scopes', () => {
  const result = validatePullRequestTitle('feat(scope with spaces): invalid scope');
  assert.equal(result.ok, false);
  assert.match(result.message, /required format/i);
});

test('rejects a title without a module', () => {
  const result = validatePullRequestTitle('[, ] feat: empty modules');
  assert.equal(result.ok, false);
  assert.match(result.message, /at least one valid module/i);
});

test('rejects duplicate modules', () => {
  const result = validatePullRequestTitle('[ci, ci] fix: duplicate module');
  assert.equal(result.ok, false);
  assert.match(result.message, /duplicate module/i);
});

test('rejects an unknown module', () => {
  const result = validatePullRequestTitle('[unknown] fix: unknown module');
  assert.equal(result.ok, false);
  assert.match(result.message, /invalid modules/i);
});

test('new commits invalidate suite GPU jobs without SHA-scoped concurrency', () => {
  const regression = readWorkflow('gpu-regression.yml');
  const invalidation = readWorkflow('gpu-invalidate.yml');
  assert.doesNotMatch(regression, /group: gpu-regression-.*head_sha/);
  for (const suite of ['mcore', 'torch']) {
    assert.match(regression, new RegExp(`group: gpu-regression-.*-${suite}`));
    assert.match(invalidation, new RegExp(`group: gpu-regression-.*-${suite}`));
  }
});

test('GPU jobs use immutable trusted scripts and revalidate the PR head on the runner', () => {
  const regression = readWorkflow('gpu-regression.yml');
  assert.doesNotMatch(regression, /ref: \$\{\{ github\.ref \}\}/);
  assert.equal((regression.match(/ref: \$\{\{ github\.sha \}\}/g) || []).length, 4);
  assert.equal((regression.match(/name: Revalidate current PR head/g) || []).length, 2);
  assert.equal(
    (regression.match(/the requested revision is no longer the PR head/g) || []).length,
    3,
  );
});

test('GPU candidate image builds derive their target from the selected runner', () => {
  const regression = readWorkflow('gpu-regression.yml');
  assert.equal((regression.match(/build_candidate_image\.sh auto/g) || []).length, 2);
  const builder = fs.readFileSync(
    path.join(repoRoot, '.github/scripts/build_candidate_image.sh'),
    'utf8',
  );
  assert.match(builder, /target="\$\{1:-auto\}"/);
  assert.match(builder, /detect_gpu_target\.sh/);
});

test('GPU invalidation recognizes a prior check or bot result comment', () => {
  assert.equal(hasGpuValidation([{ name: 'gpu-regression' }], []), true);
  assert.equal(hasGpuValidation([], [
    { user: { login: 'github-actions[bot]' }, body: '<!-- loongforge-gpu-regression:abc -->' },
  ]), true);
  assert.equal(hasGpuValidation([], [
    { user: { login: 'maintainer' }, body: '<!-- loongforge-gpu-regression:abc -->' },
  ]), false);
  assert.equal(hasGpuValidation([], []), false);
});

test('GPU invalidation reuses only an action_required check for the current SHA', () => {
  const checks = [
    { id: 1, name: 'gpu-regression', head_sha: 'old', status: 'completed', conclusion: 'action_required' },
    { id: 2, name: 'gpu-regression', head_sha: 'new', status: 'completed', conclusion: 'failure' },
    { id: 3, name: 'gpu-regression', head_sha: 'new', status: 'completed', conclusion: 'action_required' },
    { id: 4, name: 'gpu-regression', head_sha: 'new', status: 'in_progress', conclusion: null },
  ];
  assert.equal(findActionRequiredCheck(checks, 'new')?.id, 3);
  assert.equal(findActionRequiredCheck(checks, 'missing'), null);
});

test('GPU conclusion keeps cancelled runs distinct from failures', () => {
  assert.deepEqual(
    gpuConclusion({ validate: 'success', torch: 'success', mcore: 'skipped' }, 'torch'),
    { conclusion: 'success', failed: [] },
  );
  assert.deepEqual(
    gpuConclusion({ validate: 'success', torch: 'cancelled', mcore: 'skipped' }, 'torch'),
    { conclusion: 'cancelled', failed: ['torch'] },
  );
  assert.deepEqual(
    gpuConclusion({ validate: 'success', torch: 'failure', mcore: 'skipped' }, 'torch'),
    { conclusion: 'failure', failed: ['torch'] },
  );
  assert.deepEqual(
    gpuConclusion({ validate: 'failure', torch: 'skipped', mcore: 'skipped' }, 'torch'),
    { conclusion: 'failure', failed: ['validate', 'torch'] },
  );
});

test('PR check exposes queued build regression and cancellation stages', () => {
  const dispatch = readWorkflow('ok-to-test.yml');
  const regression = readWorkflow('gpu-regression.yml');
  assert.match(dispatch, /title: 'GPU validation queued'/);
  assert.match(regression, /title: 'Building candidate image'/);
  assert.match(regression, /title: `Running \$\{process\.env\.SUITE\} regression`/);
  assert.match(regression, /GPU validation cancelled/);
  assert.match(regression, /if \(regressionSummary\) checkOutput\.text = regressionSummary/);
  assert.doesNotMatch(regression, /text: process\.env\.REGRESSION_SUMMARY \|\| null/);
  assert.match(regression, /finalize:\n    timeout-minutes: (?:6[1-9]|[7-9][0-9])/);
});

test('GPU status is separate from the required CPU gate', () => {
  const dispatch = readWorkflow('ok-to-test.yml');
  const regression = readWorkflow('gpu-regression.yml');
  assert.match(dispatch, /name: 'gpu-regression'/);
  assert.match(regression, /name: 'gpu-regression'/);
  assert.match(dispatch, /CI_ENABLE_MCORE|ENABLE_MCORE/);
  assert.doesNotMatch(dispatch, /Only the Torch suite/);
  assert.equal((regression.match(/CANDIDATE_REVISION:/g) || []).length, 2);
});

test('model validation checks the generic baseline alias', () => {
  const action = fs.readFileSync(
    path.join(repoRoot, '.github/actions/validate-models/action.yml'),
    'utf8',
  );
  assert.match(action, /baseline/);
  assert.match(action, /unknown .*model/);
});

test('model validation reports a model missing its baseline', () => {
  const requested = ['known_model', 'missing_model'];
  const validated = filterModelsWithBaselines(
    requested,
    requested,
    ['known_model'],
  );
  assert.deepEqual(missingModels(requested, validated), ['missing_model']);
});

test('mcore feature gate is explicit in both trusted workflows', () => {
  assert.match(
    readWorkflow('ok-to-test.yml'),
    /request\.suite === 'mcore' && process\.env\.ENABLE_MCORE !== 'true'/,
  );
  assert.match(
    readWorkflow('gpu-regression.yml'),
    /process\.env\.SUITE === 'mcore' && process\.env\.ENABLE_MCORE !== 'true'/,
  );
});

test('manual dispatch cannot publish the required PR gate name', () => {
  const workflow = readWorkflow('static-checks.yml');
  assert.match(
    workflow,
    /github\.event_name == 'pull_request'.*'static-checks'.*'manual-static-checks'/,
  );
});

test('Claude review excludes project settings', () => {
  const workflow = readWorkflow('claude-review.yml');
  assert.match(workflow, /--setting-sources user/);
  assert.doesNotMatch(workflow, /--setting-sources project/);
});

test('Claude review uses the configured runner labels', () => {
  const workflow = readWorkflow('claude-review.yml');
  assert.match(workflow, /runs-on: \$\{\{ fromJSON\(vars\.CI_REVIEW_RUNNER\) \}\}/);
});

test('Claude review uses the configured model and reports failures', () => {
  const workflow = readWorkflow('claude-review.yml');
  assert.match(workflow, /--model "\$\{\{ vars\.CLAUDE_REVIEW_MODEL \}\}"/);
  assert.match(workflow, /path_to_claude_code_executable:/);
  assert.match(workflow, /steps\.claude\.outcome == 'failure'/);
  assert.match(workflow, /api_error_status/);
  assert.match(workflow, /terminal_reason/);
  assert.doesNotMatch(workflow, /show_full_output:\s*true/);
});

test('release uses matching context and immutable image tags', () => {
  const workflow = readWorkflow('release.yml');
  assert.match(workflow, /group: release-\$\{\{ github\.ref \}\}/);
  assert.match(workflow, /workflow_dispatch:/);
  assert.match(workflow, /path: LoongForge/);
  assert.match(workflow, /needs: \[validate, package, publish\]/);
  assert.match(workflow, /submodules: recursive/);
  assert.match(workflow, /fromJSON\(vars\.CI_RELEASE_RUNNER\)/);
  assert.match(workflow, /docker\.io\/loongforge\/loongforge:/);
  assert.match(workflow, /build_candidate_image\.sh/);
  assert.match(workflow, /--release/);
  assert.match(workflow, /--image-ref "\$IMAGE_REF"/);
  assert.match(workflow, /--target auto/);
  assert.match(workflow, /manual releases must be dispatched from master/);
  assert.match(workflow, /image_policy\.py/);
  assert.match(workflow, /if: github\.event_name == 'push'/);
  assert.match(workflow, /docker push "\$\{\{ steps\.release_image\.outputs\.image_ref \}\}"/);
  assert.doesNotMatch(workflow, /docker\/build-push-action/);
  assert.doesNotMatch(workflow, /DOCKERHUB_IMAGE.*:latest/);
});

test('GPU workflow has an immutable run identity for the watchdog', () => {
  const workflow = readWorkflow('gpu-regression.yml');
  const watchdog = readWorkflow('gpu-watchdog.yml');
  assert.match(workflow, /run-name:/);
  assert.match(workflow, /GPU Regression PR #\$\{\{ inputs\.pr_number \}\}/);
  assert.match(workflow, /\$\{\{ inputs\.head_sha \}\}/);
  assert.match(watchdog, /workflow_run:/);
  assert.match(watchdog, /workflows: \['GPU Regression'\]/);
  assert.match(watchdog, /check\.head_sha !== headSha/);
});

test('all external Actions are pinned to full commit SHAs', () => {
  for (const name of fs.readdirSync(path.join(repoRoot, '.github/workflows'))) {
    if (!name.endsWith('.yml') && !name.endsWith('.yaml')) continue;
    const workflow = readWorkflow(name);
    for (const match of workflow.matchAll(/^\s+uses:\s+([^@\s]+)@([^\s#]+)/gm)) {
      if (match[1].startsWith('./')) continue;
      assert.match(match[2], /^[0-9a-f]{40}$/, `${name}: ${match[1]} is not pinned`);
    }
  }
  const action = fs.readFileSync(
    path.join(repoRoot, '.github/actions/validate-models/action.yml'),
    'utf8',
  );
  assert.match(action, /uses:\s+actions\/github-script@[0-9a-f]{40}/);
  const sync = readWorkflow('submodule-sync.yml');
  assert.match(sync, /create-github-app-token@[0-9a-f]{40}/);
  assert.match(sync, /token: \$\{\{ steps\.app-token\.outputs\.token \}\}/);
});

test('license checks use language-appropriate comment styles', () => {
  const config = fs.readFileSync(path.join(repoRoot, '.pre-commit-config.yaml'), 'utf8');
  const workflow = readWorkflow('license.yml');
  assert.ok(config.includes('alias: spdx-check-script'));
  assert.ok(config.includes("files: '\\.(py|sh)$"));
  assert.match(config, /--comment-style\s+- ['"]#['"]/);
  assert.ok(config.includes('alias: spdx-check-cpp'));
  assert.ok(config.includes("files: '\\.(c|cc|cpp|cxx|h|hh|hpp|hxx|cu|cuh)$"));
  assert.match(config, /--comment-style\s+- ['"]\/\/['"]/);
  assert.ok(workflow.includes('spdx-check-script'));
  assert.ok(workflow.includes('spdx-check-cpp'));
  assert.ok(workflow.includes("'*.cuh'"));
});

test('doc-links is part of the blocking static checks gate', () => {
  const workflow = readWorkflow('static-checks.yml');
  const docLinks = readWorkflow('doc-links.yml');
  assert.match(workflow, /uses: \.\/\.github\/workflows\/doc-links\.yml/);
  assert.match(workflow, /needs: \[pr-title, license, secrets, lint, build, workflow-lint, sensitive, doc-links\]/);
  assert.match(workflow, /DOC_LINKS_RESULT/);
  assert.match(workflow, /"doc-links:\$DOC_LINKS_RESULT"/);
  assert.match(docLinks, /actions\/checkout@[0-9a-f]{40}/);
  assert.match(docLinks, /ref: \$\{\{ github\.event\.pull_request\.head\.sha \|\| github\.sha \}\}/);
  assert.match(docLinks, /persist-credentials: false\s+fetch-depth: 0/);
  assert.match(docLinks, /actions\/setup-python@[0-9a-f]{40}/);
  assert.match(docLinks, /ci\/check_doc_links\.py --changed-since/);
});
