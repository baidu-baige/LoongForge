// Copyright 2026 The LoongForge Authors.
// SPDX-License-Identifier: Apache-2.0

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

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
  for (const suite of ['llm_vlm', 'embodied']) {
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
    gpuConclusion({ validate: 'success', embodied: 'success', llm_vlm: 'skipped' }, 'embodied'),
    { conclusion: 'success', failed: [] },
  );
  assert.deepEqual(
    gpuConclusion({ validate: 'success', embodied: 'cancelled', llm_vlm: 'skipped' }, 'embodied'),
    { conclusion: 'cancelled', failed: ['embodied'] },
  );
  assert.deepEqual(
    gpuConclusion({ validate: 'success', embodied: 'failure', llm_vlm: 'skipped' }, 'embodied'),
    { conclusion: 'failure', failed: ['embodied'] },
  );
  assert.deepEqual(
    gpuConclusion({ validate: 'failure', embodied: 'skipped', llm_vlm: 'skipped' }, 'embodied'),
    { conclusion: 'failure', failed: ['validate', 'embodied'] },
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
  assert.match(dispatch, /CI_ENABLE_LLM_VLM|ENABLE_LLM_VLM/);
  assert.doesNotMatch(dispatch, /Only the embodied suite/);
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

test('llm_vlm feature gate is explicit in both trusted workflows', () => {
  assert.match(
    readWorkflow('ok-to-test.yml'),
    /request\.suite === 'llm_vlm' && process\.env\.ENABLE_LLM_VLM !== 'true'/,
  );
  assert.match(
    readWorkflow('gpu-regression.yml'),
    /process\.env\.SUITE === 'llm_vlm' && process\.env\.ENABLE_LLM_VLM !== 'true'/,
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

test('release is dispatched manually and builds, regresses, and promotes one immutable image', () => {
  const workflow = readWorkflow('release.yml');
  assert.match(workflow, /on:\n  workflow_dispatch:/);
  assert.doesNotMatch(workflow, /push:\s*\n\s*tags:/);
  assert.match(workflow, /__dry_run__/);
  assert.match(workflow, /needs\.validate\.outputs\.dry_run == 'false'/);
  assert.match(workflow, /  dry_run:\n/);
  assert.match(workflow, /group: release-\$\{\{ github\.ref \}\}/);
  assert.doesNotMatch(workflow, /PUBLISH_TO_PYPI/);
  assert.match(workflow, /path: LoongForge/);
  assert.match(workflow, /needs: validate\n/);
  assert.match(workflow, /submodules: recursive/);
  assert.match(workflow, /fromJSON\(vars\.CI_RELEASE_RUNNER\)/);
  assert.doesNotMatch(workflow, /CI_DOCKERHUB_PUBLISHER_RUNNER/);
  assert.match(workflow, /iregistry-release/);
  assert.match(workflow, /promote_to_dockerhub:/);
  assert.match(workflow, /promote_to_dockerhub\.sh/);
  assert.doesNotMatch(workflow, /docker_redacted\.sh push "\$DOCKERHUB_REF"/);
  assert.match(workflow, /DOCKERHUB_IMAGE:.*docker\.io\/loongforge\/loongforge/);
  assert.match(workflow, /build_candidate_image\.sh/);
  assert.match(workflow, /--release/);
  assert.match(workflow, /--image-ref "\$IMAGE_REF"/);
  assert.match(workflow, /loongforge-release:\$\{\{ env\.VERSION \}\}-\$\{\{ github\.run_id \}\}/);
  assert.match(workflow, /--target auto/);
  assert.match(workflow, /production publishing must use a matching release tag/);
  assert.match(workflow, /release tag must match pyproject version/);
  assert.match(workflow, /release tag must point to a commit on master/);
  assert.match(workflow, /release tags cannot use a debug suffix/);
  assert.match(workflow, /compare\/master\.\.\.\$\{GITHUB_SHA\}/);
  assert.match(workflow, /DEBUG_TAG_SUFFIX" == ""/);
  assert.match(workflow, /Validate release stage switches/);
  assert.match(workflow, /release_request\.py/);
  assert.match(workflow, /image_policy\.py/);
  assert.match(workflow, /run_release_regression\.sh/);
  assert.doesNotMatch(workflow, /run_release_regression_slot\.sh/);
  assert.match(workflow, /RUN_RELEASE_REGRESSION: \$\{\{ needs\.validate\.outputs\.run_regression \}\}/);
  assert.match(workflow, /PUBLISH_TO_INTERNAL_REGISTRY: \$\{\{ needs\.validate\.outputs\.publish_internal \}\}/);
  assert.match(workflow, /publish_dockerhub == 'true'/);
  assert.match(workflow, /if: env\.RUN_RELEASE_REGRESSION == 'true'/);
  assert.doesNotMatch(workflow, /LOONGFORGE_RELEASE_RUN_REGRESSION/);
  assert.match(workflow, /LOONGFORGE_RELEASE_IMAGE_TAG_PREFIX/);
  assert.match(workflow, /tag_suffix="\$\{DEBUG_TAG_SUFFIX:-\}"/);
  assert.doesNotMatch(workflow, /LOONGFORGE_RELEASE_TAG_SUFFIX/);
  assert.match(workflow, /dockerhub_ref="\$\{DOCKERHUB_IMAGE\}:\$\{VERSION\}\$\{tag_suffix\}"/);
  assert.match(workflow, /load_release_config\.sh/);
  assert.match(workflow, /load_release_credentials\.sh iregistry/);
  assert.match(workflow, /load_release_credentials\.sh dockerhub/);
  assert.doesNotMatch(workflow, /secrets\.IREGISTRY/);
  assert.doesNotMatch(workflow, /secrets\.DOCKERHUB/);
  assert.match(workflow, /LOONGFORGE_RUNNER_MASK_SCRIPT/);
  assert.match(workflow, /Protect runner-local metadata/);
  const releaseRunnerJob = workflow.slice(workflow.indexOf('build_test_push_internal:'));
  assert.ok(
    releaseRunnerJob.indexOf('Protect runner-local metadata') <
      releaseRunnerJob.indexOf('actions/checkout@'),
  );
  for (const job of workflow.split(/\n(?=  [a-z][a-z0-9_]*:\n)/)) {
    if (!job.includes('runs-on: ${{ fromJSON(vars.CI_RELEASE_RUNNER) }}')) continue;
    const configuration = job.slice(0, job.indexOf('\n    steps:'));
    assert.match(
      configuration,
      /RUNNER_PRIVATE_LOG_MASKS: \$\{\{ secrets\.RUNNER_PRIVATE_LOG_MASKS \}\}/,
    );
  }
  assert.doesNotMatch(workflow, /\/ssd4\//);
  assert.doesNotMatch(workflow, /needs\.build_test_push_internal\.outputs/);
  assert.doesNotMatch(workflow, /baidu-int\.com/);
  assert.match(workflow, /env\.PUBLISH_TO_INTERNAL_REGISTRY == 'true'/);
  assert.match(workflow, /needs\.validate\.outputs\.publish_dockerhub == 'true'/);
  assert.match(workflow, /Tag validated image for iregistry/);
  assert.match(workflow, /docker_redacted\.sh push "\$INTERNAL_IMAGE_REF"/);
  assert.match(workflow, /Pull, tag, and push image to Docker Hub/);
  assert.match(workflow, /Verify Docker Hub image/);
  assert.match(workflow, /Resolve Docker Hub image reference/);
  assert.match(workflow, /DOCKERHUB_REF: \$\{\{ steps\.hub_ref\.outputs\.dockerhub_ref \}\}/);
  assert.doesNotMatch(workflow, /docker_redacted\.sh push "\$DOCKERHUB_REF"/);
  assert.match(workflow, /cleanup_debug_tags:/);
  assert.match(workflow, /env\.BUILD_IMAGE == 'true' \|\| env\.PUBLISH_TO_INTERNAL_REGISTRY == 'true'/);
  assert.match(workflow, /cleanup_debug_tags\.py/);
  assert.match(workflow, /cleanup_debug_tags\.py \\\n\s+dockerhub/);
  const debugCleanup = workflow.slice(workflow.indexOf('  cleanup_debug_tags:'));
  assert.match(debugCleanup, /needs\.validate\.outputs\.publish_dockerhub == 'true'/);
  assert.doesNotMatch(debugCleanup, /publish_internal|INTERNAL_IMAGE|internal_ref|iregistry|--host/);
  assert.doesNotMatch(workflow, /cleanup_debug_tags\.py \\\n\s+internal/);
  assert.match(workflow, /github_release:/);
  assert.match(workflow, /permissions:\n\s+contents: write/);
  assert.match(workflow, /gh release view "\$RELEASE_TAG"/);
  assert.match(workflow, /gh release create "\$\{args\[@\]\}"/);
  assert.match(workflow, /--generate-notes/);
  assert.match(workflow, /--prerelease/);
  assert.match(workflow, /needs\.validate\.outputs\.build_image == 'true'/);
  assert.match(workflow, /needs\.validate\.outputs\.run_regression == 'true'/);
  assert.match(workflow, /needs\.validate\.outputs\.publish_internal == 'true'/);
  assert.match(workflow, /needs\.validate\.outputs\.publish_dockerhub == 'true'/);
  assert.match(workflow, /if: always\(\)/);
  assert.match(workflow, /build_image:/);
  assert.match(workflow, /run_regression:/);
  assert.match(workflow, /publish_internal:/);
  assert.match(workflow, /publish_dockerhub:/);
  assert.match(workflow, /execution_slot:/);
  assert.match(workflow, /llm_model:/);
  assert.match(workflow, /vla_model:/);
  assert.match(workflow, /validate_release_models\.py/);
  assert.match(workflow, /release_request\.py/);
  assert.match(workflow, /needs\.validate\.outputs\.build_image/);
  assert.match(workflow, /needs\.validate\.outputs\.run_regression/);
  assert.match(workflow, /needs\.validate\.outputs\.publish_internal/);
  assert.match(workflow, /needs\.validate\.outputs\.publish_dockerhub/);
  assert.match(workflow, /needs\.validate\.outputs\.execution_slot/);
  assert.match(workflow, /resolve_execution_slot\.sh/);
  assert.match(workflow, /run_release_regression\.sh/);
  assert.match(workflow, /RELEASE_EXECUTION_SLOT/);
  assert.match(workflow, /RELEASE_LLM_MODEL/);
  assert.match(workflow, /RELEASE_VLA_MODEL/);
  assert.match(workflow, /baseline_chip:/);
  assert.match(workflow, /vars\.RELEASE_BASELINE_CHIP/);
  assert.match(workflow, /BASELINE_CHIP: \$\{\{ needs\.validate\.outputs\.baseline_chip \}\}/);
  assert.match(workflow, /--baseline-chip "\$BASELINE_CHIP"/);
  assert.doesNotMatch(workflow, /VAR_BASELINE_CHIP:-[ap]/);
  assert.match(workflow, /timeout-minutes: 90/);
  assert.match(workflow, /DOCKER_CONFIG: \/tmp\/loongforge-release-docker-config-/);
  assert.match(workflow, /INTERNAL_IMAGE_REPOSITORY: \$\{\{ secrets\.INTERNAL_IMAGE_REPOSITORY \}\}/);
  assert.doesNotMatch(workflow, /INTERNAL_IMAGE_REPOSITORY: \$\{\{ vars\.INTERNAL_IMAGE_REPOSITORY \}\}/);
  assert.match(workflow, /refs\/tags\/v\(0\|\[1-9\]\[0-9\]\*\)/);
  assert.match(workflow, /-rc\\\.\(0\|\[1-9\]\[0-9\]\*\)/);
  assert.doesNotMatch(workflow, /docker\/build-push-action/);
  // A docker-container builder pulls its builder image from Docker Hub, which
  // an internal-only release runner cannot reach.
  assert.doesNotMatch(workflow, /setup-buildx-action/);
  assert.doesNotMatch(workflow, /DOCKERHUB_IMAGE.*:latest/);
});

test('release refuses self-hosted execution without startup masks', () => {
  const workflow = readWorkflow('release.yml');
  const step = workflow.match(
    /- name: Validate runner startup masking\n([\s\S]*?)\n      - id: version/,
  );
  assert.ok(step, 'startup mask validation must precede self-hosted execution');
  assert.match(step[1], /if: steps\.stages\.outputs\.skip_regression_reason != '__dry_run__'/);
  assert.match(step[1], /RUNNER_PRIVATE_LOG_MASKS: \$\{\{ secrets\.RUNNER_PRIVATE_LOG_MASKS \}\}/);
  const script = step[1].split('        run: |\n')[1];
  for (const value of ['', ' \n\t', 'http://proxy.example.invalid:8080\nproxy.example.invalid']) {
    const result = spawnSync('bash', ['-c', script], {
      env: { ...process.env, RUNNER_PRIVATE_LOG_MASKS: value },
      encoding: 'utf8',
    });
    assert.equal(result.status, value.trim() ? 0 : 1);
    assert.doesNotMatch(result.stdout + result.stderr, /proxy\.example\.invalid/);
  }
});

test('GPU regression gates and masks self-hosted execution', () => {
  const workflow = readWorkflow('gpu-regression.yml');
  assert.match(workflow, /name: Validate runner startup masking/);
  const jobs = workflow.split(/\n(?=  [a-z][a-z0-9_]*:\n)/);
  for (const jobName of ['llm_vlm', 'embodied']) {
    const job = jobs.find((candidate) => candidate.startsWith(`  ${jobName}:`));
    assert.ok(job, `missing ${jobName} job`);
    const configuration = job.slice(0, job.indexOf('\n    steps:'));
    assert.match(
      configuration,
      /RUNNER_PRIVATE_LOG_MASKS: \$\{\{ secrets\.RUNNER_PRIVATE_LOG_MASKS \}\}/,
    );
  }
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
  assert.match(docLinks, /\.github\/scripts\/check_doc_links\.py --changed-since/);
});
