// Tests for the display page's set report: score dimensions and focus lines per exercise
// (run: node --test tests/js/). The page is one inline script, so the pure block between
// SCORE_DIMS and the workout state is evaluated on its own.
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const html = readFileSync(new URL('../../src/visual/display.html', import.meta.url), 'utf8');
const block = html.slice(html.indexOf('const SCORE_DIMS'), html.indexOf('const workout = {'));
const { SCORE_DIMS, scoreDims, FOCUS_TEXT } = new Function(
    `${block}; return { SCORE_DIMS, scoreDims, FOCUS_TEXT };`,
)();

// Every deadlift fault type in .claude/deadlift/CONTRACT.md §1.
const DEADLIFT_FAULT_TYPES = [
    'deadlift_bar_position', 'deadlift_setup_hips', 'deadlift_shoulders_behind',
    'deadlift_hips_shoot', 'deadlift_bar_drift', 'deadlift_lockout', 'deadlift_lean_back',
    'deadlift_hip_shift', 'deadlift_bar_tilt', 'deadlift_bent_arms', 'deadlift_velocity_loss',
];

const keys = (dims) => dims.map(([key]) => key);

describe('score dimensions per exercise', () => {
    test('squat dimensions are unchanged', () => {
        assert.deepEqual(keys(scoreDims('squat')),
            ['depth', 'trunk_control', 'knee_tracking', 'symmetry', 'tempo']);
    });

    test('deadlift has its own five dimensions', () => {
        assert.deepEqual(keys(scoreDims('deadlift')),
            ['setup', 'coordination', 'bar_path', 'lockout', 'symmetry']);
    });

    test('an exercise without its own list falls back to the squat', () => {
        assert.equal(scoreDims(undefined), SCORE_DIMS.squat);
        assert.equal(scoreDims('overhead_press'), SCORE_DIMS.squat);
    });
});

describe('focus lines', () => {
    test('every deadlift fault has a focus line', () => {
        for (const faultType of DEADLIFT_FAULT_TYPES) {
            assert.ok(FOCUS_TEXT[faultType], `missing FOCUS_TEXT for ${faultType}`);
        }
    });

    test('no deadlift focus line claims a back shape', () => {
        for (const faultType of DEADLIFT_FAULT_TYPES) {
            assert.doesNotMatch(FOCUS_TEXT[faultType], /back flat|flat back|round/i);
        }
    });
});

describe('report tiles', () => {
    test('null depth hides the depth tiles instead of showing a dash', () => {
        assert.match(html, /t\("tile-depth-box"\)\.hidden = summary\.avg_depth == null;/);
        assert.match(html, /t\("tile-consistency-box"\)\.hidden = summary\.depth_consistency == null;/);
    });

    test('the workout start event sets the profile, squat by default', () => {
        assert.match(html, /workout\.profile = msg\.profile \|\| "squat";/);
        assert.match(html, /scoreDims\(workout\.profile\)\.forEach/);
    });
});
