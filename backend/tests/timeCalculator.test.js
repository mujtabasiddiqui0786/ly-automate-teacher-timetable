import assert from 'assert';
import { parseTimeRange, distributeDuration } from '../src/utils/timeCalculator.js';

const range = parseTimeRange('9:00-10:15');
assert.strictEqual(range.start, '9:00');
assert.strictEqual(range.end, '10:15');
assert.strictEqual(range.duration, 75);

const slices = distributeDuration('9:00', '10:00', 2);
assert.strictEqual(slices.length, 2);
assert.strictEqual(slices[0], 30);
assert.strictEqual(slices[1], 30);

// Ensure bad inputs are handled gracefully
assert.strictEqual(parseTimeRange('nope'), null);
assert.deepStrictEqual(distributeDuration('10:00', '9:00', 2), []);

console.log('timeCalculator tests passed');

