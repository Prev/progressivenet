import { encode, decode } from './quantization';

describe('test_quantization', () => {
	it('test_4_4_8_16', () => {
		const input = [0.1, 0.11, 0.20, 0.21, 0.30, 0.31, 0.49, 4.0];
		const dividingInterface = [4, 4, 8, 16];

		const encoded = encode(input, dividingInterface);
		expect(encoded.min).toBeCloseTo(0.1);
		expect(encoded.scale).toBeCloseTo(3.9);

		// There are 4 levels of decoded values
		//  0: 1/8 (4-bit per scalar)
		//  1: 2/8 (8-bit per scalar)
		//  2: 4/8 (16-bit per scalar)
		//  3: 8/8 (32-bit per scalar)
		const decoded = [];
		for (let i = 0; i < 4; i++) {
			decoded.push(decode(
				encoded.buffers.slice(0, i+1),
				encoded.scale,
				encoded.min,
				dividingInterface,
			));
		}

		expect(decoded[0][0]).toBeLessThan(decoded[0][7]);

		for (let i = 0; i < input.length; i++) {
			expect(decoded[1][i]).toBeCloseTo(input[i], 1);
			expect(decoded[2][i]).toBeCloseTo(input[i], 2);
			expect(decoded[3][i]).toBeCloseTo(input[i], 4);
		}
	});

	it('test_5_11_16', () => {
		const input = [0.1, 0.11, 0.20, 0.21, 0.30, 0.31, 0.49, 4.0];
		const dividingInterface = [5, 11, 16];

		const encoded = encode(input, dividingInterface);
		expect(encoded.min).toBeCloseTo(0.1);
		expect(encoded.scale).toBeCloseTo(3.9);

		const decoded = [];
		for (let i = 0; i < 3; i++) {
			decoded.push(decode(
				encoded.buffers.slice(0, i+1),
				encoded.scale,
				encoded.min,
				dividingInterface,
			));
		}
		for (let i = 0; i < input.length; i++) {
			expect(decoded[0][i]).toBeCloseTo(input[i], 0);
			expect(decoded[1][i]).toBeCloseTo(input[i], 2);
			expect(decoded[2][i]).toBeCloseTo(input[i], 4);
		}
	});
});
