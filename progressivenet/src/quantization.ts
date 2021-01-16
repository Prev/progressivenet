import UintNArray from 'uint-n-array';

// Default interface is to divide a buffer into 4 parts.
// | part1 | part2 |   part3    |         part4            |
// | 4bit  | 4bit  |   8bit     |         16bit            |
const DEFAULT_BUFFER_INTERFACE = [4, 4, 8, 16];

export function encode(dataList: number[] | Float32Array| Int32Array | Uint8Array, bufferInterface?: number[]) {
	if (bufferInterface == undefined) {
		bufferInterface = DEFAULT_BUFFER_INTERFACE;
	}

	let max = dataList[0];
	let min = dataList[0];
	for (const data of dataList) {
		if (data > max) max = data;
		if (data < min) min = data;
	}

	const scale = max - min;
	const numRanges = 256 * 256 * 256 * 256; // 4bytes

	let normalized = [];
	for (let i = 0; i < dataList.length; i++) {
		if (dataList[i] == max)
			normalized[i] = numRanges - 1;
		else
			normalized[i] = Math.floor((dataList[i] - min) / scale * numRanges);
	}
	
	const buffers: ArrayBuffer[] = [];
	// For example, if bufferInterface is [4, 4, 8, 16],
	// then parts would be
	// [normalized.map(d => d >>> 28),
	//  normalized.map(d => (d << 4) >>> 28),
	//  normalized.map(d => (d << 8) >>> 24),
	//  normalized.map(d => (d << 16) >>> 16)]
	let offset = 0;
	for (let i = 0; i < bufferInterface.length; i++) {
		const bitSize = bufferInterface[i];
		const slicedData = normalized.map(d => (d << offset) >>> (32 - bitSize));
		
		buffers.push(UintNArray.encode(bitSize, slicedData));
		offset += bitSize;
	}

	return {
		'buffers': buffers,
		'bufferInterface': bufferInterface,
		'scale': scale,
		'min': min, 
	};
}

export function decode(buffers: ArrayBuffer[], scale: number, min: number, bufferInterface?: number[]) {
	if (bufferInterface == undefined) {
		bufferInterface = DEFAULT_BUFFER_INTERFACE;
	}
	const views = [];
	// Accumulated buffer interface.
	// For example, if the [4, 4, 8, 16], then accInterface would be [4, 8, 16, 32]
	const accInterface: number[] = [];

	for (let k = 0; k < buffers.length; k++) {
		const bitSize = bufferInterface[k];

		accInterface[k] = k > 0 ? accInterface[k-1] + bitSize : bitSize;
		views.push(UintNArray.decode(bitSize, buffers[k]));
	}
	const data = new Uint32Array(views[0].length);
	
	for (let i = 0; i < data.length; i++) {
		data[i] = 0;
		for (let k = 0; k < views.length; k++) {
			data[i] |= views[k][i] << (32 - accInterface[k]);
		}
	}

	// ReviseFactor is a half of the smallest unit.
	const reviseFactor = scale / (Math.pow(2, accInterface[accInterface.length - 1]) * 2);
	const ret = new Array(views[0].length);

	data.forEach((value, i) => {
		ret[i] = value / (Math.pow(2, 32)) * scale + min + reviseFactor;
	});
	return ret;
}
