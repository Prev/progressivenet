import { DataType } from '@tensorflow/tfjs-core';

export interface PGNetConfig {
	layers: PGNetConfigLayerEntry[];
	files: string[];
	dividingInterface: number[];
}

export interface PGNetConfigLayerEntry {
	name: string;
	shape: number[];
	dtype: DataType;
	byteSizes: number[],
	quantization?: {
		scale: number,
		min: number,
	},
}

export type ArrayBufferMap = {[key: string]: ArrayBuffer};
