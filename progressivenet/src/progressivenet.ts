import * as tf from '@tensorflow/tfjs';
import { concat, NamedTensorMap } from '@tensorflow/tfjs-core';
import * as quantization from './quantization';
import * as io from './io';
import { PGNetConfig, ArrayBufferMap } from './types';
import { loadLayersModel } from './models/layers_model';
import { GraphModel } from './models/graph_model';

export function loadSequentially(
		config: {
			modelUrl: string,
			numProgressSteps?: number,
			concurrentMode?: boolean,
			logging?: boolean,
		},
		callback: (model: tf.LayersModel | GraphModel, isLast: boolean, progressStep: number)
			=> void | Promise<void>,
	) {
	const pNet = new ProgressiveNet(config);
	return pNet.loadSequentially(callback);
}

/**
 * ProgressiveNet Class.
 * 
 * Usage
 * --------------------
 * const pgNet = new ProgressiveNet({modelUrl: '<model_url>'});
 * await pgNet.loadSequentially((model, isLast, progressStep) => {
 *     const result = model.predict(tensor);
 * 	   console.log(result);
 * });
 * console.log('All models are transmitted and inferenced');
 */
export class ProgressiveNet {
	public model: tf.LayersModel | GraphModel;
	protected modelUrl: string;
	protected pgNetConfig: PGNetConfig;
	protected bufferMaps: ArrayBufferMap[] = [];
	protected currentStep = 0;

	protected numProgressSteps: number;
	protected concurrentMode: boolean;
	protected logging: boolean;

	constructor(config: {
		modelUrl: string,
		numProgressSteps?: number,
		concurrentMode?: boolean,
		logging?: boolean,
	}) {
		this.modelUrl = config.modelUrl;
		if (this.modelUrl[this.modelUrl.length - 1] == '/') {
			this.modelUrl = this.modelUrl.substr(0, this.modelUrl.length - 1);
		}

		this.numProgressSteps = config.numProgressSteps;
		this.concurrentMode = config.concurrentMode !== undefined ? config.concurrentMode : true;
		this.logging = config.logging !== undefined ? config.logging : false;
	}

	/**
	 * Load progressive model sequentially.
	 * @param callback: Callback function which is called when intermediate model is available.
	 * @param config: {
	 * 	- numProgressSteps: Total number of steps for intermeidate result.
	 * 		Load to the end if not defined.
	 * 	- concurrentMode: Execute transmission and inference concurrently if set to true.
	 *  - logging: Logging message if set to true
	 * }
	 * @returns Promise<tf.LayersModel | GraphModel> instance where resolveFunc is called after
	 * 	all progress is completed.
	 */
	public loadSequentially(
		callback: (model: tf.LayersModel | GraphModel, isLast: boolean, progressStep: number)
			=> (void | Promise<void>),
	) {
		const nextStep = async (step: number,
								onFinish: (model: tf.LayersModel | GraphModel) => void,
								onError: (e: any) => void) => {
			
			let buffer = null as ArrayBuffer;
			try {
				buffer = await this.loadBuffer(step);
			} catch(e) {
				onError(e);
				return;
			}
			const isLast = (step == this.numProgressSteps - 1);

			if (this.concurrentMode) {
				// If concurrentMode is enabled, we first call nextStep function
				// to make browser load the next part of the data.
				// Note that we do not use `await` keyword when calling nextStep function.
				if (!isLast) {
					nextStep(step + 1, onFinish, onError);
				}
				await this.restoreModelFromBuffer(step, buffer);
				callback(this.model, isLast, step);

			} else {
				await this.restoreModelFromBuffer(step, buffer);
				const returnVal = callback(this.model, isLast, step);
				if (Promise.resolve(returnVal) == returnVal) {
					// If callback is an async function, we wait for it before going nextStep.
					await returnVal;
				}
				if (!isLast) {
					nextStep(step + 1, onFinish, onError);
				}
			}

			if (isLast) {
				onFinish(this.model);
			}
		}

		const promise = new Promise<tf.LayersModel | GraphModel>((resolveFunc, rejectFunc) => {
			this.init().then(() => {
				nextStep(0, resolveFunc, rejectFunc);
			})
		});
		return promise;
	}

	/**
	 * Init model by loading configs of the model.
	 */
	async init() {
		// Load progressive.json config file
		this.pgNetConfig = await io.fetchJSON(this.modelUrl + '/progressive.json') as PGNetConfig;

		// Set `numProgressSteps` from the interface described in the config.
		if (this.numProgressSteps === undefined ||
			this.pgNetConfig.dividingInterface.length < this.numProgressSteps
		) {
			this.numProgressSteps = this.pgNetConfig.dividingInterface.length;
		}

		// Load model.json file, which has similar format with TFJS
		const modelJSON = await io.fetchJSON(this.modelUrl + '/model.json');
		if (modelJSON['format'] == 'layers-model') {
			this.model = loadLayersModel(modelJSON);

		} else if (modelJSON['format'] == 'graph-model') {
			this.model = new GraphModel(modelJSON);
		
		} else if (modelJSON['format'] === undefined) {
			console.warn('There is not `format` field on `model.json` file.' +
						 'Try to load model with Graph format. It may not work.');
			this.model = new GraphModel(modelJSON);

		} else {
			throw new Error('Unknown model format.' +
							'Supported formats are `layers-model` and `graph-model`');
		}
	}

		/**
	 * Load next part of the model.
	 * @returns current step of the transmission, which start from zero.
	 */
	async loadNext() {
		if (!this.pgNetConfig) {
			const message = 'Method loadNext() should be called after init() is called';
			console.error(message);
			throw new Error(message);
		}

		const curStep = this.currentStep;
		if (curStep >= this.numProgressSteps) {
			return -1;
		}

		const buffer = await this.loadBuffer(curStep);
		await this.restoreModelFromBuffer(curStep, buffer);

		this.currentStep = curStep + 1;
		return curStep;
	}

	/**
	 * Load weight file
	 */
	protected async loadBuffer(progressStep: number): Promise<ArrayBuffer> {
		return await io.fetchArrayBuffer(this.modelUrl + '/' + this.pgNetConfig.files[progressStep]);
	}

	/**
	 * Init model weights from the ArrayBuffer.
	 * Split the single array buffer to mutiple array buffers, where the number of buffers
	 * is equal to the number of layers described in `pgNetConfig`.
	 * @param progressStep: Part to download, which starts from zero.
	 */
	protected async restoreModelFromBuffer(progressStep: number, buffer: ArrayBuffer) {
		const startTime = new Date();
		// Build ArrayBufferMap from ArrayBuffer
		const bufferMap = {} as ArrayBufferMap;
		let offset = 0;
		for (const layer of this.pgNetConfig.layers) {
			const len = layer.byteSizes[progressStep];
			bufferMap[layer.name] = buffer.slice(offset, offset + len);
			offset += len;
		}

		// Save bufferMap for future use
		this.bufferMaps.push(bufferMap);

		// Build tensorMap from buffferMaps (previous buffers + recently loaded buffer)
		// by de-quantizing matrices.
		const tensorMap = this.buildTensorMap(this.bufferMaps);

		if (this.logging) {
			console.log('restore. time:', ((new Date()).getTime() - startTime.getTime()) / 1000);
		}
		this.model.loadWeights(tensorMap);
	}

	/**
	 * Load weight buffers of part #{progressStep}.
	 * Split the single array buffer to mutiple array buffers, where the number of buffers
	 * is equal to the number of layers described in `pgNetConfig`.
	 * @param progressStep: Part to download, which starts from zero.
	 * @returns An ArrayBufferMap instance
	 */
	protected async loadWeightBuffers(progressStep: number): Promise<ArrayBufferMap> {
		const buffer = await io.fetchArrayBuffer(this.modelUrl + '/' + this.pgNetConfig.files[progressStep]);
		const ret = {} as ArrayBufferMap;
		let offset = 0;

		for (const layer of this.pgNetConfig.layers) {
			const len = layer.byteSizes[progressStep];
			ret[layer.name] = buffer.slice(offset, offset + len);
			offset += len;
		}
		return ret;
	}

	/**
	 * Restore tensorMap by performing de-quantization.
	 */
	protected buildTensorMap(bufferMaps: ArrayBufferMap[]) {
		const nameToTensorMap: NamedTensorMap = {};
		for (const layer of this.pgNetConfig.layers) {
			let data: number[] | Int32Array;

			if (layer.dtype == 'float32') {
				const buffers = bufferMaps.map(map => map[layer.name]);
				data = quantization.decode(
					buffers,
					layer.quantization.scale,
					layer.quantization.min,
					this.pgNetConfig.dividingInterface,
				);

			} else if (layer.dtype == 'int32') {
				// When the tensor is int32, data is not quantized and
				// it is saved to the first file only.
				data = new Int32Array(bufferMaps[0][layer.name]);

			} else {
				throw new Error(`Currently dtype "${layer.dtype}" is not supported`);
			}
			nameToTensorMap[layer.name] = tf.tensor(data, layer.shape, layer.dtype);
		}
		return nameToTensorMap;
	}

	protected getInitTensorMap() {
		const nameToTensorMap: NamedTensorMap = {};
		for (const layer of this.pgNetConfig.layers) {
			if (layer.dtype == 'float32') {
				nameToTensorMap[layer.name] = tf.fill(
					layer.shape,
					layer.quantization.min + layer.quantization.scale * 0.5,
					layer.dtype,
				);
			} else {
				nameToTensorMap[layer.name] = tf.zeros(layer.shape, layer.dtype);
			}
		}
		return nameToTensorMap;
	}
}
