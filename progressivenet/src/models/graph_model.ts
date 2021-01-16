import { Tensor, InferenceModel } from '@tensorflow/tfjs';
import { NamedTensorMap, ModelTensorInfo, ModelPredictConfig } from '@tensorflow/tfjs-core';
import { IGraphDef, ISignatureDef } from '@tensorflow/tfjs-converter/dist/data/compiled_api';
import { OperationMapper } from '@tensorflow/tfjs-converter/dist/operations/operation_mapper';
import { GraphExecutor } from '@tensorflow/tfjs-converter/dist/executor/graph_executor';
import { NamedTensorsMap } from '@tensorflow/tfjs-converter/dist/data/types';

// Converted version of https://github.com/tensorflow/tfjs/blob/master/tfjs-converter/src/executor/graph_model.ts
export class GraphModel implements InferenceModel {
	private executor: GraphExecutor;

	get inputNodes(): string[] {
		return this.executor.inputNodes;
	}
	
	get outputNodes(): string[] {
		return this.executor.outputNodes;
	}

	get inputs(): ModelTensorInfo[] {
		return this.executor.inputs;
	}
	
	get outputs(): ModelTensorInfo[] {
		return this.executor.outputs;
	}

	get weights(): NamedTensorsMap {
		return this.executor.weightMap;
	}

	constructor(modelJSON: any) {
		const graph = modelJSON['modelTopology'] as IGraphDef;
		let signature = {};
		if (modelJSON['userDefinedMetadata'] != null) {
			signature = modelJSON['userDefinedMetadata']['signature'] as ISignatureDef;
		}
		this.executor = new GraphExecutor(
			OperationMapper.Instance.transformGraph(graph, signature)
		);
	}

	loadWeights(weightMap: NamedTensorMap) {
		this.executor.weightMap = convertTensorMapToTensorsMap(weightMap);
	}

	predict(inputs: Tensor|Tensor[]|NamedTensorMap, config: ModelPredictConfig): Tensor|Tensor[]|NamedTensorMap {
		return this.execute(inputs, this.outputNodes);
	}

	execute(inputs: Tensor|Tensor[]|NamedTensorMap, outputs: string|string[]): Tensor|Tensor[] {
		inputs = this.normalizeInputs(inputs);
		outputs = this.normalizeOutputs(outputs);
		const result = this.executor.execute(inputs, outputs);
		return result.length > 1 ? result : result[0];
	}

	async executeAsync(
			inputs: Tensor|Tensor[]|NamedTensorMap,
			outputs?: string|string[],
	): Promise<Tensor|Tensor[]> {
		inputs = this.normalizeInputs(inputs);
		outputs = this.normalizeOutputs(outputs);
		const result = await this.executor.executeAsync(inputs, outputs);
		return result.length > 1 ? result : result[0];
	}

	dispose() {
		this.executor.dispose();
	}

	private normalizeInputs(inputs: Tensor|Tensor[]|NamedTensorMap): NamedTensorMap {
		if (!(inputs instanceof Tensor) && !Array.isArray(inputs)) {
			// The input is already a NamedTensorMap.
			return inputs;
		}
		inputs = Array.isArray(inputs) ? inputs : [inputs];
		if (inputs.length !== this.inputNodes.length) {
			throw new Error(
				'Input tensor count mismatch,' +
				`the graph model has ${this.inputNodes.length} placeholders, ` +
				`while there are ${inputs.length} input tensors.`);
		}
		return this.inputNodes.reduce((map, inputName, i) => {
			map[inputName] = (inputs as Tensor[])[i];
			return map;
		}, {} as NamedTensorMap);
	}

	private normalizeOutputs(outputs: string|string[]): string[] {
		outputs = outputs || this.outputNodes;
		return !Array.isArray(outputs) ? [outputs] : outputs;
	}
}

function convertTensorMapToTensorsMap(map: NamedTensorMap): NamedTensorsMap {
    return Object.keys(map).reduce((newMap: NamedTensorsMap, key) => {
		newMap[key] = [map[key]];
		return newMap;
	}, {});
}
