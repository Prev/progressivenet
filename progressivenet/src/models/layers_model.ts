import * as tf from '@tensorflow/tfjs';
import { serialization } from '@tensorflow/tfjs-core';
import { PyJsonDict } from '@tensorflow/tfjs-layers/dist/keras_format/types';
import { convertPythonicToTs } from '@tensorflow/tfjs-layers/dist/utils/serialization_utils';
import { deserialize } from '@tensorflow/tfjs-layers/dist/layers/serialization';

export type LayersModel = tf.LayersModel;

export function loadLayersModel(modelJSON: any): LayersModel {
    let modelTopology = modelJSON['modelTopology'] as PyJsonDict;
    if (modelTopology['model_config'] != null) {
        modelTopology = modelTopology['model_config'] as PyJsonDict;
    }
    const tsConfig = convertPythonicToTs(modelTopology) as serialization.ConfigDict;
    return deserialize(tsConfig) as LayersModel;
}

