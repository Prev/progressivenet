import { promises as fs } from 'fs';
import * as path from 'path';
import * as tf from '@tensorflow/tfjs';

export async function loadModelAndWeightsConfig(modelDir: string): Promise<tf.ModelAndWeightsConfig> {
    const jsonFilePath = path.resolve(modelDir, 'model.json');
    const fileData = await fs.readFile(jsonFilePath, 'utf-8');
    return JSON.parse(fileData as string) as tf.ModelAndWeightsConfig;
}

export async function loadWeights(modelAndWeightsConfig: tf.ModelAndWeightsConfig, modelDir: string): Promise<tf.NamedTensorMap> {
    if (modelAndWeightsConfig.weightsManifest == null) {
        throw new Error('There is no weightManifest field on model.json object');
    }

    // Load weights from local files described in the manifest.
    // Replaces `@tensorflow/tfjs-core/io.loadWeights`.
    //  (Origianl file: https://github.com/tensorflow/tfjs/blob/master/tfjs-core/src/io/weights_loader.ts#L82)
    const fetchWeightsFromDisk = async (filePaths: string[]) => {
        const weightBuffers = [];
        for (let filePath of filePaths) {
            const buf = await fs.readFile(filePath);
            weightBuffers.push(buf.buffer)
        }
        return weightBuffers;
    }
    return tf.io.weightsLoaderFactory(fetchWeightsFromDisk)(modelAndWeightsConfig.weightsManifest, modelDir);
}
