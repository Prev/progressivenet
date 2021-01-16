import * as path from 'path';
import { promises as fs } from 'fs';
import { quantization, PGNetConfig } from 'progressivenet';
import * as loader from './tfjs_model_loader';


async function convert(tfjsModelPath: string, outputPath: string, progressiveInterface: number[]) {
    const modelAndWeightsConfig = await loader.loadModelAndWeightsConfig(tfjsModelPath);
    const weights = await loader.loadWeights(modelAndWeightsConfig, tfjsModelPath);

    // Remove old contents
    await fs.rmdir(outputPath, { recursive: true });
    await fs.mkdir(outputPath);

    const pgNetConfig = {
        'layers': [],
        'files': [],
        'dividingInterface': progressiveInterface,
    } as PGNetConfig;

    // Queuing data info buffers
    const bufferQueues: ArrayBuffer[][] = [];

    for (let k = 0; k < progressiveInterface.length; k++) {
        bufferQueues.push([]);
    }

    // Interate weights and convert to quantized data
    for (const name in weights) {
        const tensor = weights[name];
        const data = await tensor.data();

        if (tensor.dtype == 'float32') {
            // When the tensor is float32, quantize data and save it to multiple files.
            const encoded = quantization.encode(data, progressiveInterface);

            pgNetConfig.layers.push({
                'name': name,
                'shape': tensor.shape,
                'dtype': tensor.dtype,
                'quantization': {
                    'scale': encoded.scale,
                    'min': encoded.min,
                },
                'byteSizes': encoded.buffers.map(b => b.byteLength) as number[],
            });

            for (let i = 0; i < encoded.buffers.length; i++) {
                bufferQueues[i].push(encoded.buffers[i]);
            }
        } else if (tensor.dtype == 'int32') {
            // When the tensor is int32, we do not quantize data and save it to the first file only.
            const byteSizes = [(data as Int32Array).byteLength];
            for (let k = 1; k < progressiveInterface.length; k++) {
                byteSizes.push(0);
            }
            pgNetConfig.layers.push({
                'name': name,
                'shape': tensor.shape,
                'dtype': tensor.dtype,
                'byteSizes': byteSizes,
            });
            bufferQueues[0].push(data.buffer);
        } else {
            throw new Error(`Currently dtype "${tensor.dtype}" is not supported`);
        }
    }

    // Write weight binary files
    for (let i = 0; i < bufferQueues.length; i++) {
        const queue = bufferQueues[i];
        const fileName = `part-${i}.bin`;
        const file = await fs.open(path.resolve(outputPath, fileName), 'w');

        for (const arrayBuffer of queue) {
            await file.write(Buffer.from(arrayBuffer));
        }
        await file.close();
        pgNetConfig.files.push(fileName);
    }

    // Write config files
    const configFile = await fs.open(path.resolve(outputPath, 'progressive.json'), 'w');
    await configFile.write(JSON.stringify(pgNetConfig));
    await configFile.close();

    const modelFile = await fs.open(path.resolve(outputPath, 'model.json'), 'w');

    // Remove `weightManifest` field which is used on TFJS.
    // Progressive model uses progressive.json instead of this field.
    delete modelAndWeightsConfig.weightsManifest;
    await modelFile.write(JSON.stringify(modelAndWeightsConfig));
    await modelFile.close();
};

function main(argv: string[]) {
    if (argv.length < 4) {
        console.log('\nUsage:\n    $ progressivenet-convert MODEL_PATH OUT_DIR [INTERFACE]');
        console.log('\nExample:\n    $ progressivenet-convert ./mobilenet_v2 ./mobilenet_v2_2222 2,2,2,2')
        console.log('    $ progressivenet-convert ./mobilenet_v2 ./mobilenet_v2_44816 4,4,8,16')
        console.log('    $ progressivenet-convert ./mobilenet_v2 ./mobilenet_v2_511116 5,11,16')
        process.exit(1);
    }

    let progressiveInterface = [2, 2, 2, 2, 2, 2, 2, 2];
    if (argv.length == 5) {
        progressiveInterface = argv[4].split(',').map(e => parseInt(e));
    }

    // TODO: output dir check
    convert(argv[2], argv[3], progressiveInterface);
}

main(process.argv);
