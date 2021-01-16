import * as path from 'path';
import * as loader from './tfjs_model_loader';

// Check weight loss when we use quantization provided by TensorFlowJS
// https://github.com/tensorflow/tfjs/blob/master/tfjs-converter/python/tensorflowjs/quantization.py
// This file contains above code's algorithm, coverted from Python to TypeScript,
// then calculate RMSE from original tfjs model.

if (process.argv.length < 2) {
	console.log('Usage: npm run tfjs_quant <model_path> <quant_bit:4|8|16>');
	process.exit(1);
}

const quantBitStr = process.argv[3];
let quantBit: number;
if (['4', '8', '16'].includes(quantBitStr)) {
	quantBit = parseInt(quantBitStr);
} else {
	console.log('<quant_bit> should be either 4, 8, or 16');
	process.exit(1);
}

(async function() {
	const tfjsModelPath = path.resolve(__dirname, '..', process.argv[2]);

	const modelAndWeights = await loader.loadModelAndWeightsConfig(tfjsModelPath);
	const weights = await loader.loadWeights(modelAndWeights, tfjsModelPath);

	let errorSum = 0;
	let errorNum = 0;

	for (const name in weights) {
		const tensor = weights[name];
		const data = Array.from(await tensor.data());

		const {quantizedData, scale, minVal} = quantize(data, quantBit);
		const decoded = dequantizeWeights(quantizedData, scale, minVal);

		for (let i in data) {
			errorSum += Math.pow(data[i] - decoded[i], 2);
			errorNum++;
		}
	}

	console.log('RMSE:', Math.sqrt(errorSum / errorNum));
})();


function quantize(data: number[], quantBit: number) {
    let minVal = data[0];
    let maxVal = data[0];
    let scale: number;
    let quantizedData: number[];

    data.forEach(e => {
        if (e > maxVal) maxVal = e;
        if (e < minVal) minVal = e;
    });
    
    if (minVal == maxVal) {
        // quantized_data = np.zeros_like(data, dtype=quantization_dtype)
        quantizedData = data.map(_ => 0);
        scale = 1.0;

    }else {
        // Quantize data.
        [scale, minVal, maxVal] = _quantizationRange(minVal, maxVal, quantBit)
        
        // quantized_data = np.round((data.clip(min_val, max_val) - min_val) / scale).astype(np.uint16)
        quantizedData = data.map(e => {
            if (e < minVal) e = minVal;
            if (e > maxVal) e = maxVal;
            return Math.round((e - minVal) / scale);
        });

    }
    return {quantizedData, scale, minVal};
}


function _quantizationRange(minVal: number, maxVal: number, quantBit: number) {
    const quantMax = Math.pow(2, quantBit) - 1;
    const scale = (maxVal - minVal) / quantMax;

    let nudgedMin, nudgedMax;
    
    if (minVal <= 0 && 0 <= maxVal) {
        const quantizedZeroPoint = (0 - minVal) / scale;
        const nudgedZeroPoint = Math.floor(quantizedZeroPoint);
    
        nudgedMin = -nudgedZeroPoint * scale;
        nudgedMax = quantMax * scale + nudgedMin;

    } else {
        nudgedMin = minVal;
        nudgedMax = maxVal;
    }

    return [scale, nudgedMin, nudgedMax];
}


function dequantizeWeights(quantizedData: number[], scale: number, minVal: number) {
    return quantizedData.map(e => e * scale + minVal);
}
