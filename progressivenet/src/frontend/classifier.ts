/**
 * Object Classfier.
 * Based on https://github.com/tensorflow/tfjs-models/tree/master/mobilenet/
 *
 * Usage:
 *   import { ProgressiveNet, Classifier, GraphModel } from 'progressivenet';
 *   const pNet = new ProgressiveNet(modelUrl);
 *   await pNet.init();
 *
 *   for (let i = 0; i < iterCnt; i++) {
 *     await pNet.loadNext();
 *     const classifier = new Classifier(pNet.model as GraphModel);
 *	   const predictions = await classifier.classify(img);
 *     console.log(predictions);
 *   }
 */
import { GraphModel, tf } from '../index';
import { IMAGENET_CLASSES } from './classes/imagenet';

const IMAGE_SIZE = 224;

export class Classifier {
    model: GraphModel;

    private normalizationConstant: number;
    inputMin = -1;
    inputMax = 1;

    constructor(model: GraphModel, inputMin: number = -1, inputMax: number = 1) {
        this.model = model;
        this.inputMin = inputMin;
        this.inputMax = inputMax;
        this.normalizationConstant = (this.inputMax - this.inputMin) / 255.0;
    }

    /**
     * Computes the logits (or the embedding) for the provided image.
     *
     * @param img The image to classify. Can be a tensor or a DOM element image,
     *         video, or canvas.
     * @param embedding If true, it returns the embedding. Otherwise it returns
     *         the 1000-dim logits.
     */
    infer(img: tf.Tensor | ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement): tf.Tensor {
        return tf.tidy(() => {
            if (!(img instanceof tf.Tensor)) {
                img = tf.browser.fromPixels(img);
            }

            // Normalize the image from [0, 255] to [inputMin, inputMax].
            const normalized: tf.Tensor3D = img
                .toFloat()
                .mul(this.normalizationConstant)
                .add(this.inputMin);

            // Resize the image to
            let resized = normalized;
            if (img.shape[0] !== IMAGE_SIZE || img.shape[1] !== IMAGE_SIZE) {
                const alignCorners = true;
                resized = tf.image.resizeBilinear(
                    normalized,
                    [IMAGE_SIZE, IMAGE_SIZE],
                    alignCorners
                );
            }

            // Reshape so we can pass it to predict.
            const batched = resized.reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 3]);

            let result: tf.Tensor2D;
            const logits1001 = this.model.predict(batched, {}) as tf.Tensor2D;
            // Remove the very first logit (background noise).
            result = logits1001.slice([0, 1], [-1, 1000]);
            return result;
        });
    }

    /**
     * Classifies an image from the 1000 ImageNet classes returning a map of
     * the most likely class names to their probability.
     *
     * @param img The image to classify. Can be a tensor or a DOM element image,
     * video, or canvas.
     * @param topk How many top values to use. Defaults to 3.
     */
    async classify(
        img: tf.Tensor3D | ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement,
        topk = 3
    ): Promise<Array<{ className: string; classId: number, probability: number }>> {
        const logits = this.infer(img) as tf.Tensor2D;
        const classes = await getTopKClasses(logits, topk);
        logits.dispose();
        return classes;
    }
}

async function getTopKClasses(logits: tf.Tensor2D, topK: number):
        Promise<Array<{ className: string; classId: number, probability: number }>> {
    const softmax = logits.softmax();
    const values = await softmax.data();
    softmax.dispose();

    const valuesAndIndices = [];
    for (let i = 0; i < values.length; i++) {
        valuesAndIndices.push({ value: values[i], index: i });
    }
    valuesAndIndices.sort((a, b) => {
        return b.value - a.value;
    });
    const topkValues = new Float32Array(topK);
    const topkIndices = new Int32Array(topK);
    for (let i = 0; i < topK; i++) {
        topkValues[i] = valuesAndIndices[i].value;
        topkIndices[i] = valuesAndIndices[i].index;
    }

    const topClassesAndProbs = [];
    for (let i = 0; i < topkIndices.length; i++) {
        topClassesAndProbs.push({
            className: IMAGENET_CLASSES[topkIndices[i]],
            classId: topkIndices[i],
            probability: topkValues[i],
        });
    }
    return topClassesAndProbs;
}
