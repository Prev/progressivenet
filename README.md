# ProgressiveNet [![npm version](https://badge.fury.io/js/progressivenet.svg)](https://www.npmjs.com/package/progressivenet)

Official implementation of the paper "Progressive Transmission and Inference of Deep Learning Models", to appear at 20th IEEE International Conference on Machine
Learning and Applications (ICMLA 2021).

[[arXiv](https://arxiv.org/abs/2110.00916)]
[[demo](https://prev.github.io/progressivenet/)]

[Youngsoo Lee](https://github.com/Prev)<sup>1</sup>, [Sangdoo Yun](https://github.com/hellbell)<sup>2</sup>, [Yeonghun Kim](https://github.com/Neutrinoant)<sup>1</sup> Sunghee Choi<sup>1</sup>

<sup>1</sup> School of Computing, Korea Advanced Institute of Science and Technology(KAIST)<br>
<sup>2</sup> AI LAB, NAVER Corp.

<p align="center">
    <img src="figures/overview.png?v2" width="400" alt="Overview">
</p>

### Abstract

Modern image files are usually progressively transmitted and provide a preview before downloading the entire image for improved user experience to cope with a slow network connection.
In this paper, with a similar goal, we propose a progressive transmission framework for deep learning models, especially to deal with the scenario where pre-trained deep learning models are transmitted from servers and executed at user devices (e.g., web browser or mobile).
Our progressive transmission allows inferring approximate models in the middle of file delivery, and quickly provide an acceptable intermediate outputs.
On the server-side, a deep learning model is divided and progressively transmitted to the user devices.
Then, the divided pieces are progressively concatenated to construct approximate models on user devices.
Experiments show that our method is computationally efficient without increasing the model size and total transmission time while preserving the model accuracy.
We further demonstrate that our method can improve the user experience by providing the approximate models especially in a slow connection.


## Examples and Demo

**Image classification Model (MobileNetV2):**

<p align="center">
    <img src="figures/classification-example.jpg?v2" width="100%" alt="Object Detection Model Example">
</p>

**Object Detection Model (SSD-MobileNetV2):**

<p align="center">
    <img src="figures/detection-example.jpg?v2" width="100%" alt="Object Detection Model Example">
</p>

**Demo:**
You can try using our pre-built models in [https://prev.github.io/progressivenet](https://prev.github.io/progressivenet)


## Getting Started

ProgressiveNet is implemented based on [TensorFlowJS](https://github.com/tensorflow/tfjs), which is an open-source hardware-accelerated JavaScript library for training and deploying machine learning models.
Our implementation interacts with the TFJS core engine, by passing the intermediate weight matrices to the TFJS and receives the approximiate results in the middle of the transmission.

### How to install

```
$ npm install progressivenet
```

### How to use (Classification Model)

```js
import { ProgressiveNet, Classifier } from 'progressivenet';

const pgNet = new ProgressiveNet({
    modelUrl: '<model_url>',
    numProgressSteps: 8,
});
const img = document.getElementById('my-img');

pgNet.loadSequentially((model, isLast, progressStep) => {
    const classifier = new Classifier(model);
    const predictions = await classifier.classify(img);
    console.log(`Result from step #${progressStep}: ${predictions}`);

}).then() => {
    console.log('All models are transmitted and inferenced');
}
```

### How to use (Detection Model)

```js
import { ProgressiveNet, ObjectDetector } from 'progressivenet';

const pgNet = new ProgressiveNet({ modelUrl: '<model_url>' });
const img = document.getElementById('my-img');

pgNet.loadSequentially((model, isLast, progressStep) => {
    const detector = new ObjectDetector(model);
    const predictions = await detector.detect(img);
    console.log(`Result from step #${progressStep}: ${predictions}`);

}).then() => {
    console.log('All models are transmitted and inferenced');
}
```

You can see the detailed code in [demo](./demo). It is written in TypeScript and implemented with jQuery.

## Models

ProgressiveNet requires a dedicated model file for the progressive execution (inference).<br>
We provide MobileNetV2 and SSD-MobileNetV2 models in the [Release](hhttps://github.com/Prev/progressivenet/releases).

### Generating custom model

Rather than implementing a *training* code for ProgressiveNet, we provide a **converter** for generating progressive model from static TensorFlowJS model.

Towards making your own model beyond our pre-converted models, try visiting [converter](./converter) directory for advanced usage.
Converter is implemented with TypeScript and and we provide command line interface(CLI) to convert TFJS models into progressive models.

If you want to apply ProgressiveNet into your own service with your own models, you can follow the instructions below:

1. Writing the training code with TensorFlow.
2. Train the model and save it in a TF format (either SavedModel or Keras).
3. Convert your model into a TFJS model by following the [guide](https://www.tensorflow.org/js/guide/conversion).
4. Convert the TFJS model into Progressive model by following the [instruction on our repo](./converter).


## License

CC 4.0 Attribution-NonCommercial International.
The software is for educaitonal and academic research purpose only.
