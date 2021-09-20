# Converter

Rather than implementing a training code for ProgressiveNet, we provide a **converter** for generating progressive model from static TensorFlowJS model.

## How to install

```
$ npm install -g progressivenet-converter
```

## How to use

### Prerequisites

To convert a model, you have to prepare TFJS models for conversion.
[TensorFlow Hub]([TFHub](https://tfhub.dev/s?deployment-format=tfjs)) provides various types of pretrained models in free.
You can filter the model format in TFHub, thus you can easily find 
the TFJS models in here.
The TFJS model has the structure like below:

```
group1-shard1of5.bin
group1-shard2of5.bin
group1-shard3of5.bin
group1-shard4of5.bin
group1-shard5of5.bin
model.json
```

### Convert model

To convert TFJS model to progressive model, run command like below:

```bash
$ progressivenet-converter <TFJS_MODEL_PATH> <OUT_DIR> [<INTERFACE>]
```

`INTERFACE` indicates the *bit-width list* in the paper.
For example, if you set it to `4,4,4,4`, then the model is divided to four parts, allowing `[4,8,12,16]-bit` models progressively.

Here are the examples:

```bash
$ progressivenet-converter ./mobilenet_v2 ./mobilenet_v2_2222 2,2,2,2
$ progressivenet-converter ./mobilenet_v2 ./mobilenet_v2_44816 4,4,8,16
```


## Running in local

To build and run convert in local, run commands below:

```bash
# Link progressivenet package
$ cd ../progressivenet
$ npm link
$ cd ../converter
$ npm link progressivenet

# Install packages
$ npm install

# Compile typescript
$ npm run build

# Register command line
$ npm install -g .
```

