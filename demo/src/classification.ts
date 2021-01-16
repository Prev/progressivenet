import { loadSequentially, Classifier, GraphModel, tf } from 'progressivenet';
import './common';

const modelUrl = 'http://gcpc.prev.kr/pgnet-models/classification/mobilenet_v2_intv2/';
const numProgressSteps = 8;
let predictionResults = [] as string[];

function run(img: HTMLImageElement, numProgressSteps: number) {
	const timelineImageWidth = 160;
	const timelineImageHeight = timelineImageWidth / img.width * img.height;

	const startTime = new Date();
	predictionResults = [];

	tf.setBackend('cpu');

	return loadSequentially({
		modelUrl,
		numProgressSteps,
		concurrentMode: true,
		logging: false,
	}, async (model, isLast, step) => {
		const inferenceStartTime = new Date();
		const classifier = new Classifier(model as GraphModel);
		const predictions = await classifier.classify(img);

		const finTime = new Date();

		console.log('inference time:', (finTime.getTime() - inferenceStartTime.getTime()) / 1000);
		console.log('total time:', (finTime.getTime() - startTime.getTime()) / 1000);
		console.log('----------------------------------------------')
		const timestamp = Math.round((finTime.getTime() - startTime.getTime()) / 1000 * 10) / 10;

		const predictionResult = predictions.map(e =>
			`${e.className} ${Math.round(e.probability * 100)}%`).join('<br>');

		$('.main .classification-result').html(predictionResult);

		$(`.scene${step}`).html(`
			<div class="time">${timestamp}s</div>
			<img src="${img.src}" width="${timelineImageWidth}" height="${timelineImageHeight}" />
			<div class="message">${predictionResult}</div>
		`);
		predictionResults.push(predictionResult);
	});
}

function onSceneOver(sceneNo: number) {
	$('.main .classification-result').html(predictionResults[sceneNo]);
}

function main() {
	for (let i = 0; i < numProgressSteps; i++) {
		const scene = $(`<div class="scene scene${i}"></div>`);
		scene.on('mouseover', () => onSceneOver(i));
		$('.timeline .scenes').append(scene);
	}	

	$('.timeline .scenes').on('mouseout', () => {
		if (predictionResults.length == 0) {
			return;
		}
		const recentResult = predictionResults[predictionResults.length - 1];
		$('.main .classification-result').html(recentResult);
	});

	$('.load-btn').on('click', async () => {
		$('.load-btn').prop('disabled', true);
		$('.upload-file-btn').prop('disabled', true);
		
		const img = document.getElementById('target-image') as HTMLImageElement;

		await run(img, numProgressSteps);

		$('.load-btn').prop('disabled', false);
		$('.upload-file-btn').prop('disabled', false);	
	});
}

main();

