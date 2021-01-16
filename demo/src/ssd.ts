import { loadSequentially, ObjectDetector, GraphModel, tf } from 'progressivenet';
import './common';

const modelUrl = 'http://gcpc.prev.kr/pgnet-models/detection/ssd_mobilenet_v2_intv2/';
const numProgressSteps = 8;

interface DetectionResult {
	boxHTML: string;
	timestamp: number;
	label: string;
}

const boxColors = ['#1bc0ff', '#ff1b1b', '#ff9a33', '#d289ff'];
let detectionResults = [] as DetectionResult[];

function classToColor(className: string) {
	let hash = 0, i: number, chr: number;
	for (i = 0; i < className.length; i++) {
		chr = className.charCodeAt(i);
		hash = (hash * 31) + chr;
	}
	return boxColors[hash % boxColors.length];
}

async function run(img: HTMLImageElement, numProgressSteps: number) {
	const timelineImageWidth = 160;
	const timelineImageHeight = timelineImageWidth / img.width * img.height;

	const startTime = new Date();
	detectionResults = [];

	// tf.setBackend('cpu');

	return loadSequentially({
		modelUrl,
		numProgressSteps,
		concurrentMode: true,
	}, async (model, isLast, step) => {
		const inferenceStartTime = new Date();
		const detector = new ObjectDetector(model as GraphModel);
		const predictions = await detector.detect(img);

		const finTime = new Date();
		console.log('inference time:', (finTime.getTime() - inferenceStartTime.getTime()) / 1000);
		console.log('total time:', (finTime.getTime() - startTime.getTime()) / 1000);
		console.log('----------------------------------------------')
		const timestamp = Math.round((finTime.getTime() - startTime.getTime()) / 1000 * 10) / 10;

		let invalidObjects = 0;

		const boxHTML = predictions.map(p => {
			let [left, top, width, height] = p.bbox;
			if (left < 0 || top < 0) {
				invalidObjects++;
				return '';
			}
			width = width / img.width * 100;
			height = height / img.height * 100;
			left = left / img.width * 100;
			top = top / img.height * 100;

			const color = classToColor(p.class);
			const score = Math.round(p.score * 100) / 100;
			return `<div class="box" style="width: ${width}%; height: ${height}%; top: ${top}%; left: ${left}%; border-color: ${color}">
				<span style="background-color: ${color}">${p.class}(${score})</span>
			</div>`;
		}).join('');

		const label = 'Detected: ' + (predictions.length - invalidObjects);

		$('.main .detection-result').html(boxHTML);

		$(`.scene${step}`).html(`
			<div class="time">${timestamp}s</div>
			<img src="${img.src}" width="${timelineImageWidth}" height="${timelineImageHeight}" />
			<div class="detection-result" style="width: ${timelineImageWidth}px; height: ${timelineImageHeight}px">
				${boxHTML}
			</div>
			<div class="message">${label}</div>
		`);

		detectionResults.push({
			boxHTML,
			label,
			timestamp,
		});
	});
}

function onSceneOver(sceneNo: number) {
	$('.main .detection-result').html(detectionResults[sceneNo].boxHTML);
}

function main() {
	for (let i = 0; i < numProgressSteps; i++) {
		const scene = $(`<div class="scene scene${i}"></div>`);
		scene.on('mouseover', () => onSceneOver(i));
		$('.timeline .scenes').append(scene);
	}	

	$('.timeline .scenes').on('mouseout', () => {
		if (detectionResults.length == 0) {
			return;
		}
		const recentResult = detectionResults[detectionResults.length - 1];
		$('.main .classification-result').html(recentResult.boxHTML);
	});

	$('.load-btn').on('click', async () => {
		$('.load-btn').prop('disabled', true);
		$('.upload-file-btn').prop('disabled', true);

		const img = document.getElementById('target-image') as HTMLImageElement;

		$('.main .detection-result').css({
			width: img.width,
			height: img.height,
		});
		await run(img, numProgressSteps);

		$('.load-btn').prop('disabled', false);
		$('.upload-file-btn').prop('disabled', false);
	});
}

main();
