export async function fetchJSON(url: string) {
	const resp = await fetch(url);
	return await resp.json();
}

export async function fetchArrayBuffer(url: string) {
	const resp = await fetch(url);
	return await resp.arrayBuffer();
}
