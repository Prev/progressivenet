/**
 * Fetch JSON from given URL
 */
export async function fetchJSON(url: string) {
    const resp = await fetch(url);
    return await resp.json();
}

/**
 * Fetch ArrayBuffer from given URL
 */
export async function fetchArrayBuffer(url: string) {
    const resp = await fetch(url);
    return await resp.arrayBuffer();
}
