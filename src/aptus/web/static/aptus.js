const tileX = 200;
let cx = -0.6, cy = 0.0;
let pixsize = 3.0/600;
let canvasW = 0, canvasH = 0;

function fetchTile(tile) {
    return new Promise(resolve => {
        fetch("/tile", {method: "POST", body: JSON.stringify(tile.spec)})
        .then(response => response.json())
        .then(tiledata => {
            const img = new Image();
            tile.img = img;
            img.src = tiledata.url;
            img.onload = () => resolve(tile);
        });
    });
}

function showTile(tile) {
    tile.ctx.drawImage(tile.img, tile.tx*tileX, tile.ty*tileX);
}

function getImage(tile) {
    return fetchTile(tile).then(showTile);
}

function paint() {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    canvasW = ctx.canvas.width = window.innerWidth;
    canvasH = ctx.canvas.height = window.innerHeight;
    const imageurls = [];
    for (let tx = 0; tx < canvasW / tileX; tx++) {
        for (let ty = 0; ty < canvasH / tileX; ty++) {
            spec = {
                center: [cx, cy],
                size: [canvasW, canvasH],
                diam: [canvasW * pixsize, canvasH * pixsize],
                coords: [tx*tileX, (tx+1)*tileX, ty*tileX, (ty+1)*tileX],
            }
            imageurls.push({ctx, tx, ty, spec});
        }
    }
    Promise.all(imageurls.map(getImage));
}

function getCursorPosition(canvas, ev) {
    const rect = canvas.getBoundingClientRect()
    const x = ev.clientX - rect.left
    const y = ev.clientY - rect.top
    return {x, y};
}

function click(ev) {
    const canvas = document.getElementById("canvas");
    const {x, y} = getCursorPosition(canvas, ev);
    const x0 = cx - canvasW/2 * pixsize;
    const y0 = cy + canvasH/2 * pixsize;
    const clickx = x0 + x * pixsize;
    const clicky = y0 - y * pixsize;
    pixsize *= .5;
    const x1 = clickx - x * pixsize;
    const y1 = clicky + y * pixsize;
    cx = x1 + canvasW/2 * pixsize;
    cy = y1 - canvasH/2 * pixsize;
    paint();
}

document.body.onload = () => {
    const canvas = document.getElementById("canvas");
    canvas.addEventListener("click", click);
    paint();
}
