const canvasW = 600, tileX = 200;
let cx = -0.6, cy = 0.0;
let diam = 3.0;

function fetchTile(tile) {
    return new Promise(resolve => {
        fetch(tile.url)
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
    tile.ctx.drawImage(tile.img, tile.x*tileX, tile.y*tileX);
}

function getImage(tile) {
    return fetchTile(tile).then(showTile);
}

function paint() {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    ctx.canvas.width = ctx.canvas.height = canvasW
    const imageurls = [];
    for (let x = 0; x < canvasW / tileX; x++) {
        for (let y = 0; y < canvasW / tileX; y++) {
            const url = `/tile?center=${cx},${cy}&diam=${diam}&xmin=${x*tileX}&xmax=${(x+1)*tileX}&ymin=${y*tileX}&ymax=${(y+1)*tileX}`;
            imageurls.push({ctx, x, y, url});
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
    const x0 = cx - diam/2;
    const y0 = cy + diam/2;
    const pix = diam / canvasW;
    console.log(`x0,y0 = ${x0},${y0};  x,y = ${x},${y}; pix = ${pix}`);
    cx = x0 + x * pix;
    cy = y0 - y * pix;
    diam *= .75;
    paint();
}

document.body.onload = () => {
    const canvas = document.getElementById("canvas");
    canvas.addEventListener("click", click);
    paint();
}
