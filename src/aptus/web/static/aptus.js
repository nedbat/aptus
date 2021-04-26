const tileX = 400;

let centerr, centeri;
let pixsize;
let canvasW, canvasH;
let continuous;
let fractal_canvas, overlay_canvas;
let is_down;

function reset() {
    centerr = -0.6;
    centeri = 0.0;
    pixsize = 3.0/600;
    continuous = false;
}

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
    const imageurls = [];
    for (let tx = 0; tx < canvasW / tileX; tx++) {
        for (let ty = 0; ty < canvasH / tileX; ty++) {
            spec = {
                center: [centerr, centeri],
                diam: [canvasW * pixsize, canvasH * pixsize],
                size: [canvasW, canvasH],
                coords: [tx*tileX, (tx+1)*tileX, ty*tileX, (ty+1)*tileX],
                continuous: continuous,
            }
            imageurls.push({ctx: fractal_ctx, tx, ty, spec});
        }
    }
    Promise.all(imageurls.map(getImage));
}

function getCursorPosition(ev) {
    const rect = ev.target.getBoundingClientRect()
    const x = ev.clientX - rect.left
    const y = ev.clientY - rect.top
    return {x, y};
}

function ri4xy(x, y) {
    const r0 = centerr - canvasW/2 * pixsize;
    const i0 = centeri + canvasH/2 * pixsize;
    const r = r0 + x * pixsize;
    const i = i0 - y * pixsize;
    return {r, i};
}

function mousedown(ev) {
    is_down = true;
    rubstart = getCursorPosition(ev);
}

function mousemove(ev) {
    if (is_down) {
        overlay_ctx.clearRect(0, 0, overlay_canvas.width, overlay_canvas.height);
        const movedto = getCursorPosition(ev);
        overlay_ctx.lineWidth = 1;
        overlay_ctx.strokeStyle = "white";
        overlay_ctx.strokeRect(rubstart.x, rubstart.y, movedto.x - rubstart.x, movedto.y - rubstart.y);
    }
}

function mouseup(ev) {
    is_down = false;
    overlay_ctx.clearRect(0, 0, overlay_canvas.width, overlay_canvas.height);
    const up = getCursorPosition(ev);
    const moved = Math.abs(rubstart.x - up.x) + Math.abs(rubstart.y - up.y);
    if (moved > 20) {
        const {r: ra, i: ia} = ri4xy(rubstart.x, rubstart.y);
        const {r: rb, i: ib} = ri4xy(up.x, up.y);
        centerr = (ra + rb) / 2;
        centeri = (ia + ib) / 2;
        pixsize = Math.max(Math.abs(ra - rb) / canvasW, Math.abs(ia - ib) / canvasH);
    }
    else {
        const {r: clickr, i: clicki} = ri4xy(up.x, up.y);
        pixsize *= .5;
        const r0 = clickr - up.x * pixsize;
        const i0 = clicki + up.y * pixsize;
        centerr = r0 + canvasW/2 * pixsize;
        centeri = i0 - canvasH/2 * pixsize;
    }
    paint();
}

function keydown(e) {
    switch (e.key) {
        case "c":
            continuous = !continuous;
            paint();
            break;

        case "r":
            reset();
            paint();
            break;
    }
}

document.body.onload = () => {
    fractal_canvas = document.getElementById("fractal");
    fractal_ctx = fractal_canvas.getContext("2d");
    overlay_canvas = document.getElementById("overlay");
    overlay_ctx = overlay_canvas.getContext("2d");
    canvasW = fractal_canvas.width = overlay_canvas.width = window.innerWidth;
    canvasH = fractal_canvas.height = overlay_canvas.height = window.innerHeight;
    is_down = false;
    overlay_canvas.addEventListener("mousedown", mousedown);
    overlay_canvas.addEventListener("mousemove", mousemove);
    overlay_canvas.addEventListener("mouseup", mouseup);
    document.addEventListener("keydown", keydown);
    reset();
    paint();
}
