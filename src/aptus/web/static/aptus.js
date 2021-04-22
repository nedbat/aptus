const tileX = 200, tileW = 3;

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

function doit() {
    const ctx = document.getElementById("canvas").getContext("2d");
    ctx.canvas.width = ctx.canvas.height = tileX * tileW;
    const imageurls = [];
    for (let x = 0; x < tileW; x++) {
        for (let y = 0; y < tileW; y++) {
            const url = `/tile?xmin=${x*tileX}&xmax=${(x+1)*tileX}&ymin=${y*tileX}&ymax=${(y+1)*tileX}`;
            imageurls.push({ctx, x, y, url});
        }
    }
    Promise.all(imageurls.map(getImage));
}

document.body.onload = () => {
    document.querySelector("#doit").addEventListener("click", doit);
}
