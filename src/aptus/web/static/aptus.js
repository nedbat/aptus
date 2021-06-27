// Aptus web

const tileX = 400;

let centerr, centeri;
let pixsize;
let angle;
let continuous;
let iter_limit;
let palette_index;

let canvasW, canvasH;
let fractal_canvas, overlay_canvas;
let fractal_ctx, overlay_ctx;
let move_target = null;
let moving = false;

// sin(angle) and cos(angle)
let sina, cosa;

// Request sequence number. Requests include the sequence number and the tile
// returns it. If the sequence number has been incremented since the tile was
// requested, then the tile is no longer needed, and is not displayed.
let reqseq = 0;

function reset() {
    set_center(-0.6, 0.0);
    pixsize = 3.0/600;
    set_angle(0.0);
    continuous = false;
    set_iter_limit(999);
    palette_index = 0;
}

function fetchTile(tile) {
    return new Promise(resolve => {
        const body = {
            seq: tile.reqseq,
            spec: tile.spec,
        };
        fetch("/tile", {method: "POST", body: JSON.stringify(body)})
        .then(response => response.json())
        .then(tiledata => {
            if (tiledata.seq == reqseq) {
                const img = new Image();
                tile.img = img;
                img.src = tiledata.url;
                img.onload = () => resolve(tile);
            }
            else {
                // console.log("Discarding tile with seq " + tiledata.seq + ", only interested now in " + reqseq);
            }
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
    reqseq += 1;
    const imageurls = [];
    var palette = [...palettes[palette_index]];
    //palette.push(["adjust", {hue: 120, saturation: 0}]);
    //palette.push(["stretch", {steps: 3, hsl: true}]);
    //palette = [["spectrum", {ncolors: 16, l: [100, 150], s: [100, 175]}], ["stretch", {steps: 10, hsl: true, ease: get_input_value("ease")}]];
    //palette.push(["stretch", {steps: 2, hsl: true, ease: get_input_value("ease")}]);
    palette = [
        ["spectrum", {
            ncolors: get_input_value("ncolors"),
            h: [get_input_value("hlo"), get_input_value("hhi")],
            l: [get_input_value("llo"), get_input_value("lhi")],
            s: [get_input_value("slo"), get_input_value("shi")]
        }],
        ["stretch", {
            steps: get_input_value("stretch"),
            hsl: true,
            ease: get_input_value("ease")
        }]
    ];
    for (let tx = 0; tx < canvasW / tileX; tx++) {
        for (let ty = 0; ty < canvasH / tileX; ty++) {
            spec = {
                center: [centerr, centeri],
                diam: [canvasW * pixsize, canvasH * pixsize],
                size: [canvasW, canvasH],
                coords: [tx*tileX, (tx+1)*tileX, ty*tileX, (ty+1)*tileX],
                angle,
                continuous,
                iter_limit,
                palette,
            }
            imageurls.push({ctx: fractal_ctx, tx, ty, spec, reqseq});
        }
    }
    return Promise.all(imageurls.map(getImage));
}

function clear_ctx(ctx) {
    ctx.clearRect(0, 0, canvasW, canvasH);
}

function getCursorPosition(ev, target) {
    const rect = target.getBoundingClientRect()
    const x = ev.clientX - rect.left
    const y = ev.clientY - rect.top
    return {x, y};
}

// xrot and yrot provide rotated versions of the x,y they are given.
function xrot(x, y) {
    return x * cosa + y * sina;
}

function yrot(x, y) {
    return y * cosa - x * sina;
}

function ri4xy(x, y) {
    const r0 = centerr - xrot(canvasW, canvasH)/2 * pixsize;
    const i0 = centeri + yrot(canvasW, canvasH)/2 * pixsize;
    const r = r0 + xrot(x, y) * pixsize;
    const i = i0 - yrot(x, y) * pixsize;
    return {r, i};
}

function mainpane_mousedown(ev) {
    ev.preventDefault();
    move_target = ev.target;
    rubstart = getCursorPosition(ev, move_target);
}

function mainpane_mousemove(ev) {
    if (!move_target) {
        return;
    }
    ev.preventDefault();
    const movedto = getCursorPosition(ev, move_target);
    clear_ctx(overlay_ctx);
    if (moving) {
        fractal_canvas.style.left = (movedto.x - rubstart.x) + "px";
        fractal_canvas.style.top = (movedto.y - rubstart.y) + "px";
    }
    else {
        overlay_ctx.lineWidth = 1;
        overlay_ctx.strokeStyle = "white";
        overlay_ctx.strokeRect(rubstart.x, rubstart.y, movedto.x - rubstart.x, movedto.y - rubstart.y);
    }
}

function mainpane_mouseup(ev) {
    if (!move_target) {
        return;
    }
    ev.preventDefault();
    const up = getCursorPosition(ev, move_target);
    const dx = up.x - rubstart.x;
    const dy = up.y - rubstart.y;
    if (moving) {
        set_center(centerr - xrot(dx, dy) * pixsize, centeri + yrot(dx, dy) * pixsize);
        overlay_ctx.drawImage(fractal_canvas, dx, dy);
        fractal_canvas.style.left = "0";
        fractal_canvas.style.top = "0";
        fractal_ctx.fillStyle = "white";
        fractal_ctx.fillRect(0, 0, canvasW, canvasH);
        paint().then(() => {
            clear_ctx(overlay_ctx);
        });
    }
    else {
        clear_ctx(overlay_ctx);
        const moved = Math.abs(dx) + Math.abs(dy);
        if (moved > 20) {
            const a = ri4xy(rubstart.x, rubstart.y);
            const b = ri4xy(up.x, up.y);
            const dr = a.r - b.r, di = a.i - b.i;
            const rdr = xrot(dr, di);
            const rdi = yrot(dr, di);
            pixsize = Math.max(Math.abs(rdr) / canvasW, Math.abs(rdi) / canvasH);
            set_center((a.r + b.r) / 2, (a.i + b.i) / 2);
        }
        else {
            const {r: clickr, i: clicki} = ri4xy(up.x, up.y);

            if (ev.shiftKey) {
                pixsize *= (ev.ctrlKey ? 1.1 : 2.0);
            }
            else {
                pixsize /= (ev.ctrlKey ? 1.1 : 2.0);
            }
            const r0 = clickr - xrot(up.x, up.y) * pixsize;
            const i0 = clicki + yrot(up.x, up.y) * pixsize;
            set_center(
                r0 + xrot(canvasW, canvasH)/2 * pixsize,
                i0 - yrot(canvasW, canvasH)/2 * pixsize
            );
        }
        paint();
    }
    move_target = null;
}

function keydown(ev) {
    if (ev.target.matches("input")) {
        return;
    }

    //console.log("key:",  ev.key, "shift:", ev.shiftKey, "ctrl:", ev.ctrlKey, "meta:", ev.metaKey, "alt:", ev.altKey);
    var key = ev.key;

    // Chrome handles ctrl-lessthan as shift-ctrl-comma. Fix those combinations
    // to be what we expect.
    if (ev.shiftKey) {
        switch (key) {
            case ".":
                key = ">";
                break;
            case ",":
                key = "<";
                break;
        }
    }

    var handled = false;

    if (!ev.metaKey && !ev.altKey) {
        handled = true;
        switch (key) {
            case "a":
                new_angle = +prompt("Angle", angle);
                if (new_angle != angle) {
                    set_angle(new_angle);
                    paint();
                }
                break;

            case "c":
                continuous = !continuous;
                paint();
                break;

            case "C":
                alert(
                    `--center=${centerr},${centeri} ` +
                    (angle ? `--angle=${angle} ` : "") +
                    `--diam=${canvasW * pixsize},${canvasH * pixsize}`
                );
                break;

            case "i":
                new_limit = +prompt("Iteration limit", iter_limit);
                if (new_limit != iter_limit) {
                    set_iter_limit(new_limit);
                    paint();
                }
                break;

            case "I":
                const info_panel = document.getElementById("infopanel");
                info_panel.style.top = "5em";
                info_panel.style.left = "5em";
                info_panel.style.right = info_panel.style.bottom = null;
                info_panel.style.display = "block";
                break;

            case "m":
                moving = !moving;
                if (moving) {
                    overlay_canvas.classList.add("move");
                }
                else {
                    overlay_canvas.classList.remove("move");
                }
                break;

            case "P":
                const palette_panel = document.getElementById("palettepanel");
                palette_panel.style.top = "5em";
                palette_panel.style.left = "5em";
                palette_panel.style.right = palette_panel.style.bottom = null;
                palette_panel.style.display = "block";
                break;

            case "r":
                paint();
                break;

            case "R":
                reset();
                paint();
                break;

            case ",":
                palette_index -= 1;
                if (palette_index < 0) {
                    palette_index += palettes.length;
                }
                paint();
                break;

            case ".":
                palette_index += 1;
                palette_index %= palettes.length;
                paint();
                break;

            case ">":
                set_angle(angle + (ev.ctrlKey ? 1 : 10));
                paint();
                break;

            case "<":
                set_angle(angle - (ev.ctrlKey ? 1 : 10));
                paint();
                break;

            case "?":
                toggle_help();
                break;

            default:
                handled = false;
                break;
        }
    }

    if (handled) {
        ev.preventDefault();
    }
}

function toggle_help() {
    const help_panel = document.getElementById("helppanel");
    if (help_panel.style.display === "block") {
        help_panel.style.display = "none";
    }
    else {
        help_panel.style.top = "5em";
        help_panel.style.right = "5em";
        help_panel.style.left = help_panel.style.bottom = null;
        help_panel.style.display = "block";
    }
}

function close_panel(ev) {
    const panel = ev.target.closest(".panel");
    panel.style.display = "none";
}

function set_input_value(name, val) {
    document.getElementById(name).value = "" + val;
}

function get_input_value(name) {
    return +document.getElementById(name).value;
}

function set_size() {
    canvasW = fractal_canvas.width = overlay_canvas.width = window.innerWidth;
    canvasH = fractal_canvas.height = overlay_canvas.height = window.innerHeight;
}

function set_center(r, i) {
    centerr = r;
    centeri = i;
    set_input_value("centerr", centerr);
    set_input_value("centeri", centeri);
}

function set_angle(a) {
    angle = (a % 360 + 360) % 360;
    set_input_value("angle", angle);
    const rads = angle / 180 * Math.PI;
    sina = Math.sin(rads);
    cosa = Math.cos(rads);
}

function set_iter_limit(i) {
    iter_limit = i;
    set_input_value("iter_limit", i);
}

function spec_change(ev) {
    set_center(get_input_value("centerr"), get_input_value("centeri"));
    set_angle(get_input_value("angle"));
    set_iter_limit(get_input_value("iter_limit"));
    paint();
}

let resize_timeout = null;

function resize() {
    if (resize_timeout) {
        clearTimeout(resize_timeout);
    }
    resize_timeout = setTimeout(
        () => {
            resize_timeout = null;
            set_size();
            paint();
        },
        250
    );
}

var draggable = null;
var draggable_start;

function draggable_mousedown(ev) {
    if (ev.target.matches("input")) {
        return;
    }
    ev.preventDefault();
    ev.stopPropagation();
    const active = document.activeElement;
    if (active) {
        active.blur();
    }
    rubstart = {x: ev.clientX, y: ev.clientY};
    draggable = ev.delegate;
    draggable.classList.add("dragging");
    draggable_start = {x: draggable.offsetLeft, y: draggable.offsetTop};
    draggable.style.left = draggable.offsetLeft + "px";
    draggable.style.top = draggable.offsetTop + "px";
    draggable.style.right = null;
    draggable.style.bottom = null;
}

function draggable_mousemove(ev) {
    if (!draggable) {
        return;
    }
    ev.preventDefault();
    ev.stopPropagation();
    const movedto = {x: ev.clientX, y: ev.clientY};
    draggable.style.left = draggable_start.x - (rubstart.x - movedto.x) + "px";
    draggable.style.top = draggable_start.y - (rubstart.y - movedto.y) + "px";
}

function draggable_mouseup(ev) {
    if (!draggable) {
        return;
    }
    ev.preventDefault();
    ev.stopPropagation();
    draggable.classList.remove("dragging");
    draggable = null;
}

// From: https://gist.github.com/JustinChristensen/652bedadc92cf0aff86cc5fbcde87732
// <wroathe> You can then do on(document.body, 'pointerdown', e => console.log(e.delegate), '.draggable');

function delegatedTo(sel, fn) {
    return ev => {
        ev.delegate = ev.target.closest(sel);
        if (ev.delegate) {
            fn(ev);
        }
    };
};

function on_event(el, ev, fn, sel) {
    if (sel) {
        fn = delegatedTo(sel, fn);
    }
    if (typeof el === 'string') {
        el = document.querySelectorAll(el);
    }
    if (!el.forEach) {
        el = [el];
    }
    el.forEach(e => e.addEventListener(ev, fn));
    return Array.from(el);
}


document.body.onload = () => {
    fractal_canvas = document.getElementById("fractal");
    overlay_canvas = document.getElementById("overlay");

    fractal_ctx = fractal_canvas.getContext("2d");
    overlay_ctx = overlay_canvas.getContext("2d");

    on_event("#infopanel input", "change", spec_change);
    on_event(".panel .closebtn", "click", close_panel);

    on_event(overlay_canvas, "mousedown", mainpane_mousedown);
    on_event(document, "mousedown", draggable_mousedown, ".draggable");
    on_event(document, "mousemove", mainpane_mousemove);
    on_event(document, "mousemove", draggable_mousemove);
    on_event(document, "mouseup", mainpane_mouseup);
    on_event(document, "mouseup", draggable_mouseup);
    on_event(document, "contextmenu", ev => { ev.preventDefault(); return false; });
    on_event(document, "keydown", keydown);
    on_event(window, "resize", resize);

    set_size();
    reset();
    paint();
}
