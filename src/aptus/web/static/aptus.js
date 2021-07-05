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

// sin(angle) and cos(angle)
let sina, cosa;

// Request sequence number. Requests include the sequence number and the tile
// returns it. If the sequence number has been incremented since the tile was
// requested, then the tile is no longer needed, and is not displayed.
let reqseq = 0;

function reset() {
    set_center(-0.6, 0.0);
    set_pixsize(3.0/600);
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
    //palette = [
    //    ["spectrum", {
    //        ncolors: get_input_value("ncolors"),
    //        h: [get_input_value("hlo"), get_input_value("hhi")],
    //        l: [get_input_value("llo"), get_input_value("lhi")],
    //        s: [get_input_value("slo"), get_input_value("shi")]
    //    }],
    //    ["stretch", {
    //        steps: get_input_value("stretch"),
    //        hsl: true,
    //        ease: get_input_value("ease")
    //    }]
    //];
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

let move_target = null;
let moving = false;
let mouse_dragging = false;
let mouse_shift = false;
let rubstart = null;

function mainpane_mousedown(ev) {
    //console.log("down. shift:", ev.shiftKey, "ctrl:", ev.ctrlKey, "meta:", ev.metaKey, "alt:", ev.altKey);
    ev.preventDefault();
    move_target = ev.target;
    rubstart = getCursorPosition(ev, move_target);
    mouse_shift = ev.shiftKey;
}

const DRAGDXY = 5;

function mainpane_mousemove(ev) {
    if (!move_target) {
        return;
    }
    ev.preventDefault();
    const movedto = getCursorPosition(ev, move_target);
    const dx = movedto.x - rubstart.x;
    const dy = movedto.y - rubstart.y;
    if (!mouse_dragging && Math.abs(dx) + Math.abs(dy) > DRAGDXY) {
        mouse_dragging = true;
        set_moving(mouse_shift);
    }
    clear_ctx(overlay_ctx);
    if (mouse_dragging) {
        if (moving) {
            fractal_canvas.style.left = dx + "px";
            fractal_canvas.style.top = dy + "px";
        }
        else {
            // With anti-aliasing, 0.5 offset makes 1-pixel wide.
            overlay_ctx.lineWidth = 1;
            overlay_ctx.strokeStyle = "#ffffffc0";
            overlay_ctx.strokeRect(rubstart.x + 0.5, rubstart.y + 0.5, dx, dy);
        }
    }
}

function mainpane_mouseup(ev) {
    //console.log("up. shift:", ev.shiftKey, "ctrl:", ev.ctrlKey, "meta:", ev.metaKey, "alt:", ev.altKey);
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
        if (mouse_dragging) {
            const a = ri4xy(rubstart.x, rubstart.y);
            const b = ri4xy(up.x, up.y);
            const dr = a.r - b.r, di = a.i - b.i;
            const rdr = xrot(dr, di);
            const rdi = yrot(dr, di);
            set_pixsize(Math.max(Math.abs(rdr) / canvasW, Math.abs(rdi) / canvasH));
            set_center((a.r + b.r) / 2, (a.i + b.i) / 2);
        }
        else {
            const {r: clickr, i: clicki} = ri4xy(up.x, up.y);

            const factor = ev.altKey ? 1.1 : 2.0;
            if (ev.shiftKey) {
                set_pixsize(pixsize * factor);
            }
            else {
                set_pixsize(pixsize / factor);
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
    rubstart = null;
    mouse_dragging = false;
    set_moving(false);
}

function cancel_dragging() {
    fractal_canvas.style.left = "0";
    fractal_canvas.style.top = "0";
    clear_ctx(overlay_ctx);
    move_target = null;
    rubstart = null;
    mouse_dragging = false;
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

    // Mac option chars need to be mapped back to their original chars.
    if (platform() === "mac") {
        switch (key) {
            case "¯":
                key = "<";
                break;
            case "˘":
                key = ">";
                break;
        }
    }

    var handled = false;

    if (!ev.metaKey) {
        handled = true;
        switch (key) {
            case "Escape":
                cancel_dragging();
                break;

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
                toggle_panel("infopanel");
                break;

            case "P":
                toggle_panel("palettepanel");
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
                set_angle(angle + (ev.altKey ? 1 : 10));
                paint();
                break;

            case "<":
                set_angle(angle - (ev.altKey ? 1 : 10));
                paint();
                break;

            case "?":
                toggle_panel("helppanel");
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

function set_pixsize(ps) {
    pixsize = ps;
    set_input_value("pixsize", pixsize);
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
    set_pixsize(get_input_value("pixsize"));
    set_angle(get_input_value("angle"));
    set_iter_limit(get_input_value("iter_limit"));
    paint();
}

function set_moving(m) {
    moving = m;
    if (moving) {
        overlay_canvas.classList.add("move");
    }
    else {
        overlay_canvas.classList.remove("move");
    }
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

function bring_to_top(el, els) {
    const indexes = [...els].map(e => {
        const z = getComputedStyle(e).zIndex;
        return (z === "auto") ? 0 : z;
    });
    el.style.zIndex = Math.max(...indexes) + 1;
}

function bring_panel_to_top(el) {
    bring_to_top(el, document.querySelectorAll(".panel"));
}

function toggle_panel(panelid) {
    const panel = document.getElementById(panelid);
    if (panel.style.display === "block") {
        panel.style.display = "none";
    }
    else {
        panel.style.display = "block";
        let at_x = panel.offsetLeft, at_y = panel.offsetTop;
        if (at_x > window.innerWidth) {
            at_x = (window.innerWidth - panel.clientWidth) / 2;
        }
        if (at_y > window.innerHeight) {
            at_y = (window.innerHeight - panel.clientHeight) / 2;
        }
        position_panel(panel, at_x, at_y);
        bring_panel_to_top(panel);
    }
}

function position_panel(panel, left, top) {
    panel.style.left = left + "px";
    panel.style.top = top + "px";
    panel.style.right = "auto";
    panel.style.bottom = "auto";
}

function close_panel(ev) {
    const panel = ev.target.closest(".panel");
    panel.style.display = "none";
}

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
    draggable = ev.delegate;
    draggable.classList.add("dragging");
    rubstart = {x: ev.clientX, y: ev.clientY};
    draggable_start = {x: draggable.offsetLeft, y: draggable.offsetTop};
    bring_panel_to_top(draggable);
    position_panel(draggable, draggable.offsetLeft, draggable.offsetTop);
}

function draggable_mousemove(ev) {
    if (!draggable) {
        return;
    }
    ev.preventDefault();
    ev.stopPropagation();
    position_panel(
        draggable,
        draggable_start.x - (rubstart.x - ev.clientX),
        draggable_start.y - (rubstart.y - ev.clientY)
    );
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

function platform() {
    if (navigator.platform.indexOf("Mac") > -1) {
        return "mac";
    }
    else if (navigator.platform.indexOf("Win") > -1) {
        return "win";
    }
}

document.body.onload = () => {
    if (platform() === "mac") {
        document.querySelector("html").classList.add("mac");
    }

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
    on_event(document, "keydown", keydown);
    on_event(window, "resize", resize);

    set_size();
    reset();
    paint();
}
