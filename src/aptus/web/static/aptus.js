// Aptus web

const View = {
    tileX: 400,

    reset() {
        this.set_center(-0.6, 0.0);
        this.set_pixsize(3.0/600);
        this.set_angle(0.0);
        this.continuous = false;
        this.set_iter_limit(999);
        this.palette_index = 0;
        this.set_canvas_size("*");
    },

    set_center(r, i) {
        this.centerr = r;
        this.centeri = i;
        set_input_value("centerr", this.centerr);
        set_input_value("centeri", this.centeri);
    },

    set_pixsize(ps) {
        this.pixsize = ps;
        set_input_value("pixsize", this.pixsize);
    },

    set_angle(a) {
        this.angle = (a % 360 + 360) % 360;
        set_input_value("angle", this.angle);
        const rads = this.angle / 180 * Math.PI;
        this.sina = Math.sin(rads);
        this.cosa = Math.cos(rads);
    },

    set_iter_limit(i) {
        this.iter_limit = i;
        set_input_value("iter_limit", i);
    },

    set_canvas_size(s) {
        if (s === "*") {
            this.canvas_size_w = this.canvas_size_h = null;
        }
        else {
            const nums = s.split(/[ ,]+/);
            this.canvas_size_w = +nums[0];
            this.canvas_size_h = +nums[1];
        }
        this.set_size();
    },

    set_size() {
        if (this.canvas_size_w) {
            this.canvasW = this.canvas_size_w;
            this.canvasH = this.canvas_size_h;
        }
        else {
            this.canvasW = window.innerWidth;
            this.canvasH = window.innerHeight;
        }
        const backdrop = document.getElementById("backdrop");
        backdrop.width = fractal_canvas.width = overlay_canvas.width = this.canvasW;
        backdrop.height = fractal_canvas.height = overlay_canvas.height = this.canvasH;
        const sizer = document.querySelector(".canvas_sizer");
        sizer.style.width = this.canvasW + "px";
        sizer.style.height = this.canvasH + "px";
        checkers(document.getElementById("backdrop"));
    },

    paint() {
        reqseq += 1;
        const imageurls = [];
        var palette = [...palettes[this.palette_index]];
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
        for (let tx = 0; tx < this.canvasW / this.tileX; tx++) {
            for (let ty = 0; ty < this.canvasH / this.tileX; ty++) {
                let tile = {
                    ctx: fractal_ctx,
                    tx: tx * this.tileX,
                    ty: ty * this.tileX,
                    spec: {
                        center: [this.centerr, this.centeri],
                        diam: [
                            this.canvasW * this.pixsize,
                            this.canvasH * this.pixsize
                        ],
                        size: [this.canvasW, this.canvasH],
                        coords: [
                            tx*this.tileX, (tx+1)*this.tileX,
                            ty*this.tileX, (ty+1)*this.tileX
                        ],
                        angle: this.angle,
                        continuous: this.continuous,
                        iter_limit: this.iter_limit,
                        palette,
                    },
                    reqseq,
                };
                imageurls.push(tile);
            }
        }
        return Promise.all(imageurls.map(getImage));
    },

    // xrot and yrot provide rotated versions of the x,y they are given.
    xrot(x, y) {
        return x * this.cosa + y * this.sina;
    },

    yrot(x, y) {
        return y * this.cosa - x * this.sina;
    },

    ri4xy(x, y) {
        const r0 = this.centerr - this.xrot(this.canvasW, this.canvasH)/2 * this.pixsize;
        const i0 = this.centeri + this.yrot(this.canvasW, this.canvasH)/2 * this.pixsize;
        const r = r0 + this.xrot(x, y) * this.pixsize;
        const i = i0 - this.yrot(x, y) * this.pixsize;
        return {r, i};
    },
};

let fractal_canvas, overlay_canvas;
let fractal_ctx, overlay_ctx;

// Request sequence number. Requests include the sequence number and the tile
// returns it. If the sequence number has been incremented since the tile was
// requested, then the tile is no longer needed, and is not displayed.
let reqseq = 0;

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
        });
    });
}

function showTile(tile) {
    tile.ctx.drawImage(tile.img, tile.tx, tile.ty);
}

function getImage(tile) {
    return fetchTile(tile).then(showTile);
}

function getCursorPosition(ev, target) {
    const rect = target.getBoundingClientRect()
    const x = ev.clientX - rect.left
    const y = ev.clientY - rect.top
    return {x, y};
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
    clear_canvas(overlay_canvas);
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
        the_view.set_center(
            the_view.centerr - the_view.xrot(dx, dy) * the_view.pixsize,
            the_view.centeri + the_view.yrot(dx, dy) * the_view.pixsize
        );
        overlay_ctx.drawImage(fractal_canvas, dx, dy);
        fractal_canvas.style.left = "0";
        fractal_canvas.style.top = "0";
        fractal_ctx.clearRect(0, 0, the_view.canvasW, the_view.canvasH);
        the_view.paint().then(() => {
            clear_canvas(overlay_canvas);
        });
    }
    else {
        clear_canvas(overlay_canvas);
        if (mouse_dragging) {
            const a = the_view.ri4xy(rubstart.x, rubstart.y);
            const b = the_view.ri4xy(up.x, up.y);
            const dr = a.r - b.r, di = a.i - b.i;
            const rdr = the_view.xrot(dr, di);
            const rdi = the_view.yrot(dr, di);
            the_view.set_pixsize(Math.max(Math.abs(rdr) / the_view.canvasW, Math.abs(rdi) / the_view.canvasH));
            the_view.set_center((a.r + b.r) / 2, (a.i + b.i) / 2);
        }
        else {
            const {r: clickr, i: clicki} = the_view.ri4xy(up.x, up.y);

            const factor = ev.altKey ? 1.1 : 2.0;
            if (ev.shiftKey) {
                the_view.set_pixsize(the_view.pixsize * factor);
            }
            else {
                the_view.set_pixsize(the_view.pixsize / factor);
            }
            const r0 = clickr - the_view.xrot(up.x, up.y) * the_view.pixsize;
            const i0 = clicki + the_view.yrot(up.x, up.y) * the_view.pixsize;
            the_view.set_center(
                r0 + the_view.xrot(the_view.canvasW, the_view.canvasH)/2 * the_view.pixsize,
                i0 - the_view.yrot(the_view.canvasW, the_view.canvasH)/2 * the_view.pixsize
            );
        }
        the_view.paint();
    }
    move_target = null;
    rubstart = null;
    mouse_dragging = false;
    set_moving(false);
}

function cancel_dragging() {
    fractal_canvas.style.left = "0";
    fractal_canvas.style.top = "0";
    clear_canvas(overlay_canvas);
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
                const new_angle = +prompt("Angle", the_view.angle);
                if (new_angle != the_view.angle) {
                    the_view.set_angle(new_angle);
                    the_view.paint();
                }
                break;

            case "c":
                the_view.continuous = !the_view.continuous;
                the_view.paint();
                break;

            case "C":
                alert(
                    `--center=${the_view.centerr},${the_view.centeri} ` +
                    (the_view.angle ? `--angle=${the_view.angle} ` : "") +
                    `--diam=${the_view.canvasW * the_view.pixsize},${the_view.canvasH * the_view.pixsize}`
                );
                break;

            case "i":
                new_limit = +prompt("Iteration limit", the_view.iter_limit);
                if (new_limit != the_view.iter_limit) {
                    the_view.set_iter_limit(new_limit);
                    the_view.paint();
                }
                break;

            case "I":
                toggle_panel("infopanel");
                break;

            case "P":
                toggle_panel("palettepanel");
                break;

            case "r":
                the_view.paint();
                break;

            case "R":
                the_view.reset();
                the_view.paint();
                break;

            case "w":
                let text;
                if (!the_view.canvas_size_w) {
                    text = "*";
                }
                else {
                    text = `${the_view.canvas_size_w} ${the_view.canvas_size_h}`;
                }
                the_view.set_canvas_size(prompt("Canvas size", text));
                the_view.paint();
                break;

            case ",":
                the_view.palette_index -= 1;
                if (the_view.palette_index < 0) {
                    the_view.palette_index += palettes.length;
                }
                the_view.paint();
                break;

            case ".":
                the_view.palette_index += 1;
                the_view.palette_index %= palettes.length;
                the_view.paint();
                break;

            case ">":
                the_view.set_angle(the_view.angle + (ev.altKey ? 1 : 10));
                the_view.paint();
                break;

            case "<":
                the_view.set_angle(the_view.angle - (ev.altKey ? 1 : 10));
                the_view.paint();
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

function spec_change(ev) {
    the_view.set_center(get_input_value("centerr"), get_input_value("centeri"));
    the_view.set_pixsize(get_input_value("pixsize"));
    the_view.set_angle(get_input_value("angle"));
    the_view.set_iter_limit(get_input_value("iter_limit"));
    the_view.paint();
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
            the_view.set_size();
            the_view.paint();
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

function clear_canvas(canvas) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function checkers(canvas) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width, h = canvas.height;

    ctx.fillStyle = "#aaaaaa";
    ctx.fillRect(0, 0, w, h);

    const sqw = 50;
    ctx.fillStyle = "#999999";
    for (let col = 0; col < (w / sqw); col += 1) {
        for (let row = 0; row < (h / sqw); row += 1) {
            if ((row + col) % 2) {
                ctx.fillRect(col * sqw, row * sqw, sqw, sqw);
            }
        }
    }
}

let the_view;

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

    the_view = Object.create(View);
    the_view.reset();
    the_view.set_size();
    the_view.paint();
}
