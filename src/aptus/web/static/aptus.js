// Aptus web

const View = {
    tileX: 400,

    // One map for all views, mapping overlay canvas elements to their view.
    canvas_map: new Map(),

    init(div) {
        div.setAttribute("class", "canvas_container");

        this.canvas_sizer = document.createElement("div");
        this.canvas_sizer.setAttribute("class", "canvas_sizer");
        div.appendChild(this.canvas_sizer);

        this.backdrop_canvas = document.createElement("canvas");
        this.backdrop_canvas.setAttribute("class", "view backdrop");
        this.canvas_sizer.appendChild(this.backdrop_canvas);

        this.fractal_canvas = document.createElement("canvas");
        this.fractal_canvas.setAttribute("class", "view fractal");
        this.canvas_sizer.appendChild(this.fractal_canvas);

        this.overlay_canvas = document.createElement("canvas");
        this.overlay_canvas.setAttribute("class", "view overlay");
        this.canvas_sizer.appendChild(this.overlay_canvas);
        this.canvas_map.set(this.overlay_canvas, this);

        // Request sequence number. Requests include the sequence number and the tile
        // returns it. If the sequence number has been incremented since the tile was
        // requested, then the tile is no longer needed, and is not displayed.
        this.reqseq = 0;

        return this;
    },

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
    },

    set_pixsize(ps) {
        this.pixsize = ps;
    },

    set_angle(a) {
        this.angle = (a % 360 + 360) % 360;
        const rads = this.angle / 180 * Math.PI;
        this.sina = Math.sin(rads);
        this.cosa = Math.cos(rads);
        return this.angle;
    },

    set_iter_limit(i) {
        this.iter_limit = i;
    },

    set_continuous(c) {
        this.continuous = c;
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
        this.backdrop_canvas.width = this.fractal_canvas.width = this.overlay_canvas.width = this.canvasW;
        this.backdrop_canvas.height = this.fractal_canvas.height = this.overlay_canvas.height = this.canvasH;
        this.canvas_sizer.style.width = this.canvasW + "px";
        this.canvas_sizer.style.height = this.canvasH + "px";
        checkers(this.backdrop_canvas);
    },

    paint() {
        this.reqseq += 1;
        const imageurls = [];
        const palette = [...palettes[this.palette_index]];
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
        const dx = Math.ceil(this.canvasW / Math.floor(this.canvasW / this.tileX));
        const dy = Math.ceil(this.canvasH / Math.floor(this.canvasH / this.tileX));
        for (let tx = 0; tx < this.canvasW; tx += dx) {
            for (let ty = 0; ty < this.canvasH; ty += dy) {
                let tile = {
                    view: this,
                    reqseq: this.reqseq,
                    ctx: this.fractal_canvas.getContext("2d"),
                    tx,
                    ty,
                    spec: {
                        center: [this.centerr, this.centeri],
                        diam: [
                            this.canvasW * this.pixsize,
                            this.canvasH * this.pixsize
                        ],
                        size: [this.canvasW, this.canvasH],
                        coords: [tx, tx + dx, ty, ty + dy],
                        angle: this.angle,
                        continuous: this.continuous,
                        iter_limit: this.iter_limit,
                        palette,
                    },
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

    cli_args() {
        return (
            `--center=${this.centerr},${this.centeri} ` +
            (this.angle ? `--angle=${this.angle} ` : "") +
            `--diam=${this.canvasW * this.pixsize},${this.canvasH * this.pixsize}`
        );
    },
};

function fetchTile(tile) {
    return new Promise(resolve => {
        const body = {
            seq: tile.reqseq,
            spec: tile.spec,
        };
        fetch("/tile", {method: "POST", body: JSON.stringify(body)})
            .then(response => response.json())
            .then(tiledata => {
                if (tiledata.seq == tile.view.reqseq) {
                    tile.img = new Image();
                    tile.img.src = tiledata.url;
                    tile.img.onload = () => resolve(tile);
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

const App = {
    init() {
        this.view = Object.create(View).init(document.querySelector("#the_view"));
        this.reset();
        this.reset_dragging();
        this.resize_timeout = null;
        return this;
    },

    reset() {
        this.view.reset();
        this.set_center(-0.6, 0.0);
        this.set_pixsize(3.0/600);
        this.set_angle(0.0);
        this.set_iter_limit(999);
    },

    reset_dragging() {
        this.move_target = null;
        this.moving = false;
        this.mouse_dragging = false;
        this.mouse_shift = false;
        this.rubstart = null;
        this.set_moving(false);
    },

    set_center(r, i) {
        this.view.set_center(r, i);
        set_input_value("centerr", r);
        set_input_value("centeri", i);
    },

    set_pixsize(ps) {
        this.view.set_pixsize(ps);
        set_input_value("pixsize", ps);
    },

    set_angle(a) {
        set_input_value("angle", this.view.set_angle(a));
    },

    set_iter_limit(i) {
        this.view.set_iter_limit(i);
        set_input_value("iter_limit", i);
    },

    spec_change(ev) {
        this.set_center(get_input_value("centerr"), get_input_value("centeri"));
        this.set_pixsize(get_input_value("pixsize"));
        this.set_angle(get_input_value("angle"));
        this.set_iter_limit(get_input_value("iter_limit"));
        this.view.paint();
    },

    view_mousedown(ev) {
        //console.log("down. shift:", ev.shiftKey, "ctrl:", ev.ctrlKey, "meta:", ev.metaKey, "alt:", ev.altKey);
        ev.preventDefault();
        this.move_target = ev.target;
        this.rubstart = getCursorPosition(ev, this.move_target);
        this.mouse_shift = ev.shiftKey;
    },

    view_mousemove(ev) {
        if (!this.move_target) {
            return;
        }
        ev.preventDefault();
        const view = View.canvas_map.get(this.move_target);
        const movedto = getCursorPosition(ev, this.move_target);
        const dx = movedto.x - this.rubstart.x;
        const dy = movedto.y - this.rubstart.y;
        if (!this.mouse_dragging && Math.abs(dx) + Math.abs(dy) > 5) {
            this.mouse_dragging = true;
            this.set_moving(this.mouse_shift);
        }
        clear_canvas(view.overlay_canvas);
        if (this.mouse_dragging) {
            if (this.moving) {
                view.fractal_canvas.style.left = dx + "px";
                view.fractal_canvas.style.top = dy + "px";
            }
            else {
                // With anti-aliasing, 0.5 offset makes 1-pixel wide.
                const overlay_ctx = view.overlay_canvas.getContext("2d");
                overlay_ctx.lineWidth = 1;
                overlay_ctx.strokeStyle = "#ffffffc0";
                overlay_ctx.strokeRect(this.rubstart.x + 0.5, this.rubstart.y + 0.5, dx, dy);
            }
        }
    },

    view_mouseup(ev) {
        //console.log("up. shift:", ev.shiftKey, "ctrl:", ev.ctrlKey, "meta:", ev.metaKey, "alt:", ev.altKey);
        if (!this.move_target) {
            return;
        }
        ev.preventDefault();
        const view = View.canvas_map.get(this.move_target);
        const up = getCursorPosition(ev, this.move_target);
        const dx = up.x - this.rubstart.x;
        const dy = up.y - this.rubstart.y;
        if (this.moving) {
            this.set_center(
                view.centerr - view.xrot(dx, dy) * view.pixsize,
                view.centeri + view.yrot(dx, dy) * view.pixsize
            );
            const overlay_ctx = view.overlay_canvas.getContext("2d");
            overlay_ctx.drawImage(view.fractal_canvas, dx, dy);
            view.fractal_canvas.style.left = "0";
            view.fractal_canvas.style.top = "0";
            clear_canvas(view.fractal_canvas);
            view.paint().then(() => {
                clear_canvas(view.overlay_canvas);
            });
        }
        else {
            clear_canvas(view.overlay_canvas);
            if (this.mouse_dragging) {
                const a = view.ri4xy(this.rubstart.x, this.rubstart.y);
                const b = view.ri4xy(up.x, up.y);
                const dr = a.r - b.r, di = a.i - b.i;
                const rdr = view.xrot(dr, di);
                const rdi = view.yrot(dr, di);
                this.set_pixsize(Math.max(Math.abs(rdr) / view.canvasW, Math.abs(rdi) / view.canvasH));
                this.set_center((a.r + b.r) / 2, (a.i + b.i) / 2);
            }
            else {
                const {r: clickr, i: clicki} = view.ri4xy(up.x, up.y);

                const factor = ev.altKey ? 1.1 : 2.0;
                if (ev.shiftKey) {
                    this.set_pixsize(view.pixsize * factor);
                }
                else {
                    this.set_pixsize(view.pixsize / factor);
                }
                const r0 = clickr - view.xrot(up.x, up.y) * view.pixsize;
                const i0 = clicki + view.yrot(up.x, up.y) * view.pixsize;
                this.set_center(
                    r0 + view.xrot(view.canvasW, view.canvasH)/2 * view.pixsize,
                    i0 - view.yrot(view.canvasW, view.canvasH)/2 * view.pixsize
                );
            }
            view.paint();
        }
        this.reset_dragging();
    },

    set_moving(m) {
        if (this.moving = m) {
            this.view.overlay_canvas.classList.add("move");
        }
        else {
            this.view.overlay_canvas.classList.remove("move");
        }
    },

    cancel_dragging() {
        if (!this.move_target) {
            return;
        }
        const view = View.canvas_map.get(this.move_target);
        view.fractal_canvas.style.left = "0";
        view.fractal_canvas.style.top = "0";
        clear_canvas(view.overlay_canvas);
        this.reset_dragging();
    },

    keydown(ev) {
        if (ev.target.matches("input")) {
            return;
        }

        //console.log("key:",  ev.key, "shift:", ev.shiftKey, "ctrl:", ev.ctrlKey, "meta:", ev.metaKey, "alt:", ev.altKey);
        let key = ev.key;

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

        let handled = false;

        if (!ev.metaKey) {
            handled = true;
            switch (key) {
                case "Escape":
                    this.cancel_dragging();
                    break;

                case "a":
                    const new_angle = +prompt("Angle", this.view.angle);
                    if (new_angle != this.view.angle) {
                        this.set_angle(new_angle);
                        this.view.paint();
                    }
                    break;

                case "c":
                    this.view.set_continuous(!this.view.continuous);
                    this.view.paint();
                    break;

                case "C":
                    alert(this.view.cli_args());
                    break;

                case "i":
                    new_limit = +prompt("Iteration limit", this.view.iter_limit);
                    if (new_limit != this.view.iter_limit) {
                        this.set_iter_limit(new_limit);
                        this.view.paint();
                    }
                    break;

                case "I":
                    Panels.toggle_panel("infopanel");
                    break;

                case "P":
                    Panels.toggle_panel("palettepanel");
                    break;

                case "r":
                    this.view.paint();
                    break;

                case "R":
                    this.reset();
                    this.view.paint();
                    break;

                case "w":
                    let text;
                    if (!this.view.canvas_size_w) {
                        text = "*";
                    }
                    else {
                        text = `${this.view.canvas_size_w} ${this.view.canvas_size_h}`;
                    }
                    this.view.set_canvas_size(prompt("Canvas size", text));
                    this.view.paint();
                    break;

                case ",":
                    this.view.palette_index -= 1;
                    if (this.view.palette_index < 0) {
                        this.view.palette_index += palettes.length;
                    }
                    this.view.paint();
                    break;

                case ".":
                    this.view.palette_index += 1;
                    this.view.palette_index %= palettes.length;
                    this.view.paint();
                    break;

                case ">":
                    this.set_angle(this.view.angle + (ev.altKey ? 1 : 10));
                    this.view.paint();
                    break;

                case "<":
                    this.set_angle(this.view.angle - (ev.altKey ? 1 : 10));
                    this.view.paint();
                    break;

                case "?":
                    Panels.toggle_panel("helppanel");
                    break;

                default:
                    handled = false;
                    break;
            }
        }

        if (handled) {
            ev.preventDefault();
        }
    },

    resize() {
        if (this.resize_timeout) {
            clearTimeout(this.resize_timeout);
        }
        resize_timeout = setTimeout(
            () => {
                this.resize_timeout = null;
                this.view.set_size();
                this.view.paint();
            },
            250
        );
    },
};

function getCursorPosition(ev, target) {
    const rect = target.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    return {x, y};
}

function set_input_value(name, val) {
    document.getElementById(name).value = "" + val;
}

function get_input_value(name) {
    return +document.getElementById(name).value;
}

const Panels = {
    draggable: null,
    draggable_start: null,
    rubstart: null,

    bring_to_top(el, els) {
        const indexes = [...els].map(e => {
            const z = getComputedStyle(e).zIndex;
            return (z === "auto") ? 0 : z;
        });
        el.style.zIndex = Math.max(...indexes) + 1;
    },

    bring_panel_to_top(el) {
        this.bring_to_top(el, document.querySelectorAll(".panel"));
    },

    toggle_panel(panelid) {
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
            this.position_panel(panel, at_x, at_y);
            this.bring_panel_to_top(panel);
        }
    },

    position_panel(panel, left, top) {
        panel.style.left = left + "px";
        panel.style.top = top + "px";
        panel.style.right = "auto";
        panel.style.bottom = "auto";
    },

    close_panel(ev) {
        const panel = ev.target.closest(".panel");
        panel.style.display = "none";
    },

    draggable_mousedown(ev) {
        if (ev.target.matches("input")) {
            return;
        }
        ev.preventDefault();
        ev.stopPropagation();
        const active = document.activeElement;
        if (active) {
            active.blur();
        }
        this.draggable = ev.delegate;
        this.draggable.classList.add("dragging");
        this.rubstart = {x: ev.clientX, y: ev.clientY};
        this.draggable_start = {x: this.draggable.offsetLeft, y: this.draggable.offsetTop};
        this.bring_panel_to_top(this.draggable);
        this.position_panel(this.draggable, this.draggable.offsetLeft, this.draggable.offsetTop);
    },

    draggable_mousemove(ev) {
        if (!this.draggable) {
            return;
        }
        ev.preventDefault();
        ev.stopPropagation();
        this.position_panel(
            this.draggable,
            this.draggable_start.x - (this.rubstart.x - ev.clientX),
            this.draggable_start.y - (this.rubstart.y - ev.clientY)
        );
    },

    draggable_mouseup(ev) {
        if (!this.draggable) {
            return;
        }
        ev.preventDefault();
        ev.stopPropagation();
        this.draggable.classList.remove("dragging");
        this.draggable = null;
    },
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

function on_event(el, evname, fn, sel) {
    if (sel) {
        fn = delegatedTo(sel, fn);
    }
    if (typeof el === 'string') {
        el = document.querySelectorAll(el);
    }
    if (!el.forEach) {
        el = [el];
    }
    el.forEach(e => e.addEventListener(evname, fn));
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

function main() {
    if (platform() === "mac") {
        document.querySelector("html").classList.add("mac");
    }

    App.init();

    on_event(document, "mousedown", ev => App.view_mousedown(ev), ".view.overlay");
    on_event(document, "mousemove", ev => App.view_mousemove(ev));
    on_event(document, "mouseup", ev => App.view_mouseup(ev));
    on_event(document, "keydown", ev => App.keydown(ev));

    on_event(document, "mousedown", ev => Panels.draggable_mousedown(ev), ".draggable");
    on_event(document, "mousemove", ev => Panels.draggable_mousemove(ev));
    on_event(document, "mouseup", ev => Panels.draggable_mouseup(ev));
    on_event(".panel .closebtn", "click", ev => Panels.close_panel(ev));

    on_event("#infopanel input", "change", ev => App.spec_change(ev));
    on_event(window, "resize", ev => App.resize(ev));

    App.view.set_size();
    App.view.paint();
}
