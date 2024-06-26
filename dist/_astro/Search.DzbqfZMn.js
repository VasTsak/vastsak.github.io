import { r as N } from "./index.NEDEFKed.js";
var ne = { exports: {} }, P = {};

/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
var pe = N,
    me = Symbol.for("react.element"),
    xe = Symbol.for("react.fragment"),
    Me = Object.prototype.hasOwnProperty,
    ye = pe.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,
    _e = { key: !0, ref: !0, __self: !0, __source: !0 };

function re(s, e, t) {
    var n, r = {}, i = null, c = null;
    t !== void 0 && (i = "" + t);
    e.key !== void 0 && (i = "" + e.key);
    e.ref !== void 0 && (c = e.ref);
    for (n in e) Me.call(e, n) && !_e.hasOwnProperty(n) && (r[n] = e[n]);
    if (s && s.defaultProps)
        for (n in e = s.defaultProps, e) r[n] === void 0 && (r[n] = e[n]);
    return { $$typeof: me, type: s, key: i, ref: c, props: r, _owner: ye.current };
}
P.Fragment = xe;
P.jsx = re;
P.jsxs = re;
ne.exports = P;

function w(s) {
    return Array.isArray ? Array.isArray(s) : oe(s) === "[object Array]";
}

const Ee = 1 / 0;

function Se(s) {
    if (typeof s == "string") return s;
    let e = s + "";
    return e == "0" && 1 / s == -Ee ? "-0" : e;
}

function we(s) {
    return s == null ? "" : Se(s);
}

function E(s) {
    return typeof s == "string";
}

function ie(s) {
    return typeof s == "number";
}

function Ie(s) {
    return s === !0 || s === !1 || be(s) && oe(s) === "[object Boolean]";
}

function ce(s) {
    return typeof s == "object";
}

function be(s) {
    return ce(s) && s !== null;
}

function x(s) {
    return s != null;
}

function H(s) {
    return !s.trim().length;
}

function oe(s) {
    return s == null ? s === void 0 ? "[object Undefined]" : "[object Null]" : Object.prototype.toString.call(s);
}

const Ne = s => `Missing ${s} property in key`,
    Le = s => `Property 'weight' in key '${s}' must be a positive integer`,
    X = Object.prototype.hasOwnProperty;

function Z(s) {
    return w(s) ? s : s.split(".");
}

function K(s) {
    return w(s) ? s.join(".") : s;
}

function ke(s, e) {
    let t = [], n = !1;
    const r = (i, c, a) => {
        if (x(i))
            if (!c[a]) t.push(i);
            else {
                let o = c[a];
                const h = i[o];
                if (!x(h)) return;
                if (a === c.length - 1 && (E(h) || ie(h) || Ie(h))) t.push(we(h));
                else if (w(h)) {
                    n = !0;
                    for (let l = 0, f = h.length; l < f; l += 1) r(h[l], c, a + 1);
                } else c.length && r(h, c, a + 1);
            }
    };
    return r(s, E(e) ? e.split(".") : e, 0), n ? t : t[0];
}

const Oe = { includeMatches: !1, findAllMatches: !1, minMatchCharLength: 1 },
    $e = { isCaseSensitive: !1, includeScore: !1, keys: [], shouldSort: !0, sortFn: (s, e) => s.score === e.score ? s.idx < e.idx ? -1 : 1 : s.score < e.score ? -1 : 1 },
    Ce = { location: 0, threshold: .6, distance: 100 },
    Te = { useExtendedSearch: !1, getFn: ke, ignoreLocation: !1, ignoreFieldNorm: !1, fieldNormWeight: 1 };
var u = { ...$e, ...Oe, ...Ce, ...Te };

const Fe = /[^ ]+/g;

function Pe(s = 1, e = 3) {
    const t = new Map, n = Math.pow(10, e);
    return {
        get(r) {
            const i = r.match(Fe).length;
            if (t.has(i)) return t.get(i);
            const c = 1 / Math.pow(i, .5 * s),
                a = parseFloat(Math.round(c * n) / n);
            return t.set(i, a), a;
        },
        clear() { t.clear(); }
    };
}

class G {
    constructor({ getFn: e = u.getFn, fieldNormWeight: t = u.fieldNormWeight } = {}) {
        this.norm = Pe(t, 3);
        this.getFn = e;
        this.isCreated = !1;
        this.setIndexRecords();
    }

    setSources(e = []) {
        this.docs = e;
    }

    setIndexRecords(e = []) {
        this.records = e;
    }

    setKeys(e = []) {
        this.keys = e;
        this._keysMap = {};
        e.forEach((t, n) => { this._keysMap[t.id] = n; });
    }

    create() {
        if (!this.isCreated && this.docs.length) {
            this.isCreated = !0;
            E(this.docs[0])
                ? this.docs.forEach((e, t) => { this._addString(e, t); })
                : this.docs.forEach((e, t) => { this._addObject(e, t); });
            this.norm.clear();
        }
    }

    add(e) {
        const t = this.size();
        E(e) ? this._addString(e, t) : this._addObject(e, t);
    }

    removeAt(e) {
        this.records.splice(e, 1);
        for (let t = e, n = this.size(); t < n; t += 1) this.records[t].i -= 1;
    }

    getValueForItemAtKeyId(e, t) {
        return e[this._keysMap[t]];
    }

    size() {
        return this.records.length;
    }

    _addString(e, t) {
        if (!x(e) || H(e)) return;
        let n = { v: e, i: t, n: this.norm.get(e) };
        this.records.push(n);
    }

    _addObject(e, t) {
        let n = { i: t, $: {} };
        this.keys.forEach((r, i) => {
            let c = r.getFn ? r.getFn(e) : this.getFn(e, r.path);
            if (x(c)) {
                if (w(c)) {
                    let a = [];
                    const o = [{ nestedArrIndex: -1, value: c }];
                    for (; o.length;) {
                        const { nestedArrIndex: h, value: l } = o.pop();
                        if (x(l))
                            if (E(l) && !H(l)) {
                                let f = { v: l, i: h, n: this.norm.get(l) };
                                a.push(f);
                            } else l.forEach((f, p) => { o.push({ nestedArrIndex: p, value: f }); });
                    }
                    n.$[i] = a;
                } else if (!H(c)) {
                    let a = { v: c, n: this.norm.get(c) };
                    n.$[i] = a;
                }
            }
        });
        this.records.push(n);
    }
}

export { G as i, u as o };
