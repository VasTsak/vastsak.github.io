import{r as N}from"./index.NEDEFKed.js";var ne={exports:{}},P={};/**
 * @license React
 * react-jsx-runtime.production.min.js
 *
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */var pe=N,me=Symbol.for("react.element"),xe=Symbol.for("react.fragment"),Me=Object.prototype.hasOwnProperty,ye=pe.__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED.ReactCurrentOwner,_e={key:!0,ref:!0,__self:!0,__source:!0};function re(s,e,t){var n,r={},i=null,c=null;t!==void 0&&(i=""+t),e.key!==void 0&&(i=""+e.key),e.ref!==void 0&&(c=e.ref);for(n in e)Me.call(e,n)&&!_e.hasOwnProperty(n)&&(r[n]=e[n]);if(s&&s.defaultProps)for(n in e=s.defaultProps,e)r[n]===void 0&&(r[n]=e[n]);return{$$typeof:me,type:s,key:i,ref:c,props:r,_owner:ye.current}}P.Fragment=xe;P.jsx=re;P.jsxs=re;ne.exports=P;var g=ne.exports;function w(s){return Array.isArray?Array.isArray(s):oe(s)==="[object Array]"}const Ee=1/0;function Se(s){if(typeof s=="string")return s;let e=s+"";return e=="0"&&1/s==-Ee?"-0":e}function we(s){return s==null?"":Se(s)}function E(s){return typeof s=="string"}function ie(s){return typeof s=="number"}function Ie(s){return s===!0||s===!1||be(s)&&oe(s)=="[object Boolean]"}function ce(s){return typeof s=="object"}function be(s){return ce(s)&&s!==null}function x(s){return s!=null}function H(s){return!s.trim().length}function oe(s){return s==null?s===void 0?"[object Undefined]":"[object Null]":Object.prototype.toString.call(s)}const ve="Incorrect 'index' type",Ae=s=>`Invalid value for key ${s}`,Re=s=>`Pattern length exceeds max of ${s}.`,Ne=s=>`Missing ${s} property in key`,Le=s=>`Property 'weight' in key '${s}' must be a positive integer`,X=Object.prototype.hasOwnProperty;class je{constructor(e){this._keys=[],this._keyMap={};let t=0;e.forEach(n=>{let r=ae(n);this._keys.push(r),this._keyMap[r.id]=r,t+=r.weight}),this._keys.forEach(n=>{n.weight/=t})}get(e){return this._keyMap[e]}keys(){return this._keys}toJSON(){return JSON.stringify(this._keys)}}function ae(s){let e=null,t=null,n=null,r=1,i=null;if(E(s)||w(s))n=s,e=Z(s),t=K(s);else{if(!X.call(s,"name"))throw new Error(Ne("name"));const c=s.name;if(n=c,X.call(s,"weight")&&(r=s.weight,r<=0))throw new Error(Le(c));e=Z(c),t=K(c),i=s.getFn}return{path:e,id:t,weight:r,src:n,getFn:i}}function Z(s){return w(s)?s:s.split(".")}function K(s){return w(s)?s.join("."):s}function ke(s,e){let t=[],n=!1;const r=(i,c,a)=>{if(x(i))if(!c[a])t.push(i);else{let o=c[a];const h=i[o];if(!x(h))return;if(a===c.length-1&&(E(h)||ie(h)||Ie(h)))t.push(we(h));else if(w(h)){n=!0;for(let l=0,f=h.length;l<f;l+=1)r(h[l],c,a+1)}else c.length&&r(h,c,a+1)}};return r(s,E(e)?e.split("."):e,0),n?t:t[0]}const Oe={includeMatches:!1,findAllMatches:!1,minMatchCharLength:1},$e={isCaseSensitive:!1,includeScore:!1,keys:[],shouldSort:!0,sortFn:(s,e)=>s.score===e.score?s.idx<e.idx?-1:1:s.score<e.score?-1:1},Ce={location:0,threshold:.6,distance:100},Te={useExtendedSearch:!1,getFn:ke,ignoreLocation:!1,ignoreFieldNorm:!1,fieldNormWeight:1};var u={...$e,...Oe,...Ce,...Te};const Fe=/[^ ]+/g;function Pe(s=1,e=3){const t=new Map,n=Math.pow(10,e);return{get(r){const i=r.match(Fe).length;if(t.has(i))return t.get(i);const c=1/Math.pow(i,.5*s),a=parseFloat(Math.round(c*n)/n);return t.set(i,a),a},clear(){t.clear()}}}class G{constructor({getFn:e=u.getFn,fieldNormWeight:t=u.fieldNormWeight}={}){this.norm=Pe(t,3),this.getFn=e,this.isCreated=!1,this.setIndexRecords()}setSources(e=[]){this.docs=e}setIndexRecords(e=[]){this.records=e}setKeys(e=[]){this.keys=e,this._keysMap={},e.forEach((t,n)=>{this._keysMap[t.id]=n})}create(){this.isCreated||!this.docs.length||(this.isCreated=!0,E(this.docs[0])?this.docs.forEach((e,t)=>{this._addString(e,t)}):this.docs.forEach((e,t)=>{this._addObject(e,t)}),this.norm.clear())}add(e){const t=this.size();E(e)?this._addString(e,t):this._addObject(e,t)}removeAt(e){this.records.splice(e,1);for(let t=e,n=this.size();t<n;t+=1)this.records[t].i-=1}getValueForItemAtKeyId(e,t){return e[this._keysMap[t]]}size(){return this.records.length}_addString(e,t){if(!x(e)||H(e))return;let n={v:e,i:t,n:this.norm.get(e)};this.records.push(n)}_addObject(e,t){let n={i:t,$:{}};this.keys.forEach((r,i)=>{let c=r.getFn?r.getFn(e):this.getFn(e,r.path);if(x(c)){if(w(c)){let a=[];const o=[{nestedArrIndex:-1,value:c}];for(;o.length;){const{nestedArrIndex:h,value:l}=o.pop();if(x(l))if(E(l)&&!H(l)){let f={v:l,i:h,n:this.norm.get(l)};a.push(f)}else w(l)&&l.forEach((f,d)=>{o.push({nestedArrIndex:d,value:f})})}n.$[i]=a}else if(E(c)&&!H(c)){let a={v:c,n:this.norm.get(c)};n.$[i]=a}}}),this.records.push(n)}toJSON(){return{keys:this.keys,records:this.records}}}function he(s,e,{getFn:t=u.getFn,fieldNormWeight:n=u.fieldNormWeight}={}){const r=new G({getFn:t,fieldNormWeight:n});return r.setKeys(s.map(ae)),r.setSources(e),r.create(),r}function De(s,{getFn:e=u.getFn,fieldNormWeight:t=u.fieldNormWeight}={}){const{keys:n,records:r}=s,i=new G({getFn:e,fieldNormWeight:t});return i.setKeys(n),i.setIndexRecords(r),i}function T(s,{errors:e=0,currentLocation:t=0,expectedLocation:n=0,distance:r=u.distance,ignoreLocation:i=u.ignoreLocation}={}){const c=e/s.length;if(i)return c;const a=Math.abs(n-t);return r?c+a/r:a?1:c}function ze(s=[],e=u.minMatchCharLength){let t=[],n=-1,r=-1,i=0;for(let c=s.length;i<c;i+=1){let a=s[i];a&&n===-1?n=i:!a&&n!==-1&&(r=i-1,r-n+1>=e&&t.push([n,r]),n=-1)}return s[i-1]&&i-n>=e&&t.push([n,i-1]),t}const L=32;function He(s,e,t,{location:n=u.location,distance:r=u.distance,threshold:i=u.threshold,findAllMatches:c=u.findAllMatches,minMatchCharLength:a=u.minMatchCharLength,includeMatches:o=u.includeMatches,ignoreLocation:h=u.ignoreLocation}={}){if(e.length>L)throw new Error(Re(L));const l=e.length,f=s.length,d=Math.max(0,Math.min(n,f));let p=i,m=d;const M=a>1||o,A=M?Array(f):[];let S;for(;(S=s.indexOf(e,m))>-1;){let y=T(e,{currentLocation:S,expectedLocation:d,distance:r,ignoreLocation:h});if(p=Math.min(y,p),m=S+l,M){let I=0;for(;I<l;)A[S+I]=1,I+=1}}m=-1;let j=[],R=1,$=l+f;const ge=1<<l-1;for(let y=0;y<l;y+=1){let I=0,b=$;for(;I<b;)T(e,{errors:y,currentLocation:d+b,expectedLocation:d,distance:r,ignoreLocation:h})<=p?I=b:$=b,b=Math.floor(($-I)/2+I);$=b;let Q=Math.max(1,d-b+1),z=c?f:Math.min(d+b,f)+l,k=Array(z+2);k[z+1]=(1<<y)-1;for(let _=z;_>=Q;_-=1){let C=_-1,J=t[s.charAt(C)];if(M&&(A[C]=+!!J),k[_]=(k[_+1]<<1|1)&J,y&&(k[_]|=(j[_+1]|j[_])<<1|1|j[_+1]),k[_]&ge&&(R=T(e,{errors:y,currentLocation:C,expectedLocation:d,distance:r,ignoreLocation:h}),R<=p)){if(p=R,m=C,m<=d)break;Q=Math.max(1,2*d-m)}}if(T(e,{errors:y+1,currentLocation:d,expectedLocation:d,distance:r,ignoreLocation:h})>p)break;j=k}const D={isMatch:m>=0,score:Math.max(.001,R)};if(M){const y=ze(A,a);y.length?o&&(D.indices=y):D.isMatch=!1}return D}function Ke(s){let e={};for(let t=0,n=s.length;t<n;t+=1){const r=s.charAt(t);e[r]=(e[r]||0)|1<<n-t-1}return e}class le{constructor(e,{location:t=u.location,threshold:n=u.threshold,distance:r=u.distance,includeMatches:i=u.includeMatches,findAllMatches:c=u.findAllMatches,minMatchCharLength:a=u.minMatchCharLength,isCaseSensitive:o=u.isCaseSensitive,ignoreLocation:h=u.ignoreLocation}={}){if(this.options={location:t,threshold:n,distance:r,includeMatches:i,findAllMatches:c,minMatchCharLength:a,isCaseSensitive:o,ignoreLocation:h},this.pattern=o?e:e.toLowerCase(),this.chunks=[],!this.pattern.length)return;const l=(d,p)=>{this.chunks.push({pattern:d,alphabet:Ke(d),startIndex:p})},f=this.pattern.length;if(f>L){let d=0;const p=f%L,m=f-p;for(;d<m;)l(this.pattern.substr(d,L),d),d+=L;if(p){const M=f-L;l(this.pattern.substr(M),M)}}else l(this.pattern,0)}searchIn(e){const{isCaseSensitive:t,includeMatches:n}=this.options;if(t||(e=e.toLowerCase()),this.pattern===e){let m={isMatch:!0,score:0};return n&&(m.indices=[[0,e.length-1]]),m}const{location:r,distance:i,threshold:c,findAllMatches:a,minMatchCharLength:o,ignoreLocation:h}=this.options;let l=[],f=0,d=!1;this.chunks.forEach(({pattern:m,alphabet:M,startIndex:A})=>{const{isMatch:S,score:j,indices:R}=He(e,m,M,{location:r+A,distance:i,threshold:c,findAllMatches:a,minMatchCharLength:o,includeMatches:n,ignoreLocation:h});S&&(d=!0),f+=j,S&&R&&(l=[...l,...R])});let p={isMatch:d,score:d?f/this.chunks.length:1};return d&&n&&(p.indices=l),p}}class v{constructor(e){this.pattern=e}static isMultiMatch(e){return q(e,this.multiRegex)}static isSingleMatch(e){return q(e,this.singleRegex)}search(){}}function q(s,e){const t=s.match(e);return t?t[1]:null}class Ve extends v{constructor(e){super(e)}static get type(){return"exact"}static get multiRegex(){return/^="(.*)"$/}static get singleRegex(){return/^=(.*)$/}search(e){const t=e===this.pattern;return{isMatch:t,score:t?0:1,indices:[0,this.pattern.length-1]}}}class We extends v{constructor(e){super(e)}static get type(){return"inverse-exact"}static get multiRegex(){return/^!"(.*)"$/}static get singleRegex(){return/^!(.*)$/}search(e){const n=e.indexOf(this.pattern)===-1;return{isMatch:n,score:n?0:1,indices:[0,e.length-1]}}}class Be extends v{constructor(e){super(e)}static get type(){return"prefix-exact"}static get multiRegex(){return/^\^"(.*)"$/}static get singleRegex(){return/^\^(.*)$/}search(e){const t=e.startsWith(this.pattern);return{isMatch:t,score:t?0:1,indices:[0,this.pattern.length-1]}}}class Ue extends v{constructor(e){super(e)}static get type(){return"inverse-prefix-exact"}static get multiRegex(){return/^!\^"(.*)"$/}static get singleRegex(){return/^!\^(.*)$/}search(e){const t=!e.startsWith(this.pattern);return{isMatch:t,score:t?0:1,indices:[0,e.length-1]}}}class Ye extends v{constructor(e){super(e)}static get type(){return"suffix-exact"}static get multiRegex(){return/^"(.*)"\$$/}static get singleRegex(){return/^(.*)\$$/}search(e){const t=e.endsWith(this.pattern);return{isMatch:t,score:t?0:1,indices:[e.length-this.pattern.length,e.length-1]}}}class Ge extends v{constructor(e){super(e)}static get type(){return"inverse-suffix-exact"}static get multiRegex(){return/^!"(.*)"\$$/}static get singleRegex(){return/^!(.*)\$$/}search(e){const t=!e.endsWith(this.pattern);return{isMatch:t,score:t?0:1,indices:[0,e.length-1]}}}class ue extends v{constructor(e,{location:t=u.location,threshold:n=u.threshold,distance:r=u.distance,includeMatches:i=u.includeMatches,findAllMatches:c=u.findAllMatches,minMatchCharLength:a=u.minMatchCharLength,isCaseSensitive:o=u.isCaseSensitive,ignoreLocation:h=u.ignoreLocation}={}){super(e),this._bitapSearch=new le(e,{location:t,threshold:n,distance:r,includeMatches:i,findAllMatches:c,minMatchCharLength:a,isCaseSensitive:o,ignoreLocation:h})}static get type(){return"fuzzy"}static get multiRegex(){return/^"(.*)"$/}static get singleRegex(){return/^(.*)$/}search(e){return this._bitapSearch.searchIn(e)}}class fe extends v{constructor(e){super(e)}static get type(){return"include"}static get multiRegex(){return/^'"(.*)"$/}static get singleRegex(){return/^'(.*)$/}search(e){let t=0,n;const r=[],i=this.pattern.length;for(;(n=e.indexOf(this.pattern,t))>-1;)t=n+i,r.push([n,t-1]);const c=!!r.length;return{isMatch:c,score:c?0:1,indices:r}}}const V=[Ve,fe,Be,Ue,Ge,Ye,We,ue],ee=V.length,Qe=/ +(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)/,Je="|";function Xe(s,e={}){return s.split(Je).map(t=>{let n=t.trim().split(Qe).filter(i=>i&&!!i.trim()),r=[];for(let i=0,c=n.length;i<c;i+=1){const a=n[i];let o=!1,h=-1;for(;!o&&++h<ee;){const l=V[h];let f=l.isMultiMatch(a);f&&(r.push(new l(f,e)),o=!0)}if(!o)for(h=-1;++h<ee;){const l=V[h];let f=l.isSingleMatch(a);if(f){r.push(new l(f,e));break}}}return r})}const Ze=new Set([ue.type,fe.type]);class qe{constructor(e,{isCaseSensitive:t=u.isCaseSensitive,includeMatches:n=u.includeMatches,minMatchCharLength:r=u.minMatchCharLength,ignoreLocation:i=u.ignoreLocation,findAllMatches:c=u.findAllMatches,location:a=u.location,threshold:o=u.threshold,distance:h=u.distance}={}){this.query=null,this.options={isCaseSensitive:t,includeMatches:n,minMatchCharLength:r,findAllMatches:c,ignoreLocation:i,location:a,threshold:o,distance:h},this.pattern=t?e:e.toLowerCase(),this.query=Xe(this.pattern,this.options)}static condition(e,t){return t.useExtendedSearch}searchIn(e){const t=this.query;if(!t)return{isMatch:!1,score:1};const{includeMatches:n,isCaseSensitive:r}=this.options;e=r?e:e.toLowerCase();let i=0,c=[],a=0;for(let o=0,h=t.length;o<h;o+=1){const l=t[o];c.length=0,i=0;for(let f=0,d=l.length;f<d;f+=1){const p=l[f],{isMatch:m,indices:M,score:A}=p.search(e);if(m){if(i+=1,a+=A,n){const S=p.constructor.type;Ze.has(S)?c=[...c,...M]:c.push(M)}}else{a=0,i=0,c.length=0;break}}if(i){let f={isMatch:!0,score:a/i};return n&&(f.indices=c),f}}return{isMatch:!1,score:1}}}const W=[];function et(...s){W.push(...s)}function B(s,e){for(let t=0,n=W.length;t<n;t+=1){let r=W[t];if(r.condition(s,e))return new r(s,e)}return new le(s,e)}const F={AND:"$and",OR:"$or"},U={PATH:"$path",PATTERN:"$val"},Y=s=>!!(s[F.AND]||s[F.OR]),tt=s=>!!s[U.PATH],st=s=>!w(s)&&ce(s)&&!Y(s),te=s=>({[F.AND]:Object.keys(s).map(e=>({[e]:s[e]}))});function de(s,e,{auto:t=!0}={}){const n=r=>{let i=Object.keys(r);const c=tt(r);if(!c&&i.length>1&&!Y(r))return n(te(r));if(st(r)){const o=c?r[U.PATH]:i[0],h=c?r[U.PATTERN]:r[o];if(!E(h))throw new Error(Ae(o));const l={keyId:K(o),pattern:h};return t&&(l.searcher=B(h,e)),l}let a={children:[],operator:i[0]};return i.forEach(o=>{const h=r[o];w(h)&&h.forEach(l=>{a.children.push(n(l))})}),a};return Y(s)||(s=te(s)),n(s)}function nt(s,{ignoreFieldNorm:e=u.ignoreFieldNorm}){s.forEach(t=>{let n=1;t.matches.forEach(({key:r,norm:i,score:c})=>{const a=r?r.weight:null;n*=Math.pow(c===0&&a?Number.EPSILON:c,(a||1)*(e?1:i))}),t.score=n})}function rt(s,e){const t=s.matches;e.matches=[],x(t)&&t.forEach(n=>{if(!x(n.indices)||!n.indices.length)return;const{indices:r,value:i}=n;let c={indices:r,value:i};n.key&&(c.key=n.key.src),n.idx>-1&&(c.refIndex=n.idx),e.matches.push(c)})}function it(s,e){e.score=s.score}function ct(s,e,{includeMatches:t=u.includeMatches,includeScore:n=u.includeScore}={}){const r=[];return t&&r.push(rt),n&&r.push(it),s.map(i=>{const{idx:c}=i,a={item:e[c],refIndex:c};return r.length&&r.forEach(o=>{o(i,a)}),a})}class O{constructor(e,t={},n){this.options={...u,...t},this.options.useExtendedSearch,this._keyStore=new je(this.options.keys),this.setCollection(e,n)}setCollection(e,t){if(this._docs=e,t&&!(t instanceof G))throw new Error(ve);this._myIndex=t||he(this.options.keys,this._docs,{getFn:this.options.getFn,fieldNormWeight:this.options.fieldNormWeight})}add(e){x(e)&&(this._docs.push(e),this._myIndex.add(e))}remove(e=()=>!1){const t=[];for(let n=0,r=this._docs.length;n<r;n+=1){const i=this._docs[n];e(i,n)&&(this.removeAt(n),n-=1,r-=1,t.push(i))}return t}removeAt(e){this._docs.splice(e,1),this._myIndex.removeAt(e)}getIndex(){return this._myIndex}search(e,{limit:t=-1}={}){const{includeMatches:n,includeScore:r,shouldSort:i,sortFn:c,ignoreFieldNorm:a}=this.options;let o=E(e)?E(this._docs[0])?this._searchStringList(e):this._searchObjectList(e):this._searchLogical(e);return nt(o,{ignoreFieldNorm:a}),i&&o.sort(c),ie(t)&&t>-1&&(o=o.slice(0,t)),ct(o,this._docs,{includeMatches:n,includeScore:r})}_searchStringList(e){const t=B(e,this.options),{records:n}=this._myIndex,r=[];return n.forEach(({v:i,i:c,n:a})=>{if(!x(i))return;const{isMatch:o,score:h,indices:l}=t.searchIn(i);o&&r.push({item:i,idx:c,matches:[{score:h,value:i,norm:a,indices:l}]})}),r}_searchLogical(e){const t=de(e,this.options),n=(a,o,h)=>{if(!a.children){const{keyId:f,searcher:d}=a,p=this._findMatches({key:this._keyStore.get(f),value:this._myIndex.getValueForItemAtKeyId(o,f),searcher:d});return p&&p.length?[{idx:h,item:o,matches:p}]:[]}const l=[];for(let f=0,d=a.children.length;f<d;f+=1){const p=a.children[f],m=n(p,o,h);if(m.length)l.push(...m);else if(a.operator===F.AND)return[]}return l},r=this._myIndex.records,i={},c=[];return r.forEach(({$:a,i:o})=>{if(x(a)){let h=n(t,a,o);h.length&&(i[o]||(i[o]={idx:o,item:a,matches:[]},c.push(i[o])),h.forEach(({matches:l})=>{i[o].matches.push(...l)}))}}),c}_searchObjectList(e){const t=B(e,this.options),{keys:n,records:r}=this._myIndex,i=[];return r.forEach(({$:c,i:a})=>{if(!x(c))return;let o=[];n.forEach((h,l)=>{o.push(...this._findMatches({key:h,value:c[l],searcher:t}))}),o.length&&i.push({idx:a,item:c,matches:o})}),i}_findMatches({key:e,value:t,searcher:n}){if(!x(t))return[];let r=[];if(w(t))t.forEach(({v:i,i:c,n:a})=>{if(!x(i))return;const{isMatch:o,score:h,indices:l}=n.searchIn(i);o&&r.push({score:h,key:e,value:i,idx:c,norm:a,indices:l})});else{const{v:i,n:c}=t,{isMatch:a,score:o,indices:h}=n.searchIn(i);a&&r.push({score:o,key:e,value:i,norm:c,indices:h})}return r}}O.version="7.0.0";O.createIndex=he;O.parseIndex=De;O.config=u;O.parseQuery=de;et(qe);const se={lang:"en",langTag:["en-EN"]};function ot({pubDatetime:s,modDatetime:e,size:t="sm",className:n}){return g.jsxs("div",{className:`flex items-center space-x-2 opacity-80 ${n}`,children:[g.jsxs("svg",{xmlns:"http://www.w3.org/2000/svg",className:`${t==="sm"?"scale-90":"scale-100"} inline-block h-6 w-6 min-w-[1.375rem] fill-skin-base`,"aria-hidden":"true",children:[g.jsx("path",{d:"M7 11h2v2H7zm0 4h2v2H7zm4-4h2v2h-2zm0 4h2v2h-2zm4-4h2v2h-2zm0 4h2v2h-2z"}),g.jsx("path",{d:"M5 22h14c1.103 0 2-.897 2-2V6c0-1.103-.897-2-2-2h-2V2h-2v2H9V2H7v2H5c-1.103 0-2 .897-2 2v14c0 1.103.897 2 2 2zM19 8l.001 12H5V8h14z"})]}),e?g.jsx("span",{className:`italic ${t==="sm"?"text-sm":"text-base"}`,children:"Updated:"}):g.jsx("span",{className:"sr-only",children:"Published:"}),g.jsx("span",{className:`italic ${t==="sm"?"text-sm":"text-base"}`,children:g.jsx(at,{pubDatetime:s,modDatetime:e})})]})}const at=({pubDatetime:s,modDatetime:e})=>{const t=new Date(e||s),n=t.toLocaleDateString(se.langTag,{year:"numeric",month:"short",day:"numeric"}),r=t.toLocaleTimeString(se.langTag,{hour:"2-digit",minute:"2-digit"});return g.jsxs(g.Fragment,{children:[g.jsx("time",{dateTime:t.toISOString(),children:n}),g.jsx("span",{"aria-hidden":"true",children:" | "}),g.jsx("span",{className:"sr-only",children:" at "}),g.jsx("span",{className:"text-nowrap",children:r})]})};function ht({href:s,frontmatter:e}){const{title:t,pubDatetime:n,modDatetime:r,description:i}=e,c={className:"text-lg line-clamp-2-inline font-medium decoration-dashed hover:underline"};return g.jsxs("li",{className:"my-6",children:[g.jsx("a",{href:s,className:"inline-block text-lg font-medium text-skin-accent decoration-dashed underline-offset-4 focus-visible:no-underline focus-visible:underline-offset-0",children:g.jsx("h2",{...c,children:t})}),g.jsx(ot,{pubDatetime:n,modDatetime:r}),g.jsx("p",{className:"line-clamp-3-inline sm:line-clamp-2-inline",children:i})]})}function ft({searchList:s}){const e=N.useRef(null),[t,n]=N.useState(""),[r,i]=N.useState(null),c=o=>{n(o.currentTarget.value)},a=N.useMemo(()=>new O(s,{keys:["title","description"],includeMatches:!0,minMatchCharLength:2,threshold:.5}),[s]);return N.useEffect(()=>{const h=new URLSearchParams(window.location.search).get("q");h&&n(h),setTimeout(function(){e.current.selectionStart=e.current.selectionEnd=h?.length||0},50)},[]),N.useEffect(()=>{let o=t.length>1?a.search(t):[];if(i(o),t.length>0){const h=new URLSearchParams(window.location.search);h.set("q",t);const l=window.location.pathname+"?"+h.toString();history.replaceState(history.state,"",l)}else history.replaceState(history.state,"",window.location.pathname)},[t]),g.jsxs(g.Fragment,{children:[g.jsxs("label",{className:"relative block",children:[g.jsxs("span",{className:"absolute inset-y-0 left-0 flex items-center pl-2 opacity-75",children:[g.jsx("svg",{xmlns:"http://www.w3.org/2000/svg","aria-hidden":"true",children:g.jsx("path",{d:"M19.023 16.977a35.13 35.13 0 0 1-1.367-1.384c-.372-.378-.596-.653-.596-.653l-2.8-1.337A6.962 6.962 0 0 0 16 9c0-3.859-3.14-7-7-7S2 5.141 2 9s3.14 7 7 7c1.763 0 3.37-.66 4.603-1.739l1.337 2.8s.275.224.653.596c.387.363.896.854 1.384 1.367l1.358 1.392.604.646 2.121-2.121-.646-.604c-.379-.372-.885-.866-1.391-1.36zM9 14c-2.757 0-5-2.243-5-5s2.243-5 5-5 5 2.243 5 5-2.243 5-5 5z"})}),g.jsx("span",{className:"sr-only",children:"Search"})]}),g.jsx("input",{className:`block w-full rounded border border-skin-fill 
        border-opacity-40 bg-skin-fill py-3 pl-10
        pr-3 placeholder:italic placeholder:text-opacity-75 
        focus:border-skin-accent focus:outline-none`,placeholder:"Search posts...",type:"text",name:"search",value:t,onChange:c,autoComplete:"off",ref:e})]}),t.length>1&&g.jsxs("div",{className:"mt-8",children:["Found ",r?.length,r?.length&&r?.length===1?" result":" results"," ","for '",t,"'"]}),g.jsx("ul",{children:r&&r.map(({item:o,refIndex:h})=>g.jsx(ht,{href:`/posts/${o.slug}/`,frontmatter:o.data},`${h}-${o.slug}`))})]})}export{ft as default};