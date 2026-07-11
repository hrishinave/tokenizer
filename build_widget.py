"""Bake the trained BPE tokenizer into a single self-contained web/index.html.

Re-run this whenever you retrain (it reads tokenizer.json + data/*.txt):
    python3 train.py && python3 build_widget.py
The output web/index.html has no external dependencies and can be deployed
anywhere (GitHub Pages, Netlify, S3, or just opened in a browser).
"""
import json
import os

import bpe

ROOT = os.path.dirname(os.path.abspath(__file__))
LANGS = [("en", "English"), ("hi", "Hindi"), ("te", "Telugu"), ("kn", "Kannada")]


def load_tokenizer():
    with open(os.path.join(ROOT, "tokenizer.json"), encoding="utf-8") as f:
        return json.load(f)


def script_of(tok):
    t = tok.replace("</w>", "")
    has = set()
    for ch in t:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F:
            has.add("Hindi")
        elif 0x0C00 <= o <= 0x0C7F:
            has.add("Telugu")
        elif 0x0C80 <= o <= 0x0CFF:
            has.add("Kannada")
        elif ("a" <= ch <= "z") or ("A" <= ch <= "Z"):
            has.add("English")
    if not has:
        return "shared"
    if len(has) == 1:
        return next(iter(has))
    return "mixed"


def vocab_stats(vocab):
    counts = {}
    for t in vocab:
        s = script_of(t)
        counts[s] = counts.get(s, 0) + 1
    tot = len(vocab)
    return {k: {"tokens": v, "pct": round(100 * v / tot, 1)} for k, v in counts.items()}


def samples(n_words=45):
    out = {}
    for code, name in LANGS:
        path = os.path.join(ROOT, "data", f"{code}.txt")
        with open(path, encoding="utf-8") as f:
            words = f.read().split()
        out[name] = " ".join(words[:n_words])
    return out


def per_language_fertility(merges):
    """Fertility (tokens/words) of each language on its full reference article."""
    ferts = {}
    for code, name in LANGS:
        with open(os.path.join(ROOT, "data", f"{code}.txt"), encoding="utf-8") as f:
            text = f.read()
        ferts[name] = round(len(bpe.encode(text, merges)) / len(text.split()), 4)
    return ferts


def build():
    tok = load_tokenizer()
    merges = tok["merges"]
    stats = vocab_stats(tok["vocab"])
    data = {
        "merges": merges,
        "vocab": tok["vocab"],
        "vocabSize": len(tok["vocab"]),
        "baseSymbols": len(tok["vocab"]) - len(merges),
        "stats": stats,
        "fertilities": per_language_fertility(merges),
        "samples": samples(),
    }
    html = TEMPLATE.replace("/*__DATA__*/", json.dumps(data, ensure_ascii=False))
    # Guard: refuse to write a file with control characters that would break the
    # inline <script> (this is exactly the NUL-byte bug that silently killed it).
    bad = sorted({ord(c) for c in html if ord(c) < 0x20 and c not in "\t\n\r"}
                 | {ord(c) for c in html if ord(c) in (0x2028, 0x2029)})
    if bad:
        raise SystemExit(f"ABORT: output contains illegal control chars {[hex(b) for b in bad]}")
    os.makedirs(os.path.join(ROOT, "web"), exist_ok=True)
    out_path = os.path.join(ROOT, "web", "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    kb = os.path.getsize(out_path) / 1024
    print(f"Wrote {out_path} ({kb:.0f} KB, {len(merges)} merges embedded)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>India Wikipedia BPE Tokenizer</title>
<style>
  :root{
    --bg:#f7f7f8; --panel:#ffffff; --ink:#1b1b1f; --muted:#6b6b76;
    --line:#e4e4ea; --accent:#3b5bdb; --accent-soft:#e7ecff;
    --chip-fg:#1b1b1f;
    --radius:14px; --shadow:0 1px 2px rgba(0,0,0,.05),0 8px 24px rgba(0,0,0,.06);
  }
  @media (prefers-color-scheme: dark){
    :root{
      --bg:#0f1014; --panel:#181a20; --ink:#e9e9ee; --muted:#9a9aa7;
      --line:#2a2d36; --accent:#7c93ff; --accent-soft:#20263f; --chip-fg:#e9e9ee;
      --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
    }
  }
  :root[data-theme="dark"]{
      --bg:#0f1014; --panel:#181a20; --ink:#e9e9ee; --muted:#9a9aa7;
      --line:#2a2d36; --accent:#7c93ff; --accent-soft:#20263f; --chip-fg:#e9e9ee;
      --shadow:0 1px 2px rgba(0,0,0,.4),0 10px 30px rgba(0,0,0,.35);
  }
  :root[data-theme="light"]{
    --bg:#f7f7f8; --panel:#ffffff; --ink:#1b1b1f; --muted:#6b6b76;
    --line:#e4e4ea; --accent:#3b5bdb; --accent-soft:#e7ecff; --chip-fg:#1b1b1f;
    --shadow:0 1px 2px rgba(0,0,0,.05),0 8px 24px rgba(0,0,0,.06);
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    line-height:1.5;-webkit-font-smoothing:antialiased;}
  .wrap{max-width:860px;margin:0 auto;padding:32px 20px 64px;}
  header{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;}
  h1{font-size:1.5rem;margin:0 0 4px;letter-spacing:-.02em;}
  .sub{color:var(--muted);font-size:.92rem;margin:0;}
  .theme-btn{flex:0 0 auto;background:var(--panel);border:1px solid var(--line);
    color:var(--ink);border-radius:10px;padding:8px 12px;cursor:pointer;font-size:.85rem;}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);
    box-shadow:var(--shadow);padding:18px;margin-top:18px;}
  .samples{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px;}
  .samples span{color:var(--muted);font-size:.85rem;align-self:center;margin-right:2px;}
  .pill{background:var(--panel);border:1px solid var(--line);color:var(--ink);
    padding:6px 13px;border-radius:999px;cursor:pointer;font-size:.85rem;transition:.15s;}
  .pill:hover{border-color:var(--accent);color:var(--accent);}
  textarea{width:100%;min-height:120px;resize:vertical;border:1px solid var(--line);
    border-radius:12px;background:var(--bg);color:var(--ink);padding:14px;font-size:1rem;
    font-family:inherit;line-height:1.6;}
  textarea:focus{outline:2px solid var(--accent);outline-offset:1px;border-color:transparent;}
  .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:18px;}
  .stat{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px;}
  .stat .n{font-size:1.5rem;font-weight:650;letter-spacing:-.02em;font-variant-numeric:tabular-nums;}
  .stat .l{color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.04em;margin-top:2px;}
  .stat.hl{background:var(--accent-soft);border-color:transparent;}
  .stat.hl .n{color:var(--accent);}
  .toolbar{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px;flex-wrap:wrap;}
  .toolbar h2{font-size:.95rem;margin:0;}
  .toggle{display:inline-flex;align-items:center;gap:7px;color:var(--muted);font-size:.85rem;cursor:pointer;user-select:none;}
  .tokens{font-size:1.02rem;line-height:2.2;word-break:break-word;}
  .word{display:inline;}
  .gap{display:inline-block;width:.4ch;}
  .tok{border-radius:6px;padding:2px 3px;margin:0 0;white-space:pre-wrap;
    box-shadow:inset 0 0 0 1px rgba(128,128,128,.18);color:var(--chip-fg);}
  .tokens.spaced .tok{margin:1px;}
  .detected{color:var(--muted);font-size:.88rem;margin-top:14px;}
  .detected b{color:var(--ink);}
  .budget{margin-top:8px;}
  .brow{display:flex;align-items:center;gap:10px;margin:9px 0;font-size:.85rem;}
  .brow .name{width:78px;color:var(--muted);}
  .bar{flex:1;height:9px;background:var(--bg);border:1px solid var(--line);border-radius:999px;overflow:hidden;}
  .bar i{display:block;height:100%;background:var(--accent);border-radius:999px;}
  .brow .val{width:112px;text-align:right;color:var(--muted);font-variant-numeric:tabular-nums;}
  .foot{color:var(--muted);font-size:.8rem;margin-top:28px;text-align:center;}
  .foot code{background:var(--panel);border:1px solid var(--line);padding:1px 6px;border-radius:6px;}
  @media (max-width:560px){ .stats{grid-template-columns:repeat(2,1fr);} h1{font-size:1.3rem;} }
</style>
</head>
<body>
<div class="wrap">
  <div id="loaderr" role="alert" style="display:none;margin-bottom:14px;padding:12px 14px;border-radius:10px;background:#5a1620;color:#ffd7dc;border:1px solid #7a2230;font-size:.9rem;line-height:1.45;"></div>
  <header>
    <div>
      <h1>India Wikipedia BPE Tokenizer</h1>
    </div>
    <div style="display:flex;gap:8px;flex-shrink:0;">
      <button class="theme-btn" id="dlBtn" title="Download tokenizer.json">&#8681; Tokenizer</button>
    </div>
  </header>

  <div class="samples">
    <span>Try a sample:</span>
    <button class="pill" data-lang="English">English</button>
    <button class="pill" data-lang="Hindi">Hindi</button>
    <button class="pill" data-lang="Telugu">Telugu</button>
    <button class="pill" data-lang="Kannada">Kannada</button>
  </div>

  <div class="card" style="margin-top:14px;">
    <textarea id="input" placeholder="Type or paste text in English, Hindi, Telugu, or Kannada&hellip;"></textarea>
  </div>

  <div class="stats">
    <div class="stat"><div class="n" id="s-chars">0</div><div class="l">Characters</div></div>
    <div class="stat"><div class="n" id="s-words">0</div><div class="l">Words</div></div>
    <div class="stat"><div class="n" id="s-tokens">0</div><div class="l">Tokens</div></div>
    <div class="stat hl"><div class="n" id="s-fert">&mdash;</div><div class="l">Fertility (tok/word)</div></div>
  </div>

  <div class="card">
    <div class="toolbar">
      <h2>Tokens</h2>
      <label class="toggle"><input type="checkbox" id="spaceToggle"> space out tokens</label>
    </div>
    <div class="tokens" id="out"></div>
    <div class="detected" id="detected"></div>
  </div>

  <div class="card">
    <div class="toolbar"><h2>How the 10k vocabulary is split</h2></div>
    <div class="budget" id="budget"></div>
  </div>

  <div class="card">
    <div class="toolbar"><h2>Self Score</h2></div>
    <div class="n" id="selfscore" style="font-size:2.4rem;font-weight:650;letter-spacing:-.02em;color:var(--accent);">&mdash;</div>
    <p class="sub" id="selfscore-detail" style="margin-top:6px;"></p>
  </div>

</div>

<script>
function showFatal(msg){
  var b=document.getElementById("loaderr");
  if(b){ b.style.display="block";
    b.textContent="⚠ Widget error — "+msg+". Try a hard refresh (Cmd/Ctrl+Shift+R) or re-upload the latest index.html."; }
}
window.addEventListener("error",function(ev){ showFatal((ev.error&&ev.error.message)||ev.message||"script failed to load"); });

const DATA = /*__DATA__*/;
const END = "</w>";

// ---- tokenizer (JS port of bpe.py, byte-for-byte compatible) ----
const RANKS = new Map();
DATA.merges.forEach((p,i)=>RANKS.set(p[0]+"\u0001"+p[1], i));

function pretokenize(text){ return text.split(/\s+/).filter(s=>s.length>0); }

function encodeWord(word){
  let sym = Array.from(word);
  sym.push(END);
  while(sym.length>=2){
    let bestRank=Infinity, bestI=-1;
    for(let i=0;i<sym.length-1;i++){
      const r = RANKS.get(sym[i]+"\u0001"+sym[i+1]);
      if(r!==undefined && r<bestRank){ bestRank=r; bestI=i; }
    }
    if(bestI<0) break;
    sym.splice(bestI,2, sym[bestI]+sym[bestI+1]);
  }
  return sym;
}

function tokenizeDetailed(text){
  return pretokenize(text).map(w=>({word:w, tokens:encodeWord(w)}));
}

// ---- script detection (for the "detected language" hint) ----
function dominantScript(text){
  const c={English:0,Hindi:0,Telugu:0,Kannada:0};
  for(const ch of text){
    const o=ch.codePointAt(0);
    if(o>=0x0900&&o<=0x097F)c.Hindi++;
    else if(o>=0x0C00&&o<=0x0C7F)c.Telugu++;
    else if(o>=0x0C80&&o<=0x0CFF)c.Kannada++;
    else if((ch>='a'&&ch<='z')||(ch>='A'&&ch<='Z'))c.English++;
  }
  let best=null,n=0;
  for(const k in c){ if(c[k]>n){n=c[k];best=k;} }
  return n>0?best:null;
}

// ---- token colouring (theme-agnostic translucent tints) ----
const HUES=[210,145,28,340,265,175,45,305];
function tint(i){ return `hsla(${HUES[i%HUES.length]},70%,55%,.20)`; }

// ---- rendering ----
const $=id=>document.getElementById(id);
const nf=n=>n.toLocaleString();

function render(){
  const text=$("input").value;
  const detailed=tokenizeDetailed(text);
  const nWords=detailed.length;
  const nTokens=detailed.reduce((a,d)=>a+d.tokens.length,0);
  $("s-chars").textContent=nf(Array.from(text).length);
  $("s-words").textContent=nf(nWords);
  $("s-tokens").textContent=nf(nTokens);
  $("s-fert").textContent=nWords? (nTokens/nWords).toFixed(3) : "—";

  const out=$("out"); out.innerHTML="";
  let ci=0;
  detailed.forEach((d,wi)=>{
    const g=document.createElement("span"); g.className="word";
    d.tokens.forEach(t=>{
      const span=document.createElement("span");
      span.className="tok";
      span.style.background=tint(ci++);
      span.textContent=t.split(END).join("");
      g.appendChild(span);
    });
    out.appendChild(g);
    if(wi<detailed.length-1){
      const gap=document.createElement("span"); gap.className="gap"; gap.textContent=" ";
      out.appendChild(gap);
    }
  });

  const det=$("detected"); const scr=dominantScript(text);
  if(scr && DATA.stats[scr]){
    const s=DATA.stats[scr];
    det.innerHTML=`Detected script: <b>${scr}</b> &middot; owns <b>${nf(s.tokens)}</b> of the ${nf(DATA.vocabSize)} vocab tokens (${s.pct}%).`;
  } else { det.textContent=""; }
}

// ---- vocab budget bars ----
function budget(){
  const order=["English","Hindi","Telugu","Kannada","shared"];
  const max=Math.max(...order.map(k=>DATA.stats[k]?DATA.stats[k].tokens:0));
  const el=$("budget"); el.innerHTML="";
  order.forEach(k=>{
    const s=DATA.stats[k]; if(!s) return;
    const row=document.createElement("div"); row.className="brow";
    row.innerHTML=`<div class="name">${k}</div>
      <div class="bar"><i style="width:${100*s.tokens/max}%"></i></div>
      <div class="val">${nf(s.tokens)} &middot; ${s.pct}%</div>`;
    el.appendChild(row);
  });
}

// ---- self score: 1000 / (worst fertility - best fertility) ----
function selfScore(){
  const vals=Object.values(DATA.fertilities);
  const worst=Math.max(...vals), best=Math.min(...vals);
  const score=1000/(worst-best);
  $("selfscore").textContent=score.toFixed(1);
  $("selfscore-detail").innerHTML=
    `1000 &divide; (largest fertility ${worst.toFixed(3)} &minus; best fertility ${best.toFixed(3)}) `+
    `= 1000 &divide; ${(worst-best).toFixed(3)}`;
}

// ---- wiring ----
$("input").addEventListener("input",render);
$("spaceToggle").addEventListener("change",e=>{
  $("out").classList.toggle("spaced", e.target.checked);
});
document.querySelectorAll(".pill").forEach(b=>b.addEventListener("click",()=>{
  $("input").value=DATA.samples[b.dataset.lang]||""; render();
}));
$("dlBtn").addEventListener("click",()=>{
  const json=JSON.stringify({merges:DATA.merges, vocab:DATA.vocab});
  const blob=new Blob([json],{type:"application/json"});
  const url=URL.createObjectURL(blob);
  const a=document.createElement("a");
  a.href=url; a.download="tokenizer.json";
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

$("input").value=DATA.samples["English"]||"";
try { budget(); selfScore(); render(); window.__widgetReady=true; }
catch(e){ showFatal("init failed: "+((e&&e.message)||e)); }
</script>
<script>
// Runs even if the main script above fails to parse — turns a silent dead
// widget into a visible message (this is the failure mode we hit before).
setTimeout(function(){
  if(!window.__widgetReady){
    var b=document.getElementById("loaderr");
    if(b){ b.style.display="block";
      b.textContent="⚠ The tokenizer did not initialize — you may be viewing a stale cached copy. Hard-refresh with Cmd/Ctrl+Shift+R (or re-upload the latest index.html)."; }
  }
}, 250);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    build()
