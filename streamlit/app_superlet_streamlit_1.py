#!/usr/bin/env python3
"""
Aplicație Streamlit — Vizualizare interactivă Superlet Transform pe date EEG reale.

Moduri de funcționare:
    A) Upload simplu  — un singur .bin EEG → fereastră glisantă pe semnal continuu
    B) Upload complet — EEG .bin + Event-Codes .bin + Event-Timestamps .bin (+ CSV opțional)
                        → extragere automată de trial-uri cu label Seen/Uncertain/Nothing
    C) Demo           — semnal sintetic cu ERP-uri simulate

Utilizare:
    pip install streamlit numpy pandas
    streamlit run app_superlet_streamlit.py
"""

import streamlit as st
import numpy as np
import json
import os
import io

# ══════════════════════════════════════════════════════════════════════
# CONSTANTE
# ══════════════════════════════════════════════════════════════════════

FS = 1024
DTYPE_F32 = np.float32
DTYPE_I32 = np.int32
CLASS_NAMES = {1: "Seen", 2: "Uncertain", 3: "Nothing"}
CLASS_COLORS = {1: "#5b8df9", 2: "#f9c55b", 3: "#f9605b"}


# ══════════════════════════════════════════════════════════════════════
# FUNCȚII DE ÎNCĂRCARE DATE
# ══════════════════════════════════════════════════════════════════════

def load_bin_as_signal(uploaded_file, dtype=np.float32):
    """Citește un fișier .bin uploadat și returnează un numpy array."""
    raw_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    return np.frombuffer(raw_bytes, dtype=dtype)


def extract_trials_from_events(signal, ev_codes, ev_timestamps, t_pre=0.2, t_post=0.8, quality_mask=None):
    """
    Extrage epoci stimulus-locked din semnalul continuu.
    Event codes: 129 = stimulus ON, 1 = Seen, 2 = Uncertain, 3 = Nothing
    """
    n_samp = len(signal)
    stim_idx = np.where(ev_codes == 129)[0]
    pre_s = int(round(t_pre * FS))
    post_s = int(round(t_post * FS))
    ep_len = pre_s + post_s + 1
    times = (np.arange(ep_len) - pre_s) / FS

    epochs_list, y_list, rt_list = [], [], []
    csv_ctr = 0

    for k, si in enumerate(stim_idx):
        nxt = stim_idx[k + 1] if k < len(stim_idx) - 1 else len(ev_codes)
        resp_j = None
        for j in range(si + 1, nxt):
            if ev_codes[j] in {1, 2, 3}:
                resp_j = j
                break

        q_ok = True
        if quality_mask is not None and csv_ctr < len(quality_mask):
            q_ok = quality_mask[csv_ctr] == 1
        csv_ctr += 1

        if resp_j is None or not q_ok:
            continue

        s0 = ev_timestamps[si] - pre_s
        s1 = ev_timestamps[si] + post_s

        if s0 < 0 or s1 >= n_samp:
            continue

        epoch = signal[s0 : s1 + 1].copy()
        b_idx = np.where((times >= -t_pre) & (times <= 0.0))[0]
        epoch -= epoch[b_idx].mean()

        epochs_list.append(epoch)
        y_list.append(int(ev_codes[resp_j]))
        rt_list.append(int(ev_timestamps[resp_j] - ev_timestamps[si]))

    if not epochs_list:
        return None, None, times, None

    return (
        np.stack(epochs_list).astype(np.float32),
        np.array(y_list, dtype=int),
        times,
        np.array(rt_list, dtype=int),
    )


def extract_windows_from_continuous(signal, window_sec=1.0, step_sec=0.5, max_windows=30):
    """Extrage ferestre din semnalul continuu (fără events)."""
    win_samp = int(window_sec * FS)
    step_samp = int(step_sec * FS)
    times = np.arange(win_samp) / FS

    windows = []
    starts_list = []
    for s0 in range(0, len(signal) - win_samp, step_samp):
        if len(windows) >= max_windows:
            break
        chunk = signal[s0 : s0 + win_samp].copy()
        chunk -= chunk[: win_samp // 5].mean()
        windows.append(chunk)
        starts_list.append(s0)

    if not windows:
        return None, None, times, None

    labels = np.zeros(len(windows), dtype=int)
    rt = np.array(starts_list, dtype=int)
    return np.stack(windows).astype(np.float32), labels, times, rt


def generate_demo_data(n_trials=10, n_samples=1025):
    """Date demo sintetice cu ERP-uri."""
    np.random.seed(42)
    times = (np.arange(n_samples) - 205) / FS
    epochs = np.random.randn(n_trials, n_samples).astype(np.float32) * 5

    for t_idx in range(n_trials):
        alpha = 3.0 * np.sin(2 * np.pi * 10 * times + np.random.rand() * 2 * np.pi)
        beta = 1.5 * np.sin(2 * np.pi * 25 * times + np.random.rand() * 2 * np.pi)
        erp = np.zeros(n_samples)
        t0 = int(0.2 * FS)
        if t0 + 130 < n_samples:
            erp[t0 + 80 : t0 + 130] = -8 * np.exp(-0.5 * ((np.arange(50) - 25) / 10) ** 2)
        if t0 + 400 < n_samples:
            erp[t0 + 250 : t0 + 400] = 5 * np.exp(-0.5 * ((np.arange(150) - 75) / 30) ** 2)
        epochs[t_idx] += alpha + beta + erp * (0.5 + 0.5 * np.random.rand())

    y = np.array([1, 1, 1, 3, 3, 3, 1, 3, 2, 2], dtype=int)[:n_trials]
    rt = np.random.randint(500, 5000, size=n_trials)
    return epochs, y, times, rt


# ══════════════════════════════════════════════════════════════════════
# SERIALIZARE JSON
# ══════════════════════════════════════════════════════════════════════

def prepare_json(epochs, y, times, rt, source_name, is_continuous=False, max_trials=20):
    trial_data = []

    if is_continuous:
        for i in range(min(len(epochs), max_trials)):
            t_start_sec = rt[i] / FS if rt is not None else i
            trial_data.append({
                "trial_idx": int(i),
                "class": 0,
                "class_name": "Window",
                "rt_ms": int(round(t_start_sec * 1000)),
                "signal": [round(float(v), 2) for v in epochs[i]],
            })
    else:
        for cls in [1, 3, 2]:
            idx = np.where(y == cls)[0][:max_trials]
            for i in idx:
                trial_data.append({
                    "trial_idx": int(i),
                    "class": int(cls),
                    "class_name": CLASS_NAMES.get(cls, "?"),
                    "rt_ms": int(rt[i] / FS * 1000) if rt[i] > 100 else int(rt[i]),
                    "signal": [round(float(v), 2) for v in epochs[i]],
                })

    return json.dumps({
        "source": source_name,
        "fs": FS,
        "n_samples": int(epochs.shape[1]),
        "times_ms": [round(t * 1000, 1) for t in times.tolist()],
        "is_continuous": is_continuous,
        "trials": trial_data,
    }, separators=(",", ":"))


# ══════════════════════════════════════════════════════════════════════
# COMPONENTA HTML INTERACTIVĂ (tot engine-ul de vizualizare)
# ══════════════════════════════════════════════════════════════════════

def build_html(data_json):
    return f'''
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

  .slt *{{box-sizing:border-box;margin:0;padding:0;}}
  .slt{{
    font-family:'DM Sans',-apple-system,sans-serif;
    background:transparent;color:#c8cad0;padding:8px 0;
  }}
  .slt .row{{display:flex;align-items:center;gap:10px;margin:0 0 7px;}}
  .slt .row label{{
    font-size:11px;color:#6b7080;min-width:60px;
    text-transform:uppercase;letter-spacing:0.06em;font-weight:500;
  }}
  .slt .row input[type="range"]{{
    flex:1;height:3px;
    -webkit-appearance:none;appearance:none;
    background:#1e2230;border-radius:2px;outline:none;
  }}
  .slt .row input[type="range"]::-webkit-slider-thumb{{
    -webkit-appearance:none;width:15px;height:15px;
    border-radius:50%;background:#12151c;
    border:2px solid #5b8df9;cursor:pointer;
  }}
  .slt .row span{{
    font-family:'JetBrains Mono',monospace;
    font-size:11px;font-weight:400;min-width:56px;
    text-align:right;color:#8a8f9e;
  }}
  .slt select{{
    font-size:12px;padding:4px 8px;
    border:1px solid #252a36;border-radius:5px;
    background:#12151c;color:#c8cad0;cursor:pointer;
    font-family:inherit;
  }}
  .slt .btn{{
    display:inline-block;padding:4px 10px;margin:2px;
    font-size:10px;border-radius:5px;cursor:pointer;
    border:1px solid #252a36;background:#12151c;color:#8a8f9e;
    transition:all 0.12s;font-family:inherit;
  }}
  .slt .btn:hover{{background:#1a1f2c;border-color:#354060;}}
  .slt .btn.active{{background:#5b8df9;color:#fff;border-color:#5b8df9;}}
  .slt .btn.c1{{border-left:3px solid #5b8df9;}}
  .slt .btn.c3{{border-left:3px solid #f9605b;}}
  .slt .btn.c2{{border-left:3px solid #f9c55b;}}
  .slt .btn.c0{{border-left:3px solid #8a8f9e;}}
  .slt .lbl{{
    font-size:10px;font-weight:500;color:#4a4f5c;
    margin:10px 0 4px;text-transform:uppercase;letter-spacing:0.08em;
  }}
  .slt canvas{{
    width:100%;border-radius:6px;
    border:1px solid #181c26;margin-bottom:2px;
    background:#0c0f16;
  }}
  .slt .leg{{
    display:flex;gap:12px;margin:0 0 8px;
    font-size:9px;color:#4a4f5c;flex-wrap:wrap;
  }}
  .slt .leg i{{
    display:inline-block;width:12px;height:2px;
    border-radius:1px;vertical-align:middle;margin-right:3px;
  }}
  .slt .nfo{{
    font-family:'JetBrains Mono',monospace;
    font-size:10px;color:#4a5068;margin:2px 0 5px;
  }}
  .slt .stats{{display:flex;gap:6px;margin:6px 0 10px;flex-wrap:wrap;}}
  .slt .stat{{
    background:#111420;border:1px solid #1e2230;border-radius:5px;
    padding:5px 10px;font-size:10px;color:#6b7080;
    font-family:'JetBrains Mono',monospace;
  }}
  .slt .stat b{{color:#c8cad0;font-weight:500;}}
</style>

<div class="slt">

<div class="stats" id="stats-bar"></div>

<p class="lbl">Selectează segmentul</p>
<div id="btns" style="max-height:80px;overflow-y:auto;margin-bottom:4px;"></div>
<p class="nfo" id="info">&nbsp;</p>

<div class="row">
  <label>Frecvența</label>
  <input type="range" id="freq" min="4" max="100" value="30" step="1" oninput="update()">
  <span id="freq-out">30 Hz</span>
</div>
<div class="row">
  <label>Poziția</label>
  <input type="range" id="pos" min="0" max="1023" value="512" step="1" oninput="update()">
  <span id="pos-out">300 ms</span>
</div>

<p class="lbl">Semnal EEG + Wavelet Morlet</p>
<canvas id="c1" height="150"></canvas>
<div class="leg">
  <span><i style="background:#5b8df9"></i>Semnal EEG</span>
  <span><i style="background:#f97a5b"></i>Wavelet Morlet</span>
  <span>Linia punctată = t₀ (stimulus ON)</span>
</div>

<p class="lbl">Putere la frecvența selectată (|convoluție|²)</p>
<canvas id="c2" height="100"></canvas>
<div class="leg">
  <span><i style="background:#3dd9a0"></i>Putere</span>
  <span><i style="background:#f97a5b"></i>Cursor wavelet</span>
</div>

<p class="lbl">Spectrogramă completă (4–100 Hz)</p>
<canvas id="c3" height="200"></canvas>
<div class="leg">
  <span>Fiecare rând = convoluție la o frecvență · Chenar alb = frecvența selectată</span>
</div>

</div>

<script>
const D={data_json};

const c1=document.getElementById('c1'),c2=document.getElementById('c2'),c3=document.getElementById('c3');
const ctx1=c1.getContext('2d'),ctx2=c2.getContext('2d'),ctx3=c3.getContext('2d');

function resize(c){{const d=window.devicePixelRatio||1;const r=c.getBoundingClientRect();c.width=r.width*d;c.height=r.height*d;c.getContext('2d').scale(d,d);}}
[c1,c2,c3].forEach(resize);

let W=c1.getBoundingClientRect().width;
let H1=c1.getBoundingClientRect().height;
let H2=c2.getBoundingClientRect().height;
let H3=c3.getBoundingClientRect().height;
const N=D.n_samples;

document.getElementById('pos').max=N-1;

const muted='#4a5068';
const grid='rgba(255,255,255,0.03)';
const stimC='rgba(255,255,255,0.1)';

let cur=0;
let sig=new Float64Array(N);
let spec=null;

// Stats
const sb=document.getElementById('stats-bar');
if(D.is_continuous){{
  sb.innerHTML='<div class="stat">Ferestre: <b>'+D.trials.length+'</b></div><div class="stat">Samples: <b>'+N+'</b></div><div class="stat">Fs: <b>'+D.fs+' Hz</b></div>';
}}else{{
  const cc={{}};D.trials.forEach(t=>{{cc[t.class_name]=(cc[t.class_name]||0)+1;}});
  let h='<div class="stat">Trial-uri: <b>'+D.trials.length+'</b></div>';
  for(const[k,v] of Object.entries(cc))h+='<div class="stat">'+k+': <b>'+v+'</b></div>';
  h+='<div class="stat">Fs: <b>'+D.fs+' Hz</b></div><div class="stat">Epocă: <b>'+D.times_ms[0]+'→'+D.times_ms[D.times_ms.length-1]+' ms</b></div>';
  sb.innerHTML=h;
}}

// Buttons
const bd=document.getElementById('btns');
D.trials.forEach((tr,i)=>{{
  const b=document.createElement('button');
  b.className='btn c'+tr.class;
  if(D.is_continuous)b.textContent='W'+i+' (@'+(tr.rt_ms/1000).toFixed(1)+'s)';
  else b.textContent='T'+tr.trial_idx+' ('+tr.class_name+')';
  b.onclick=()=>sel(i);
  bd.appendChild(b);
}});

function sel(i){{
  cur=i;
  document.querySelectorAll('.btn').forEach((b,j)=>b.classList.toggle('active',j===i));
  const tr=D.trials[i];
  if(D.is_continuous)document.getElementById('info').textContent='Fereastră '+i+' · Start: '+(tr.rt_ms/1000).toFixed(2)+' s · '+N+' samples';
  else document.getElementById('info').textContent='Trial '+tr.trial_idx+' · '+tr.class_name+' · RT: '+tr.rt_ms+' ms';
  load();
}}

function load(){{
  const raw=D.trials[cur].signal;if(!raw)return;
  for(let i=0;i<N;i++)sig[i]=raw[i];
  buildSpec();update();
}}

function mkW(fc,nc,fs){{
  const sd=(nc/2)*(1/fc)/2.5;
  const sz=Math.floor(Math.round(sd*fs*6)/2)*2+1;
  const h=Math.floor(sz/2);
  const w=new Float64Array(sz);
  let gs=0;
  for(let i=0;i<sz;i++){{const t=(i-h)*(3/(h||1));gs+=Math.exp(-t*t*0.5);}}
  for(let i=0;i<sz;i++){{
    const tg=(i-h)*(3/(h||1));const g=Math.exp(-tg*tg*0.5)/gs;
    const ts=(i-h)/fs;w[i]=g*Math.cos(2*Math.PI*fc*ts);
  }}
  return w;
}}

function conv(s,w){{
  const o=new Float64Array(s.length);const h=Math.floor(w.length/2);
  for(let i=0;i<s.length;i++){{let v=0;for(let j=0;j<w.length;j++){{const si=i-h+j;if(si>=0&&si<s.length)v+=s[si]*w[j];}}o[i]=v*v;}}
  return o;
}}

function buildSpec(){{
  spec=[];for(let f=4;f<=100;f++)spec.push(conv(sig,mkW(f,3,1024)));
}}

function update(){{
  const freq=+document.getElementById('freq').value;
  const pos=+document.getElementById('pos').value;

  let posLabel;
  if(D.is_continuous)posLabel=Math.round(pos/D.fs*1000)+' ms';
  else{{const tMs=D.times_ms;const idx=Math.min(pos,tMs.length-1);posLabel=tMs[idx]+' ms';}}
  document.getElementById('freq-out').textContent=freq+' Hz';
  document.getElementById('pos-out').textContent=posLabel;

  const wav=mkW(freq,3,1024);
  const half=Math.floor(wav.length/2);
  const pwr=conv(sig,wav);
  const px=i=>i/N*W;

  let t0px;
  if(D.is_continuous)t0px=-10;
  else{{const t0i=D.times_ms.findIndex(t=>t>=0);t0px=px(t0i>=0?t0i:Math.round(0.2*N));}}

  // ═══ G1: Signal + Wavelet ═══
  let sMin=Infinity,sMax=-Infinity;
  for(let i=0;i<N;i++){{if(sig[i]<sMin)sMin=sig[i];if(sig[i]>sMax)sMax=sig[i];}}
  if(sMax===sMin)sMax=sMin+1;
  const sY=v=>14+(1-(v-sMin)/(sMax-sMin))*(H1-28);

  ctx1.clearRect(0,0,W,H1);
  ctx1.fillStyle=grid;for(let g=0;g<=4;g++)ctx1.fillRect(0,14+g*(H1-28)/4,W,0.5);

  if(t0px>0){{ctx1.strokeStyle=stimC;ctx1.lineWidth=1;ctx1.setLineDash([4,4]);ctx1.beginPath();ctx1.moveTo(t0px,0);ctx1.lineTo(t0px,H1);ctx1.stroke();ctx1.setLineDash([]);}}

  ctx1.strokeStyle='#5b8df9';ctx1.lineWidth=1.1;ctx1.beginPath();
  for(let i=0;i<N;i++){{const x=px(i),y=sY(sig[i]);i?ctx1.lineTo(x,y):ctx1.moveTo(x,y);}}
  ctx1.stroke();

  const ws=pos-half,we=pos+half;
  ctx1.fillStyle='rgba(249,122,91,0.07)';
  ctx1.fillRect(px(Math.max(0,ws)),0,px(Math.min(N,we)-Math.max(0,ws)),H1);

  const wM=Math.max(...wav.map(Math.abs))*0.7;
  ctx1.strokeStyle='#f97a5b';ctx1.lineWidth=1.4;ctx1.beginPath();let st=false;
  for(let j=0;j<wav.length;j++){{const si=ws+j;if(si<0||si>=N)continue;const x=px(si),wN=wav[j]/wM,y=H1/2-wN*(H1/2-14);st?ctx1.lineTo(x,y):ctx1.moveTo(x,y);st=true;}}
  ctx1.stroke();

  ctx1.fillStyle=muted;ctx1.font='10px JetBrains Mono,monospace';
  const tMs=D.times_ms;
  if(!D.is_continuous&&tMs.length>0){{ctx1.fillText(tMs[0]+'ms',2,H1-3);if(t0px>0)ctx1.fillText('0',t0px+3,H1-3);ctx1.fillText(tMs[tMs.length-1]+'ms',W-52,H1-3);}}
  else{{ctx1.fillText('0 ms',2,H1-3);ctx1.fillText(Math.round(N/1024*1000)+' ms',W-52,H1-3);}}
  ctx1.fillText('wavelet: '+wav.length+' smp · '+Math.round(wav.length/1024*1000)+' ms',Math.min(px(pos)-50,W-200),12);

  // ═══ G2: Power ═══
  let pM=0;for(let i=0;i<N;i++)if(pwr[i]>pM)pM=pwr[i];pM=pM*1.1||1;
  const pY=v=>7+(1-v/pM)*(H2-18);

  ctx2.clearRect(0,0,W,H2);ctx2.fillStyle=grid;ctx2.fillRect(0,pY(0),W,0.5);
  if(t0px>0){{ctx2.strokeStyle=stimC;ctx2.lineWidth=1;ctx2.setLineDash([4,4]);ctx2.beginPath();ctx2.moveTo(t0px,0);ctx2.lineTo(t0px,H2);ctx2.stroke();ctx2.setLineDash([]);}}

  ctx2.fillStyle='rgba(61,217,160,0.08)';ctx2.beginPath();ctx2.moveTo(0,pY(0));
  for(let i=0;i<N;i++)ctx2.lineTo(px(i),pY(pwr[i]));ctx2.lineTo(px(N-1),pY(0));ctx2.fill();

  ctx2.strokeStyle='#3dd9a0';ctx2.lineWidth=1.1;ctx2.beginPath();
  for(let i=0;i<N;i++){{const x=px(i),y=pY(pwr[i]);i?ctx2.lineTo(x,y):ctx2.moveTo(x,y);}}
  ctx2.stroke();

  ctx2.strokeStyle='#f97a5b';ctx2.lineWidth=1;ctx2.setLineDash([4,3]);ctx2.beginPath();ctx2.moveTo(px(pos),0);ctx2.lineTo(px(pos),H2);ctx2.stroke();ctx2.setLineDash([]);
  ctx2.fillStyle='#f97a5b';ctx2.beginPath();ctx2.arc(px(pos),pY(pwr[Math.min(pos,N-1)]),3.5,0,Math.PI*2);ctx2.fill();
  ctx2.fillStyle=muted;ctx2.font='10px JetBrains Mono,monospace';ctx2.fillText('power @'+freq+' Hz',4,H2-3);

  // ═══ G3: Spectrogram ═══
  if(!spec)return;
  ctx3.clearRect(0,0,W,H3);
  const nF=spec.length,cH=H3/nF;
  let gM=0;for(const row of spec)for(const v of row)if(v>gM)gM=v;

  for(let fi=0;fi<nF;fi++){{
    const row=spec[fi],y=H3-(fi+1)*cH;
    for(let ti=0;ti<N;ti+=2){{
      const v=Math.sqrt(row[ti]/(gM||1));
      const r=Math.round(v<0.5?v*2*180:180+(v-0.5)*2*75);
      const g=Math.round(v<0.5?v*2*30:30+(v-0.5)*2*200);
      const b=Math.round(v<0.3?40+v/0.3*80:v<0.7?120-((v-0.3)/0.4)*80:40+(v-0.7)/0.3*40);
      ctx3.fillStyle=`rgb(${{r}},${{g}},${{b}})`;
      ctx3.fillRect(px(ti),y,Math.ceil(W/N*2)+1,Math.ceil(cH)+1);
    }}
  }}

  if(t0px>0){{ctx3.strokeStyle='rgba(255,255,255,0.2)';ctx3.lineWidth=1;ctx3.setLineDash([4,4]);ctx3.beginPath();ctx3.moveTo(t0px,0);ctx3.lineTo(t0px,H3);ctx3.stroke();ctx3.setLineDash([]);}}

  const fIdx=freq-4,rY=H3-(fIdx+1)*cH;
  ctx3.strokeStyle='rgba(255,255,255,0.5)';ctx3.lineWidth=1;ctx3.strokeRect(0,rY,W,cH);
  ctx3.strokeStyle='#f97a5b';ctx3.lineWidth=1;ctx3.setLineDash([4,3]);ctx3.beginPath();ctx3.moveTo(px(pos),0);ctx3.lineTo(px(pos),H3);ctx3.stroke();ctx3.setLineDash([]);

  ctx3.fillStyle='rgba(255,255,255,0.6)';ctx3.font='10px JetBrains Mono,monospace';
  ctx3.fillText('4 Hz',4,H3-3);ctx3.fillText('100 Hz',4,11);
  ctx3.fillText(freq+' Hz \\u25B8',4,rY+cH/2+3);
  if(t0px>0)ctx3.fillText('stim',t0px+3,H3-3);
}}

window.addEventListener('resize',()=>{{[c1,c2,c3].forEach(resize);W=c1.getBoundingClientRect().width;H1=c1.getBoundingClientRect().height;H2=c2.getBoundingClientRect().height;H3=c3.getBoundingClientRect().height;update();}});

sel(0);
</script>
'''


# ══════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Superlet Transform — EEG Explorer",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## 🧠 Superlet Explorer")
        st.caption("Vizualizare interactivă Superlet Transform pe date EEG")
        st.divider()

        mode = st.radio(
            "Mod de funcționare",
            ["📁 Upload fișiere .bin", "🧪 Demo (semnal sintetic)"],
            index=0,
        )

        epochs = y = times = rt = None
        source_name = ""
        is_continuous = False

        # Variabile implicite pt slidere
        t_pre = 0.2
        t_post = 0.8
        max_trials = 10
        win_sec = 1.0
        max_windows = 20

        if mode == "📁 Upload fișiere .bin":
            st.markdown("#### 1️⃣ Fișierul EEG (.bin)")
            st.caption("Un singur canal · float32 · 1024 Hz")
            eeg_file = st.file_uploader(
                "Încarcă fișierul EEG .bin",
                type=["bin"],
                key="eeg_bin",
                help="Ex: Dots_30_002-A6.bin",
            )

            st.divider()
            st.markdown("#### 2️⃣ Event files (opțional)")
            st.caption(
                "Adaugă-le pentru extragerea automată de trial-uri.\n"
                "Fără ele → mod fereastra glisantă pe semnal continuu."
            )
            ev_codes_file = st.file_uploader(
                "Event-Codes .bin (int32)",
                type=["bin"],
                key="ev_codes",
            )
            ev_ts_file = st.file_uploader(
                "Event-Timestamps .bin (int32)",
                type=["bin"],
                key="ev_ts",
            )

            st.divider()
            st.markdown("#### 3️⃣ Trial CSV (opțional)")
            st.caption("Pentru masca GoodTrialsManual")
            csv_file = st.file_uploader("Trialinfo CSV", type=["csv"], key="csv_info")

            st.divider()
            st.markdown("#### ⚙️ Parametri")

            if eeg_file and ev_codes_file and ev_ts_file:
                t_pre = st.slider("Pre-stimulus (s)", 0.1, 0.5, 0.2, 0.05)
                t_post = st.slider("Post-stimulus (s)", 0.5, 2.0, 0.8, 0.1)
                max_trials = st.slider("Max trial-uri per clasă", 3, 30, 10)
            elif eeg_file:
                win_sec = st.slider("Fereastră (sec)", 0.5, 3.0, 1.0, 0.1)
                max_windows = st.slider("Nr. ferestre", 5, 50, 20)

            source_name = eeg_file.name if eeg_file else ""
        else:
            eeg_file = None
            ev_codes_file = None
            ev_ts_file = None
            csv_file = None
            source_name = "Demo (sintetic)"

        st.divider()
        st.markdown(
            '<div style="font-size:10px;color:#555;line-height:1.5;">'
            "<b>Ref:</b> Moca et al. 2021 · Volcov S. · Del Pup et al. 2025<br>"
            "<b>Pipeline:</b> .bin → epoching → baseline → Superlet → vizualizare"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Main ──
    st.markdown("### Superlet Transform Explorer")

    if mode == "📁 Upload fișiere .bin":
        if not eeg_file:
            st.info(
                "👈 **Încarcă un fișier `.bin` din sidebar.**\n\n"
                "Fișierul conține un singur canal EEG, stocat ca `float32` la 1024 Hz.\n\n"
                "**Exemplu:** `Dots_30_002-A6.bin` (canalul A6 al subiectului 002)\n\n"
                "---\n\n"
                "**Opțional** — pentru trial-uri cu label-uri, adaugă și:\n"
                "- `*-Event-Codes.bin` (int32: 129=stimulus, 1=Seen, 2=Uncertain, 3=Nothing)\n"
                "- `*-Event-Timestamps.bin` (int32: indexul sample-ului)\n"
                "- CSV-ul trialinfo (pentru masca de calitate)"
            )
            return

        # Citim EEG
        signal = load_bin_as_signal(eeg_file, DTYPE_F32)

        col1, col2, col3 = st.columns(3)
        col1.metric("Samples", f"{len(signal):,}")
        col2.metric("Durată", f"{len(signal)/FS:.1f} s")
        col3.metric("Fișier", eeg_file.name)

        if ev_codes_file and ev_ts_file:
            # ── MOD TRIAL ──
            ev_codes = load_bin_as_signal(ev_codes_file, DTYPE_I32)
            ev_timestamps = load_bin_as_signal(ev_ts_file, DTYPE_I32)

            quality_mask = None
            if csv_file:
                import pandas as pd
                csv_file.seek(0)
                csv_text = csv_file.read().decode("utf-8")
                csv_file.seek(0)
                lines = csv_text.split("\n")
                skip = 0
                for i, ln in enumerate(lines):
                    if ln.startswith("Trial,"):
                        skip = i
                        break
                df_q = pd.read_csv(io.StringIO(csv_text), skiprows=skip)
                if "GoodTrialsManual" in df_q.columns:
                    quality_mask = df_q["GoodTrialsManual"].values
                    st.success(
                        f"CSV încărcat — {int(quality_mask.sum())}/{len(quality_mask)} Good trials"
                    )

            with st.spinner("Se extrag trial-urile..."):
                epochs, y, times, rt = extract_trials_from_events(
                    signal, ev_codes, ev_timestamps,
                    t_pre=t_pre, t_post=t_post,
                    quality_mask=quality_mask,
                )

            if epochs is None or len(epochs) == 0:
                st.error("Nu s-au putut extrage trial-uri. Verifică fișierele de events.")
                return

            is_continuous = False
            unique, counts = np.unique(y, return_counts=True)
            dist = dict(zip(unique, counts))
            cols = st.columns(5)
            cols[0].metric("Trial-uri", epochs.shape[0])
            cols[1].metric("Seen", dist.get(1, 0))
            cols[2].metric("Uncertain", dist.get(2, 0))
            cols[3].metric("Nothing", dist.get(3, 0))
            cols[4].metric("Samples/trial", epochs.shape[1])

        else:
            # ── MOD CONTINUU ──
            with st.spinner("Se extrag ferestrele..."):
                epochs, y, times, rt = extract_windows_from_continuous(
                    signal, window_sec=win_sec, max_windows=max_windows,
                )

            if epochs is None:
                st.error("Semnalul e prea scurt.")
                return

            is_continuous = True
            st.info(
                f"**Mod continuu** — {epochs.shape[0]} ferestre × {epochs.shape[1]} samples "
                f"({win_sec}s fiecare) · Fără events → navigare pe semnal brut"
            )

    else:
        # ── DEMO ──
        epochs, y, times, rt = generate_demo_data()
        is_continuous = False
        st.caption("Date sintetice — oscilații alpha + beta + ERP simulate")

    # ── Vizualizare ──
    data_json = prepare_json(
        epochs, y, times, rt, source_name,
        is_continuous=is_continuous, max_trials=30,
    )

    html = build_html(data_json)
    st.components.v1.html(html, height=750, scrolling=False)

    with st.expander("ℹ️ Despre vizualizare & structura fișierelor"):
        st.markdown(f"""
**Fișier:** `{source_name}`  
**Mod:** {"Continuu (ferestre glisante)" if is_continuous else "Trial-uri stimulus-locked"}  
**Fs:** {FS} Hz

---

**Graficul 1 — Semnal + Wavelet:** Semnalul EEG (albastru) cu waveletul Morlet suprapus (portocaliu).

**Graficul 2 — Putere:** |convoluție(semnal, wavelet)|² — un rând din spectrogramă.

**Graficul 3 — Spectrograma:** Toate frecvențele 4–100 Hz.  
Wavelet Morlet cu c₁ = 3 cicluri (ordin 1 — vizualizare simplificată).

---

**Structura fișierelor `.bin` (dataset Dots_30):**

| Fișier | Tip | Conținut |
|--------|-----|---------|
| `Dots_30_0XX-A6.bin` | float32 | Canal EEG A6, valori continue (µV) |
| `Dots_30_0XX-Event-Codes.bin` | int32 | 129 = stimulus ON, 1/2/3 = răspuns |
| `Dots_30_0XX-Event-Timestamps.bin` | int32 | Index sample pentru fiecare event |
| `Dots_30_0XX-trialinfo.csv` | CSV | GoodTrialsManual, ResponseID, etc. |
        """)


if __name__ == "__main__":
    main()
