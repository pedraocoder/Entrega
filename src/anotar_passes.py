# ── Ferramenta de anotação manual de passes (gabarito para validar_passes) ──
import base64, json, subprocess, pandas as pd
from pathlib import Path
from google.colab.output import eval_js
from IPython.display import display, Javascript

display(Javascript(r"""
async function anotarPasses(b64, nome) {
  const div = document.createElement('div'); div.style.fontFamily = 'sans-serif';
  const vid = document.createElement('video'); vid.src = 'data:video/mp4;base64,' + b64;
  vid.controls = true; vid.style.cssText = 'width:854px;max-width:100%;display:block';
  const ajuda = document.createElement('div'); ajuda.style.margin = '6px 0';
  ajuda.innerHTML = `<b>${nome}</b> — no instante em que a bola SAI do pé, aperte: ` +
    '<b>1</b> Náutico certo · <b>2</b> Náutico errado · <b>3</b> Sport certo · <b>4</b> Sport errado · ' +
    '<b>Z</b> desfazer · <b>espaço</b> play/pausa · <b>←/→</b> volta/avança 1 s';
  const vel = document.createElement('select');
  [0.25, 0.5, 0.75, 1].forEach(v => { const o = document.createElement('option'); o.value = v; o.text = v + 'x'; if (v === 0.5) o.selected = true; vel.add(o); });
  vel.onchange = () => { vid.playbackRate = parseFloat(vel.value); };
  const bOk = document.createElement('button'); bOk.textContent = 'Concluir e salvar';
  const barra = document.createElement('div'); barra.append('Velocidade: ', vel, '  ', bOk);
  const lista = document.createElement('div'); lista.style.cssText = 'font-family:monospace;white-space:pre;margin-top:6px';
  div.append(ajuda, barra, vid, lista); document.body.appendChild(div);
  vid.playbackRate = 0.5;
  const ann = [];
  const rot = {'0,1': 'Náutico ✓', '0,0': 'Náutico ✗', '1,1': 'Sport ✓', '1,0': 'Sport ✗'};
  const mostrar = () => { lista.textContent = ann.length + ' passe(s)\n' +
    ann.slice().sort((a, b) => a[0] - b[0]).map(a => `${a[0].toFixed(1)} s  ${rot[a[1] + ',' + a[2]]}`).join('\n'); };
  const tecla = e => {
    const k = e.key.toLowerCase();
    const mapa = {'1': [0, 1], '2': [0, 0], '3': [1, 1], '4': [1, 0]};
    if (mapa[k]) { ann.push([vid.currentTime, ...mapa[k]]); mostrar(); e.preventDefault(); }
    else if (k === 'z') { ann.pop(); mostrar(); e.preventDefault(); }
    else if (k === ' ') { vid.paused ? vid.play() : vid.pause(); e.preventDefault(); }
    else if (k === 'arrowleft') { vid.currentTime = Math.max(0, vid.currentTime - 1); e.preventDefault(); }
    else if (k === 'arrowright') { vid.currentTime += 1; e.preventDefault(); }
  };
  document.addEventListener('keydown', tecla);
  await new Promise(r => bOk.onclick = r);
  document.removeEventListener('keydown', tecla);
  div.remove();
  return ann;
}
"""))

def anotar(seg):
    pasta = Path(SAIDA) / seg
    meta = json.load(open(pasta / 'meta.json'))
    dur = meta['n_frames'] / meta['fps']
    mini = f'/content/anotar_{seg}.mp4'
    # recorte exato do segmento (mesmo relógio do rastreado.mp4), em 480p para caber no navegador
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-ss', str(meta['inicio_s']), '-i', meta['video'],
                    '-t', f'{dur:.3f}', '-vf', 'scale=854:-2', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
                    '-crf', '26', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', mini], check=True)
    b64 = base64.b64encode(Path(mini).read_bytes()).decode()
    ann = eval_js(f'anotarPasses({json.dumps(b64)}, {json.dumps(seg)})')
    df = pd.DataFrame(ann, columns=['t_s', 'time', 'sucesso']).sort_values('t_s')
    df['t_s'] = df['t_s'].round(2)
    destino = pasta / 'passes_anotados.csv'
    if destino.exists():
        destino.rename(pasta / 'passes_anotados_anterior.csv')
        print('(a anotação anterior foi guardada em passes_anotados_anterior.csv)')
    df.to_csv(destino, index=False)
    print(f'{len(df)} passes salvos em {destino}')
    print(df.groupby(['time', 'sucesso']).size().rename({0: 'Náutico', 1: 'Sport'}, level=0).to_string())
