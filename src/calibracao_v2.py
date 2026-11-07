# ── Calibração por cliques (v2: aceita arco do círculo e pontos sobre linhas) ──
import base64, json, numpy as np, cv2, campo, homografia
from pathlib import Path
from scipy.optimize import least_squares
from google.colab.output import eval_js
from IPython.display import display, Javascript, Image

campo.MARCOS.update({"circulo_esq": (campo.COMPRIMENTO/2 - 9.15, campo.LARGURA/2),
                     "circulo_dir": (campo.COMPRIMENTO/2 + 9.15, campo.LARGURA/2)})

C_CENTRO = np.array([52.5, 34.0]); R_CIRC = 9.15
ANG = {'circulo_sup': -np.pi/2, 'circulo_dir': 0.0, 'circulo_inf': np.pi/2, 'circulo_esq': np.pi}
# pontos "sobre uma linha": só uma coordenada é conhecida
LINHAS = {'sobre_lateral_sup': (1, 0.0), 'sobre_lateral_inf': (1, 68.0), 'sobre_linha_meio': (0, 52.5)}

def _aplicar(H, p):
    p = np.asarray(p, float).reshape(-1, 2); q = np.c_[p, np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:3]

def homografia_com_arco(nomeados, arco, sobre, marcos):
    """nomeados: [(nome,u,v)] marcos exatos · arco: [(u,v)] sobre o círculo central ·
    sobre: [(tipo,u,v)] pontos quaisquer sobre uma linha (tipo em LINHAS)."""
    src_n = np.array([[u, v] for _, u, v in nomeados], float)
    dst_n = np.array([marcos[n] for n, _, _ in nomeados], float)
    arco = np.array(arco, float).reshape(-1, 2)
    # 1) chute inicial: ângulo de cada ponto do arco a partir da elipse ajustada na imagem
    circ = [(n, u, v) for n, u, v in nomeados if n in ANG]
    pts_circ = np.vstack([arco, np.array([[u, v] for _, u, v in circ]).reshape(-1, 2)])
    (cx, cy), (a, b), rot = cv2.fitEllipse(pts_circ.astype(np.float32))
    t = np.deg2rad(rot); R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    def phi(P):
        q = (np.asarray(P, float).reshape(-1, 2) - [cx, cy]) @ R.T
        return np.arctan2(q[:, 1] / (b / 2), q[:, 0] / (a / 2))
    phc = phi([[u, v] for _, u, v in circ]); thc = np.array([ANG[n] for n, _, _ in circ])
    melhor = None
    for s in (1, -1):
        off = np.angle(np.mean(np.exp(1j * (thc - s * phc))))
        err = np.abs(np.angle(np.exp(1j * (s * phc + off - thc)))).sum()
        if melhor is None or err < melhor[0]: melhor = (err, s, off)
    _, s, off = melhor
    th = s * phi(arco) + off
    dst_a = C_CENTRO + R_CIRC * np.c_[np.cos(th), np.sin(th)]
    H, _ = cv2.findHomography(np.vstack([src_n, arco]), np.vstack([dst_n, dst_a]), 0)
    src_s = np.array([[u, v] for _, u, v in sobre], float).reshape(-1, 2)
    eixo = np.array([LINHAS[k][0] for k, _, _ in sobre], int); val = np.array([LINHAS[k][1] for k, _, _ in sobre])
    # 2) refinamento: marcos no lugar, arco a 9,15 m do centro, pontos "sobre" na linha certa
    def resid(h):
        Hh = np.append(h, 1).reshape(3, 3)
        r = [(_aplicar(Hh, src_n) - dst_n).ravel(),
             np.linalg.norm(_aplicar(Hh, arco) - C_CENTRO, axis=1) - R_CIRC]
        if len(src_s): r.append(_aplicar(Hh, src_s)[np.arange(len(src_s)), eixo] - val)
        return np.concatenate(r)
    sol = least_squares(resid, (H / H[2, 2]).ravel()[:8], method='lm')
    H = np.append(sol.x, 1).reshape(3, 3)
    erros = {'marcos': np.linalg.norm(_aplicar(H, src_n) - dst_n, axis=1),
             'arco': np.abs(np.linalg.norm(_aplicar(H, arco) - C_CENTRO, axis=1) - R_CIRC)}
    if len(src_s): erros['sobre_linhas'] = np.abs(_aplicar(H, src_s)[np.arange(len(src_s)), eixo] - val)
    return H, erros

MULTI = ['arco_circulo'] + list(LINHAS)      # tipos que aceitam vários cliques

display(Javascript(r"""
async function clicarMarcos(b64, nomes, multi) {
  const div = document.createElement('div');
  const sel = document.createElement('select');
  multi.concat(nomes).forEach(n => { const o = document.createElement('option');
    o.text = n; o.value = n; if (multi.includes(n)) o.text = n + '  (vários cliques)'; sel.add(o); });
  const bDes = document.createElement('button'); bDes.textContent = 'Desfazer';
  const bOk = document.createElement('button'); bOk.textContent = 'Concluir';
  const barra = document.createElement('div'); barra.style.margin = '6px 0';
  barra.append('Marco: ', sel, ' ', bDes, ' ', bOk);
  const info = document.createElement('div'); info.style.fontFamily = 'monospace';
  const wrap = document.createElement('div'); wrap.style.cssText = 'position:relative;display:inline-block';
  const img = document.createElement('img'); img.src = 'data:image/jpeg;base64,' + b64;
  img.style.cssText = 'max-width:100%;display:block;cursor:crosshair';
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.style.cssText = 'position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none';
  wrap.append(img, svg); div.append(barra, info, wrap); document.body.appendChild(div);
  await img.decode();
  svg.setAttribute('viewBox', `0 0 ${img.naturalWidth} ${img.naturalHeight}`);
  let pts = [];
  const desenhar = () => {
    svg.innerHTML = pts.map(([n, u, v]) => multi.includes(n)
      ? `<circle cx="${u}" cy="${v}" r="5" fill="cyan" stroke="black" stroke-width="1"/>`
      : `<circle cx="${u}" cy="${v}" r="7" fill="none" stroke="yellow" stroke-width="2"/>` +
        `<text x="${u + 9}" y="${v - 9}" fill="yellow" stroke="black" stroke-width="0.6" font-size="18">${n}</text>`).join('');
    const cont = {}; pts.forEach(p => cont[p[0]] = (cont[p[0]] || 0) + 1);
    info.textContent = Object.entries(cont).map(([k, c]) => c > 1 ? `${k} ×${c}` : k).join(', ');
  };
  img.addEventListener('click', e => {
    const r = img.getBoundingClientRect();
    if (!multi.includes(sel.value)) pts = pts.filter(p => p[0] !== sel.value);
    pts.push([sel.value, (e.clientX - r.left) * img.naturalWidth / r.width,
                         (e.clientY - r.top) * img.naturalHeight / r.height]);
    desenhar();
  });
  bDes.onclick = () => { pts.pop(); desenhar(); };
  await new Promise(r => bOk.onclick = r);
  div.remove();
  return pts;
}
"""))

def _grade(pasta, H0):
    img = cv2.imread(str(pasta / 'frame0.jpg')); Hinv = np.linalg.inv(H0)
    L, W = campo.COMPRIMENTO, campo.LARGURA
    linhas = [[[x, 0], [x, W]] for x in np.arange(0, L + 0.1, 5)] + [[[0, y], [L, y]] for y in np.arange(0, W + 0.1, 5)]
    for seg in linhas:
        a, b = np.clip(homografia.aplicar_h(Hinv, np.array(seg, float)), -1e4, 1e4).astype(int)
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), (0, 255, 255), 1)
    t = np.linspace(0, 2 * np.pi, 120)
    circ = homografia.aplicar_h(Hinv, C_CENTRO + R_CIRC * np.c_[np.cos(t), np.sin(t)]).astype(np.int32)
    cv2.polylines(img, [circ], True, (255, 0, 255), 2)       # círculo central reprojetado (rosa)
    cv2.imwrite(str(pasta / 'calibracao_grade.jpg'), img)

def calibrar_segmento(nome):
    pasta = Path(SAIDA) / nome
    b64 = base64.b64encode((pasta / 'frame0.jpg').read_bytes()).decode()
    pts = eval_js(f'clicarMarcos({json.dumps(b64)}, {json.dumps(list(campo.MARCOS))}, {json.dumps(MULTI)})')
    nomeados = [(n, u, v) for n, u, v in pts if n in campo.MARCOS]
    arco = [(u, v) for n, u, v in pts if n == 'arco_circulo']
    sobre = [(n, u, v) for n, u, v in pts if n in LINHAS]
    if arco:
        if sum(n in ANG for n, _, _ in nomeados) < 2 or len(arco) < 4:
            print('❌ Com o arco, marque também ≥ 2 marcos do círculo (circulo_sup/inf/dir/esq) e ≥ 4 pontos de arco.'); return
        H0, erros = homografia_com_arco(nomeados, arco, sobre, campo.MARCOS)
    else:
        if len(nomeados) < 4:
            print('❌ São necessários pelo menos 4 marcos (ou use arco_circulo).'); return
        src = np.array([[u, v] for _, u, v in nomeados]); dst = np.array([campo.MARCOS[n] for n, _, _ in nomeados])
        H0, _ = cv2.findHomography(src, dst, 0)
        erros = {'marcos': np.linalg.norm(homografia.aplicar_h(H0, src) - dst, axis=1)}
    json.dump({'pixels': [[round(u, 1), round(v, 1)] for _, u, v in nomeados], 'marcos': [n for n, _, _ in nomeados],
               'arco_circulo': [[round(u, 1), round(v, 1)] for u, v in arco],
               'pontos_sobre_linhas': [[n, round(u, 1), round(v, 1)] for n, u, v in sobre],
               'H0': H0.tolist(), 'erro_reprojecao_m': erros['marcos'].tolist(),
               'erros_m': {k: [round(float(x), 3) for x in v] for k, v in erros.items()}},
              open(pasta / 'calibracao.json', 'w'), indent=2)
    print(f'H0 salva em {pasta}/calibracao.json')
    for (n, _, _), e in zip(nomeados, erros['marcos']): print(f'  {n:18s} erro = {e:.2f} m')
    for k in ('arco', 'sobre_linhas'):
        if k in erros: print(f'  {k:18s} erro médio = {erros[k].mean():.2f} m (máx {erros[k].max():.2f} m, {len(erros[k])} pontos)')
    _grade(pasta, H0)
    display(Image(str(pasta / 'calibracao_grade.jpg'), width=1000))
