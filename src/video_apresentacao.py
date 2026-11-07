# ── Vídeo de apresentação: times corrigidos + bola (YOLO × Kalman) + minimapa + posse + passes ──
import json, subprocess, cv2, numpy as np, pandas as pd
from pathlib import Path

COR = {0: (255, 255, 255), 1: (40, 40, 40), -2: (0, 230, 255), -1: (150, 150, 150)}   # BGR
NOME = {0: 'Nautico', 1: 'Sport'}
L, W = 105.0, 68.0

def _campo_base(esc=3.2, mg=8):
    w, h = int(L * esc) + 2 * mg, int(W * esc) + 2 * mg
    img = np.full((h, w, 3), (60, 120, 60), np.uint8)
    P = lambda x, y: (int(mg + x * esc), int(mg + y * esc))
    c = (230, 230, 230)
    cv2.rectangle(img, P(0, 0), P(L, W), c, 1); cv2.line(img, P(L/2, 0), P(L/2, W), c, 1)
    cv2.circle(img, P(L/2, W/2), int(9.15 * esc), c, 1)
    for x0, s in ((0, 1), (L, -1)):
        cv2.rectangle(img, P(x0, W/2 - 20.16), P(x0 + s * 16.5, W/2 + 20.16), c, 1)
        cv2.rectangle(img, P(x0, W/2 - 9.16), P(x0 + s * 5.5, W/2 + 9.16), c, 1)
    return img, P

def _texto(img, t, org, esc=0.6, cor=(255, 255, 255), esp=1):
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, esc, (0, 0, 0), esp + 3, cv2.LINE_AA)
    cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, esc, cor, esp, cv2.LINE_AA)

def _ler(p, nome):
    f = p / nome
    return pd.read_csv(f) if f.exists() else pd.DataFrame()

def gerar_video(segs, destino, max_s=None, saida=None):
    """segs: lista de nomes ('seg1', ...); max_s: corta cada segmento em N segundos (opcional)."""
    saida = Path(saida or SAIDA)
    meta0 = json.load(open(saida / segs[0] / 'meta.json'))
    Wv, Hv = meta0['largura'] if 'largura' in meta0 else 1280, meta0['altura'] if 'altura' in meta0 else 720
    fps = meta0['fps']
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                           '-s', f'{Wv}x{Hv}', '-r', str(fps), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast',
                           '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(destino)],
                          stdin=subprocess.PIPE)
    base, P = _campo_base()
    for seg in segs:
        p = saida / seg; meta = json.load(open(p / 'meta.json'))
        tr = pd.read_csv(p / 'tracks.csv'); tr = tr[tr.classe != 'bola']
        moda = tr[tr.time != -1].groupby('track_id').time.agg(lambda s: s.value_counts().idxmax())
        tr['t'] = tr.track_id.map(moda).fillna(-1).astype(int)
        tc = _ler(p, 'tracks_campo.csv')
        if len(tc): tc['t'] = tc.track_id.map(moda).fillna(-1).astype(int)
        bola, bc, posse = _ler(p, 'bola.csv'), _ler(p, 'bola_campo.csv'), _ler(p, 'posse_por_quadro.csv')
        passes, anot = _ler(p, 'passes.csv'), _ler(p, 'passes_anotados.csv')
        por_f = {f: g for f, g in tr.groupby('frame')}
        tc_f = {f: g for f, g in tc.groupby('frame')} if len(tc) else {}
        b_f = bola.set_index('frame') if len(bola) else None
        bc_f = bc.set_index('frame') if len(bc) else None
        ps_f = posse.set_index('frame') if len(posse) else None
        f_anot = (anot.t_s * meta['fps']).round().astype(int).tolist() if len(anot) else []
        cont = {0: 0, 1: 0}
        n = meta['n_frames'] if not max_s else min(meta['n_frames'], int(max_s * meta['fps']))
        cap = cv2.VideoCapture(meta['video']); cap.set(cv2.CAP_PROP_POS_FRAMES, round(meta['inicio_s'] * meta['fps']))
        for f in range(n):
            ok, img = cap.read()
            if not ok: break
            if img.shape[1] != Wv: img = cv2.resize(img, (Wv, Hv))
            # jogadores
            for r in por_f.get(f, pd.DataFrame()).itertuples():
                c = COR.get(r.t, COR[-1]); esp = 1 if r.t < 0 else 2
                cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), c, esp)
                if r.t >= 0: _texto(img, str(r.track_id), (int(r.x1), int(r.y1) - 4), 0.4, c)
            # bola: verde = YOLO, laranja = estimada pelo Kalman
            if b_f is not None and f in b_f.index:
                rb = b_f.loc[f]; rb = rb.iloc[0] if isinstance(rb, pd.DataFrame) else rb
                if pd.notna(rb.px):
                    det = rb.fonte == 'detectada'
                    cv2.circle(img, (int(rb.px), int(rb.py)), 9, (0, 255, 0) if det else (0, 140, 255), 2 if det else 1)
            # posse acumulada
            dono_txt = ''
            if ps_f is not None and f in ps_f.index:
                rp = ps_f.loc[f]; rp = rp.iloc[0] if isinstance(rp, pd.DataFrame) else rp
                if pd.notna(rp.time_posse): cont[int(rp.time_posse)] += 1
                if pd.notna(rp.get('time_dono', np.nan)): dono_txt = f'  | com a bola: {NOME[int(rp.time_dono)]} #{int(rp.dono)}'
            tot = cont[0] + cont[1]
            cv2.rectangle(img, (0, Hv - 64), (Wv, Hv), (0, 0, 0), -1)
            if tot:
                _texto(img, f'Posse  Nautico {100*cont[0]/tot:.0f}%  x  {100*cont[1]/tot:.0f}% Sport{dono_txt}', (10, Hv - 12), 0.6)
            _texto(img, f'{seg}  t={f/meta["fps"]:5.1f}s', (Wv - 200, Hv - 12), 0.6, (200, 200, 200))
            _texto(img, 'caixa branca: Nautico | escura: Sport | amarela: arbitro/outros   bola: verde = YOLO, laranja = Kalman', (10, Hv - 48), 0.5, (230, 230, 230))
            # passes: detectado (pipeline) e anotado (gabarito manual)
            if len(passes):
                ativos = passes[(passes.frame_ini <= f) & (f <= passes.frame_ini + int(1.2 * meta['fps']))]
                for r in ativos.itertuples():
                    _texto(img, f'PASSE DETECTADO ({NOME[r.time]}, {"certo" if r.sucesso else "errado"})', (20, 230), 0.8, (0, 255, 255), 2)
            if any(0 <= f - fa <= int(1.2 * meta['fps']) for fa in f_anot):
                _texto(img, 'passe no gabarito manual', (20, 265), 0.7, (255, 200, 0), 2)
            # minimapa
            mm = base.copy()
            for r in tc_f.get(f, pd.DataFrame()).itertuples():
                if pd.notna(r.x_m) and r.t >= 0: cv2.circle(mm, P(r.x_m, r.y_m), 4, COR[r.t], -1)
                elif pd.notna(r.x_m) and r.t == -2: cv2.circle(mm, P(r.x_m, r.y_m), 3, COR[-2], -1)
            if bc_f is not None and f in bc_f.index:
                rb = bc_f.loc[f]; rb = rb.iloc[0] if isinstance(rb, pd.DataFrame) else rb
                if pd.notna(rb.x_m): cv2.circle(mm, P(rb.x_m, rb.y_m), 4, (0, 165, 255), -1)
            h, w = mm.shape[:2]; x0, y0 = Wv - w - 10, 10
            img[y0:y0 + h, x0:x0 + w] = cv2.addWeighted(img[y0:y0 + h, x0:x0 + w], 0.25, mm, 0.75, 0)
            ff.stdin.write(img.tobytes())
        cap.release()
        print(f'{seg}: {n} quadros')
    ff.stdin.close(); ff.wait()
    print('vídeo salvo em', destino)
