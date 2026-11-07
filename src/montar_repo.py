# ── Monta a pasta do repositório (sem vídeos) e gera entrega_repo.zip no Drive ──
import shutil, glob, os, json
from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

def montar_repo(segs=('seg1', 'seg2', 'seg4'), destino='/content/entrega'):
    D = Path(destino); base, saida = Path(BASE), Path(SAIDA)
    if D.exists(): shutil.rmtree(D)
    (D / 'src').mkdir(parents=True); (D / 'figuras').mkdir(); (D / 'notebook').mkdir()
    # código: scripts do professor + ferramentas da equipe
    for f in glob.glob('/content/src/*.py'): shutil.copy(f, D / 'src')
    for f in ('calibracao_v2.py', 'anotar_passes.py', 'video_apresentacao.py', 'montar_repo.py'):
        if (base / f).exists(): shutil.copy(base / f, D / 'src')
    # notebook (o Colab salva em "Colab Notebooks")
    nbs = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/*P3*.ipynb'), key=os.path.getmtime)
    if nbs: shutil.copy(nbs[-1], D / 'notebook' / 'P3_futebol_colab.ipynb')
    else: print('⚠️ notebook não encontrado: baixe pelo menu Arquivo → Baixar .ipynb e coloque em notebook/')
    # saídas sem vídeos nem pesos
    shutil.copytree(saida, D / 'saidas', ignore=shutil.ignore_patterns('*.mp4', '*.avi', '*.pt', '__pycache__'))
    # figuras/ reunidas com prefixo do segmento
    for s in segs:
        for f in glob.glob(str(saida / s / 'figuras' / '*.png')): shutil.copy(f, D / 'figuras' / f'{s}_{Path(f).name}')
        if (saida / s / 'calibracao_grade.jpg').exists():
            shutil.copy(saida / s / 'calibracao_grade.jpg', D / 'figuras' / f'{s}_calibracao_grade.jpg')
    # gráfico da sensibilidade ao raio (Q5.2)
    sp = saida / 'sensibilidade_raio.csv'
    if sp.exists():
        sens = pd.read_csv(sp); fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for s, g in sens.groupby('seg'):
            ax[0].plot(g.raio, g.posse_nautico, 'o-', label=s); ax[1].plot(g.raio, g.passes_det, 'o-', label=f'{s} detectados')
        tot = sens.groupby('raio').agg(det=('passes_det', 'sum'), anot=('passes_anot', 'sum'), vp=('VP', 'sum'))
        P, R = tot.vp / tot.det, tot.vp / tot.anot
        ax[1].plot(tot.index, 2 * P * R / (P + R) * 10, 'k--', label='F1 total (×10)')
        ax[0].set(xlabel='raio de posse (m)', ylabel='posse do Náutico (%)', title='Posse × raio')
        ax[1].set(xlabel='raio de posse (m)', ylabel='nº de passes', title='Passes detectados × raio')
        for a in ax: a.set_xticks([1, 2, 3]); a.legend(fontsize=8); a.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(D / 'figuras' / 'sensibilidade_raio.png', dpi=130); plt.close(fig)
    # README e .gitignore
    if (base / 'README.md').exists(): shutil.copy(base / 'README.md', D / 'README.md')
    else: print('⚠️ README.md não encontrado em P3_VC/: coloque lá o esqueleto e rode de novo')
    (D / '.gitignore').write_text('*.mp4\n*.avi\n*.mov\n*.pt\ndados/\n__pycache__/\n.ipynb_checkpoints/\n')
    # checagens
    arqs = [p for p in D.rglob('*') if p.is_file()]
    videos = [p for p in arqs if p.suffix.lower() in ('.mp4', '.avi', '.mov')]
    grandes = [(p.relative_to(D), round(p.stat().st_size / 1e6, 1)) for p in arqs if p.stat().st_size > 50e6]
    print(f'{len(arqs)} arquivos, {sum(p.stat().st_size for p in arqs) / 1e6:.1f} MB')
    print('vídeos no repositório:', videos or 'nenhum ✅')
    if grandes: print('⚠️ arquivos > 50 MB (o GitHub recusa acima de 100 MB):', grandes)
    zipf = shutil.make_archive(str(base / 'entrega_repo'), 'zip', D)
    print('zip salvo em', zipf)
