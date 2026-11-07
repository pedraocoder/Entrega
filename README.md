# Prática 3 — Futebol Analytics com YOLO · Náutico × Sport

> **Como usar este esqueleto:** os números e tabelas abaixo vieram dos arquivos em `saidas/`. Não edite os números à mão: se rodarem algo de novo, atualizem a partir dos arquivos. Os trechos marcados com ✍️ são a **análise autoral** da equipe. Os comentários `<!-- ... -->` indicam onde está a evidência de cada resposta, não aparecem no GitHub e podem ser apagados no final.

## Equipe

| Nome | Email |
|---|---|
| ✍️ | ✍️@cesar.school |
| ✍️ | ✍️@cesar.school |

## Vídeo

- **Jogo:** Náutico 0 × 3 Sport — Final do Campeonato Pernambucano 2026, 2º jogo (transmissão Globo, 1º tempo)
- **URL:** ✍️ (link do YouTube)
- **Intervalo baixado:** `1:00:30 – 1:04:30` (4 min) → `python baixar_video.py "URL" --inicio 60.5 --fim 64.5`
- **Vídeo no Google Drive:** ✍️ (link do clipe + `video_apresentacao.mp4`)
- **Resolução e fps originais:** 1280×720 a 59,94 fps, 14.386 quadros (4,00 min). Convertido para **29,97 fps** (`ffmpeg -r 30000/1001`), o que corta o custo de processamento pela metade.

### Segmentos de câmera principal usados

A transmissão tem 13 cortes de câmera no trecho (closes e replays). Cada corte quebra a homografia encadeada (`H0·Gt`), então cada trecho de câmera principal foi processado e calibrado separadamente.

| Segmento | Tempo no clipe (s) | Tempo no vídeo original | Duração | Uso |
|---|---|---|---|---|
| seg1 | 0,0 – 62,0 | 1:00:30 – 1:01:32 | 62,0 s | ✅ |
| seg2 | 68,9 – 109,0 | 1:01:38,9 – 1:02:19 | 40,1 s | ✅ |
| seg3 | 152,0 – 182,5 | 1:03:02 – 1:03:32,5 | 30,5 s | ❌ descartado (plano rasante na área, calibração inviável) |
| seg4 | 186,2 – 208,0 | 1:03:36,2 – 1:03:58 | 21,8 s | ✅ |

Total analisado: **123,9 s** (`saidas/resumo_final.json`).

## Pipeline

```
vídeo 60 fps → 30 fps → detecção de cortes → segmentos
  → YOLOv8m + ByteTrack (detectar_rastrear.py)
  → times: descritor de listras branco/preto/flúor (substitui o K-Means)
  → Kalman da bola (bola_kalman.py)
  → calibração por cliques: marcos + pontos sobre o círculo e as linhas (calibracao_v2.py)
  → movimento de câmera + projeção em metros (homografia.py camera / projetar)
  → posse, passes, distância (metricas.py, raio 2 m) → validação manual (validar_passes.py)
  → eXP (xpass.py, passes dos 3 segmentos juntos)
```

### Comandos executados

Todos foram rodados no notebook `notebook/P3_futebol_colab.ipynb` (Colab, GPU T4):

```bash
python detectar_rastrear.py --video jogo_clip30.mp4 --saida saidas/segN --modelo yolov8m.pt --inicio A --fim B --imgsz 1280
python bola_kalman.py --saida saidas/segN
python homografia.py camera   --saida saidas/segN
python homografia.py projetar --saida saidas/segN
python metricas.py --saida saidas/segN --raio-posse 2.0 --ataque-time0 esquerda
python validar_passes.py --saida saidas/segN --anotado saidas/segN/passes_anotados.csv
python xpass.py --saida saidas/seg1 --extra saidas/seg2/passes_features.csv saidas/seg4/passes_features.csv --what-if
```

Além disso, no notebook: separação de times por descritor de listras (`times_descritores.csv`, `times_regra.json`), calibração (`src/calibracao_v2.py`), anotação de passes (`src/anotar_passes.py`) e vídeo de apresentação (`src/video_apresentacao.py`).

---

## Parte 1 — Aquisição

**Q1.1 · Escolha do jogo e do trecho** ✍️
<!-- evidências: câmera principal alta; jogo noturno com iluminação uniforme; uniformes com o mesmo vermelho (Náutico com listras brancas, Sport com listras pretas), o que é um desafio deliberado; 13 cortes no trecho; a dificuldade de achar 3–5 min sem cortes -->

**Q1.2 · Resolução, fps e custo**

| | Valor |
|---|---|
| Resolução / fps | 1280×720 · 59,94 fps (processado a 29,97 fps) |
| Quadros processados | seg1 1859 · seg2 1202 · seg4 654 (+ seg3 915, descartado) |
| Velocidade (T4) | yolov8n 19,4 fps · yolov8m 5,5–6,2 fps |

✍️ <!-- custo estimado: 4 min a 60 fps com o yolov8m ≈ 37 min; a 30 fps e só com os segmentos ≈ 13 min -->

## Parte 2 — Detecção

**Q2.1 · Comparação de modelos** (mesmo subtrecho de 30 s, `saidas/comparacao_modelos.csv`)

| Modelo | fps | Jogadores/quadro | % quadros c/ bola | % bola conf ≥ 0,3 | % quadros > 1 candidata | conf mediana da bola |
|---|---|---|---|---|---|---|
| yolov8n | 19,4 | 10,7 | 70,5 | 39,5 | 10,1 | 0,29 |
| yolov8m | 6,2 | 11,1 | 75,2 | 50,8 | 20,9 | 0,33 |

**Q2.2 · Por que a bola é difícil** ✍️
<!-- confiança mediana ~0,3 contra ~0,7+ dos jogadores; poucos pixels; borrão de movimento; relação com o F1 de 0,31 de Ahmad Yousef et al. -->

**Q2.3 · `--imgsz` e `--conf-bola`** ✍️
<!-- se rodaram a célula da Q2.3: `saidas/comparacao_imgsz_confbola.csv`; exemplos de falsos positivos (cabeças, placas) com prints do rastreado.mp4 -->

## Parte 3 — Rastreamento e times

**Q3.1 · Trocas de ID** ✍️
<!-- observação da equipe: o ID se mantém enquanto o jogador está na imagem e muda quando ele sai e volta (track_buffer ≈ 1 s); seg1: 107 tracks com ≥ 30 quadros em 62 s, contra ~11 jogadores visíveis por quadro -->

**Q3.2 · K-Means de cor** ✍️
<!-- evidências: a galeria do seg1 mostrou os dois clusters com jogadores dos dois times misturados; os centróides (meta.json → times_bgr) ficaram marrons/oliva porque a média de uma camisa listrada não é nenhuma das duas cores; o árbitro flúor e os goleiros caíram no "Sport"; o classificador descarta pixels verdes como se fossem grama.
     Solução proposta: proporção de pixels BRANCOS × PRETOS × FLÚOR no tronco (times_descritores.csv, times_regra.json), com Náutico=0 e Sport=1 fixos em todos os segmentos.
     Erros que sobraram: dois jogadores na mesma caixa, borrão de movimento, detecções de torcida e placas.
     (S·cos H, S·sin H, V): o vermelho fica nas duas pontas da escala de matiz (H≈0 e H≈179), e a média simples de H dá um ciano -->

**Q3.3 · `--k 3`** ✍️

## Parte 4 — Bola e homografia

**Q4.1 · Kalman** (`bola.csv`, coluna `fonte`)

| Segmento | Bola vista pelo YOLO | Bola estimada pelo Kalman | Cobertura total |
|---|---|---|---|
| seg1 | 42,0% | 47,7% | 89,6% |
| seg2 | 37,7% | 50,4% | 88,1% |
| seg4 | 56,3% | 38,5% | 94,8% |

✍️ <!-- o gating barrou ~30% das detecções do YOLO; mais da metade da bola é estimada (velocidade constante, lacunas de até 0,5 s); cobertura ≠ acerto; exemplos no video_apresentacao.mp4: círculo verde = YOLO, laranja = Kalman -->

**Q4.2 · Calibração e erro de reprojeção** (`segN/calibracao.json`, `figuras/segN_calibracao_grade.jpg`)

| Segmento | Erro máx. nos marcos | Erro médio no arco do círculo | Marco com mais erro |
|---|---|---|---|
| seg1 | 0,58 m | 0,07 m | centro (estimado, sem marca visível) |
| seg2 | 0,96 m | 0,16 m | circulo_esq |
| seg4 | 1,12 m | 0,27 m | circulo_esq / meio_sup |

✍️ <!-- o quadro do seg1 só mostrava a linha do meio e metade do círculo: com 3 marcos na mesma reta a homografia fica indeterminada. Solução: pontos sobre o arco (todos a 9,15 m do centro) e sobre as linhas, ajustados por mínimos quadrados. Os maiores erros ficam nos marcos distantes (1 px ≈ 0,3–0,5 m) e nas "pontas" da elipse (perspectiva). Teste com pontos não usados: ~1–2 m. Por que o seg3 foi descartado. -->

**Q4.3 · `H0 · Gt`, deriva e re-ancoragem** ✍️
<!-- falhas de câmera: seg1 0, seg2 33/1202, seg4 0 quadros (saída de homografia.py camera) -->

**Q4.4 · Pé do jogador × bola no ar** ✍️
<!-- bola no campo: seg1 95,7%, seg2 86,7%, seg4 76,9% (a bola alta é projetada fora do lugar); jogadores no campo: 98,0 / 87,6 / 83,0% -->

## Parte 5 — Métricas

**Q5.1 · Posse, distância e velocidade** (`segN/metricas_times.json`, `segN/metricas_jogadores.csv`)

| Segmento | Posse Náutico | Posse Sport | Passes detectados |
|---|---|---|---|
| seg1 | 43,1% | 56,9% | 3 |
| seg2 | 46,7% | 53,3% | 6 |
| seg4 | 84,3% | 15,7% | 7 |
| **Total (ponderado)** | **50,8%** | **49,2%** | **16** |

✍️ <!-- plausibilidade: a velocidade máxima de 33,9 km/h (track 8, seg1) é irreal, provavelmente uma caixa que "pulou" de jogador; a distância é por track e não por jogador, porque as trocas de ID quebram o somatório -->

**Q5.2 · Sensibilidade ao raio de posse** (`saidas/sensibilidade_raio.csv`, `sensibilidade_raio_total.csv`, `figuras/sensibilidade_raio.png`)

| Raio | Detectados | Anotados | Casados | Precisão | Revocação | F1 |
|---|---|---|---|---|---|---|
| 1 m | 7 | 22 | 1 | 0,14 | 0,05 | 0,07 |
| **2 m** | 16 | 22 | 5 | **0,31** | 0,23 | **0,26** |
| 3 m | 25 | 22 | 6 | 0,24 | 0,27 | 0,26 |

✍️

**Q5.3 · Validação manual** (`segN/passes_anotados.csv`)

| Segmento | Anotados | Detectados | Casados | Precisão | Revocação | F1 |
|---|---|---|---|---|---|---|
| seg1 | 16 | 3 | 2 | 0,67 | 0,12 | 0,21 |
| seg2 | 1 | 6 | 1 | 0,17 | 1,00 | 0,29 |
| seg4 | 5 | 7 | 2 | 0,29 | 0,40 | 0,33 |

**Critério de anotação** ✍️ (o que contou como passe)

✍️ <!-- erros que dominam: no jogo corrido (seg1) o detector perde passes; na bola parada (seg2, falta) ele inventa passes com o amontoado de jogadores. Proposta: filtro de "bola em jogo" -->

**Q5.4 · Direção de ataque, trocas de ID** ✍️
<!-- a direção foi forçada (--ataque-time0 esquerda: o Náutico ataca para a esquerda no 1º tempo, conferido no vídeo) -->

## Parte 6 — eXP

**Q6.1 · Features e canais de Gomes et al.** ✍️

**Q6.2 · Comparação de modelos** (`seg1/resultados_xp.csv`, 16 passes dos 3 segmentos, 43,8% certos)

| Modelo | AUC ↑ | Brier ↓ | Log-Loss ↓ |
|---|---|---|---|
| Baseline (distância) | 0,413 | 0,241 | 0,674 |
| Heurístico (β fixos) | 0,444 | 0,456 | 1,385 |
| Logístico (validação cruzada) | 0,524 | 0,295 | 0,845 |
| SoccerMap · Gomes et al. | 0,622 | 0,138 | 0,444 |

✍️

**Q6.3 · Calibração** (`seg1/figuras/calibracao_xp.png`) ✍️
<!-- média prevista × taxa observada: heurístico 0,756 × 0,438; logístico 0,398 × 0,438 -->

**Q6.4 · Superfície de eXP** (`seg1/figuras/superficie_xp_f85.png`) ✍️

**Q6.5 · What-if e PAx** (`seg1/what_if.csv`, `seg1/jogadores_xp.csv`) ✍️

---

## Limitações e propostas ✍️

## Declaração de uso de IA ✍️
<!-- usem o modelo da disciplina. Sejam específicos e honestos sobre o que a IA fez, por exemplo:
Usamos o Claude (Anthropic) como assistente no código: notebook do Colab, separação de times por descritor de listras, ferramenta de calibração com arco e linhas (src/calibracao_v2.py), ferramenta de anotação de passes (src/anotar_passes.py), gerador do vídeo de apresentação (src/video_apresentacao.py) e células de resumo. As anotações manuais dos passes, a escolha do jogo e dos segmentos, os cliques de calibração, a interpretação dos resultados e o texto deste README são da equipe. -->
