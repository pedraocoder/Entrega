# Prática 3 — Futebol Analytics com YOLO · Náutico × Sport

Pipeline que parte de um vídeo de transmissão do YouTube e chega a métricas de jogo: detecção e rastreamento de jogadores e bola (YOLOv8 + ByteTrack), separação dos times, projeção no campo em metros (homografia), posse de bola, passes validados à mão e eXP (expected pass). Todos os números deste README foram tirados dos arquivos em `saidas/`.

## Equipe

| Nome | Email |
|---|---|
| Pedro [sobrenome] | psp2@cesar.school |
| [nome] | [email]@cesar.school |

## Vídeo

- **Jogo:** Náutico 0 × 3 Sport, final do Campeonato Pernambucano 2026, 2º jogo (transmissão Globo, 1º tempo)
- **URL:** [link do YouTube]
- **Intervalo baixado:** `1:00:30 – 1:04:30` (4 min), com `python baixar_video.py "URL" --inicio 60.5 --fim 64.5`
- **Vídeos no Google Drive** (não versionados): clipe original `jogo_clip.mp4` [link] · vídeo de apresentação com as anotações: https://drive.google.com/file/d/1mk94Coqnco0mAj-v4rFhhJgXjWo7FL1O/view
- **Formato original:** 1280×720 a 59,94 fps, 14.386 quadros (4,00 min). Convertido para **29,97 fps** (`ffmpeg -r 30000/1001`).

### Segmentos de câmera principal usados

A transmissão tem 13 cortes de câmera no trecho (closes e replays), detectados automaticamente por correlação de histograma HSV. Cada corte quebra a homografia encadeada (`H0·Gt`), então cada trecho de câmera principal foi processado e calibrado separadamente.

| Segmento | Tempo no clipe (s) | Tempo no vídeo original | Duração | Uso |
|---|---|---|---|---|
| seg1 | 0,0 – 62,0 | 1:00:30 – 1:01:32 | 62,0 s | ✅ |
| seg2 | 68,9 – 109,0 | 1:01:38,9 – 1:02:19 | 40,1 s | ✅ (início ajustado: o corte em 65,2 s ainda era um close) |
| seg3 | 152,0 – 182,5 | 1:03:02 – 1:03:32,5 | 30,5 s | ❌ descartado (plano rasante na área, calibração inviável) |
| seg4 | 186,2 – 208,0 | 1:03:36,2 – 1:03:58 | 21,8 s | ✅ (início ajustado para pular a transição do corte) |

Total analisado: **123,9 s** (`saidas/resumo_final.json`).

## Pipeline

```
vídeo 60 fps → 30 fps → detecção de cortes → segmentos
  → YOLOv8m + ByteTrack (detectar_rastrear.py)
  → times: descritor de listras branco/preto/flúor (substitui o K-Means)
  → Kalman da bola (bola_kalman.py)
  → calibração por cliques: marcos + pontos sobre o círculo e as linhas (src/calibracao_v2.py)
  → movimento de câmera + projeção em metros (homografia.py camera / projetar)
  → posse, passes, distância (metricas.py, raio 2 m) → validação manual (validar_passes.py)
  → eXP (xpass.py, passes dos 3 segmentos juntos)
```

### Comandos executados

Todos rodaram no notebook `notebook/P3_futebol_colab.ipynb` (Google Colab, GPU T4):

```bash
python detectar_rastrear.py --video jogo_clip30.mp4 --saida saidas/segN --modelo yolov8m.pt --inicio A --fim B --imgsz 1280
python bola_kalman.py --saida saidas/segN
python homografia.py camera   --saida saidas/segN
python homografia.py projetar --saida saidas/segN
python metricas.py --saida saidas/segN --raio-posse 2.0 --ataque-time0 esquerda
python validar_passes.py --saida saidas/segN --anotado saidas/segN/passes_anotados.csv
python xpass.py --saida saidas/seg1 --extra saidas/seg2/passes_features.csv saidas/seg4/passes_features.csv --what-if
```

Também estão no notebook: a separação de times por descritor de listras (`times_descritores.csv`, `times_regra.json`), a calibração (`src/calibracao_v2.py`), a anotação manual de passes (`src/anotar_passes.py`), a sensibilidade ao raio (`saidas/sensibilidade_raio*.csv`), o resumo final (`saidas/resumo_segmentos.csv`, `saidas/resumo_final.json`) e o vídeo de apresentação (`src/video_apresentacao.py`).

---

## Parte 1 — Aquisição

**Q1.1 · Escolha do jogo e do trecho.** Escolhemos a final do Pernambucano entre Náutico e Sport por ser uma transmissão com câmera principal alta, em plano aberto e com iluminação noturna uniforme, sem a metade do campo na sombra. O trecho do 1º tempo foi escolhido por ter jogo corrido e as linhas do meio-campo visíveis, necessárias para a homografia. Sabíamos de dois desafios: os dois uniformes têm o mesmo vermelho (Náutico com listras brancas, Sport com listras pretas), o que dificulta a separação por cor, e não existe um trecho de 3–5 min sem cortes. O clipe de 4 min tem 13 cortes. Em vez de procurar outro jogo, tratamos os cortes dividindo o clipe em segmentos de câmera principal.

**Q1.2 · Resolução, fps e custo.**

| | Valor |
|---|---|
| Resolução / fps | 1280×720 · 59,94 fps (processado a 29,97 fps) |
| Quadros processados | seg1 1859 · seg2 1202 · seg4 654 (+ seg3 915, descartado) |
| Velocidade (T4) | yolov8n 19,4 fps · yolov8m 5,5–6,2 fps |

Com o yolov8m a ~6 fps, os 14.386 quadros originais levariam cerca de 40 min. Reduzindo para 30 fps e processando só os segmentos, foram cerca de 14 min de detecção (4.630 quadros, incluindo o seg3 depois descartado), mais ~7 min de estimativa de movimento de câmera. Rodar a 30 fps não prejudica o rastreamento: a maioria das transmissões é nessa taxa.

## Parte 2 — Detecção

**Q2.1 · Comparação de modelos** (mesmo subtrecho de 30 s, `saidas/comparacao_modelos.csv`)

| Modelo | fps | Jogadores/quadro | % quadros c/ bola | % bola conf ≥ 0,3 | % quadros > 1 candidata | conf mediana da bola |
|---|---|---|---|---|---|---|
| yolov8n | 19,4 | 10,7 | 70,5 | 39,5 | 10,1 | 0,29 |
| yolov8m | 6,2 | 11,1 | 75,2 | 50,8 | 20,9 | 0,33 |

O yolov8m é três vezes mais lento e ganha pouco em jogadores (+0,4 por quadro), mas ganha 11 pontos percentuais em detecções de bola confiáveis. Como posse e passes dependem da bola, usamos o yolov8m.

**Q2.2 · Por que a bola é difícil.** A bola ocupa poucos pixels (cerca de 8–12 px numa imagem 1280×720 em plano aberto), se move rápido e aparece borrada nos quadros de passe e chute. Com isso, a confiança mediana da bola ficou em ~0,3, e a contagem bruta (70–75%) inclui falsos positivos: o script aceita bola a partir de 0,10 e guarda até 5 candidatas por quadro. No yolov8m, 21% dos quadros tinham mais de uma candidata, o que prova que algumas são falsas. Isso bate com Ahmad Yousef et al., em que a bola tem F1 de 0,31 sem filtro temporal e só chega a 0,96 com o filtro de Kalman.

**Q2.3 · `--imgsz` e `--conf-bola`.** Por limitação de tempo, não rodamos a grade completa de `imgsz` 640 × 1280. O efeito do limiar de confiança foi medido de forma indireta na tabela da Q2.1: subir o limiar de 0,10 para 0,30 derruba a cobertura de 75,2% para 50,8% no yolov8m. A diferença são justamente as detecções duvidosas. No vídeo rastreado, os falsos positivos típicos foram cabeças, chuteiras, marcações brancas do gramado e objetos nas placas de publicidade. Por isso deixamos o limiar baixo e delegamos a filtragem ao gating do Kalman.

## Parte 3 — Rastreamento e times

**Q3.1 · Trocas de ID.** O ID se mantém enquanto o jogador continua na imagem, inclusive em cruzamentos rápidos. Ele muda quando o jogador sai do quadro e volta, porque o ByteTrack só "lembra" de um track por ~1 s (`track_buffer` de 30 quadros), e em aglomerações. No seg1 surgiram **107 tracks com 30 quadros ou mais em 62 s**, para cerca de 11 jogadores visíveis por quadro. Isso dá dezenas de trocas por minuto, sendo uma parte desses tracks detecções de torcida e placas. A consequência é que a distância percorrida só pode ser calculada por track, e não por jogador.

**Q3.2 · K-Means de cor.** Não funcionou para este jogo. A galeria de recortes do seg1 (um recorte por track, agrupado pelo time atribuído) mostrou jogadores do Náutico e do Sport misturados nos dois grupos. Os centróides (`meta.json → times_bgr`) ficaram marrons e oliva: a cor média de uma camisa listrada, somada à pele e à grama nas bordas, não é nenhuma das duas cores do uniforme, e as duas médias ficam parecidas. Além disso, o árbitro de amarelo-flúor caía no Sport, porque o classificador descarta pixels verdes como se fossem grama e sobra a bermuda preta, e os goleiros eram forçados a um dos dois grupos (k = 2).

**Nossa solução:** um descritor específico para esses uniformes. Para cada track, medimos no tronco (18–50% da altura da caixa, sem 20% das laterais) a proporção de pixels **brancos** (V > 150, S < 70), **pretos** (V < 60) e **amarelo-flúor** (H 20–40, S > 140, V > 170), e usamos a mediana ao longo do track. A regra, com limiares tirados dos casos-limite da galeria (`times_regra.json`): flúor > 15% → árbitro (−2); preto > 20% → comissão técnica e torcida (−2); branco ≥ 15% → Náutico (0); o resto → Sport (1). Isso fixou **Náutico = 0 e Sport = 1 em todos os segmentos**, resolvendo o alinhamento entre eles. Na galeria corrigida, todos os tracks longos ficaram certos. Os erros restantes são tracks curtos com dois jogadores na mesma caixa, borrão de movimento (o Náutico correndo perde o branco nítido) e detecções de torcida e placas, que a homografia descarta por cair fora do campo.

**Sobre (S·cos H, S·sin H, V):** o matiz é circular, e o vermelho fica nas duas pontas da escala do OpenCV (H ≈ 0 e H ≈ 179). A média simples de H daria ~90, um ciano que não existe na camisa. Usar seno e cosseno trata o matiz como ângulo, e multiplicar por S evita que pixels sem cor (branco, cinza) puxem o matiz. No nosso caso o problema não era o matiz, que é igual nos dois times, mas o branco × preto das listras, que só aparece separadamente em V e S.

**Q3.3 · `--k 3`.** Não executado por limitação de tempo. Pelo diagnóstico da Q3.2, esperamos que não resolva o árbitro: a camisa flúor é descartada como grama antes do agrupamento. O risco do k = 3 é o terceiro grupo capturar sombra ou um dos lados das listras, em vez do árbitro e dos goleiros, dividindo um time em dois.

## Parte 4 — Bola e homografia

**Q4.1 · Kalman** (`bola.csv`, coluna `fonte`)

| Segmento | Bola vista pelo YOLO | Bola estimada pelo Kalman | Cobertura total |
|---|---|---|---|
| seg1 | 42,0% | 47,7% | 89,6% |
| seg2 | 37,7% | 50,4% | 88,1% |
| seg4 | 56,3% | 38,5% | 94,8% |

O gating descartou cerca de 30% das detecções do YOLO por serem incoerentes com a trajetória (no seg1, 1121 → 780 quadros), eliminando a maior parte dos falsos positivos da Q2.3. A cobertura final passa de 88% em todos os segmentos, mas no seg1 e no seg2 mais da metade dos quadros com bola foram **estimados**: o filtro supõe velocidade constante e preenche lacunas de até 15 quadros (0,5 s). Ele acerta em passes retos e "inventa" a bola logo depois de um chute ou desvio, quando a direção muda. No `video_apresentacao.mp4`, o círculo verde é a bola vista e o laranja a estimada, e os dois casos aparecem. Cobertura não é acerto: Ahmad Yousef et al. medem F1 contra anotação manual, o que não fizemos para a bola.

**Q4.2 · Calibração e erro de reprojeção** (`segN/calibracao.json`, `figuras/segN_calibracao_grade.jpg`)

| Segmento | Erro máx. nos marcos | Erro médio no arco do círculo | Marco com mais erro |
|---|---|---|---|
| seg1 | 0,58 m | 0,07 m | centro (estimado, sem marca visível) |
| seg2 | 0,96 m | 0,16 m | circulo_esq |
| seg4 | 1,12 m | 0,27 m | circulo_esq / meio_sup |

O primeiro quadro do seg1 só mostrava a linha do meio e metade do círculo. Os marcos com nome visíveis (`meio_sup`, `circulo_sup`, `centro`, `circulo_inf`) ficam todos na mesma reta, e com 3 de 4 marcos colineares a homografia fica indeterminada: a primeira tentativa deu 0 m de erro em três marcos e 49 m no quarto. A solução foi uma ferramenta de calibração (`src/calibracao_v2.py`) que aceita, além dos marcos, **vários pontos clicados sobre o arco do círculo** (todos a 9,15 m do centro) e **sobre a linha do meio** (x = 52,5 m), ajustando a homografia por mínimos quadrados (`scipy.optimize.least_squares`). Os maiores erros ficam nos marcos mais distantes da câmera, onde 1 pixel vale 0,3–0,5 m, e nas "pontas" da elipse do círculo, que pela perspectiva não coincidem exatamente com o ponto do campo. Esses erros medem o ajuste nos pontos clicados. Em pontos fora do ajuste, longe do círculo, o erro esperado é de 1–2 m.

O seg3 foi descartado: a câmera estava rasante, olhando a área de lado, então 15 m de campo cabiam em ~40 px de altura, a lateral estava escondida atrás das placas e as linhas da área ficaram ambíguas. Nenhuma interpretação das linhas produziu uma calibração consistente.

**Q4.3 · `H0 · Gt`, deriva e re-ancoragem.** `H0` leva os pixels do quadro 0 para metros (calibração manual). `Gt` leva os pixels do quadro t para os pixels do quadro 0: é estimado por ORB + RANSAC entre quadros, ignorando jogadores e placar, e encadeado em quadros-chave. A posição em metros é `H0 · Gt · [u, v, 1]ᵀ`. Como `Gt` é um produto de estimativas, pequenos erros se acumulam e geram deriva ao longo do clipe. Quando a estimativa falha num quadro, o script reaproveita a anterior: foram 0 falhas no seg1, 33 em 1202 quadros no seg2 e 0 no seg4. Um corte de câmera quebra a cadeia de vez, por isso processamos por segmento. Para re-ancorar, propomos detectar automaticamente as linhas e o círculo (transformada de Hough mais ajuste de elipse) a cada N segundos e recalcular `H0` com a mesma otimização da nossa calibração, sem cliques.

**Q4.4 · Pé do jogador × bola no ar.** A homografia só é válida para pontos no plano do campo (altura zero). O pé do jogador, a base da caixa, toca o chão, então é projetado corretamente. O centro da caixa fica cerca de 0,9 m acima do chão e seria projetado mais longe da câmera, deslocando o jogador alguns metros. A bola num lançamento está no ar, e é projetada no ponto onde o raio da câmera cruza o gramado, atrás da posição real. Isso aparece nos dados: só 76,9% das posições da bola no seg4 caíram dentro do campo, contra 95,7% no seg1. Jogadores dentro do campo: 98,0% / 87,6% / 83,0% (os descartados são torcida, banco, comissão técnica e bandeirinha).

## Parte 5 — Métricas

**Q5.1 · Posse, distância e velocidade** (`segN/metricas_times.json`, `segN/metricas_jogadores.csv`)

| Segmento | Posse Náutico | Posse Sport | Passes detectados |
|---|---|---|---|
| seg1 | 43,1% | 56,9% | 3 |
| seg2 | 46,7% | 53,3% | 6 |
| seg4 | 84,3% | 15,7% | 7 |
| **Total (ponderado pelos quadros com bola)** | **50,8%** | **49,2%** | **16** |

A posse é plausível no jogo corrido (no seg4, o Náutico trocou passes com a bola o tempo todo, como na anotação manual). No seg2 a bola estava parada por uma falta durante quase todo o trecho, e a posse calculada ali não tem significado. As distâncias por track são plausíveis em parte. O track 327 (seg1) andou 31,0 m em 23,6 s, cerca de 79 m/min, abaixo da média de ~120 m/min de um jogo inteiro e compatível com um meia em fase de pouca movimentação. O track 66 (seg4) fez 77,7 m em 20,8 s, cerca de 225 m/min, típico de um trecho de corrida. Já a velocidade máxima de **33,9 km/h** (track 8, seg1) é irreal: equivale ao pico de um velocista de elite e quase certamente vem de uma caixa que "pulou" de um jogador para outro. O filtro de 12 m/s (43 km/h) não pega esse caso. As trocas de ID impedem somar a distância por jogador.

**Q5.2 · Sensibilidade ao raio de posse** (`saidas/sensibilidade_raio.csv`, `saidas/sensibilidade_raio_total.csv`, `figuras/sensibilidade_raio.png`)

| Raio | Detectados | Anotados | Casados | Precisão | Revocação | F1 |
|---|---|---|---|---|---|---|
| 1 m | 7 | 22 | 1 | 0,14 | 0,05 | 0,07 |
| **2 m** | 16 | 22 | 5 | **0,31** | 0,23 | **0,26** |
| 3 m | 25 | 22 | 6 | 0,24 | 0,27 | 0,26 |

Com 1 m, o erro de ~1 m da homografia faz o jogador quase nunca ficar "colado" na bola, e posse e passes somem (no seg4 a posse do Náutico salta para 95,6%). Com 3 m, o detector acha um passe real a mais, mas cria oito detecções falsas, e a precisão cai. O F1 empata entre 2 e 3 m, e escolhemos **2 m** pela melhor precisão. A posse é muito mais estável que a contagem de passes: no seg1 ela varia entre 38,8% e 43,1% nos três raios, enquanto o número total de passes vai de 7 para 25.

**Q5.3 · Validação manual** (`segN/passes_anotados.csv`)

| Segmento | Anotados | Detectados | Casados | Precisão | Revocação | F1 |
|---|---|---|---|---|---|---|
| seg1 | 16 | 3 | 2 | 0,67 | 0,12 | 0,21 |
| seg2 | 1 | 6 | 1 | 0,17 | 1,00 | 0,29 |
| seg4 | 5 | 7 | 2 | 0,29 | 0,40 | 0,33 |
| **Total** | **22** | **16** | **5** | **0,31** | **0,23** | **0,26** |

**Critério de anotação:** contou como passe todo toque intencional da bola para um companheiro, incluindo lançamento e cruzamento, marcado no instante em que a bola sai do pé, com o time do passador e se um companheiro recebeu. Não contaram condução, chutão de afastamento sem alvo nem finalização. A anotação foi feita assistindo cada segmento em câmera lenta (0,5x) com `src/anotar_passes.py`, e a tolerância de casamento é de 1 s.

**Erros que dominam:** são dois, opostos. No **jogo corrido (seg1)**, o detector **perde passes**: só 3 dos 16. As causas são as trocas de ID (o mesmo jogador vira dois "donos"), a bola estimada pelo Kalman, que erra o momento do toque, e o erro de 1–2 m da homografia, que tira o jogador do raio de posse. Na **bola parada (seg2, falta)**, o detector **inventa passes**: 6 detectados para 1 real, porque com os jogadores amontoados em volta da bola o "dono" troca a cada instante. O acordo certo/errado nos passes casados foi de apenas 50%.

**Q5.4 · Direção de ataque e trocas de ID.** Forçamos a direção com `--ataque-time0 esquerda`, conferida no vídeo: o Náutico ataca para a esquerda da tela no 1º tempo. A estimativa automática usa os jogadores mais recuados de cada time e falharia aqui, porque os goleiros raramente aparecem nos segmentos e a classificação de cor original os colocava no time errado. As trocas de ID afetam as métricas de duas formas: quebram a distância percorrida, que fica por track e não por jogador, e criam passes falsos quando o ID do portador muda. O filtro de distância mínima da bola só evita parte disso. Propomos re-identificação por aparência para unir tracks do mesmo jogador e um filtro de bola em jogo: com bola e jogadores parados por mais de alguns segundos, não se conta posse nem passe.

## Parte 6 — eXP

**Q6.1 · Features e canais de Gomes et al.** O `xpass.py` usa seis features geométricas que resumem, num único número por passe, os canais espaciais da Tabela 1 de Gomes et al. (2026):

| Feature | O que mede | Canais correspondentes (Tabela 1) |
|---|---|---|
| `distancia` | comprimento do passe | 3 (distância à bola), 17 (distância ao início do evento) |
| `adversarios_na_linha` | adversários entre a bola e o destino | 11 (defensores entre bola e célula) |
| `pressao_passador` | adversários perto do passador | 2 (posição do time defensor) |
| `espaco_destino` | distância do destino ao adversário mais próximo | 1 e 2 (ocupação dos dois times) |
| `progressao_x` | avanço em direção ao gol adversário | localização e direção de ataque |
| `dist_destino_gol` | distância do destino ao gol | 4 (distância ao gol) |

Não conseguimos extrair com confiança: **os canais de velocidade** (8–9 bola, 12 portador, 13–16 atacantes e defensores), porque as trocas de ID e a bola estimada pelo Kalman tornam as velocidades muito ruidosas; **a ocupação do campo inteiro**, porque a câmera mostra só metade do campo e cerca de 11 dos 22 jogadores; e os **ângulos bola–gol** (5–7), que poderiam ser derivados das posições mas não fazem parte do modelo. A representação de Gomes et al. é uma superfície 104×68 por canal, enquanto a nossa resume cada passe em seis números.

**Q6.2 · Comparação de modelos** (`seg1/resultados_xp.csv`, 16 passes detectados nos 3 segmentos, 43,8% certos)

| Modelo | AUC ↑ | Brier ↓ | Log-Loss ↓ |
|---|---|---|---|
| Baseline (distância) | 0,413 | 0,241 | 0,674 |
| Heurístico (β fixos) | 0,444 | 0,456 | 1,385 |
| Logístico (validação cruzada) | 0,524 | 0,295 | 0,845 |
| SoccerMap · Gomes et al. | 0,622 | 0,138 | 0,444 |

Os três modelos ficam próximos de AUC 0,5, o desempenho do acaso, e o baseline fica abaixo disso. Com 16 passes (7 certos e 9 errados), uma única previsão a mais ou a menos muda a AUC em vários pontos, então essas métricas têm intervalo de confiança muito largo e não permitem dizer que um modelo é melhor que outro. Os rótulos também são ruidosos: o "certo/errado" do detector concorda com a anotação manual em só 50% dos passes casados. Os coeficientes do logístico não são interpretáveis; "adversários na linha" saiu com sinal positivo, o que é contraintuitivo. Já juntamos os passes dos três segmentos com `--extra`. Para uma avaliação significativa, seria preciso juntar passes de outros trechos ou grupos, de preferência com rótulos corrigidos pela anotação manual. A SoccerMap foi treinada com 52 partidas de tracking profissional.

**Q6.3 · Calibração** (`seg1/figuras/calibracao_xp.png`). O heurístico é otimista demais: prevê em média 75,6% de acerto, e a taxa observada foi de 43,8%. Por isso tem o pior Brier e o pior Log-Loss. O logístico prevê em média 39,8%, próximo dos 43,8% observados, mas com 16 passes cada faixa da curva de calibração tem 3 ou 4 passes, e a curva oscila sem forma definida. Em Gomes et al., a média predita coincide com a taxa observada no teste, mas com milhares de passes.

**Q6.4 · Superfície de eXP** (`seg1/figuras/superficie_xp_f85.png`). A superfície mostra o eXP de um passe hipotético para cada ponto do campo, a partir da posição do passador no quadro 85 do seg1. As regiões quentes ficam perto do passador e em espaços sem adversários, e as frias ficam atrás de linhas de marcação e longe da bola. Isso reproduz a lógica da Figura 1 de Gomes et al.: passes curtos e para espaços livres são mais seguros. Como o modelo que gera a superfície tem AUC ≈ 0,5, o desenho reflete as features codificadas (distância e ocupação) mais do que algo aprendido dos dados, e não deve ser lido como uma recomendação tática.

**Q6.5 · What-if e PAx** (`seg1/what_if.csv`, `seg1/jogadores_xp.csv`). O what-if desloca o destino dos passes errados em 3–6 m e mede o ganho de eXP: o ganho médio foi de 0,217, com 100% dos passes errados acima de 0,10. Esse "sempre melhora" é sintoma do modelo fraco, e não uma conclusão tática. No PAx (acertos − soma dos eXP), só três passadores do seg1 aparecem, cada um com um único passe (PAx entre −0,73 e 0,00). Nenhum jogador se destaca, e não poderia: em 2 minutos de jogo, com um passe por jogador e IDs que mudam, o PAx mede ruído. Conclusões por jogador exigiriam partidas inteiras e identidades estáveis.

---

## Limitações e propostas

| Limitação | Evidência | Proposta |
|---|---|---|
| Trocas de ID | 107 tracks em 62 s, distância só por track | Re-identificação por aparência para unir tracks |
| Bola no ar | só 76,9% da bola no campo no seg4 | Estimar a altura pela parábola da trajetória antes de projetar |
| Cortes de câmera | 13 cortes, um segmento descartado, calibração manual por segmento | Detecção automática de linhas e círculo para recalibrar |
| Bola parada | seg2: 6 passes detectados para 1 real | Filtro de bola em jogo × parada antes das métricas |
| Poucos passes | 16 passes, AUC ≈ 0,5 | Mais trechos e grupos com `--extra`, rótulos corrigidos à mão |
| Classes do COCO | sem árbitro e goleiro | Modelo de futebol (Roboflow Universe, Q2.4) |

**Conclusão:** o pipeline funciona de ponta a ponta, e a posse saiu estável e plausível (50,8% × 49,2%). A validação manual mostrou que os passes detectados ainda são pouco confiáveis (F1 0,26), e por isso o eXP não aprende nada. O maior ganho viria de melhorar os eventos detectados (IDs estáveis, bola parada e bola alta), e não de um modelo de eXP mais sofisticado.

## Declaração de uso de IA

Usamos o Claude (Anthropic) como assistente durante todo o trabalho: no código (notebook do Colab, separação de times por descritor de listras, ferramenta de calibração com arco e linhas em `src/calibracao_v2.py`, ferramenta de anotação de passes em `src/anotar_passes.py`, gerador do vídeo de apresentação em `src/video_apresentacao.py` e células de resumo) e na redação de um rascunho deste README e dos slides, a partir dos nossos resultados. São da equipe: a escolha do jogo e dos segmentos, os cliques de calibração, a anotação manual dos 22 passes, a execução de todo o pipeline, a verificação dos resultados e a revisão final do texto.
