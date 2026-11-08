# 논문 초안 메모 (paper/main.tex)

초안 작성 2026-11-07. 결과 수치는 전부 `\TBD{}`이고, 그 밖의 수치와 수식은 아래 근거로 확인했다.

## 수식·수치 검토 (코드·config 대조)

| 항목 | 근거 | 상태 |
|---|---|---|
| GDN 갱신 (식 2): 감쇠 후 읽기, α가 괄호 안에도 있음 | transformers 5.22.2 `modeling_qwen3_5.py` recurrent 커널 (감쇠 → kv_mem → delta → 쓰기) | 확인 |
| q·k L2 정규화, q에 d_k^{-1/2}, α=exp(−e^A softplus(a+b)), β=sigmoid | 같은 파일 605·607·621·634행 | 확인 |
| state shape [B, value 헤드, d_k, d_v], value 헤드 32 / key 헤드 16 (4B·9B) | 같은 파일 309·319행, config | 확인 |
| 캐시 크기 (식 4), 4B·9B 동일: L_a 8, L_g 24, H_kv 4, d 256, H 32, d_k=d_v 128 | `AutoConfig` + `hybridquant.budget.cache_sizes` | 확인 |
| state 50.3 MB(fp32), KV 32.8 kB/token(bf16), 교차점 1,536토큰 (식 5) | 위 값으로 계산. 첫 논문의 768은 state를 bf16으로 셈 | 확인 |
| n_v = HW/(pm)², p=16, m=2, 상한 1280 | vision config, 전제 2, max_pixels | 확인 |
| M-RoPE: partial_rotary_factor 0.25(헤드 차원의 1/4), 섹션 [11,11,10] | text config | 확인 |
| 키는 RoPE 적용 후 캐시됨 | 같은 파일 784·787행 | 확인 |
| 양자화기 (식 7), 그룹 128, scale 하한 1e-8, fp32 계산 | `hybridquant/quant.py` `_rtn`, `fake_quant` | 확인 |
| 노이즈 (식 8): RMS는 텐서 전체, seed는 텐서마다 0으로 재설정 → 같은 shape이면 같은 Z | `hybridquant` `fake_quant`, `twomem/quant.py` `_gen` | 확인, 논문에 명시 |
| EF (식 9): 청크 ≤256, C ≤ 5, 마지막 잔차 버림, n_v ≤ 256이면 RTN과 동일 | `twomem/quant.py` `prefill_image`, `_quant_states_ef` | 확인 |
| 토큰 삭제: n_v' = max(1, round(κ n_v)), uniform ⌊j n_v / n_v'⌋, random seed 0 | `twomem/tokens.py` | 확인 |
| retained (식 11), 대응 부트스트랩 2000회 seed 0, 분모 0인 재표본 제외 | `twomem/metrics.py` | 확인 |
| ANLS τ=0.5, relaxed 5%, VQA 정확도(10개 중 9개로 leave-one-out), POPE yes 양성 | `twomem/metrics.py` | 확인 |
| MB (식 12), scale·zero 미포함(fp16 쌍이면 0.25 bit/원소) | `twomem/budget.py` | 확인 |
| 템플릿: 빈 think 블록(non-thinking) | `apply_chat_template` 출력 | 확인 |
| 4B·9B는 post-trained, `-Base`가 따로 있음 | HF 모델 목록 | 확인 |

## 첫 논문(hybridquant) 정정 사항
- 식 (3) `S_t = α_t S_{t−1} + β_t k_t (v_t − S_{t−1}^T k_t)^T`는 괄호 안 α_t가 빠졌다. 구현은 `α_t S_{t−1}^T k_t`이고, 같은 논문 식 (8) `M_j = α_j(I − β_j k_j k_j^T)`도 이쪽과 맞는다. 결론에는 영향 없음.

## 방법론 주의 (논문 4.5절)
- KV zero: 0인 키도 로짓 0으로 softmax 가중치를 받는다 → 정보 제거 + attention 희석. Afendulev et al.(2609.04434)은 이 이유로 0 키를 쓰지 않고 KV를 아예 뺐다. **KV-drop(시각 행을 캐시에서 제거) 조건 추가 여부 결정 필요.**
- state zero: conv cache(커널 4)를 두므로 이미지 뒤 첫 3토큰에 마지막 시각 토큰이 섞인다. Afendulev et al.은 conv를 state와 함께 버렸다. **conv까지 0으로 하는 조건 추가 여부도 같이 결정.**
- state zero는 prefix(템플릿 헤더) 요약도 지운다.

## 가장 가까운 선행연구 (원문 초록 확인)
- Afendulev et al., EMNLP 2026 Findings, arXiv:2609.04434: Qwen3.5·Falcon-H1, 텍스트만. split-prefill(KV만 / state만), state-swap. 정확한 검색은 attention, 언어·페르소나는 recurrent.
- Jiang et al., arXiv:2603.00198: 하이브리드 Mamba–Transformer 비디오 VLM, 토큰 삭제가 state 때문에 압축처럼 동작.
- Kwon et al., arXiv:2609.27470 (DeltaS): GDN state 변화로 KV eviction(Qwen3.5-9B 포함).
- 첫 논문은 arXiv에 없음 → `@misc` 미발표 원고로 인용.

## 2026-11-07 2차 수정 (코드 변경 없이 원고만)
- Figure 1 개요 그림 추가(이후 TikZ 아키텍처 그림으로 교체, 아래 3차 수정).
- 관련 연구: knockout 3편(Kaduri, Zhang, Neo)은 원문 초록으로 확인한 한 문장씩. Salazar et al.(2607.03358, 경로 대체·fallback), DTP(ICLR 2026, Mamba MLLM 가지치기), InfiniteVL(2512.08829, GDN 하이브리드 VLM) 추가.
- 양자화 오차 상한 |Q_b(x)−x| ≤ σ/2 = 그룹 범위의 1/(2(2^b−1)): 4bit 3.3%, 6bit 0.79%, 8bit 0.20%.
- retained를 샘플 가중 평균으로 분해(식 12). 두 번째 등호는 모든 샘플에서 B_i ≠ F_i일 때만 성립한다고 명시.
- 개입별로 남는 정보 경로 표(Table 2). KV 0 + state 0 셀이 conv 누출과 attention 희석의 상한을 잰다.
- 사전 예측 P1–P4(4.6절). 결과와 대조해 유지, 수정, 삭제할 것.
- **phase 3의 같은 메모리 비교 짝 정정**: 이전 TBD의 "κ=0.5@16bit vs κ=1@4bit"는 8 vs 4 bit라 같지 않다. 맞는 짝은 κ=0.25@16bit vs κ=1@4bit(둘 다 4 bit). 텍스트 행은 16 vs 4 bit로 다르다.
- 한계 (iv): 제거 실험은 평소 경로가 아니라 대체(fallback) 가능성을 잴 수 있다(Salazar et al.).
- 부록 B 조건 표(22개, phase, 비트), 부록 C 컴퓨팅 표(preflight 행만 채움).

## 2026-11-07 3차 수정 (사용자 요청: 형식)
- Figure 1을 TikZ 아키텍처 흐름도로 교체: `paper/figures/architecture.tex`. "Attention Is All You Need" Figure 1 형식(층을 쌓은 블록, 잔차 경로, 반복 횟수 3×·8×). 블록 구성은 모델 코드 기준: pre-norm(RMSNorm → 믹서 → +, RMSNorm → FFN → +), dense FFN, Gated Attention(query 16 / KV 4 헤드). 오른쪽에 두 메모리(KV cache, recurrent state + conv)와 개입 f·g, 아래에 2단계 prefill 타임라인.
- 형식을 NeurIPS 2017 스타일로: 1단, 본문 5.5×9인치, Times 본문 + Computer Modern 수식, 제목 위 4pt·아래 1pt 가로선, 가운데 "Abstract"와 좌우 0.5인치 들여쓴 초록, 문단 들여쓰기 없음·문단 간격 5.5pt, 섹션 제목 12pt 굵게. 공식 스타일 파일이 아니라 그 설정을 재현한 것이다. 키워드 줄은 그 스타일에 없어 뺐다.

## 코드 검토에서 찾은 문제 (지금은 고치지 않음, Colab 실행 중)
- **whole-scope EF의 잔차 공유**: hybridquant의 EF 잔차는 `stats.ef_residuals` 하나이고, `Session.fork()`의 deepcopy는 캐시만 복사한다. 그래서 정답 NLL용 fork의 forward가 잔차를 덮어쓰고, 원본 캐시에서 하는 greedy 디코딩 첫 단계가 그 잔차를 이어받는다. 영향 범위는 `KV4_S8ef_whole` 셀의 pred·score뿐이다. gold NLL은 fork 이전의 2단계 잔차를 쓰므로 정상이다. 논문 4.2절에 적었다. 다음 태그에서 고칠지 그대로 보고할지 결정해야 한다.

## 남은 TBD (결과 외)
- 4B Colab에서 2단 경로 vs 단일 forward 일치 확인 수치(로컬 0.8B 수치는 쓰지 않는다).
- scope=whole의 EF 리셋 시점(2단 prefill과의 관계) 정확한 서술.
- Colab에서 fla 커널 경로를 타지 않았음 확인, torch/transformers 버전(결과 줄 meta).
- 지식 차단 연구 3편(Kaduri, Zhang, Neo) 각 한 문장 요약은 본문 확인 후.

## 제목 (결과 후 확정) — D-34에서 후보 A로 확정
현재 제목 "The Image Lives in Two Memories: Visual Information in Hybrid Linear-Attention Vision-Language Models"는 결과 전에 결론을 단정하고 부제가 길다. 결과를 보고 확정한다. 후보:
- A. Where Does the Image Live? KV Cache and Recurrent State in Hybrid Vision-Language Models
- B. Two Memories of an Image: KV Cache and Recurrent State in Hybrid Vision-Language Models
- C. What a Hybrid Vision-Language Model Remembers of an Image
- D. 결론형(예: The Image Lives in the KV Cache: …), 결과가 분명할 때

## 2단 전환 (2026-11-09, 사용자 요청)
- 1단(NeurIPS 2017 재현) → 2단: 본문 7×9인치, 단 간격 0.25인치, 제목·저자만 두 단에 걸침. 21쪽 → 16쪽(본문 11쪽).
- 넓은 것은 figure*/table*: Fig 1(구조도), Fig 2(전체 retained), 주 결과 표, 부록 결과 표와 조건 표. 나머지 그림(Fig 3–5)은 단 폭(3.4인치, 글자 7pt)으로 다시 그려 축소 없이 넣는다.
- 표는 `adjustbox`의 `max width`로, 넘칠 때만 줄인다(이전 `resizebox`는 좁은 표를 키웠다).
