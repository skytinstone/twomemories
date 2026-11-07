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

## 남은 TBD (결과 외)
- 4B Colab에서 2단 경로 vs 단일 forward 일치 확인 수치(로컬 0.8B 수치는 쓰지 않는다).
- scope=whole의 EF 리셋 시점(2단 prefill과의 관계) 정확한 서술.
- Colab에서 fla 커널 경로를 타지 않았음 확인, torch/transformers 버전(결과 줄 meta).
- 지식 차단 연구 3편(Kaduri, Zhang, Neo) 각 한 문장 요약은 본문 확인 후.
