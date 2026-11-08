# twomemories

"The Image Lives in Two Memories: Visual Information in Hybrid Linear-Attention VLMs" 실험 코드. 하이브리드 VLM(Qwen3.5: GDN 3층 + Gated Attention 1층 반복, early fusion)에서 이미지 정보가 KV cache와 recurrent state 중 어디에 저장되는지를 training-free 개입으로 잰다.

## 규칙

- 양자화는 hybridquant를 **import만** 한다(`pip install -e`). 코드 복사·수정 금지. 맞지 않으면 `src/twomem/quant.py`에 어댑터를 둔다.
- 저장 시점 양자화(fake-quant). 커널 구현 금지.
- 개입 위치 (scope=image):
  - state 개입은 이미지 span 마지막 토큰 직후 1회 (EF 조건은 256 시각 토큰 청크 경계마다).
  - KV 개입은 시각 토큰 행에만.
- prefill은 2단: (prefix + image) → 개입 → (question). gold NLL은 그 뒤 forward 1회, decode는 greedy 16토큰.
- scope=whole: 첫 논문 설정(전 토큰, decode 매 토큰, hybridquant 기본 훅).
- retained = (NLL_blind − NLL_cond) / (NLL_blind − NLL_full). 데이터셋 수준 비율만, CI는 대응(paired) 부트스트랩(n=2000, seed 0). 샘플별 비율은 만들지 않는다.
- 논문·보고서 수치는 `results/**/*.jsonl`에서만 가져온다. 로컬(맥) preflight(`results/local`, 5샘플) 수치는 검증용이며 논문에 쓰지 않는다. 0.8B 본 실험은 맥에서 전체 subset으로 `results/mac`에 새로 돌린 결과만 쓴다(D-30). 표·그림은 `--results results/mac --allow-local`로 만든다.
- 모든 결과 줄에 commit 해시. Colab에서는 코드 편집 금지: 맥에서 고치고 새 태그를 push한 뒤 Colab에서 재체크아웃.
- 확인 안 된 것은 UNVERIFIED로 적는다. 결정 번복 시 decision_log에서 지우지 않고 새 항목으로 덧붙인다.
- 코드 주석은 영어, 문서는 한국어.

## 축

| 축 | 수준 |
|---|---|
| KV | full, 4, noise, zero, drop |
| State | full, 8, 8+EF, 6, 6+EF, noise, zero, zeroconv |
| Tokens | 100%, 50%, 25% (uniform / random) |
| Task | DocVQA, ChartQA, TextVQA, POPE |

- EF: prefill 청크(시각 토큰 256개) 경계에서 Q(S + r) 저장, r 이월.
- noise: 같은 텐서에 KV 4bit / state 8bit RTN 오차와 같은 RMS의 가우시안.

## 예산식

MB = KV(n_vis + n_text, kv_bits) + State(state_bits). state는 토큰 수와 무관. KV full 16bit, state full 32bit(fp32 저장, docs/assumptions.md 전제 3). 회계식이며 측정값이 아니다(`src/twomem/budget.py`).

## 컴퓨팅

- Colab 유닛 350, 단발. phase별 상한은 `configs/compute.yaml`의 `phase_caps`(preflight 15 / phase1 60 / phase2 80 / phase3 60 / phase4 105). `run_phase.py`가 상한에 닿으면 멈춘다.
- 단계별 중단 규칙은 각 `configs/phase*.yaml`의 `stop_rule`. phase가 끝날 때마다 결과를 보고 다음 phase를 수동으로 승인한다.
- 0.8B는 맥에서(유닛 0): 전 경로 검증과 preprint 본 실험(D-30). Colab은 4B 이상 확인 실험만.

## 명령

```bash
source .venv/bin/activate
pytest -q                                              # CPU smoke (tiny model)
python scripts/check_l1.py                             # 0.8B, MPS
python scripts/run_phase.py --phase preflight --limit 5
python scripts/run_phase.py --phase phase1_probe --dry-run
python scripts/status.py --out results/local --phases preflight --model Qwen/Qwen3.5-0.8B --limit 5   # progress (--watch 3)
python analysis/make_tables.py
```

## 문서

- `docs/assumptions.md` 전제 확인(파일:줄 근거), 구조 수치, 데이터, L1 결과
- `docs/decision_log.md` 결정 기록(번복 시 삭제 금지)
- `docs/experiment_log.md` 실행 기록
