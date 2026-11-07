# 결정 기록 (decision log)

원칙: 결정을 번복할 때 기존 항목을 지우거나 고치지 않는다. 새 항목을 덧붙이고 "번복: D-n"으로 이전 항목을 가리킨다. 상태는 `확정`, `제안`(사용자 확인 필요) 중 하나.

## 2026-11-07 세션 1 (로컬 구축 A~D, Colab 준비 E)

### D-1. 레포명 `twomemories` — 확정
논문 제목 "The Image Lives in Two Memories"의 핵심어. 패키지명은 짧게 `twomem`. 소유자 skytinstone, 비공개, MIT(hybridquant와 동일).

### D-2. 전제 확인 결과로 갈린 구현 규칙 — 확정
- 전제 1(입력 임베딩에서만 주입): 개입 정의 그대로. 결과 줄 `meta.inject_layers = [0]`. 중간 층 주입 분기 미적용.
- 전제 4(캐시 메서드명 동일): 어댑터 없이 hybridquant `quantized_cache`, `fake_quant`, `make_cache`를 그대로 import.
- 전제 5(forward 끝 1회 쓰기): `8ef`/`6ef`는 시각 토큰 256개씩 forward를 반복 호출하고 청크마다 Q(S + r) 저장 + 잔차 이월. 청크 경계 훅 방식은 해당 없음. 라이브러리 내부 chunk_size 64는 쓰기 시점과 무관하므로 EF 청크와 별개(기록만).
- 근거는 docs/assumptions.md.

### D-3. 2단 prefill을 수동 LM forward로 구현 — 확정
`model(**inputs)`는 이미지 span 직후에서 멈출 수 없어서, 입력 임베딩(이미지 특징 scatter)과 4행 position ids(text row + M-RoPE 3행)를 샘플당 한 번 만들고 language model에 슬라이스로 넣는다. 단일 forward 기준 stock 경로와 logits 차이 0(bf16, MPS). 2단 분할은 fp32에서 6e-5, bf16에서 마지막 행 logits 0.25 차이(반올림 순서). 모든 조건이 같은 2단 경로를 쓴다. L1에서 stock 단일 forward와의 gold NLL/token 차이 3.8e-3을 기록.

### D-4. 이미지 해상도 상한 `max_pixels = 1,310,720` (시각 토큰 ≤ 1280) — 제안
기본 processor는 `longest_edge = 16,777,216` 픽셀이라 DocVQA 원본 페이지가 수천 토큰이 된다. 4B 본 실험 유닛(350)을 고려해 상한을 둔다. 1280토큰이면 EF 청크가 최대 5개. 상한은 n_vis 분포와 태스크 난이도(특히 DocVQA 작은 글씨)에 영향을 준다. 사용자 확인 필요. 바꾸면 cell_hash가 달라져 기존 결과와 섞이지 않는다.

### D-5. "22조건"의 정의 — 제안
작업 지시의 Colab preflight "전 22조건"은 목록이 없었다. phase 1(11) + phase 2(12) + phase 3(7)의 고유 조건 합집합을 정의했고 그 수가 22다(`run_phase.py --phase preflight --set full --dry-run`이 목록 출력). phase 4(9B)는 phase 1–3 조건의 부분집합이라 추가 경로가 없다. 의도한 22개와 다르면 configs만 고치면 된다.

### D-6. 예산 회계 — 확정
KV full 16bit, state full 32bit(전제 3: fp32 저장), noise는 맞춘 비트(KV 4, state 8), zero 0bit. KV는 작업 지시 식대로 모든 행(n_vis_kept + n_text)을 kv_bits로 계산한다. scope=image에서 실제로는 텍스트 행이 16bit로 남지만, 표 5의 같은 MB 비교는 배포 회계 기준이다. 양자화 scale/zero, conv state는 제외. MB = 10^6 바이트.

### D-7. Colab 결과 저장: symlink 대신 Drive 직접 쓰기 + rsync — 확정
작업 지시는 results를 Drive로 symlink하라고 했으나, symlink 디렉터리에 `git add results`를 하면 git이 링크만 커밋하고 jsonl은 커밋되지 않는다. 대신 `TWOMEM_RESULTS=/content/drive/MyDrive/twomemories/results`에 직접 쓰고(세션이 끊겨도 보존), `run_all.sh`가 phase 끝마다 Drive → `results/`로 rsync한 뒤 커밋한다. HF 캐시는 작업 지시대로 Drive(`HF_HOME`). Colab push는 `results/<tag>` 브랜치로 하고 맥에서 main에 병합한다(Colab의 detached tag에서 main으로 직접 push하면 충돌 가능).

### D-8. 로컬 결과 분리 — 확정
로컬 L2 결과는 `results/local/`에 쓴다(git 추적). `analysis/make_tables.py`는 gpu가 mps/cpu인 줄을 기본 제외하고, `--allow-local`일 때만 포함하며 표 머리에 "LOCAL CHECK"를 찍는다.

### D-9. conv1d state 미개입 — 확정
hybridquant 관례를 따른다. state zero에서도 마지막 몇 개 시각 토큰의 국소 정보가 conv state(커널 4)로 질문에 전달될 수 있다. 논문에서 명시할 것.

### D-10. scope=whole에서 zero 조건 미지원 — 확정
whole은 첫 논문 설정(비트폭/노이즈)의 재현용이다. zero를 매 토큰 적용하면 모델이 동작하지 않는 자명한 조건이라 정의하지 않는다(`Condition`이 예외).

### D-11. gold와 프롬프트 — 확정
네 태스크 공통 lmms-eval식 지시 한 줄, system 없음, 기본 chat template(빈 think 블록 포함) 그대로. gold는 DocVQA answers[0], TextVQA 최빈값, ChartQA/POPE 원문. POPE gold는 데이터의 소문자 "yes"/"no" 그대로.

### D-12. phase별 유닛 상한 — 제안
총 350: preflight 15, phase1 60, phase2 80, phase3 60, phase4 105(9B). 근거 없는 초안이다. Colab preflight의 초/샘플로 다시 정한다. `configs/compute.yaml`의 `units_per_hour`도 UNVERIFIED 자리값이며 Colab 리소스 패널 값으로 고친다.

### D-13. batch 1 고정 — 확정
L2-local(0.8B, MPS, DocVQA 5샘플, full): 같은 샘플을 batch 2로 복제했을 때 gold_nll(합)과 batch 1의 차이 최대 7.2e-3(5개 중 4개가 1e-5 ~ 7.2e-3), 기준 1e-3 불통과. 서로 다른 샘플의 padded batch는 구현하지 않았다(샘플마다 이미지 span 위치가 달라 2단 prefill 분할점이 다름). `run_cell(batch=N>1)`은 NotImplementedError. greedy 두 번 실행은 5/5 동일(pred와 gold_nll 완전 일치). 근거: `scripts/check_l2.py`.

### D-14. 로컬 L2 소요 시간 — 기록
0.8B MPS 평균 초/샘플(5조건 + blind, 2단 prefill + gold + decode 포함): DocVQA 1.37, TextVQA 0.78, ChartQA 0.33, POPE 0.33. preflight 전체 120샘플 4분 22초(shard 다운로드 포함). n_vis 최대 1276(max_pixels 상한 동작 확인), EF 청크 >1인 줄 16개.

## 2026-11-07 세션 2 (Colab 실행 전 확인)

### D-15. 해상도 상한 확정 — 확정 (D-4 확정)
사용자가 D-4 제안대로 `max_pixels = 1,310,720`(시각 토큰 ≤ 1280, EF 청크 최대 5개)을 확정했다. 코드·config 변경 없음.

### D-16. "22조건" 확정 — 확정 (D-5 확정)
사용자가 D-5 정의(phase 1–3 고유 조건 합집합, `run_phase.py --phase preflight --set full --dry-run` 출력 22개)를 확정했다. 코드·config 변경 없음.

### D-17. 태그 유지 — 확정
문서만 바뀌었으므로 `v0.1-preflight`(188a6dd)를 옮기지 않는다. 결과 줄의 commit은 src/scripts/configs/analysis 기준이라 이 문서 커밋과 무관하다. 남은 제안은 D-12(유닛 상한, units_per_hour)이며 Colab preflight 셀 5 결과로 정한다.
