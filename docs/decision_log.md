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

### D-18. HF 캐시를 Drive에서 Colab 로컬 디스크로 — 확정 (작업 지시 이탈)
작업 지시는 HF 캐시를 Drive에 두라고 했다. Colab preflight(v0.1-preflight)에서 DocVQA blind 셀은 `validation-00000-of-00006.parquet`을 읽었는데, 바로 다음 셀이 Drive 마운트로 같은 파일을 다시 읽자 `ArrowInvalid: Parquet magic bytes not found in footer`가 났다. Drive 마운트를 거친 큰 파일 쓰기/읽기가 잘린 것으로 본다(원인 세부는 UNVERIFIED). 조치: (1) `HF_HOME=/content/hf_cache`(로컬 디스크). 세션마다 모델·shard를 다시 받는다. (2) `data._read_shard`가 파싱에 실패한 캐시 shard를 `force_download`로 한 번 다시 받는다(백엔드 무관). 결과(jsonl, compute_log)는 그대로 Drive에 쓴다. Drive의 `MyDrive/twomemories/hf_cache`는 더 쓰지 않으니 지워도 된다.

### D-19. A100 시간당 유닛 8.88 — 확정 (D-12 일부)
같은 세션의 Colab 리소스 패널: "사용률 약 8.88/시간"(Pro+, GPU RAM 80 GB, 시스템 RAM 167 GB). `configs/compute.yaml`의 A100 값을 11.8 → 8.88로 고쳤다. GPU 이름이 A100인지는 셀 2 출력 `GPU:` 줄로 확인할 것(자리값 표에서 A100과 default가 모두 11.8이라 셀 5의 11.80만으로는 구분 불가, UNVERIFIED). phase별 상한(D-12 나머지)은 preflight 완료 후 정한다.

### D-20. 노트북 셀 5 dry-run 루프를 subprocess로 — 확정
`!... --phase {ph} ... "$TWOMEM_RESULTS"`에서 IPython이 `$TWOMEM_RESULTS`를 파이썬 변수로 확장하려다 실패하면 줄 전체를 확장하지 않아 `{ph}`가 그대로 넘어갔다(`configs/{ph}.yaml` 없음). `subprocess.run(['bash', '-c', ...])`로 바꿨다. 셀 4는 `{}`가 없어 영향 없음.

### D-21. 본 실험은 백그라운드 실행 + 모니터 셀 — 확정
셀 6이 `!bash scripts/run_all.sh`를 포그라운드로 돌리면 몇 시간 동안 커널이 막혀 진행을 볼 수 없다. 셀 6은 `subprocess.Popen(start_new_session=True)`로 띄우고 바로 끝나며, 셀 7이 60초마다 `scripts/status.py`를 호출한다. status는 결과 파일만 읽는다(셀별 jsonl 줄 수를 cell_hash로 세고, 유닛은 cap과 같은 compute_log의 phase 태그 합). 시작 전 phase의 남은 시간은 같은 모델의 이전 셀 평균(`--dry-run`과 같은 추정). 로그는 `MyDrive/twomemories/logs/`에 둔다. `results/` 안에 두면 run_all이 커밋하므로 push 오류 메시지 같은 것이 레포에 들어갈 수 있다. 셀 8은 프로세스 그룹에 SIGTERM을 보낸다. 끊긴 셀의 유닛은 compute_log에 남지 않는다(셀 끝에 기록). Colab이 실행 중인 셀이 없을 때 런타임을 끊는지는 UNVERIFIED라 셀 7을 켜 두도록 안내한다. status.py는 다음 태그(D-12 확정 후 본 실험 태그)부터 체크아웃에 들어간다.

### D-22. 세션 단절 후 이어하기 — 확정
- 결과 줄 읽기는 잘린 줄을 건너뛰고(`runner.read_jsonl`), 쓰기는 파일이 잘린 줄로 끝나면 새 줄에서 시작한다(`runner.append_jsonl`). 분석(`Index`)은 같은 (셀, 샘플)이 두 번 있으면 처음 것만 쓴다.
- compute_log의 phase 태그를 runner가 줄을 쓸 때 함께 넣는다. 이전에는 `run_phase._tag_phase`가 compute_log 전체를 다시 썼는데, 그 도중 끊기면 로그(=상한 계산 근거)를 잃을 수 있었다.
- `run_phase`는 남은 셀이 없는 모델·phase에서 모델을 로드하지 않는다(새 세션에서 끝난 phase를 빠르게 지나감).
- `run_all.sh`는 체크아웃을 태그에 둔 채, `.results_wt` worktree를 원격 `results/<tag>` 브랜치 끝(없으면 태그)에서 만들고 Drive 결과를 rsync해 커밋·push한다. 새 세션의 새 clone에서도 push가 fast-forward다. 결과 줄의 commit은 계속 태그 커밋이다. 로컬 bare 원격으로 두 세션(새 clone 두 번)을 흉내 내어 확인했다.
- 끊긴 셀에서 이미 쓴 샘플의 유닛은 compute_log에 남지 않는다(셀 끝에 기록). 상한 계산이 최대 한 셀만큼 덜 잡힌다. 4B 셀 하나 200샘플 × 약 2초 = 약 7분, 약 1유닛.

### D-23. 모니터 3초 갱신, 커널 안에서 증분 읽기 — 확정 (번복: D-21의 "60초마다 status.py 호출")
사용자 요청으로 3초 갱신. 매번 subprocess로 python을 띄우면 torch import만 수 초라, 셀 7이 `status.Monitor`를 커널에 한 번 import하고 각 jsonl의 늘어난 바이트만 읽는다(로컬 측정 렌더 3 ms). 표시: 전체·phase별 % (phase 사이에 공유되는 셀은 전체 %에서 한 번만 센다), 현재 단계, 완료 예정 시각(KST). 속도는 같은 모델의 최근 15분 결과 줄의 벽시계 간격(모델 로드·다운로드 포함). 그 모델 줄이 아직 없으면(예: phase 4의 9B) compute_log / `sec_per_sample_guess`로 추정하고 "prior"로 표시한다.

### D-24. phase마다 표·그림 생성, 논문은 맥에서 — 확정
`run_all.sh`가 phase마다 결과 브랜치 worktree에서 `make_tables`, `figures`(pdf + png), `report`를 모델별로 돌려 `paper/`에 넣고 함께 push한다(CPU, 수 초). 논문용 최종 표·그림은 맥에서 병합 후 다시 만든다(hybridquant와 같은 `paper/figures`, `paper/tables` 구조).
