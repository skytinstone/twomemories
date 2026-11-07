# twomemories

하이브리드 linear-attention VLM(Qwen3.5)에서 이미지 정보가 KV cache와 recurrent state 중 어디에 저장되는지 training-free 개입으로 재는 실험 코드. 논문: *The Image Lives in Two Memories: Visual Information in Hybrid Linear-Attention VLMs*.

양자화·노이즈 구현은 [hybridquant](https://github.com/skytinstone/hybridquant)를 패키지로 import한다(복사하지 않음).

> **로컬(맥) 수치는 논문에 쓰지 않는다.** `results/local/`과 `docs/assumptions.md`의 L1 결과는 경로 검증용이다. 논문 수치는 Colab에서 4B 이상으로 돌린 `results/**/*.jsonl`에서만 가져온다.

## 무엇을 재는가

이미지를 읽은 직후(이미지 span 마지막 토큰 다음) 두 메모리 중 하나를 바꾸고, 질문과 답 생성이 그 메모리를 읽게 한다.

- KV: 시각 토큰 행만 {full, 4bit, noise, zero}
- State: 모든 GDN 층 recurrent state {full, 8, 8+EF, 6, 6+EF, noise, zero}
- 시각 토큰 {100, 50, 25}% (uniform / random 삭제)
- 태스크: DocVQA, ChartQA, TextVQA, POPE

지표는 정답 teacher-forced NLL로 만든 데이터셋 수준 비율
retained = (NLL_blind − NLL_cond) / (NLL_blind − NLL_full)과 대응 부트스트랩 95% CI, 그리고 greedy 답의 태스크 점수.

## 로컬 검증 (맥, MPS)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ../path/to/hybridquant -e ".[dev]"

pytest -q                                                            # 1) tiny 모델 CPU 스모크 14개
python scripts/check_l1.py                                           # 2) 0.8B 실모델 L1 점검
python scripts/run_phase.py --phase preflight --limit 5 --out results/local   # 3) 0.8B 4태스크 × 5조건
```

## Colab 실행 절차

1. GitHub fine-grained PAT 발급: 저장소 `twomemories`, `hybridquant` 두 개만, 권한 Contents read/write, 만료 90일. Colab 왼쪽 열쇠(Secrets)에 이름 `GH_TOKEN`으로 등록하고 노트북 접근 허용. 토큰 값은 레포·문서·셀에 적지 않는다.
2. Google Drive에 `MyDrive/twomemories/` 폴더가 없어도 된다(setup이 `results/`를 만든다). HF 캐시는 Drive가 아니라 Colab 로컬 디스크 `/content/hf_cache`에 두며 세션마다 다시 받는다(D-18).
3. Colab에서 `notebooks/colab_runner.ipynb`를 연다(GitHub 탭, 비공개 레포 접근 허용). 런타임 GPU 선택.
4. 셀 1–5 실행: Drive 마운트 → setup → `v0.1.1-preflight` 체크아웃 → 4B preflight(22조건 × 4태스크 × 5샘플) → 셀당 초, VRAM 피크, phase별 예상 유닛.
5. 5번 결과를 보고 승인한 뒤에만 셀 6에서 `APPROVED = True`로 바꿔 실행한다. `run_all.sh`가 백그라운드로 돌고, phase마다 결과가 `results/<tag>` 브랜치로 push된다. 로그는 `MyDrive/twomemories/logs/run_all.log`.
6. 셀 7(모니터)을 켜 둔다. 60초마다 `scripts/status.py`로 phase별 셀·샘플·유닛/상한·남은 시간, 최근 셀(마지막 줄 이후 경과 초), GPU, 로그 끝 줄을 보여준다. 셀 7을 멈춰도 실험은 계속된다. 셀 8은 실험 중지(재실행 시 resume).

Colab에서는 코드를 고치지 않는다. 수정은 맥에서 하고 새 태그를 push한 뒤 셀 3에서 다시 체크아웃한다.

## 구조

```
src/twomem/   model(로드·인코딩·2단 forward) quant(개입) tokens(토큰 삭제) budget(MB 회계)
              data(subset 로더) metrics(NLL·점수·retained) runner(셀 실행·resume)
scripts/      check_l1 check_l2 make_subsets run_cell run_phase run_all.sh status colab_setup.sh
configs/      phase*.yaml preflight.yaml compute.yaml subsets/*.json
analysis/     schema make_tables figures report
docs/         assumptions(전제 확인) decision_log experiment_log
```

## 라이선스

MIT. 데이터셋 라이선스는 `docs/assumptions.md` "데이터" 절.
