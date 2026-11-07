# 실행 기록 (experiment log)

수치의 출처는 `results/**/*.jsonl`과 `results/**/compute_log.jsonl`이다. 이 문서는 무엇을 언제 어떤 커밋으로 돌렸는지만 적는다.

## 2026-11-07 로컬 (MacBook Pro M3 Pro, MPS, 유닛 0)

| 단계 | 명령 | 커밋 | 결과 |
|---|---|---|---|
| C1 | `pytest -q` | 4c7387e 이후 작업 트리 | 14 passed, 6.35s |
| C2 | `python scripts/check_l1.py` | (L1 절 참조) | docs/assumptions.md "L1 결과" |
| C3 | `python scripts/make_subsets.py` | — | configs/subsets/*.json (한 번만 실행) |
| C3 | `python scripts/run_phase.py --phase preflight --limit 5 --out results/local` | 8db117f79e | results/local/, 24셀 120줄, 스키마 통과, 4분 22초. 재실행 시 전부 skip(resume) |
| C3 | `python analysis/make_tables.py --model Qwen/Qwen3.5-0.8B --allow-local --results results/local` | 8db117f79e | Table 2/5 생성, 미평가 셀 n/e |
| C3 | `python scripts/check_l2.py` | 8db117f79e | greedy 5/5 동일, batch2 최대 차이 7.2e-3 → batch 1 고정(D-13) |

로컬 수치는 경로 검증용이며 논문에 쓰지 않는다.

## 2026-11-07 Colab preflight 1차 (v0.1-preflight, 중단)

| 단계 | 결과 |
|---|---|
| 셀 2 setup | 완료. 리소스 패널 약 8.88 유닛/시간, GPU RAM 80 GB |
| 셀 4 preflight | 4B DocVQA blind 셀 5샘플 완료(0.46 s/sample, VRAM 피크 8.7 GB) 후 KVfull_Sfull 셀에서 Drive 캐시 parquet 손상으로 중단(D-18) |
| 셀 5 요약 | dry-run 루프 `{ph}` 미확장(D-20). 사용 유닛 약 3.2(설치·다운로드 포함) |

수정 후 `v0.1.1-preflight`로 재실행. blind DocVQA 5줄은 cell_hash가 같아 resume으로 건너뛴다(그 줄의 commit은 188a6dd).
