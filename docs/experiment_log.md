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

## 2026-11-07 맥 작업 사고 기록

run_all.sh 이어하기 시뮬레이션 중 임시 clone이 실패한 채 명령이 이어져, 실제 레포에서 테스트용 sed와 run_all이 실행됐다. 가짜 결과 6줄(`results/m/...`)이 원격 브랜치 `results/0432a52`로 push됐다. 확인 후 브랜치를 삭제하고 run_all.sh를 되돌렸다. main과 태그는 영향 없음. 이후 시뮬레이션은 `set -e` 서브셸, 로컬 bare 원격, `GH_TOKEN` 제거로 다시 했다.

## 2026-11-07 Colab preflight 2차 (v0.1.1-preflight, 완료)

| 항목 | 값 |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB, 8.88 유닛/시간(리소스 패널) |
| 셀 | compute_log 91셀(기대 92). 빠진 1셀은 1차 실행의 DocVQA blind 기록으로 추정. 1차의 `_tag_phase`가 Drive 위에서 compute_log를 다시 쓰다 잃은 것으로 보이나 UNVERIFIED. 결과 jsonl은 덧붙이기만 하므로 영향 없음 |
| 속도 | 1.47 s/sample(행 기준 평균). blind 0.87, EF 1.66–1.99, 토큰 삭제 1.34–1.37 |
| VRAM 피크 | 9.3 GB |
| 유닛 | preflight 1.64 |

상한은 D-25.

## 2026-11-07 로컬 phase 5 점검 (0.8B, MPS, 검증용, 논문 미사용)

`python scripts/run_phase.py --phase phase5_channels --model Qwen/Qwen3.5-0.8B --limit 5 --out results/local`: 12셀 신규, full·blind 8셀 skip. 5샘플 retained(참고만): KV drop ≈ KV zero(DocVQA 0.14 대 0.13, TextVQA 0.31 대 0.32), state zeroconv는 state zero와 비슷하거나 약간 높음, drop+zeroconv는 DocVQA −0.10, TextVQA 0.02(0 근처). ChartQA는 CI가 너무 넓어 판단 불가.
