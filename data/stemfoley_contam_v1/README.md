# StemFoley 오염/clean 검수 데이터 v1 (팀 내부용)

UnAV 101클립에서 SAM-Audio로 목표 소리 하나씩 분리한 stem 767개와 사람 검수 라벨이다. **오염**은 목표 소리 외에 **다른 종류의 소리**가 섞인 것이다(같은 종류의 다른 개체는 오염이 아니다. 말소리는 다른 화자가 섞이면 오염이다).

**팀 내부 연구용이다. 외부 공유·배포 금지**(UnAV 영상에서 나온 소리다).

## 구성

```
labels.csv
audio/
  test/{contaminated,clean,undetermined}/<sample_id>.wav   509개
  dev/{contaminated,clean,undetermined}/<sample_id>.wav    258개
```

- 오디오: 16 kHz mono 24-bit WAV, 약 10초. 판별 모델에 넣은 파일 그대로다. 사람 재검수 때 들은 파일은 32 kHz 16-bit 재생본이었다(같은 stem).
- 클립은 test와 dev에 겹치지 않는다. 같은 클립의 stem(2~3개)은 서로 독립이 아니므로 평가·분할은 `clip_id` 단위로 한다.

## split별 라벨의 품질이 다르다

| split | stem | 라벨 근거 | 확정 |
|---|---:|---|---|
| test | 509 | 원래 평가자 1명 + 새 평가자 2명의 **3인 다수결** | 오염 247 / clean 234 / undetermined 28 |
| dev | 258 | 원래 평가자 **1명** (재검수 전, 잠정) | 오염 122 / clean 95 / undetermined 41 |

- test와 dev를 섞어서 "정답"으로 쓰지 않는다. dev는 더 노이즈가 많다.
- `undetermined`는 clean도 오염도 아니다. 학습·평가에서 빼고, 뺀 수를 같이 보고한다. negative로 쓰지 않는다.

## labels.csv 열

| 열 | 뜻 |
|---|---|
| `sample_id`, `split`, `clip_id`, `target` | stem ID, 분할, 클립, 분리 목표 소리 이름 |
| `label` | `contaminated` / `clean` / `undetermined` |
| `label_source` | test: `3-rater majority`, dev: `single rater (provisional)` |
| `votes_original` | (test) 원래 평가자의 오염 표. `yes`=오염, `no`=clean, 빈칸=표 없음. **원래 폼의 S3(목표 외 소리 혼입) 기준**이다. |
| `votes_new_1`, `votes_new_2` | (test) 새 평가자 2명의 Q2 답. `yes`=다른 종류 소리가 들림(오염) |
| `unanimous` | (test) 위 3표가 모두 있고 모두 같으면 `true` |
| `target_audible` | 목표 소리가 들리는가. `yes`/`no`/`undetermined`. test는 3인 다수결, dev는 1인 |
| `original_rater`, `new_rater_1`, `new_rater_2` | 평가자 코드 |
| `original_S3_types` | 원래 평가자가 적은 혼입 소리 유형(복수, `;` 구분) |
| `new_notes` | 새 평가자가 Q2=예일 때 적은 메모(`\|` 구분) |
| `review_status` | 원래 StemFoley 검수 결과(chosen / not_chosen / source_dropped 등) |
| `in_final_v3` | 원래 최종 수록 여부 |
| `audio_path`, `sha256` | 패키지 안 오디오 경로와 해시 |

## 사용 전에 꼭 볼 것

1. **`target_audible`과 `label`은 따로 본다.** 목표 소리가 안 들리는 stem(`target_audible=no`)도 `label=contaminated`일 수 있다. 오염이 아니라 "분리 실패"로 따로 다루려면 두 열을 같이 써야 한다.
2. **방향 주의:** 새 평가자의 Q1은 `예`=좋음(목표가 들림), Q2는 `예`=나쁨(다른 소리가 들림)이다. 이 파일에서는 `votes_*`가 Q2이고 `yes`=오염이다.
3. **사람 간 일치도의 한계:** 오염 판정의 Fleiss κ는 0.59다. 3인 다수결도 정답을 확정한 것이 아니다. 더 믿을 수 있는 부분은 `unanimous=true`다.
4. **다수결은 독립 정답이 아니다.** 원래 평가자의 표가 한 표로 들어간다.
5. **dev의 `contaminated`는 S3 또는 S4**(원본에 없던 소리) 기준이고, test의 `votes_original`은 S3 기준이다. 정의가 약간 다르다.
6. 자세한 해석과 수치는 팀 공유 문서(`report_team_share_20261015.md`)를 본다.
