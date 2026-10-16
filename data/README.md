# Experiment data

각 TSV 행은 `(clip_id, MID)` 한 쌍입니다. 두 파일 모두 `evaluation` split, 110개 caption/MID를 포함합니다.

## 분포

| 파일 | label | 행 수 | 고유 clip 수 | caption별 행 수 (최소 / 중앙값 / 평균 / 최대) |
|---|---|---:|---:|---:|
| `caption_positive_samples.tsv` | `yes` | 11,745 | 8,646 | 17 / 53 / 106.8 / 1,398 |
| `caption_negative_samples.tsv` | `no` | 13,262 | 7,496 | 40 / 123 / 120.6 / 192 |

전체 25,007행입니다. Negative는 해당 clip/class의 기록된 10개 프레임이 모두 `NOT_PRESENT`인 쌍만 포함합니다. TSV의 `positive_intervals`, `not_present_frames`, `overlapping_mids` 값은 JSON 문자열입니다.

## 공통 열

| 열 | 설명 |
|---|---|
| `split` | 데이터 split. 현재 모든 행은 `evaluation`. |
| `clip_id` | AudioSet segment ID (`video_id`와 segment 시작 시각으로 구성). |
| `video_id` | YouTube 동영상 ID. |
| `clip_start_seconds`, `clip_end_seconds` | 동영상에서 평가할 오디오 segment의 시작·끝 시각(초). |
| `caption` | 클래스의 영문 이름. |
| `caption_ko` | 클래스의 한국어 이름. |
| `mid` | AudioSet ontology의 클래스 ID. |
| `label` | 정답 레이블: positive 파일은 `yes`, negative 파일은 `no`. |
| `youtube_url` | segment 시작 위치로 이동하는 YouTube 링크. |

## Positive 파일 열

파일: `caption_positive_samples.tsv`

| 열 | 설명 |
|---|---|
| `positive_event_count` | 해당 clip/class의 strong positive 이벤트 수. |
| `positive_intervals` | 대상 클래스 positive 이벤트의 시작·끝 시각 목록(초). 각 원소는 `start`, `end`. |
| `overlap_event_count` | 대상 positive 이벤트 시간과 겹치는 다른 클래스의 주석 이벤트 수. |
| `overlap_class_count` | 겹친 다른 클래스의 고유 MID 수. |
| `max_simultaneous_other_class_count` | 대상 이벤트가 나는 동안 동시에 겹친 다른 클래스 수의 최댓값. |
| `overlapping_mids` | 겹친 다른 클래스의 MID 목록. |

## Negative 파일 열

파일: `caption_negative_samples.tsv`

| 열 | 설명 |
|---|---|
| `frame_count` | `NOT_PRESENT`로 확인된 프레임 수. 이 파일은 모든 행이 10. |
| `not_present_frames` | 프레임별 `start`, `end`, `status` 목록. 각 상태는 `NOT_PRESENT`. |

`ontology.json`은 MID와 AudioSet 클래스 이름을 연결하는 파일이며 TSV가 아닙니다.
