# StemFoley 감쇠 후 저에너지 구간 분석

476개 음원에 실제 평가와 동일하게 16kHz mono float32 변환 및 gain 적용. PCM 재양자화 없음. 20ms 비중첩 프레임 RMS, 연속 100ms 이상인 구간만 집계. 마지막 20ms 미만 꼬리는 구간 분석에서 제외.

완전히 0으로 새롭게 변한 샘플은 모든 gain에서 0개. 100ms 이상 연속 완전 0 구간도 0개. 아래는 디지털 RMS 기준이며 사람 또는 모델의 무음 지각 판정이 아님.

|Gain|−60dBFS 이하 구간을 포함한 음원|감쇠로 새로 −60dBFS 이하가 된 구간을 포함한 음원|새 구간 총 길이(초)|
|---:|---:|---:|---:|
|-30dB|456|441|1236.82|
|-20dB|410|370|563.98|
|-10dB|373|249|162.80|
|-5dB|363|134|40.44|

`intervals.tsv`: gain, sample_id, caption, label, threshold_dbfs, kind, start_s, end_s, duration_s, audio_path. `below`는 감쇠 후 기준 이하 전체, `newly_below`는 원본에서 기준 초과였지만 감쇠 후 기준 이하가 된 프레임으로만 이어진 구간. 두 집합은 중첩되며 합산하면 안 됨.

−80/−100dBFS 결과는 summary.json과 samples.tsv 참조.
