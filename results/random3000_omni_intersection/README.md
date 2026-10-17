# random3000 — Omni 모델 성공 샘플 교집합

원본 3,000개 중 **2,482개**를 선택했습니다. 제외: **518개**.

Audio Flamingo 3, Kimi Audio 7B, Qwen3 Omni 모두 `status=ok`이고 `prediction`이 `yes` 또는 `no`인 샘플의 `sample_id` 교집합입니다. 정답 여부는 선택 조건에 포함하지 않습니다.

원본은 `../random3000/`에 보존했습니다. CLAP/CNN14와 샘플 목록에도 동일한 교집합을 적용했으며, CLAP/CNN14의 실패 여부는 선택 조건이 아닙니다. Qwen TSV의 `nsample_id` 열 이름은 파생 파일에서 `sample_id`로 정정했습니다.

재생성: `python3 scripts/build_random3000_intersection.py`

# AudioSet caption presence evaluation results

Input examples: **2,482** (positive and complete-frame-negative evaluation clip/class pairs).

## LLM yes/no predictions

Metrics use `yes` as the positive class. Invalid outputs and failed inferences are reported separately and excluded from the metric denominator.

| Model | N scored | Invalid/failed | Accuracy | Precision (yes) | Recall (yes) | F1 (yes) | TP (yes→yes) | FN (yes→no) | FP (no→yes) | TN (no→no) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| audio_flamingo_3 | 2482 | 0 | 0.5471 | 0.5846 | 0.0985 | 0.1686 | 114 | 1043 | 81 | 1244 |
| kimi_audio_7b | 2482 | 0 | 0.6148 | 0.5539 | 0.8920 | 0.6834 | 1032 | 125 | 831 | 494 |
| qwen3_omni_instruct | 2482 | 0 | 0.5983 | 0.5512 | 0.7442 | 0.6333 | 861 | 296 | 701 | 624 |

## CLAPScore / CNN14 mean scores by reference label

CLAPScore is audio/text cosine similarity. CNN14 is the AudioSet class score for the mapped MID. Neither score is thresholded into a yes/no prediction.

Pooled mean is the mean over all scored samples; pooled SD is their standard deviation. Per-MID macro mean averages each MID's mean equally. Class-level tables below show yes/no means and their difference for each MID.

| Score | Reference label | N | MIDs scored | Pooled mean | Pooled SD | Per-MID macro mean |
|---|---|---:|---:|---:|---:|---:|
| clap | yes | 1157 | 109 | 0.202073 | 0.163519 | 0.272960 |
| clap | no | 1325 | 110 | 0.099775 | 0.147383 | 0.100136 |
| cnn14 | yes | 1157 | 109 | 0.179741 | 0.259838 | 0.292530 |
| cnn14 | no | 1325 | 110 | 0.030863 | 0.075301 | 0.031162 |

## CLAP scores by class

| MID | Caption | Yes N | Yes mean | No N | No mean | Yes minus no |
|---|---|---:|---:|---:|---:|---:|
| /m/07q2z82 | Accelerating, revving, vroom | 49 | 0.257662 | 10 | 0.131247 | 0.126415 |
| /m/0gvgw0 | Air brake | 5 | 0.203626 | 8 | 0.166824 | 0.036802 |
| /m/025wky1 | Air conditioning | 2 | 0.022552 | 12 | 0.142487 | -0.119935 |
| /m/05x_td | Air horn, truck horn | 10 | 0.299957 | 15 | 0.139597 | 0.160360 |
| /m/0k5j | Aircraft | 10 | 0.268368 | 10 | 0.070487 | 0.197882 |
| /m/014yck | Aircraft engine | 8 | 0.258627 | 5 | 0.135727 | 0.122900 |
| /m/07pp_mv | Alarm | 11 | 0.266435 | 2 | 0.106645 | 0.159790 |
| /m/046dlr | Alarm clock | 7 | 0.259940 | 18 | 0.190712 | 0.069229 |
| /m/028ght | Applause | 11 | 0.215300 | 7 | 0.004039 | 0.211261 |
| /m/0dl83 | Arrow | 5 | 0.078361 | 16 | 0.030212 | 0.048149 |
| /m/0_1c | Artillery fire | 10 | 0.406365 | 17 | 0.127346 | 0.279019 |
| /m/0261r1 | Babbling | 4 | 0.267702 | 13 | 0.154709 | 0.112993 |
| /t/dd00002 | Baby cry, infant cry | 10 | 0.213286 | 8 | 0.084530 | 0.128756 |
| /t/dd00001 | Baby laughter | 7 | 0.265228 | 7 | 0.086162 | 0.179066 |
| /m/05tny_ | Bark | 16 | 0.208409 | 16 | 0.030088 | 0.178321 |
| /m/018w8 | Basketball bounce | 3 | 0.277686 | 9 | 0.034454 | 0.243233 |
| /m/03dnzn | Bathtub (filling or washing) | 2 | 0.478292 | 10 | 0.149817 | 0.328475 |
| /m/04gy_2 | Battle cry | 2 | 0.341462 | 10 | 0.116573 | 0.224889 |
| /m/0395lw | Bell | 21 | 0.245336 | 3 | 0.133792 | 0.111544 |
| /m/07sq110 | Belly laugh | 5 | 0.231209 | 5 | 0.107007 | 0.124202 |
| /m/0199g | Bicycle | 2 | 0.322089 | 9 | 0.035841 | 0.286248 |
| /m/0gy1t2s | Bicycle bell | 4 | 0.434346 | 16 | 0.075480 | 0.358866 |
| /m/015p6 | Bird | 31 | 0.079120 | 10 | -0.010469 | 0.089588 |
| /m/05_wcq | Bird flight, flapping wings | 7 | 0.176252 | 9 | 0.016557 | 0.159695 |
| /m/020bb7 | Bird vocalization, bird call, bird song | 51 | 0.158734 | 9 | -0.030270 | 0.189003 |
| /m/07pdhp0 | Biting | 3 | 0.105923 | 7 | 0.103268 | 0.002655 |
| /m/07q0h5t | Bleat | 12 | 0.293052 | 12 | 0.066008 | 0.227044 |
| /m/02pjr4 | Blender | 4 | 0.381992 | 20 | 0.140190 | 0.241802 |
| /m/019jd | Boat, Water vehicle | 9 | 0.306670 | 19 | 0.056230 | 0.250440 |
| /m/0dv3j | Boiling | 2 | 0.144209 | 13 | -0.014009 | 0.158218 |
| /m/07pqn27 | Bouncing | 6 | 0.129942 | 12 | 0.051057 | 0.078885 |
| /m/07pc8lb | Breaking | 8 | 0.215673 | 7 | 0.131636 | 0.084037 |
| /m/03q5_w | Burping, eructation | 3 | 0.332643 | 16 | 0.019977 | 0.312666 |
| /m/01bjv | Bus | 3 | 0.236590 | 11 | 0.229581 | 0.007009 |
| /m/01z47d | Busy signal | 0 | NA | 7 | 0.202041 | NA |
| /m/0dv5r | Camera | 7 | 0.125653 | 14 | 0.090124 | 0.035529 |
| /m/073cg4 | Cap gun | 4 | 0.240042 | 14 | 0.104152 | 0.135890 |
| /m/0k4j | Car | 23 | 0.197064 | 13 | 0.040770 | 0.156295 |
| /m/02mfyn | Car alarm | 2 | 0.236645 | 10 | 0.193525 | 0.043120 |
| /t/dd00134 | Car passing by | 13 | 0.197245 | 24 | -0.007819 | 0.205064 |
| /m/024dl | Cash register | 4 | 0.298508 | 17 | 0.156390 | 0.142117 |
| /m/01yrx | Cat | 3 | 0.415941 | 12 | 0.022878 | 0.393063 |
| /m/07r81j2 | Caterwaul | 3 | 0.346006 | 17 | 0.013043 | 0.332962 |
| /m/01xq0k1 | Cattle, bovinae | 3 | 0.320779 | 7 | -0.006885 | 0.327664 |
| /m/01j4z9 | Chainsaw | 6 | 0.393948 | 12 | 0.055974 | 0.337974 |
| /m/0150b9 | Change ringing (campanology) | 5 | 0.225020 | 9 | 0.119171 | 0.105849 |
| /m/053hz1 | Cheering | 29 | 0.203073 | 11 | 0.142053 | 0.061020 |
| /m/03cczk | Chewing, mastication | 4 | 0.364678 | 12 | 0.063138 | 0.301540 |
| /m/09b5t | Chicken, rooster | 6 | 0.250504 | 12 | 0.002959 | 0.247545 |
| /t/dd00005 | Child singing | 14 | 0.238634 | 10 | 0.142615 | 0.096019 |
| /m/0ytgt | Child speech, kid speaking | 45 | 0.087572 | 8 | 0.048133 | 0.039440 |
| /t/dd00013 | Children playing | 3 | 0.410084 | 4 | 0.149086 | 0.260998 |
| /t/dd00135 | Children shouting | 5 | 0.246855 | 11 | 0.097297 | 0.149558 |
| /m/0hdsk | Chirp tone | 3 | 0.309724 | 16 | 0.152040 | 0.157684 |
| /m/07pn_8q | Chopping (food) | 2 | 0.267261 | 14 | 0.015516 | 0.251744 |
| /m/07m2kt | Chorus effect | 8 | 0.157470 | 9 | 0.119770 | 0.037699 |
| /m/03w41f | Church bell | 10 | 0.427974 | 12 | 0.150526 | 0.277447 |
| /m/0dgbq | Civil defense siren | 7 | 0.536644 | 15 | 0.208173 | 0.328470 |
| /m/0242l | Coin (dropping) | 3 | 0.350415 | 17 | 0.042814 | 0.307601 |
| /m/01m2v | Computer keyboard | 7 | 0.304422 | 12 | -0.017331 | 0.321753 |
| /m/015jpf | Dial tone | 1 | 0.154230 | 11 | 0.251770 | -0.097540 |
| /m/0fqfqc | Drawer open or close | 3 | 0.256767 | 19 | 0.112750 | 0.144017 |
| /m/04fgwm | Electric toothbrush | 2 | 0.155585 | 6 | 0.169270 | -0.013685 |
| /m/0b_fwt | Electronic tuner | 2 | 0.040948 | 4 | 0.365204 | -0.324256 |
| /m/03j1ly | Emergency vehicle | 2 | 0.435214 | 18 | 0.091067 | 0.344147 |
| /m/01h82_ | Engine knocking | 8 | 0.222520 | 11 | 0.104182 | 0.118338 |
| /t/dd00130 | Engine starting | 10 | 0.233154 | 6 | 0.030948 | 0.202206 |
| /m/06_y0by | Environmental noise | 5 | 0.096599 | 5 | -0.015924 | 0.112524 |
| /t/dd00004 | Female singing | 70 | 0.129193 | 8 | 0.118180 | 0.011013 |
| /m/02p01q | Filing (rasp) | 2 | 0.265165 | 19 | 0.155354 | 0.109811 |
| /m/07p7b8y | Fill (with liquid) | 6 | 0.291042 | 8 | -0.016520 | 0.307561 |
| /m/025_jnm | Finger snapping | 10 | 0.159101 | 10 | 0.006507 | 0.152594 |
| /m/0c3f7m | Fire alarm | 5 | 0.535661 | 11 | 0.221958 | 0.313702 |
| /m/0dxrf | Frying (food) | 11 | 0.336396 | 14 | 0.081733 | 0.254663 |
| /m/07rn7sz | Glass shatter | 16 | 0.282583 | 13 | 0.023607 | 0.258976 |
| /m/03wvsk | Hair dryer | 2 | 0.439275 | 19 | 0.119630 | 0.319645 |
| /m/04229 | Jet engine | 3 | 0.413062 | 17 | 0.120312 | 0.292750 |
| /m/027m70_ | Jingle bell | 1 | 0.506959 | 10 | 0.164180 | 0.342778 |
| /m/03v3yw | Keys jangling | 7 | 0.221390 | 13 | -0.005780 | 0.227170 |
| /m/01yg9g | Lawn mower | 5 | 0.454288 | 15 | 0.199715 | 0.254573 |
| /m/04zjc | Machine gun | 5 | 0.254829 | 13 | 0.107843 | 0.146985 |
| /t/dd00003 | Male singing | 166 | 0.070857 | 12 | 0.066611 | 0.004245 |
| /m/02x984l | Mechanical fan | 5 | 0.198390 | 12 | 0.085738 | 0.112651 |
| /m/012f08 | Motor vehicle (road) | 30 | 0.145246 | 7 | 0.053086 | 0.092159 |
| /m/04qvtq | Police car (siren) | 2 | 0.387091 | 17 | 0.186664 | 0.200427 |
| /m/0_ksk | Power tool | 22 | 0.245185 | 13 | 0.157046 | 0.088139 |
| /m/07ptzwd | Pump (liquid) | 5 | 0.111505 | 19 | 0.095979 | 0.015525 |
| /m/06d_3 | Rail transport | 6 | 0.272418 | 9 | 0.084820 | 0.187598 |
| /t/dd00038 | Rain on surface | 6 | 0.192411 | 10 | 0.078220 | 0.114191 |
| /m/02rhddq | Reversing beeps | 10 | 0.139361 | 15 | -0.049122 | 0.188483 |
| /m/0llzx | Sewing machine | 1 | 0.464410 | 14 | 0.127533 | 0.336877 |
| /m/05rj2 | Shuffling cards | 8 | 0.258204 | 15 | 0.018542 | 0.239662 |
| /m/0130jx | Sink (filling or washing) | 6 | 0.258684 | 11 | 0.087122 | 0.171562 |
| /m/02y_763 | Sliding door | 3 | 0.256830 | 12 | 0.046255 | 0.210575 |
| /m/06hck5 | Steam whistle | 7 | 0.526367 | 9 | 0.117490 | 0.408877 |
| /m/01g90h | Stomach rumble | 4 | 0.595313 | 6 | 0.156731 | 0.438583 |
| /m/07pp8cl | Telephone bell ringing | 4 | 0.247639 | 13 | 0.107932 | 0.139707 |
| /m/0dl9sf8 | Throat clearing | 4 | 0.148460 | 5 | 0.150718 | -0.002259 |
| /m/01jt3m | Toilet flush | 4 | 0.276059 | 9 | 0.167951 | 0.108108 |
| /m/0284vy3 | Train horn | 6 | 0.324037 | 27 | 0.134121 | 0.189916 |
| /t/dd00048 | Train wheels squealing | 5 | 0.429034 | 12 | 0.149662 | 0.279372 |
| /m/04zmvq | Train whistle | 10 | 0.390041 | 15 | 0.170742 | 0.219299 |
| /m/07n_g | Tuning fork | 4 | 0.442094 | 20 | 0.210835 | 0.231259 |
| /m/0d31p | Vacuum cleaner | 4 | 0.398561 | 21 | 0.178220 | 0.220341 |
| /m/032n05 | Whale vocalization | 3 | 0.371884 | 8 | 0.245291 | 0.126593 |
| /t/dd00136 | Whimper (dog) | 2 | 0.222872 | 7 | 0.088991 | 0.133881 |
| /m/01280g | Wild animals | 8 | 0.159208 | 13 | -0.057141 | 0.216349 |
| /m/026fgl | Wind chime | 4 | 0.336522 | 17 | 0.223175 | 0.113347 |
| /t/dd00092 | Wind noise (microphone) | 78 | 0.138653 | 20 | 0.149710 | -0.011058 |
| /m/01s0vc | Zipper (clothing) | 2 | 0.326141 | 17 | 0.062386 | 0.263756 |

## CNN14 scores by class

| MID | Caption | Yes N | Yes mean | No N | No mean | Yes minus no |
|---|---|---:|---:|---:|---:|---:|
| /m/07q2z82 | Accelerating, revving, vroom | 49 | 0.132464 | 10 | 0.036527 | 0.095936 |
| /m/0gvgw0 | Air brake | 5 | 0.256213 | 8 | 0.081746 | 0.174467 |
| /m/025wky1 | Air conditioning | 2 | 0.001555 | 12 | 0.014730 | -0.013176 |
| /m/05x_td | Air horn, truck horn | 10 | 0.323012 | 15 | 0.010362 | 0.312650 |
| /m/0k5j | Aircraft | 10 | 0.213753 | 10 | 0.054749 | 0.159004 |
| /m/014yck | Aircraft engine | 8 | 0.340806 | 5 | 0.043939 | 0.296867 |
| /m/07pp_mv | Alarm | 11 | 0.176979 | 2 | 0.005060 | 0.171920 |
| /m/046dlr | Alarm clock | 7 | 0.237486 | 18 | 0.025751 | 0.211736 |
| /m/028ght | Applause | 11 | 0.154087 | 7 | 0.001208 | 0.152879 |
| /m/0dl83 | Arrow | 5 | 0.075595 | 16 | 0.011393 | 0.064202 |
| /m/0_1c | Artillery fire | 10 | 0.459657 | 17 | 0.046168 | 0.413490 |
| /m/0261r1 | Babbling | 4 | 0.163779 | 13 | 0.036648 | 0.127131 |
| /t/dd00002 | Baby cry, infant cry | 10 | 0.261701 | 8 | 0.026487 | 0.235214 |
| /t/dd00001 | Baby laughter | 7 | 0.336067 | 7 | 0.069661 | 0.266406 |
| /m/05tny_ | Bark | 16 | 0.263796 | 16 | 0.017381 | 0.246415 |
| /m/018w8 | Basketball bounce | 3 | 0.431664 | 9 | 0.002381 | 0.429283 |
| /m/03dnzn | Bathtub (filling or washing) | 2 | 0.412699 | 10 | 0.043561 | 0.369138 |
| /m/04gy_2 | Battle cry | 2 | 0.830256 | 10 | 0.022374 | 0.807882 |
| /m/0395lw | Bell | 21 | 0.152006 | 3 | 0.002587 | 0.149419 |
| /m/07sq110 | Belly laugh | 5 | 0.131298 | 5 | 0.135505 | -0.004208 |
| /m/0199g | Bicycle | 2 | 0.036375 | 9 | 0.024493 | 0.011883 |
| /m/0gy1t2s | Bicycle bell | 4 | 0.712333 | 16 | 0.039173 | 0.673159 |
| /m/015p6 | Bird | 31 | 0.173969 | 10 | 0.036891 | 0.137078 |
| /m/05_wcq | Bird flight, flapping wings | 7 | 0.077698 | 9 | 0.023990 | 0.053709 |
| /m/020bb7 | Bird vocalization, bird call, bird song | 51 | 0.066396 | 9 | 0.024932 | 0.041464 |
| /m/07pdhp0 | Biting | 3 | 0.082732 | 7 | 0.009587 | 0.073145 |
| /m/07q0h5t | Bleat | 12 | 0.287522 | 12 | 0.023084 | 0.264437 |
| /m/02pjr4 | Blender | 4 | 0.512701 | 20 | 0.028227 | 0.484474 |
| /m/019jd | Boat, Water vehicle | 9 | 0.282458 | 19 | 0.055403 | 0.227055 |
| /m/0dv3j | Boiling | 2 | 0.005870 | 13 | 0.011868 | -0.005998 |
| /m/07pqn27 | Bouncing | 6 | 0.002548 | 12 | 0.000923 | 0.001625 |
| /m/07pc8lb | Breaking | 8 | 0.079593 | 7 | 0.005357 | 0.074235 |
| /m/03q5_w | Burping, eructation | 3 | 0.927078 | 16 | 0.082922 | 0.844155 |
| /m/01bjv | Bus | 3 | 0.057271 | 11 | 0.051677 | 0.005594 |
| /m/01z47d | Busy signal | 0 | NA | 7 | 0.000745 | NA |
| /m/0dv5r | Camera | 7 | 0.222060 | 14 | 0.013032 | 0.209028 |
| /m/073cg4 | Cap gun | 4 | 0.255395 | 14 | 0.028871 | 0.226524 |
| /m/0k4j | Car | 23 | 0.386777 | 13 | 0.170930 | 0.215847 |
| /m/02mfyn | Car alarm | 2 | 0.114096 | 10 | 0.013860 | 0.100236 |
| /t/dd00134 | Car passing by | 13 | 0.078533 | 24 | 0.008956 | 0.069577 |
| /m/024dl | Cash register | 4 | 0.461492 | 17 | 0.021627 | 0.439864 |
| /m/01yrx | Cat | 3 | 0.750505 | 12 | 0.023519 | 0.726986 |
| /m/07r81j2 | Caterwaul | 3 | 0.687058 | 17 | 0.021757 | 0.665301 |
| /m/01xq0k1 | Cattle, bovinae | 3 | 0.685940 | 7 | 0.003085 | 0.682855 |
| /m/01j4z9 | Chainsaw | 6 | 0.424594 | 12 | 0.011301 | 0.413293 |
| /m/0150b9 | Change ringing (campanology) | 5 | 0.418432 | 9 | 0.015785 | 0.402647 |
| /m/053hz1 | Cheering | 29 | 0.058861 | 11 | 0.022946 | 0.035915 |
| /m/03cczk | Chewing, mastication | 4 | 0.178762 | 12 | 0.046466 | 0.132296 |
| /m/09b5t | Chicken, rooster | 6 | 0.370930 | 12 | 0.032027 | 0.338904 |
| /t/dd00005 | Child singing | 14 | 0.140494 | 10 | 0.040994 | 0.099500 |
| /m/0ytgt | Child speech, kid speaking | 45 | 0.106771 | 8 | 0.050533 | 0.056238 |
| /t/dd00013 | Children playing | 3 | 0.292610 | 4 | 0.035214 | 0.257396 |
| /t/dd00135 | Children shouting | 5 | 0.164946 | 11 | 0.033406 | 0.131540 |
| /m/0hdsk | Chirp tone | 3 | 0.023550 | 16 | 0.015449 | 0.008102 |
| /m/07pn_8q | Chopping (food) | 2 | 0.140837 | 14 | 0.003559 | 0.137278 |
| /m/07m2kt | Chorus effect | 8 | 0.049900 | 9 | 0.021673 | 0.028227 |
| /m/03w41f | Church bell | 10 | 0.505746 | 12 | 0.037717 | 0.468029 |
| /m/0dgbq | Civil defense siren | 7 | 0.714461 | 15 | 0.005124 | 0.709337 |
| /m/0242l | Coin (dropping) | 3 | 0.156430 | 17 | 0.010060 | 0.146370 |
| /m/01m2v | Computer keyboard | 7 | 0.573393 | 12 | 0.013904 | 0.559489 |
| /m/015jpf | Dial tone | 1 | 0.216073 | 11 | 0.120106 | 0.095967 |
| /m/0fqfqc | Drawer open or close | 3 | 0.133554 | 19 | 0.106841 | 0.026713 |
| /m/04fgwm | Electric toothbrush | 2 | 0.025491 | 6 | 0.015758 | 0.009733 |
| /m/0b_fwt | Electronic tuner | 2 | 0.005461 | 4 | 0.082333 | -0.076872 |
| /m/03j1ly | Emergency vehicle | 2 | 0.700992 | 18 | 0.009469 | 0.691522 |
| /m/01h82_ | Engine knocking | 8 | 0.129592 | 11 | 0.026641 | 0.102950 |
| /t/dd00130 | Engine starting | 10 | 0.342910 | 6 | 0.017044 | 0.325866 |
| /m/06_y0by | Environmental noise | 5 | 0.004264 | 5 | 0.082925 | -0.078661 |
| /t/dd00004 | Female singing | 70 | 0.051654 | 8 | 0.052662 | -0.001009 |
| /m/02p01q | Filing (rasp) | 2 | 0.309050 | 19 | 0.034852 | 0.274197 |
| /m/07p7b8y | Fill (with liquid) | 6 | 0.118245 | 8 | 0.022337 | 0.095908 |
| /m/025_jnm | Finger snapping | 10 | 0.159839 | 10 | 0.001061 | 0.158778 |
| /m/0c3f7m | Fire alarm | 5 | 0.577182 | 11 | 0.010441 | 0.566741 |
| /m/0dxrf | Frying (food) | 11 | 0.572769 | 14 | 0.029884 | 0.542884 |
| /m/07rn7sz | Glass shatter | 16 | 0.273283 | 13 | 0.036746 | 0.236537 |
| /m/03wvsk | Hair dryer | 2 | 0.376649 | 19 | 0.014316 | 0.362333 |
| /m/04229 | Jet engine | 3 | 0.437544 | 17 | 0.069260 | 0.368284 |
| /m/027m70_ | Jingle bell | 1 | 0.791794 | 10 | 0.040406 | 0.751388 |
| /m/03v3yw | Keys jangling | 7 | 0.357008 | 13 | 0.007596 | 0.349412 |
| /m/01yg9g | Lawn mower | 5 | 0.132997 | 15 | 0.023541 | 0.109456 |
| /m/04zjc | Machine gun | 5 | 0.439547 | 13 | 0.053322 | 0.386225 |
| /t/dd00003 | Male singing | 166 | 0.015248 | 12 | 0.012249 | 0.002999 |
| /m/02x984l | Mechanical fan | 5 | 0.014525 | 12 | 0.006664 | 0.007861 |
| /m/012f08 | Motor vehicle (road) | 30 | 0.069464 | 7 | 0.034456 | 0.035008 |
| /m/04qvtq | Police car (siren) | 2 | 0.389350 | 17 | 0.053935 | 0.335415 |
| /m/0_ksk | Power tool | 22 | 0.099693 | 13 | 0.024070 | 0.075623 |
| /m/07ptzwd | Pump (liquid) | 5 | 0.072706 | 19 | 0.015685 | 0.057021 |
| /m/06d_3 | Rail transport | 6 | 0.604744 | 9 | 0.019347 | 0.585397 |
| /t/dd00038 | Rain on surface | 6 | 0.376081 | 10 | 0.026101 | 0.349980 |
| /m/02rhddq | Reversing beeps | 10 | 0.340960 | 15 | 0.019664 | 0.321296 |
| /m/0llzx | Sewing machine | 1 | 0.987897 | 14 | 0.037945 | 0.949952 |
| /m/05rj2 | Shuffling cards | 8 | 0.332545 | 15 | 0.003381 | 0.329164 |
| /m/0130jx | Sink (filling or washing) | 6 | 0.100605 | 11 | 0.056901 | 0.043704 |
| /m/02y_763 | Sliding door | 3 | 0.184532 | 12 | 0.017088 | 0.167444 |
| /m/06hck5 | Steam whistle | 7 | 0.556698 | 9 | 0.024093 | 0.532604 |
| /m/01g90h | Stomach rumble | 4 | 0.920385 | 6 | 0.012158 | 0.908227 |
| /m/07pp8cl | Telephone bell ringing | 4 | 0.487104 | 13 | 0.048890 | 0.438214 |
| /m/0dl9sf8 | Throat clearing | 4 | 0.130871 | 5 | 0.042364 | 0.088508 |
| /m/01jt3m | Toilet flush | 4 | 0.537859 | 9 | 0.026766 | 0.511092 |
| /m/0284vy3 | Train horn | 6 | 0.517460 | 27 | 0.050315 | 0.467146 |
| /t/dd00048 | Train wheels squealing | 5 | 0.544241 | 12 | 0.025119 | 0.519121 |
| /m/04zmvq | Train whistle | 10 | 0.274395 | 15 | 0.062811 | 0.211584 |
| /m/07n_g | Tuning fork | 4 | 0.467826 | 20 | 0.010279 | 0.457546 |
| /m/0d31p | Vacuum cleaner | 4 | 0.102855 | 21 | 0.033646 | 0.069209 |
| /m/032n05 | Whale vocalization | 3 | 0.214410 | 8 | 0.018079 | 0.196330 |
| /t/dd00136 | Whimper (dog) | 2 | 0.107340 | 7 | 0.017728 | 0.089612 |
| /m/01280g | Wild animals | 8 | 0.021553 | 13 | 0.004520 | 0.017034 |
| /m/026fgl | Wind chime | 4 | 0.237607 | 17 | 0.028543 | 0.209065 |
| /t/dd00092 | Wind noise (microphone) | 78 | 0.044059 | 20 | 0.021804 | 0.022255 |
| /m/01s0vc | Zipper (clothing) | 2 | 0.354868 | 17 | 0.006422 | 0.348447 |

## Notes

- `yes`/`no` are the reference labels from the existing evaluation TSVs; each row is one clip/class pair.
- Sample duration follows the source TSV. Some positive clips are shorter than 10 seconds; no padding is applied to model input audio.
- Failed downloads/inferences and unmapped CNN14 classes remain visible in result TSVs and are excluded from score means where no score exists.
- CLAPScore values are similarity scores, not calibrated probabilities.
