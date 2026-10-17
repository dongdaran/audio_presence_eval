# AudioSet caption presence evaluation results

Input examples: **3,000** (positive and complete-frame-negative evaluation clip/class pairs).

## LLM yes/no predictions

Metrics use `yes` as the positive class. Invalid outputs and failed inferences are reported separately and excluded from the metric denominator.

| Model | N scored | Invalid/failed | Accuracy | Precision (yes) | Recall (yes) | F1 (yes) | TP (yes→yes) | FN (yes→no) | FP (no→yes) | TN (no→no) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| audio_flamingo_3 | 2558 | 442 | 0.5442 | 0.5771 | 0.0969 | 0.1660 | 116 | 1081 | 85 | 1276 |
| qwen3_omni_instruct | 2643 | 357 | 0.5978 | 0.5526 | 0.7431 | 0.6338 | 920 | 318 | 745 | 660 |

## Configured model IDs

- `gemini_3_pro`: `google/gemini-3.1-pro-preview` (max new tokens: `4096`)
- `gemini_argon`: `google/gemini-4-argon` (max new tokens: `4096`)
- `kimi_audio_7b`: `moonshotai/Kimi-Audio-7B-Instruct` (max new tokens: `1`)
- `audio_flamingo_3`: `nvidia/audio-flamingo-3-hf` (max new tokens: `1`)
- `qwen3_omni_instruct`: `Qwen/Qwen3-Omni-30B-A3B-Instruct` (max new tokens: `1`)

- Maximum new tokens: `1`
- Score batch size: `16`
- LLM Transformers micro-batch size: `128`
- Requested GPU: `NVIDIA A100 80GB`

## CLAPScore / CNN14 mean scores by reference label

CLAPScore is audio/text cosine similarity. CNN14 is the AudioSet class score for the mapped MID. Neither score is thresholded into a yes/no prediction.

Pooled mean is the mean over all scored samples; pooled SD is their standard deviation. Per-MID macro mean averages each MID's mean equally. Class-level tables below show yes/no means and their difference for each MID.

| Score | Reference label | N | MIDs scored | Pooled mean | Pooled SD | Per-MID macro mean |
|---|---|---:|---:|---:|---:|---:|
| clap | yes | 1198 | 109 | 0.201424 | 0.163592 | 0.271537 |
| clap | no | 1357 | 110 | 0.099562 | 0.147713 | 0.100204 |
| cnn14 | yes | 1197 | 109 | 0.178156 | 0.258317 | 0.286946 |
| cnn14 | no | 1356 | 110 | 0.031201 | 0.078559 | 0.031525 |

## CLAP scores by class

| MID | Caption | Yes N | Yes mean | No N | No mean | Yes minus no |
|---|---|---:|---:|---:|---:|---:|
| /m/07q2z82 | Accelerating, revving, vroom | 52 | 0.264269 | 10 | 0.131247 | 0.133021 |
| /m/0gvgw0 | Air brake | 5 | 0.203626 | 8 | 0.166824 | 0.036802 |
| /m/025wky1 | Air conditioning | 2 | 0.022552 | 12 | 0.142487 | -0.119935 |
| /m/05x_td | Air horn, truck horn | 11 | 0.282625 | 17 | 0.135008 | 0.147617 |
| /m/0k5j | Aircraft | 10 | 0.268368 | 10 | 0.070487 | 0.197882 |
| /m/014yck | Aircraft engine | 8 | 0.258627 | 6 | 0.153580 | 0.105046 |
| /m/07pp_mv | Alarm | 11 | 0.266435 | 2 | 0.106645 | 0.159790 |
| /m/046dlr | Alarm clock | 7 | 0.259940 | 18 | 0.190712 | 0.069229 |
| /m/028ght | Applause | 11 | 0.215300 | 8 | 0.029864 | 0.185436 |
| /m/0dl83 | Arrow | 5 | 0.078361 | 16 | 0.030212 | 0.048149 |
| /m/0_1c | Artillery fire | 10 | 0.406365 | 18 | 0.125713 | 0.280653 |
| /m/0261r1 | Babbling | 5 | 0.280038 | 14 | 0.155324 | 0.124714 |
| /t/dd00002 | Baby cry, infant cry | 10 | 0.213286 | 8 | 0.084530 | 0.128756 |
| /t/dd00001 | Baby laughter | 7 | 0.265228 | 8 | 0.105439 | 0.159789 |
| /m/05tny_ | Bark | 20 | 0.180995 | 16 | 0.030088 | 0.150907 |
| /m/018w8 | Basketball bounce | 3 | 0.277686 | 9 | 0.034454 | 0.243233 |
| /m/03dnzn | Bathtub (filling or washing) | 2 | 0.478292 | 10 | 0.149817 | 0.328475 |
| /m/04gy_2 | Battle cry | 2 | 0.341462 | 10 | 0.116573 | 0.224889 |
| /m/0395lw | Bell | 21 | 0.245336 | 3 | 0.133792 | 0.111544 |
| /m/07sq110 | Belly laugh | 5 | 0.231209 | 5 | 0.107007 | 0.124202 |
| /m/0199g | Bicycle | 2 | 0.322089 | 9 | 0.035841 | 0.286248 |
| /m/0gy1t2s | Bicycle bell | 4 | 0.434346 | 16 | 0.075480 | 0.358866 |
| /m/015p6 | Bird | 33 | 0.086610 | 10 | -0.010469 | 0.097079 |
| /m/05_wcq | Bird flight, flapping wings | 7 | 0.176252 | 9 | 0.016557 | 0.159695 |
| /m/020bb7 | Bird vocalization, bird call, bird song | 54 | 0.156729 | 9 | -0.030270 | 0.186998 |
| /m/07pdhp0 | Biting | 3 | 0.105923 | 7 | 0.103268 | 0.002655 |
| /m/07q0h5t | Bleat | 13 | 0.277429 | 13 | 0.066586 | 0.210842 |
| /m/02pjr4 | Blender | 6 | 0.327818 | 20 | 0.140190 | 0.187628 |
| /m/019jd | Boat, Water vehicle | 9 | 0.306670 | 19 | 0.056230 | 0.250440 |
| /m/0dv3j | Boiling | 2 | 0.144209 | 13 | -0.014009 | 0.158218 |
| /m/07pqn27 | Bouncing | 6 | 0.129942 | 14 | 0.043741 | 0.086201 |
| /m/07pc8lb | Breaking | 9 | 0.225491 | 7 | 0.131636 | 0.093855 |
| /m/03q5_w | Burping, eructation | 3 | 0.332643 | 16 | 0.019977 | 0.312666 |
| /m/01bjv | Bus | 3 | 0.236590 | 11 | 0.229581 | 0.007009 |
| /m/01z47d | Busy signal | 0 | NA | 7 | 0.202041 | NA |
| /m/0dv5r | Camera | 7 | 0.125653 | 14 | 0.090124 | 0.035529 |
| /m/073cg4 | Cap gun | 4 | 0.240042 | 14 | 0.104152 | 0.135890 |
| /m/0k4j | Car | 23 | 0.197064 | 14 | 0.039144 | 0.157920 |
| /m/02mfyn | Car alarm | 2 | 0.236645 | 11 | 0.168015 | 0.068630 |
| /t/dd00134 | Car passing by | 13 | 0.197245 | 25 | -0.014506 | 0.211751 |
| /m/024dl | Cash register | 5 | 0.309999 | 17 | 0.156390 | 0.153609 |
| /m/01yrx | Cat | 3 | 0.415941 | 13 | 0.031120 | 0.384822 |
| /m/07r81j2 | Caterwaul | 3 | 0.346006 | 17 | 0.013043 | 0.332962 |
| /m/01xq0k1 | Cattle, bovinae | 3 | 0.320779 | 8 | 0.005781 | 0.314998 |
| /m/01j4z9 | Chainsaw | 6 | 0.393948 | 13 | 0.077404 | 0.316545 |
| /m/0150b9 | Change ringing (campanology) | 5 | 0.225020 | 10 | 0.128683 | 0.096337 |
| /m/053hz1 | Cheering | 31 | 0.202605 | 11 | 0.142053 | 0.060552 |
| /m/03cczk | Chewing, mastication | 4 | 0.364678 | 12 | 0.063138 | 0.301540 |
| /m/09b5t | Chicken, rooster | 6 | 0.250504 | 13 | 0.008774 | 0.241730 |
| /t/dd00005 | Child singing | 14 | 0.238634 | 10 | 0.142615 | 0.096019 |
| /m/0ytgt | Child speech, kid speaking | 47 | 0.088892 | 9 | 0.042356 | 0.046536 |
| /t/dd00013 | Children playing | 3 | 0.410084 | 4 | 0.149086 | 0.260998 |
| /t/dd00135 | Children shouting | 8 | 0.231251 | 11 | 0.097297 | 0.133955 |
| /m/0hdsk | Chirp tone | 3 | 0.309724 | 16 | 0.152040 | 0.157684 |
| /m/07pn_8q | Chopping (food) | 2 | 0.267261 | 14 | 0.015516 | 0.251744 |
| /m/07m2kt | Chorus effect | 8 | 0.157470 | 9 | 0.119770 | 0.037699 |
| /m/03w41f | Church bell | 11 | 0.434041 | 12 | 0.150526 | 0.283515 |
| /m/0dgbq | Civil defense siren | 8 | 0.534973 | 16 | 0.197527 | 0.337446 |
| /m/0242l | Coin (dropping) | 3 | 0.350415 | 17 | 0.042814 | 0.307601 |
| /m/01m2v | Computer keyboard | 7 | 0.304422 | 12 | -0.017331 | 0.321753 |
| /m/015jpf | Dial tone | 1 | 0.154230 | 11 | 0.251770 | -0.097540 |
| /m/0fqfqc | Drawer open or close | 3 | 0.256767 | 20 | 0.104638 | 0.152129 |
| /m/04fgwm | Electric toothbrush | 2 | 0.155585 | 7 | 0.153682 | 0.001903 |
| /m/0b_fwt | Electronic tuner | 2 | 0.040948 | 4 | 0.365204 | -0.324256 |
| /m/03j1ly | Emergency vehicle | 2 | 0.435214 | 18 | 0.091067 | 0.344147 |
| /m/01h82_ | Engine knocking | 8 | 0.222520 | 11 | 0.104182 | 0.118338 |
| /t/dd00130 | Engine starting | 10 | 0.233154 | 6 | 0.030948 | 0.202206 |
| /m/06_y0by | Environmental noise | 5 | 0.096599 | 5 | -0.015924 | 0.112524 |
| /t/dd00004 | Female singing | 72 | 0.122999 | 8 | 0.118180 | 0.004819 |
| /m/02p01q | Filing (rasp) | 2 | 0.265165 | 19 | 0.155354 | 0.109811 |
| /m/07p7b8y | Fill (with liquid) | 6 | 0.291042 | 8 | -0.016520 | 0.307561 |
| /m/025_jnm | Finger snapping | 11 | 0.158597 | 10 | 0.006507 | 0.152091 |
| /m/0c3f7m | Fire alarm | 5 | 0.535661 | 12 | 0.201412 | 0.334249 |
| /m/0dxrf | Frying (food) | 11 | 0.336396 | 14 | 0.081733 | 0.254663 |
| /m/07rn7sz | Glass shatter | 16 | 0.282583 | 13 | 0.023607 | 0.258976 |
| /m/03wvsk | Hair dryer | 2 | 0.439275 | 19 | 0.119630 | 0.319645 |
| /m/04229 | Jet engine | 3 | 0.413062 | 17 | 0.120312 | 0.292750 |
| /m/027m70_ | Jingle bell | 2 | 0.428930 | 11 | 0.168158 | 0.260772 |
| /m/03v3yw | Keys jangling | 7 | 0.221390 | 13 | -0.005780 | 0.227170 |
| /m/01yg9g | Lawn mower | 5 | 0.454288 | 15 | 0.199715 | 0.254573 |
| /m/04zjc | Machine gun | 5 | 0.254829 | 13 | 0.107843 | 0.146985 |
| /t/dd00003 | Male singing | 168 | 0.071726 | 13 | 0.075506 | -0.003780 |
| /m/02x984l | Mechanical fan | 5 | 0.198390 | 12 | 0.085738 | 0.112651 |
| /m/012f08 | Motor vehicle (road) | 31 | 0.143272 | 7 | 0.053086 | 0.090186 |
| /m/04qvtq | Police car (siren) | 2 | 0.387091 | 17 | 0.186664 | 0.200427 |
| /m/0_ksk | Power tool | 22 | 0.245185 | 14 | 0.158290 | 0.086895 |
| /m/07ptzwd | Pump (liquid) | 5 | 0.111505 | 19 | 0.095979 | 0.015525 |
| /m/06d_3 | Rail transport | 6 | 0.272418 | 10 | 0.091710 | 0.180708 |
| /t/dd00038 | Rain on surface | 6 | 0.192411 | 10 | 0.078220 | 0.114191 |
| /m/02rhddq | Reversing beeps | 10 | 0.139361 | 15 | -0.049122 | 0.188483 |
| /m/0llzx | Sewing machine | 1 | 0.464410 | 14 | 0.127533 | 0.336877 |
| /m/05rj2 | Shuffling cards | 9 | 0.278057 | 16 | 0.012740 | 0.265316 |
| /m/0130jx | Sink (filling or washing) | 6 | 0.258684 | 11 | 0.087122 | 0.171562 |
| /m/02y_763 | Sliding door | 3 | 0.256830 | 13 | 0.018601 | 0.238229 |
| /m/06hck5 | Steam whistle | 7 | 0.526367 | 9 | 0.117490 | 0.408877 |
| /m/01g90h | Stomach rumble | 4 | 0.595313 | 6 | 0.156731 | 0.438583 |
| /m/07pp8cl | Telephone bell ringing | 5 | 0.258351 | 13 | 0.107932 | 0.150419 |
| /m/0dl9sf8 | Throat clearing | 4 | 0.148460 | 5 | 0.150718 | -0.002259 |
| /m/01jt3m | Toilet flush | 4 | 0.276059 | 9 | 0.167951 | 0.108108 |
| /m/0284vy3 | Train horn | 6 | 0.324037 | 27 | 0.134121 | 0.189916 |
| /t/dd00048 | Train wheels squealing | 5 | 0.429034 | 12 | 0.149662 | 0.279372 |
| /m/04zmvq | Train whistle | 10 | 0.390041 | 17 | 0.158406 | 0.231635 |
| /m/07n_g | Tuning fork | 4 | 0.442094 | 20 | 0.210835 | 0.231259 |
| /m/0d31p | Vacuum cleaner | 4 | 0.398561 | 22 | 0.196636 | 0.201926 |
| /m/032n05 | Whale vocalization | 3 | 0.371884 | 8 | 0.245291 | 0.126593 |
| /t/dd00136 | Whimper (dog) | 2 | 0.222872 | 7 | 0.088991 | 0.133881 |
| /m/01280g | Wild animals | 9 | 0.143691 | 13 | -0.057141 | 0.200831 |
| /m/026fgl | Wind chime | 4 | 0.336522 | 17 | 0.223175 | 0.113347 |
| /t/dd00092 | Wind noise (microphone) | 81 | 0.133464 | 20 | 0.149710 | -0.016247 |
| /m/01s0vc | Zipper (clothing) | 2 | 0.326141 | 17 | 0.062386 | 0.263756 |

## CNN14 scores by class

| MID | Caption | Yes N | Yes mean | No N | No mean | Yes minus no |
|---|---|---:|---:|---:|---:|---:|
| /m/07q2z82 | Accelerating, revving, vroom | 52 | 0.131460 | 10 | 0.036527 | 0.094932 |
| /m/0gvgw0 | Air brake | 5 | 0.256213 | 8 | 0.081746 | 0.174467 |
| /m/025wky1 | Air conditioning | 2 | 0.001555 | 12 | 0.014730 | -0.013176 |
| /m/05x_td | Air horn, truck horn | 11 | 0.295431 | 17 | 0.011598 | 0.283833 |
| /m/0k5j | Aircraft | 10 | 0.213753 | 10 | 0.054749 | 0.159004 |
| /m/014yck | Aircraft engine | 8 | 0.340806 | 6 | 0.037180 | 0.303626 |
| /m/07pp_mv | Alarm | 11 | 0.176979 | 2 | 0.005060 | 0.171920 |
| /m/046dlr | Alarm clock | 7 | 0.237486 | 18 | 0.025751 | 0.211736 |
| /m/028ght | Applause | 11 | 0.154087 | 8 | 0.001280 | 0.152807 |
| /m/0dl83 | Arrow | 5 | 0.075595 | 16 | 0.011393 | 0.064202 |
| /m/0_1c | Artillery fire | 10 | 0.459657 | 18 | 0.044064 | 0.415593 |
| /m/0261r1 | Babbling | 5 | 0.199630 | 14 | 0.041832 | 0.157799 |
| /t/dd00002 | Baby cry, infant cry | 10 | 0.261701 | 8 | 0.026487 | 0.235214 |
| /t/dd00001 | Baby laughter | 7 | 0.336067 | 7 | 0.069661 | 0.266406 |
| /m/05tny_ | Bark | 20 | 0.235281 | 16 | 0.017381 | 0.217900 |
| /m/018w8 | Basketball bounce | 3 | 0.431664 | 9 | 0.002381 | 0.429283 |
| /m/03dnzn | Bathtub (filling or washing) | 2 | 0.412699 | 10 | 0.043561 | 0.369138 |
| /m/04gy_2 | Battle cry | 2 | 0.830256 | 10 | 0.022374 | 0.807882 |
| /m/0395lw | Bell | 21 | 0.152006 | 3 | 0.002587 | 0.149419 |
| /m/07sq110 | Belly laugh | 5 | 0.131298 | 5 | 0.135505 | -0.004208 |
| /m/0199g | Bicycle | 2 | 0.036375 | 9 | 0.024493 | 0.011883 |
| /m/0gy1t2s | Bicycle bell | 4 | 0.712333 | 16 | 0.039173 | 0.673159 |
| /m/015p6 | Bird | 33 | 0.166690 | 10 | 0.036891 | 0.129799 |
| /m/05_wcq | Bird flight, flapping wings | 7 | 0.077698 | 9 | 0.023990 | 0.053709 |
| /m/020bb7 | Bird vocalization, bird call, bird song | 54 | 0.063661 | 9 | 0.024932 | 0.038729 |
| /m/07pdhp0 | Biting | 3 | 0.082732 | 7 | 0.009587 | 0.073145 |
| /m/07q0h5t | Bleat | 13 | 0.267799 | 13 | 0.022409 | 0.245389 |
| /m/02pjr4 | Blender | 6 | 0.362631 | 20 | 0.028227 | 0.334404 |
| /m/019jd | Boat, Water vehicle | 9 | 0.282458 | 19 | 0.055403 | 0.227055 |
| /m/0dv3j | Boiling | 2 | 0.005870 | 13 | 0.011868 | -0.005998 |
| /m/07pqn27 | Bouncing | 6 | 0.002548 | 14 | 0.000886 | 0.001662 |
| /m/07pc8lb | Breaking | 9 | 0.073823 | 7 | 0.005357 | 0.068466 |
| /m/03q5_w | Burping, eructation | 3 | 0.927078 | 16 | 0.082922 | 0.844155 |
| /m/01bjv | Bus | 3 | 0.057271 | 11 | 0.051677 | 0.005594 |
| /m/01z47d | Busy signal | 0 | NA | 7 | 0.000745 | NA |
| /m/0dv5r | Camera | 7 | 0.222060 | 15 | 0.074717 | 0.147343 |
| /m/073cg4 | Cap gun | 4 | 0.255395 | 14 | 0.028871 | 0.226524 |
| /m/0k4j | Car | 23 | 0.386777 | 13 | 0.170930 | 0.215847 |
| /m/02mfyn | Car alarm | 2 | 0.114096 | 11 | 0.012654 | 0.101442 |
| /t/dd00134 | Car passing by | 13 | 0.078533 | 25 | 0.008601 | 0.069932 |
| /m/024dl | Cash register | 5 | 0.430630 | 17 | 0.021627 | 0.409003 |
| /m/01yrx | Cat | 3 | 0.750505 | 13 | 0.022535 | 0.727970 |
| /m/07r81j2 | Caterwaul | 3 | 0.687058 | 17 | 0.021757 | 0.665301 |
| /m/01xq0k1 | Cattle, bovinae | 3 | 0.685940 | 8 | 0.002759 | 0.683181 |
| /m/01j4z9 | Chainsaw | 6 | 0.424594 | 13 | 0.010542 | 0.414053 |
| /m/0150b9 | Change ringing (campanology) | 5 | 0.418432 | 10 | 0.014266 | 0.404166 |
| /m/053hz1 | Cheering | 31 | 0.057739 | 11 | 0.022946 | 0.034793 |
| /m/03cczk | Chewing, mastication | 4 | 0.178762 | 12 | 0.046466 | 0.132296 |
| /m/09b5t | Chicken, rooster | 6 | 0.370930 | 13 | 0.029664 | 0.341267 |
| /t/dd00005 | Child singing | 14 | 0.140494 | 10 | 0.040994 | 0.099500 |
| /m/0ytgt | Child speech, kid speaking | 47 | 0.111102 | 9 | 0.054993 | 0.056109 |
| /t/dd00013 | Children playing | 3 | 0.292610 | 4 | 0.035214 | 0.257396 |
| /t/dd00135 | Children shouting | 7 | 0.177315 | 11 | 0.033406 | 0.143909 |
| /m/0hdsk | Chirp tone | 3 | 0.023550 | 16 | 0.015449 | 0.008102 |
| /m/07pn_8q | Chopping (food) | 2 | 0.140837 | 14 | 0.003559 | 0.137278 |
| /m/07m2kt | Chorus effect | 8 | 0.049900 | 9 | 0.021673 | 0.028227 |
| /m/03w41f | Church bell | 11 | 0.485993 | 12 | 0.037717 | 0.448276 |
| /m/0dgbq | Civil defense siren | 8 | 0.718092 | 16 | 0.004875 | 0.713217 |
| /m/0242l | Coin (dropping) | 3 | 0.156430 | 17 | 0.010060 | 0.146370 |
| /m/01m2v | Computer keyboard | 7 | 0.573393 | 12 | 0.013904 | 0.559489 |
| /m/015jpf | Dial tone | 1 | 0.216073 | 11 | 0.120106 | 0.095967 |
| /m/0fqfqc | Drawer open or close | 3 | 0.133554 | 20 | 0.101506 | 0.032048 |
| /m/04fgwm | Electric toothbrush | 2 | 0.025491 | 7 | 0.013790 | 0.011702 |
| /m/0b_fwt | Electronic tuner | 2 | 0.005461 | 4 | 0.082333 | -0.076872 |
| /m/03j1ly | Emergency vehicle | 2 | 0.700992 | 18 | 0.009469 | 0.691522 |
| /m/01h82_ | Engine knocking | 8 | 0.129592 | 11 | 0.026641 | 0.102950 |
| /t/dd00130 | Engine starting | 10 | 0.342910 | 6 | 0.017044 | 0.325866 |
| /m/06_y0by | Environmental noise | 5 | 0.004264 | 5 | 0.082925 | -0.078661 |
| /t/dd00004 | Female singing | 72 | 0.050244 | 8 | 0.052662 | -0.002418 |
| /m/02p01q | Filing (rasp) | 2 | 0.309050 | 19 | 0.034852 | 0.274197 |
| /m/07p7b8y | Fill (with liquid) | 6 | 0.118245 | 8 | 0.022337 | 0.095908 |
| /m/025_jnm | Finger snapping | 11 | 0.194205 | 10 | 0.001061 | 0.193145 |
| /m/0c3f7m | Fire alarm | 5 | 0.577182 | 12 | 0.009614 | 0.567568 |
| /m/0dxrf | Frying (food) | 11 | 0.572769 | 14 | 0.029884 | 0.542884 |
| /m/07rn7sz | Glass shatter | 16 | 0.273283 | 13 | 0.036746 | 0.236537 |
| /m/03wvsk | Hair dryer | 2 | 0.376649 | 19 | 0.014316 | 0.362333 |
| /m/04229 | Jet engine | 3 | 0.437544 | 17 | 0.069260 | 0.368284 |
| /m/027m70_ | Jingle bell | 2 | 0.430940 | 11 | 0.037192 | 0.393748 |
| /m/03v3yw | Keys jangling | 7 | 0.357008 | 13 | 0.007596 | 0.349412 |
| /m/01yg9g | Lawn mower | 5 | 0.132997 | 15 | 0.023541 | 0.109456 |
| /m/04zjc | Machine gun | 5 | 0.439547 | 13 | 0.053322 | 0.386225 |
| /t/dd00003 | Male singing | 168 | 0.015213 | 13 | 0.012134 | 0.003079 |
| /m/02x984l | Mechanical fan | 5 | 0.014525 | 12 | 0.006664 | 0.007861 |
| /m/012f08 | Motor vehicle (road) | 31 | 0.067656 | 7 | 0.034456 | 0.033200 |
| /m/04qvtq | Police car (siren) | 2 | 0.389350 | 17 | 0.053935 | 0.335415 |
| /m/0_ksk | Power tool | 22 | 0.099693 | 14 | 0.023248 | 0.076445 |
| /m/07ptzwd | Pump (liquid) | 5 | 0.072706 | 19 | 0.015685 | 0.057021 |
| /m/06d_3 | Rail transport | 6 | 0.604744 | 10 | 0.019293 | 0.585451 |
| /t/dd00038 | Rain on surface | 6 | 0.376081 | 10 | 0.026101 | 0.349980 |
| /m/02rhddq | Reversing beeps | 10 | 0.340960 | 15 | 0.019664 | 0.321296 |
| /m/0llzx | Sewing machine | 1 | 0.987897 | 14 | 0.037945 | 0.949952 |
| /m/05rj2 | Shuffling cards | 9 | 0.385521 | 16 | 0.006789 | 0.378733 |
| /m/0130jx | Sink (filling or washing) | 6 | 0.100605 | 11 | 0.056901 | 0.043704 |
| /m/02y_763 | Sliding door | 3 | 0.184532 | 13 | 0.015793 | 0.168739 |
| /m/06hck5 | Steam whistle | 7 | 0.556698 | 9 | 0.024093 | 0.532604 |
| /m/01g90h | Stomach rumble | 4 | 0.920385 | 6 | 0.012158 | 0.908227 |
| /m/07pp8cl | Telephone bell ringing | 5 | 0.397378 | 13 | 0.048890 | 0.348488 |
| /m/0dl9sf8 | Throat clearing | 4 | 0.130871 | 5 | 0.042364 | 0.088508 |
| /m/01jt3m | Toilet flush | 4 | 0.537859 | 9 | 0.026766 | 0.511092 |
| /m/0284vy3 | Train horn | 6 | 0.517460 | 27 | 0.050315 | 0.467146 |
| /t/dd00048 | Train wheels squealing | 5 | 0.544241 | 12 | 0.025119 | 0.519121 |
| /m/04zmvq | Train whistle | 10 | 0.274395 | 17 | 0.055445 | 0.218950 |
| /m/07n_g | Tuning fork | 4 | 0.467826 | 20 | 0.010279 | 0.457546 |
| /m/0d31p | Vacuum cleaner | 4 | 0.102855 | 22 | 0.035810 | 0.067045 |
| /m/032n05 | Whale vocalization | 3 | 0.214410 | 8 | 0.018079 | 0.196330 |
| /t/dd00136 | Whimper (dog) | 2 | 0.107340 | 7 | 0.017728 | 0.089612 |
| /m/01280g | Wild animals | 9 | 0.019168 | 13 | 0.004520 | 0.014649 |
| /m/026fgl | Wind chime | 4 | 0.237607 | 17 | 0.028543 | 0.209065 |
| /t/dd00092 | Wind noise (microphone) | 81 | 0.042487 | 20 | 0.021804 | 0.020684 |
| /m/01s0vc | Zipper (clothing) | 2 | 0.354868 | 17 | 0.006422 | 0.348447 |

## Notes

- `yes`/`no` are the reference labels from the existing evaluation TSVs; each row is one clip/class pair.
- Sample duration follows the source TSV. Some positive clips are shorter than 10 seconds; no padding is applied to model input audio.
- Failed downloads/inferences and unmapped CNN14 classes remain visible in result TSVs and are excluded from score means where no score exists.
- CLAPScore values are similarity scores, not calibrated probabilities.
