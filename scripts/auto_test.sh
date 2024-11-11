#!/bin/bash
scripts_dir=$(dirname $(readlink -f "$0"))
top_dir=$scripts_dir/../
pushd $top_dir

#default config
TARGET="BM1684X"
MODE="pcie_test"
TPUID=0
ALL_PASS=1
PYTEST="auto_test"
ECHO_LINES=20
if [ -f "tools/benchmark.txt" ]; then
  rm tools/benchmark.txt
fi

usage() 
{
  echo "Usage: $0 [ -m MODE compile_mlir|pcie_test|soc_test] [ -t TARGET BM1684X] [ -s SOCSDK] [ -d TPUID] [ -p PYTEST auto_test|pytest]" 1>&2 
}

while getopts ":m:t:s:d:p:" opt
do
  case $opt in 
    m)
      MODE=${OPTARG}
      echo "mode is $MODE";;
    t)
      TARGET=${OPTARG}
      echo "target is $TARGET";;
    s)
      SOCSDK=${OPTARG}
      echo "soc-sdk is $SOCSDK";;
    d)
      TPUID=${OPTARG}
      echo "using tpu $TPUID";;
    p)
      PYTEST=${OPTARG}
      echo "generate logs for $PYTEST";;
    ?)
      usage
      exit 1;;
  esac
done

if [ -f "tools/benchmark.txt" ]; then
  rm tools/benchmark.txt
fi
echo "|    测试平台   |  测试程序    |     测试模型      |  preprocess_time(s)  |  inference_time(s)  |" >> tools/benchmark.txt
PLATFORM=$TARGET
if test $MODE = "soc_test"; then
  if test $TARGET = "BM1684X"; then
    PLATFORM="SE7-32"
  else
    echo "Unknown TARGET type: $TARGET"
  fi
fi

function bmrt_test_case(){
    bmrt_test_log=$(bmrt_test --bmodel $1 --devid $TPUID 2>&1)
    readarray -t calculate_times < <(echo "$bmrt_test_log" | grep -oP 'calculate  time\(s\): \K\d+\.\d+' | awk '{printf "%.2f \n", $1 * 1000}')
    readarray -t model_names < <(echo "$bmrt_test_log" | grep -oP 'running network #\d+, name: \K[^,]+')
    local index=0
    for time in "${calculate_times[@]}"
    do
        if [ $index -lt ${#model_names[@]} ]; then
            printf "| %-35s | %-20s | % 15s |\n" "$1" "${model_names[$index]}" "$time"
        else
            printf "| %-35s | %-20s | % 15s |\n" "$1" "Unknown Model" "$time"
        fi
        ((index++))
    done
}

function bmrt_test_benchmark(){
    pushd models
    printf "| %-35s | %-20s | % 15s |\n" "测试模型" "测试子模型" "calculate time(ms)"
    printf "| %-35s | %-20s | % 15s |\n" "-------------------" "-------------------" "--------------"
   
    if test $TARGET = "BM1684X"; then
      bmrt_test_case BM1684X/faceformer_f32.bmodel
    fi
    popd
}


if test $PYTEST = "pytest"
then
  >${top_dir}auto_test_result.txt
fi

function judge_ret()
{
  if [[ $1 == 0 ]]; then
    echo "Passed: $2"
    echo ""
    if test $PYTEST = "pytest"
    then
      echo "Passed: $2" >> ${top_dir}auto_test_result.txt
      echo "#######Debug Info Start#######" >> ${top_dir}auto_test_result.txt
    fi
  else
    echo "Failed: $2"
    ALL_PASS=0
    if test $PYTEST = "pytest"
    then
      echo "Failed: $2" >> ${top_dir}auto_test_result.txt
      echo "#######Debug Info Start#######" >> ${top_dir}auto_test_result.txt
    fi
  fi

  if test $PYTEST = "pytest"
  then
    if [[ $3 != 0 ]] && [[ $3 != "" ]];then
      tail -n ${ECHO_LINES} $3 >> ${top_dir}auto_test_result.txt
    fi
    echo "########Debug Info End########" >> ${top_dir}auto_test_result.txt
  fi

  sleep 3
}

function download()
{
  chmod -R +x scripts/
  ./scripts/download.sh
  judge_ret $? "download" 0
}

function test_python()
{
  
  if [ ! -d log ];then
    mkdir log
  fi
  pushd python
  python3 faceformer.py --wav_path $2 --bmodel ../models/$TARGET/$1 --model_name vocaset --dataset vocaset --dev_id $TPUID > ../log/$1_python_test.log 2>&1
  judge_ret $? "python3 faceformer.py --wav_path $2 --bmodel ../models/$TARGET/$1 --model_name vocaset --dataset vocaset --dev_id $TPUID" ../log/$1_python_test.log
  echo "==================="
  tail -n 20 ../log/$1_python_test.log

  preprocess_time=$(grep "preprocess_time:" ../log/$1_python_test.log | awk '{print $2}')
  inference_time=$(grep "inference_time:" ../log/$1_python_test.log | awk '{print $2}')

  # 格式化输出
  printf "| %-12s | %-14s | %-20s | %8.4f | %8.4f |\n" "$PLATFORM" "faceformer.py" "$1" "$(printf "%.4f" $preprocess_time)" "$(printf "%.4f" $inference_time)" >> ../tools/benchmark.txt
  echo -e "########################\nCase End: test python\n########################\n"
  popd
}

function compile_mlir()
{
  ./scripts/gen_bmodel_mlir.sh $TARGET
  judge_ret $? "generate $TARGET fp32bmodel" 0
}

if test $MODE = "compile_mlir"
then
  download
  compile_mlir
elif test $MODE = "pcie_test"
then
  download
  apt-get install libsndfile1
  pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  if test $TARGET = "BM1684X"
  then
    test_python faceformer_f32.bmodel ../Data/wav/test1.wav
  fi
elif test $MODE = "soc_test"
then
  download
  apt-get install libsndfile1
  pip3 install -r python/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
  if test $TARGET = "BM1684X"
  then
    test_python  faceformer_f32.bmodel ../Data/wav/test1.wav
  fi
fi

if [ x$MODE == x"pcie_test" ] || [ x$MODE == x"soc_test" ]; then
  echo "--------bmrt_test performance-----------"
  bmrt_test_benchmark
  echo "--------faceformer performance-----------"
  cat tools/benchmark.txt
fi

if [ $ALL_PASS -eq 0 ]
then
    echo "====================================================================="
    echo "Some process produced unexpected results, please look out their logs!"
    echo "====================================================================="
else
    echo "===================="
    echo "Test cases all pass!"
    echo "===================="
fi

popd