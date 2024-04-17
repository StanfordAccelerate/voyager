DATATYPES="BF16 \
  BF16_NS \
  BF16_TRN \
  BF16_NS_TRN \
  BF16_ONLY \
  BF16_ONLY_NS \
  BF16_ONLY_TRN \
  BF16_ONLY_NS_TRN \
  E4M3 \
  E4M3_NS \
  E5M2 \
  P8_1"

mkdir -p accuracy

for datatype in $DATATYPES; do
  echo "Running $datatype"
  make -j MobileBERTAccuracy DATATYPE=$datatype DIMENSION=8 > accuracy_logs/accuracy_$datatype.log 2>&1
done