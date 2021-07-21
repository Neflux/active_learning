file=maia/dataset/data3.csv
set -- `ls -l "$file"`
size=$5
while true; do
  sleep 5
  set -- `ls -l "$file"`
  case $5 in $size) break;; esac
  size=$5
  echo $5
done