#target = /usr/local/opt/ilmbase/lib/libImath-2_4.24.dylib
#/usr/local/opt/opencv@3/lib/
#/usr/local/opt/protobuf/lib/libprotobuf.22.dylib

pushd /usr/local/opt/protobuf/lib/ || exit
for i in *.dylib
do
  if [ ! -L $i ]; then
  x=$(echo $i | awk -F'.' '{print $1}')
  echo "$i --> $x.22.dylib"
  ln -s "$i" "$x.22.dylib"
  fi
done
popd || ewit


