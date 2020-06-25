cd /usr/local/opt/icu4c/lib/
for i in *67.*.dylib
do
echo "$i --> ${i::${#i}-9}6.dylib"
ln -s "$i" "${i::${#i}-9}6.dylib"
done