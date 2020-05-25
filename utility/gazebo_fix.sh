cd /usr/local/opt/bullet/lib/
for i in *89.dylib
do
ln -s "$i" "${i::${#i}-7}8.dylib"
done