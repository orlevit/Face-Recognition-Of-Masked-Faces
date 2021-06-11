# Run this inside the 03_Protocol.. directory
rm *.pts
ls -1| awk -F _ '{print $1}'| sort -u| grep -v / |xargs -I %  mkdir %/
ls -1| awk -F _ '{print $1}'| grep -v /|sort -ur| xargs -I % sh -c 'mv %*.jpg %/'
for dir in *; do  if [ -d "$dir" ]; then cd "$dir"; for filename in *; do echo "$filename" |cut -d'_' -f3|xargs -I % mv "${filename}"  "%.jpg"; done; cd ..; fi; done
