for i in {1..8}
do
	if [ -f $i.csv ]; then
		awk '!a[$0]++' $i.csv > $i-2.csv
		mv $i-2.csv $i.csv
	fi
done
