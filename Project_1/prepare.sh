# fill in pts.tr.csv 
patients_file="./train_data/pts.tr.csv"
echo "1" > $patients_file
for i in {2..8267}
do
    echo $i >> $patients_file
done

# reindex test data to begin with 6001
for i in {1..2267}
do 
    new=$(($i+6000))
    cp ./test_data/test_with_missing/$i.csv ./train_data/train_with_missing/$new.csv
done

# create micegp_log folder for running 3D-mice
mkdir -p ./train_data/micegp_log/
