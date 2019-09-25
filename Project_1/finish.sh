###################################################################
#Script Name	: finish.sh                                                                                             
#Description	: reindex the last 2267 files and move them to 'test_output/3d-mice-test' folder
#Author       	:Yuhang Lin                                               
#Email         	:ylin34@ncsu.edu                                          
###################################################################
# reindex the files back to original format
mkdir -p ./test_output/3d-mice-test
for i in {6001..8267}
do
	new=$(($i-6000))
	cp ./train_data/micegp_log/$i.csv ./test_output/3d-mice-test/$new.csv
done
