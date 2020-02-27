# root=/path/to/your/root/
root=/home/chenzhaowei/Project/Light-Condition-Style-Transfer/
# data_dir=/path/to/your/data/
data_dir=/home/chenzhaowei/data/CULane/
exp=ERFNet
detect_dir=${root}tools/prob2lines/output/${exp}/
w_lane=30;
iou=0.5;  # Set iou to 0.3 or 0.5
im_w=1640
im_h=590
frame=1
list=${data_dir}list/test.txt
out=./output/${exp}_iou${iou}.txt
./evaluate -a $data_dir -d $detect_dir -i $data_dir -l $list -w $w_lane -t $iou -c $im_w -r $im_h -f $frame -o $out
