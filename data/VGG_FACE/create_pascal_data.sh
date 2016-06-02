cur_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )
root_dir=$cur_dir/../..

cd $root_dir

redo=1
data_root_dir="/host-ssd1TB/pascal_faces/"
dataset_name="VGG_FACE"
mapfile="$root_dir/data/$dataset_name/labelmap_face.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=jpg --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in test
do
  python $root_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $data_root_dir $data_root_dir/$subset.txt $data_root_dir/$db/$subset"_"$db $root_dir/data/VGG_FACE/ssd/examples
done
