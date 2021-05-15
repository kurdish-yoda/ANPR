#!/bin/bash

# Assume coco folder from YOLO

coco_original_path="C:\Users\kurdish_yoda\OneDrive\Documents\ANPR\yolo-object-detection\yolo-coco"
coco_new_path="C:\Users\kurdish_yoda\OneDrive\Documents\ANPR\yolo-object-detection\yolo-coco_custom"

mkdir -p ${coco_new_path}
cp -R ${coco_original_path}/annotations ${coco_new_path}/
cp -R ${coco_original_path}/common ${coco_new_path}/
cp -R ${coco_original_path}/*.txt ${coco_new_path}/
cp -R ${coco_original_path}/results ${coco_new_path}/
cp -R ${coco_original_path}/*API ${coco_new_path}/
ln -s ${coco_original_path}/images ${coco_new_path}/ # Images are not duplicated
mkdir -p ${coco_new_path}/labels/train2014
mkdir -p ${coco_new_path}/labels/val2014

class_to_keep=(
    2 #  car
    5 #  bus
    7 #  truck
    3 #  motorbike
)

path_=(
    "labels/train2014"
    "labels/val2014"
)

# https://stackoverflow.com/a/15028821/7036639
get_index_fct(){
    value=$1
    for i in "${!class_to_keep[@]}"; do
        if [[ "${class_to_keep[$i]}" = "${value}" ]]; then
            echo "${i}";
        fi
    done
}

update_label_fct(){
    filepath=$1
    filename=$(basename -- "$filepath")
    filepath_new="${coco_new_path}/${current_path}/${filename}"
    touch "${filepath_new}"
    echo "$filepath -> ${filepath_new}"

    while read -r line; do
        currClass=$(echo "${line}" | sed -r 's/^([^.]+).*$/\1/; s/^[^0-9]*([0-9]+).*$/\1/')
        if [[ "${class_to_keep[@]}" =~ "${currClass}" ]]; then
            # whatever you want to do when arr contains value #https://stackoverflow.com/a/15394738/7036639
            newClass=$(get_index_fct ${currClass})
            new_line=$(echo "${line}" | sed '0,/'$currClass'/s//'$newClass'/')
            echo "${new_line}" >> "${filepath_new}"
            #echo "$line -> $new_line"
        fi
    done < "${filepath}" 
}

sed -i "s#${coco_original_path}#${coco_new_path}#g" ${coco_new_path}/*5k.txt

for current_path in "${path_[@]}" ; do
    #https://unix.stackexchange.com/a/216466/217235
    N=256
    (
    for filepath in "${coco_original_path}/${current_path}"/*.txt; do
       ((i=i%N)); ((i++==0)) && wait

        update_label_fct "$filepath" &
    done
    )
done