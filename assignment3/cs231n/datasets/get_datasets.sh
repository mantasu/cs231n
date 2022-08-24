#!/bin/bash
if [ ! -d "coco_captioning" ]; then
    sh get_coco_captioning.sh
    sh get_imagenet_val.sh
fi