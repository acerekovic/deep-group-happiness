#!/usr/bin/env bash

# This scripts downloads the prerequisites and trained TF models.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading models..."

cd data
mkdir -p models
cd models
wget https://www.dropbox.com/s/mdae915tytzvmq9/googlenet.pb
wget https://www.dropbox.com/s/xputod3d2591sog/gnet-fc.ckpt-6744
wget https://www.dropbox.com/s/qp6aipxyi24o0ds/vgg16.ckpt-6601
wget https://www.dropbox.com/s/hnc5enspze6wbw2/face-detector.ckpt-25000
wget https://www.dropbox.com/s/u1job5v89gcj6b9/face-centrist.ckpt-47
wget https://www.dropbox.com/s/jiuftd48ofmlecj/face-detector-hypes.json
echo "Done."
