#!/bin/bash

declare file_name;
declare download_link;

file_name="resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt";
download_link="https://drive.usercontent.google.com/download?id=1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L&export=download&authuser=0&confirm=t&uuid=b1037fd1-d3a7-4160-a7ab-3ac337355e81&at=AEz70l5NmbFw96xAPG4yIAzGloNN%3A1741776525528";

# Download models
echo Downloading $file_name...;
curl "$download_link" > $file_name;
