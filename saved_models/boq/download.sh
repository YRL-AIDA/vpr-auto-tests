#!/bin/bash

declare file_name;
declare download_link;

file_name="dinov2_12288.pth";
download_link="https://github.com/amaralibey/Bag-of-Queries/releases/download/v1.0/dinov2_12288.pth";

# Download models
echo Downloading $file_name...;
wget "$download_link" -O $file_name;
