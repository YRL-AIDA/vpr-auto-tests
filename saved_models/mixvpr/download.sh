#!/bin/bash

declare file_name;
declare download_link;


# dim = 128
file_name="resnet50_MixVPR_128_channels(64)_rows(2).ckpt";
download_link="https://drive.usercontent.google.com/download?id=1DQnefjk1hVICOEYPwE4-CZAZOvi1NSJz&export=download&authuser=0&confirm=t&uuid=05aaac3d-35d7-4b50-985e-1a47a831b984&at=AEz70l6QmwOq7EWNja6D0_5sZFuM%3A1743053576906";

# Download models
echo Downloading $file_name...;
wget "$download_link" -O $file_name;

echo "--- --- ---";

# dim = 512
file_name="resnet50_MixVPR_512_channels(256)_rows(2).ckpt";
download_link="https://drive.usercontent.google.com/download?id=1khiTUNzZhfV2UUupZoIsPIbsMRBYVDqj&export=download&authuser=0&confirm=t&uuid=25315725-3da9-4bf2-9ddb-102695c869aa&at=AEz70l7ZOZO60QLBZOZwfz7y3q7c%3A1743053496799";

# Download models
echo Downloading $file_name...;
wget "$download_link" -O $file_name;

echo "--- --- ---";

# dim = 04096
file_name="resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt";
download_link="https://drive.usercontent.google.com/download?id=1vuz3PvnR7vxnDDLQrdHJaOA04SQrtk5L&export=download&authuser=0&confirm=t&uuid=b1037fd1-d3a7-4160-a7ab-3ac337355e81&at=AEz70l5NmbFw96xAPG4yIAzGloNN%3A1741776525528";

# Download models
echo Downloading $file_name...;
wget "$download_link" -O $file_name;
