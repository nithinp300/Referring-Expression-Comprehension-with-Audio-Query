nvidia-smi
pip install transformers

pip install --upgrade --no-cache-dir gdown
echo "export PATH=\"`python3 -m site --user-base`/bin:\$PATH\"" >> ~/.bashrc
source ~/.bashrc
pip install torch==1.5.0
pip install librosa
pip install numpy==1.20
pip install tensorflow==2.8.0

pip install torchvision==0.6.0

python -c "import torch; print(torch.__version__)"

pip install pytorch-pretrained-bert

sudo apt-get install aria2
git clone https://github.com/Muktan/TransVG.git
pwd

cd ./TransVG/ln_data
ls
sudo bash download_data.sh --path .
pwd

cd ../

pwd

gdown https://drive.google.com/uc?id=1fVwdDvXNbH8uuq_pHD_o5HI7yqeuz0yS

tar -xvf data.tar
mv ./my_train_split.pth ./data/unc/unc_train.pth
mv ./my_test_split.pth ./data/unc/unc_testA.pth
mv ./my_val_split.pth ./data/unc/unc_val.pth

sudo bash ./checkpoints/download_detr_model.sh
gdown --folder --no-cookies --id 1SOHPCCR6yElQmVp96LGJhfTP46RxVwzF

mv -v ./pretrained_detr_params/* ./checkpoints

pwd

# wget https://utdallas.box.com/shared/static/oqide5zr8ee3i002kuhffzisokwhu1i1.pth
# mv oqide5zr8ee3i002kuhffzisokwhu1i1.pth unc_train_my.pth
# wget https://utdallas.box.com/shared/static/o9nyr1lmfr4m7oftk8exrh3rfu90zp6t.txt

gdown https://drive.google.com/uc?id=1BsqntINUc9u67cgU_bLTFlk5wrGcLIwo

unzip -q data_new.zip -d .

pwd

# gdown https://drive.google.com/uc?id=1vdQVFs7qS1YkwjKut6hokMnoFyred-9S
# mv unc_train_my.pth ./TransVG/data/unc/unc_train.pth

# cd TransVG
# mv -v ./data_new/data_new* ./data_audio
# rm -r ./data_new
# mv -v ./data_audio ./data_new 