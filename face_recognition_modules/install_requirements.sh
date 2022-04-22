# On t2.micro EC2 instance with Ubuntu 20.04.3 LTS
sudo apt update
sudo apt install python3-pip
pip3 --version

pip3 install matplotlib

pip3 install torch==1.10.2+cpu torchvision==0.11.3+cpu torchaudio==0.10.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
