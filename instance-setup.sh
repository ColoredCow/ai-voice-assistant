# steps to set up the server
# use g4dn.xlarge EC2 instance with Deep Learning Pytorch AMI (Ubuntu 22.04)

# by default the pytorch will be installed and not enabled
# enable pytorch env with source activate pytorch

sudo mkdir -p /var/www/html
sudo chown -R ubuntu:ubuntu /var/www/html

sudo apt update
sudo apt install python3-pip python3-venv git -y

git clone https://github.com/coloredcow/ai-voice-assistant.git
cd ai-voice-assistant/
