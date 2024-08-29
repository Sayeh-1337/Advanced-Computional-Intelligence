#!/bin/bash

# Step 1: Change directory to root


# Step 2: Clone the FinRL-Meta repository
git clone https://github.com/AI4Finance-Foundation/FinRL-Meta

# Step 3: Change directory to FinRL-Meta
cd FinRL-Meta/

# Step 4: Install required Python packages and dependencies
pip install git+https://github.com/AI4Finance-Foundation/ElegantRL.git
pip install wrds
pip install swig
pip install -q condacolab

# Step 5: Install Conda using condacolab
python -c "import condacolab; condacolab.install()"

# Step 6: Install system dependencies
sudo apt-get update -y -qq && sudo apt-get install -y -qq cmake libopenmpi-dev python3-dev zlib1g-dev libgl1-mesa-glx swig

# Step 7: Install more Python packages
pip install git+https://github.com/AI4Finance-Foundation/FinRL.git
pip install yfinance stockstats
pip install alpaca_trade_api
pip install ray[default]
pip install lz4
pip install ray[tune]
pip install tensorboardX
pip install gputil
pip install trading_calendars
pip install wrds
pip install rqdatac
pip install sqlalchemy==1.2.19
pip install tushare

# Step 8: Install TA-Lib
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar xvzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
sudo make install
cd ../
pip install TA-Lib

# Step 9: Install additional Python packages
pip install baostock
pip install quandl
pip install shimmy
pip install neat-python torch torch-geometric snntorch scipy neuralprophet stable-baselines3 sb3-contrib gym
pip install autoqubo dynex
pip install dimod

# Step 10: Clone and build DynexSDK testnet
rm -rf dynexsdk_testnet
git clone https://github.com/dynexcoin/dynexsdk_testnet.git
cd dynexsdk_testnet
chmod +x build.sh
./build.sh

# Step 11: Copy dynex-testnet-bnb file
cp dynex-testnet-bnb ../.
mkdir -p testnet
cp dynex-testnet-bnb ./testnet
# Step 12: Copy dynex.ini file
cd ..
cp ../dynex.ini .
cp ../neat-config.ini .
cp ../CryptoAgent.py .
cp ../QuantumCryptoAgent.py .