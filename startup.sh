

echo "Downloading vocab"
mkdir -p  source/main/vocab/output
wget http://213.246.38.101:10002/source/main/vocab/output/v1.model \
    -O source/main/vocab/output/

echo "Downloading model"
mkdir -p  source/main/train/output/saved_models/Model/1.4
wget http://213.246.38.101:10002/source/main/train/output/saved_models/Model/1.4/200000.pt \
    -O source/main/train/output/saved_models/Model/1.4/

ln -s "`pwd`/source/" /source

