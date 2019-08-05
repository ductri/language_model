ln -s "`pwd`/source/" /source

echo "Downloading vocab"
mkdir -p  source/main/vocab/output
wget http://213.246.38.101:8002/source/main/vocab/output/v1.model \
    -P source/main/vocab/output/

echo "Downloading model"
mkdir -p  source/main/train/output/saved_models/Model/1.4
wget http://213.246.38.101:8002/source/main/train/output/saved_models/Model/1.4/200000.pt \
    -P source/main/train/output/saved_models/Model/1.4/



