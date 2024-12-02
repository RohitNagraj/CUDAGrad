wget -P ../data/ https://archive.ics.uci.edu/static/public/280/higgs.zip
unzip ../data/higgs.zip -d ../data/higgs/
gunzip -c ../data/higgs/HIGGS.csv.gz > ../data/higgs/higgs.csv
rm ../data/higgs.zip
rm ../data/higgs/HIGGS.csv.gz