sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
python BVSM_M.py D1 | tee D1_BVSM_M.txt
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
python BVSM_M.py D2 | tee D2_BVSM_M.txt
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
python BVSM_G.py D3 | tee D3_BVSM_G.txt
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
python BVSM_G.py D4 | tee D4_BVSM_G.txt
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
python BVSM_G.py D5 | tee D5_BVSM_G.txt
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
python BVSM_G.py AIDS | tee AIDS_BVSM_G.txt
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
python BVSM_G.py BZR | tee BZR_BVSM_G.txt
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
python BVSM_G.py COX2 | tee COX2_BVSM_G.txt
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
python BVSM_G.py DHFR | tee DHFR_BVSM_G.txt
