###################################
# 20260529 IDF
###################################

tmux new-session -s idf
nproc_value=$(nproc) && cd /home/admin_climatecharted_com/GitHub/hazard-heavy-rain && { time ./run.sh; } &> ./exec_output_${nproc_value}.txt

idf: ETA 
 c4-highcpu-96
tmux kill-session -t idf

tmux ls
tmux attach -t idf

# how to verify permission =======
ls -l ./run.sh
chmod +x ./run.sh
ls -l ./run.sh

# otherwise explicit run =========
bash ./run.sh

# ================================

##################################