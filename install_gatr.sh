url=https://github.com/Qualcomm-AI-research/geometric-algebra-transformer.git
project_dir=geometric-algebra-transformer
# commit ID of version 1.2.0
commit_id=87fa7d0c5cec5d874b13814816c628843e348866
mkdir build
cd build
echo "Cloning gatr repository"
git clone $url
cd $project_dir
git reset --hard $commit_id
echo "Applying patch 1/2: remove dependency on xformers"
git apply ../../patches/gatr_xformers.patch
echo "Applying patch 2/2: improvements of runtime performance"
git apply ../../patches/gatr_performance.patch
echo "Installing gatr package"
pip install --no-deps .
