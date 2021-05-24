# Create necessary directories
mkdir -p data/train/raw data/train/processed
mkdir -p data/test/raw data/test/processed
mkdir -p data/reads
mkdir trained_models
mkdir figures

# Compile Raven
cd vendor/raven
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && make

# Back to the project directory
cd ../../..