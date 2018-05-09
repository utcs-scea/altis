#bfs
for i in 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608
do
    ./src/cuda/level1/bfs/bfs -i data/bfs/input/bfs_$i -m features/bfs/$i
done

#pathfinder
#for i in 2 4 8 16 32 64 128 256 512 1024 2048
#do
    #./src/cuda/level1/pathfinder/pathfinder -s 1 -m features/pathfinder/$i --instances $i
#done

#srad
#for i in 16 32 48 64 80 96 112 128 144 160 176 192 208 224 240 256
#do
    #./src/cuda/level2/srad/srad --imageSize $i --speckleSize 8 --iterations 50 -m features/srad/$i
#done

#mandelbrot
#for i in 32 64 128 256 512 1024 2048 4096 8192 16384 32768
#do
    #./src/cuda/level2/mandelbrot/mandelbrot --imageSize $i --iterations 2048 -m features/mandelbrot/$i
#done

