#!/usr/bin/env bash



python ex4.py mat1.npy cat cat > out1.txt
diff out1.txt test1.out
python ex4.py mat2.npy dog dog > out2.txt
diff out2.txt test2.out
python ex4.py mat3.npy rabbit rabitc > out3.txt
diff out3.txt test3.out
python ex4.py mat4.npy a abc > out4.txt
diff out4.txt test4.out
python ex4.py mat5.npy b abc > out5.txt
diff out5.txt test5.out

read -p "Press enter to continue"