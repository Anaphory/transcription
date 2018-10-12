for i in `seq 50`
do
    rename '%' '' *%*
done

for ending in .wav
do
    for file in *$ending
    do
        mv $file $file~
        ffmpeg -i $file~ -ac 1 -ar 44100 -y `basename $file $ending`.wav
    done
done

for ending in .ogg .oga
do
    for file in *$ending
    do
        ffmpeg -i $file -ac 1 -ar 44100 -y `basename $file $ending`.wav
    done
done

for file in *.txt
do
    (echo -n "KAN: 0 "
     python3.6 -m segments tokenize < $file
     echo) | tee `basename $file .txt`.par
done
