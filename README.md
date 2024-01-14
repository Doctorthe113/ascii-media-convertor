# ASCII-media-convertor
A ASCII art/video maker written in Python. It uses numpy, openCV to convert a video to greyscaled frames to pixels to long ASCII strings. Those are then rendered with PIL library. Unfortunately PIL library is quite slow for this task and it's what slowing the convertor. In the next release, I will use rust to speed up the text rendering. Untill then, enjoy :3

To use it, download the .exe file
```
cd <to the .exe directory>
./ascii -i <input file name> -o <output file name>
```