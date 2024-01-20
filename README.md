# ASCII-media-convertor
A ASCII art/video maker written in Python. It uses numpy, openCV to convert a video to greyscaled frames to pixels to long ASCII strings. ~~Those are then rendered with PIL library. Unfortunately PIL library is quite slow for this task and it's what slowing the convertor. In the next release, I will use rust to speed up the text rendering. Untill then, enjoy :3~~
In the new version, it uses pygame instead. Which is much faster than PIL. Currently it only supports windows and videos. In the next version, I will add linux support and ability to convert images to ASCII.

#### To use it 
Download the executeables and install ffmpeg and add the path to it's binary to PATH variable of your windows.
```
cd <to the .exe directory>
./ascii -i <input file name> -o <output file name> -c <codec. Set to default if not specified>
```

Default codec is hevc_amf. Run `ffmpeg -codecs` for a list of codecs.

#### Preview
https://github.com/Doctorthe113/ascii-media-convertor/assets/51150805/4c7d013d-7f5f-4a38-ba23-6a30781d0335