# Generation_memoji

I converted this lua code (https://github.com/patniemeyer/vgg-memoji) to python code.
I used weight converter and the VGG face model of this github (https://github.com/chi0tzp/PyVGGFace).

## Explanation
It is an algorithm that finds the most similar memoji among memojis that we have through similarity scoring.
Obtain the layer immediately preceding the output layer from the pre-trained model and calculate it.
It shows the top 3 of the memojis we have.
The more memoji we have, we can get the more accurate memoji about the pictures.
I'm also designing a function to generate memoj.

## How to run code.
1. Download this repository.
2. Download pre-trained model, vggface.pth on release tab. And, put the file in "models" folder.
3. Run the "compare-memoji-multi.py"
4. You can add the other sample or pictures of new people by modifying "refsFolder" in "compare-memoji-multi.py" file.

## Results
I found Jonathan Ive's the most similar memoji.  

<p align="center">
<img src=https://user-images.githubusercontent.com/53460541/130322330-9de5af12-e492-4d46-895f-0b7dddb5c624.png>
</p>

Enjoy!

## TO DO
How can we make memoji, not select in images pool?
