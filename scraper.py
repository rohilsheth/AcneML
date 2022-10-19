# scrape all images from https://dermnetnz.org/topics/acne-face-images?stage=Live and save them to a folder called images

import requests
from bs4 import BeautifulSoup
import os
from PIL import Image
from io import BytesIO

# create a folder called images if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

# get the html from the website
r = requests.get('https://dermnetnz.org/topics/acne-face-images?stage=Live')
soup = BeautifulSoup(r.text, 'html.parser')

# find all the images
images = soup.find_all('img')

#save txt file with all the image urls
i=0
for image in images:
    try:
        print('https://dermnetnz.org'+image['src'])
        response = requests.get('https://dermnetnz.org'+image['src'])
        bytes = BytesIO(response.content)
        image = Image.open(bytes)
        image.save('images/'+str(i)+'.jpg')
        i+=1
    except:
        pass
