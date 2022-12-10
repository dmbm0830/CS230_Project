from selenium import webdriver
from selenium.webdriver.common.by import By
import urllib
import time
import requests

# Install the chrome web driver from selenium. 

chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--headless')
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome('chromedriver',chrome_options=chrome_options)# Create url variable containing the webpage for a Google image search.
url = ("https://www.fortniteskin.com/all")# Launch the browser and open the given url in the webdriver.
driver.get(url)# Scroll down the body of the web page and load the images.
driver.execute_script("window.scrollTo(0,document.body.scrollHeight);")
time.sleep(10)# Find the images.
imgResults = driver.find_elements(By.XPATH,"//li[@data-type='outfit']//following::img")# Access and store the scr list of image url's.
src = []
print(len(imgResults))
for img in imgResults:
    src.append(img.get_attribute('data-src'))# Retrieve and download the images.
print("started grabbing images")
for i in range(len(src)):
    urllib.request.urlretrieve(str(src[i]),"scraped_images/skin_num{}.png".format(i))
    if (i % 100) == 0:
        print(i)
