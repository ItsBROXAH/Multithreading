import os
import requests
import random
import cv2
import threading
import numpy as np

from PIL import Image
from sklearn.cluster import KMeans
from bs4 import BeautifulSoup
from multiprocessing import Process, Queue
 
# Define the URL from which to download the images
url = "https://www.google.com/search?client=ubuntu-sn&hs=Jvz&sca_esv=ddbbdcdace42deb0&channel=fs&q=images&tbm=isch&source=lnms&prmd=ivsnbz&sa=X&ved=2ahUKEwjZqNzCvJ-EAxU3avEDHfyFD24Q0pQJegQIEBAB&biw=954&bih=656&dpr=1"
 
# Define the number of images to download
n = 17
 
# Define the folder to store the images
folder = "images"
 
# Create the folder if it doesn't exist
if not os.path.exists(folder):
    os.mkdir(folder)
 
# Function to download an image from a URL
def download_image(image_url, filename):

    # Check if the URL is absolute
    if not image_url.startswith(("http://", "https://")):

        #print(f"Skipping image with invalid URL: {image_url}")
        return

    # If the URL is absolute, proceed with downloading
    image_data = requests.get(image_url).content
    with open(os.path.join(folder, filename), "wb") as f:
        f.write(image_data)
    print(f"Link Downloaded: {image_url}")
 
# Function executed by each thread

def thread_function(img_tags, q):
    selected_imgs = random.sample(img_tags, min(n, len(img_tags)))
    for i, img_tag in enumerate(selected_imgs):
        image_url = img_tag["src"]
        filename = f"image_{i}.jpg"
        print(f"\nLaunching thread for image: {filename}")
        download_image(image_url, filename)
        q.put(filename)
        print(f"Image {filename} added to the queue.")
 
# Function executed by each process for clustering
def process_function(q_in, q_out):
    while not q_in.empty():
        filename = q_in.get()
        file_path = os.path.join(folder, filename)
        print(f"Processing image: {filename}")
        image = Image.open(file_path)
        image_array = np.array(image)
        pixels = image_array.reshape(-1, 3)
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]
        image_gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        image_canny = cv2.Canny(image_gray, 100, 200)
        contours, hierarchy = cv2.findContours(image_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shapes = []

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
            n_vertices = len(approx)
            if n_vertices == 3:
                shape = "Triangle"
            elif n_vertices == 4:
                shape = "Rectangle"
            elif n_vertices == 5:
                shape = "Pentagon"
            else:
                shape = "Circle"

            shapes.append(shape)
        q_out.put((filename, hex_colors, shapes))
        print(f"Clustering completed for image: {filename}")
 
# Send an HTTP request to get the content of the web page
response = requests.get(url)
 
# Check if the request was successful
if response.status_code == 200:

    # Extract the HTML content from the response
    html_content = response.text
    
    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Find all img elements on the web page
    img_tags = soup.find_all("img")
    
    # Create a queue to retrieve file names
    q_img = Queue()
    
    # Create and start threads to download images
    threads = []
    for _ in range(5):  # Number of threads: 5
        thread = threading.Thread(target=thread_function, args=(img_tags, q_img))
        threads.append(thread)
        thread.start()
 
    # Wait for all threads to finish
    for thread in threads:
        thread.join()
 
    # Create a queue for clustering results
    q_cluster = Queue()
 
    # Create and start processes for clustering
    processes = []
    for _ in range(n*2):
        process = Process(target=process_function, args=(q_img, q_cluster))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()
    print("All processes have finished.")

 
    # Retrieve clustering results from the queue
    clustering_results = {}
    while not q_cluster.empty():
        filename, colors, shapes = q_cluster.get()
        clustering_results[filename] = (colors, shapes)

    # Définir la taille des images à afficher dans la page HTML
    width = 200
    height = 200
 
    # Generate HTML with clustering results
    html = ""
    html += "<!DOCTYPE html>\n"
    html += "<html>\n"
    html += "<head>\n"
    html += "<meta charset=\"utf-8\">\n"
    html += "<title>Projet SE</title>\n"
    html += "<style>\n"
    html += "img {\n"
    html += "}\n"
    html += "</style>\n"
    html += "</head>\n"
    html += "<body>\n"
    html += "<p>Validation Projet Système D'exploitation Avancé</p>\n"
    html += "<p>Med Amine Khalofaoui & Houssem Baccar & Med Amine Hidri.</p>\n"
    html += "<p>1ALSLEAM2.</p>\n"
    html += "<table>\n"
 
    # Iterate over the clustering results and add HTML for each image
    for filename, (colors, shapes) in clustering_results.items():
        file_path = os.path.join(folder, filename)
 
        # Charger l'image à partir du chemin de fichier
        image = Image.open(file_path)
 
        # Redimensionner l'image
        resized_image = image.resize((width, height))
        resized_image.save(file_path)
        html += "<tr>\n"
        html += f"<td><img src=\"{file_path}\"></td>\n"
        html += "<td>\n"
        html += "<ul>\n"

        for color in colors:
            html += f"<li style=\"color: {color};\">{color}</li>\n"
        html += "</ul>\n"
        html += "</td>\n"
        html += "<td>\n"
        html += "<ul>\n"

        for shape in set(shapes):
            html += f"<li>{shape}</li>\n"
        html += "</ul>\n"
        html += "</td>\n"
        html += "</tr>\n"
    html += "</table>\n"
    html += "</body>\n"
    html += "</html>\n"
 
    # Write HTML to a file

    html_file = "SE.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)
    print("All Threads have finished.")
    print(f"HTML page created successfully.")
 
else:
    print("Failed to retrieve the content of the web page.")
