import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
import datetime

BASE_URL = "https://books.toscrape.com/catalogue/page-{}.html"

all_books = []

for page in range(1, 6):  
    print(f"Scraping page {page}...")
    url = BASE_URL.format(page)
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to fetch page {page}")
        break

    soup = BeautifulSoup(response.text, "html.parser")
    books = soup.find_all("article", class_="product_pod")

    for book in books:
        title = book.h3.a["title"]
        price = book.find("p", class_="price_color").text.strip()
        stock = book.find("p", class_="instock availability").text.strip()

        all_books.append({
            "Title": title,
            "Price": price,
            "Stock Status": stock
        })

    time.sleep(1)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


csv_file = f"books_{timestamp}.csv"
df = pd.DataFrame(all_books)
df.to_csv(csv_file, index=False)
print(f" Data saved to {csv_file}")


json_file = f"books_{timestamp}.json"
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(all_books, f, indent=4, ensure_ascii=False)
print(f" Data saved to {json_file}")

print(f"Total books scraped: {len(all_books)}")
