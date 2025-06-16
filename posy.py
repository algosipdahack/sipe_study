import requests
import re
from bs4 import BeautifulSoup

def get_latest_post_ids_from_rss(blog_id, max_posts=10):
    url = f"https://rss.blog.naver.com/{blog_id}.xml"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)

    soup = BeautifulSoup(res.content, "xml")
    items = soup.find_all("item")[:max_posts]

    post_ids = []
    for item in items:
        link = item.find("link").text
        match = re.search(r"/(\d+)", link)
        if match:
            post_ids.append(match.group(1))

    return post_ids
post_ids = get_latest_post_ids_from_rss("sophia5460")
print(post_ids)
