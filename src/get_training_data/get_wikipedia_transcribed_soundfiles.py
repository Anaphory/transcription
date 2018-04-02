#!/usr/bin/env python

"""Download vowel audio file from Wikipedia and put them in place with
transcription files."""

import re
import urllib.request
from bs4 import BeautifulSoup
from pathlib import Path
from hashlib import md5

URL_TEMPLATE = "https://en.wikipedia.org/{page}"

fileurl = re.compile("/wiki/File:(.*)")
ipastring = re.compile("\\[(.*)\\]")
uploadurl = re.compile(
    "//upload.wikimedia.org/wikipedia/commons/[0-9a-f]/[0-9a-f][0-9a-f]/(.*)")
def download_url_from_filename(wikilink):
    wikiname = fileurl.match(wikilink)[1]
    hash = md5(wikiname.encode("ascii")).hexdigest()
    return "//upload.wikimedia.org/wikipedia/commons/{:}/{:}/{:}".format(
        hash[0], hash[:2], wikiname)

try:
    PATH = Path(__file__)
except NameError:
    PATH = Path("./get_wikipedia_vowels.py")
PATH = PATH.absolute().parent.parent.parent / "data"

def get_raw_from_wikipedia(article_name):
    return urllib.request.urlopen(
        URL_TEMPLATE.format(page="wiki/" + article_name)
        + "?action=raw")


with urllib.request.urlopen(
        URL_TEMPLATE.format(
            page="wiki/Template:IPA_vowels")) as v:
    vowel_soup = BeautifulSoup(v.read())

vowel_link = re.compile("/wiki/([^:]*_vowel)")
articles = set()
for link in vowel_soup.find_all("a"):
    address = link.get("href")
    if address and vowel_link.match(address):
        articles.add(vowel_link.match(address)[1])

for article in articles:
    with urllib.request.urlopen(
            URL_TEMPLATE.format(page="wiki/" + article)) as v:
        vowel_soup = BeautifulSoup(v.read())

    ipa = vowel_soup.find_all("span", class_="IPA")[0].text
    for link in vowel_soup.find_all("a"):
        address = link.get("href")
        if link.text == "source" and fileurl.match(address):
            transcription = ipa
            address = download_url_from_filename(address)
        elif ipastring.match(link.text) and uploadurl.match(address):
            transcription = link.text[1:-1]
        else:
            continue
        with urllib.request.urlopen(
                "https:" + address) as remoteoggfile:
            name = Path(address).name
            with open(PATH / name, "wb") as localoggfile:
                localoggfile.write(remoteoggfile.read())
            with open(PATH / (name[:-4] + ".txt"),
                      "w") as localtxtfile:
                localtxtfile.write(transcription)

with urllib.request.urlopen(
        URL_TEMPLATE.format(
            page="wiki/Template:IPA_pulmonic_consonants")) as v:
    soup = BeautifulSoup(
        v.read())
soup = soup.find_all("table", class_="wikitable")[0]

articles = set()
for link in soup.find_all("a"):
    address = link.get("href")
    if address:
        address = address.split("#")[0]
    if address:
        articles.add(address)

for article in articles:
    with urllib.request.urlopen(
            URL_TEMPLATE.format(page=article[1:])) as v:
        consonant_soup = BeautifulSoup(v.read())
    try:
        ipa = consonant_soup.find_all("span", class_="IPA")[0].text
    except IndexError:
        continue
    print(article)
    for link in consonant_soup.find_all("a"):
        address = link.get("href")
        if link.text == "source" and fileurl.match(address):
            transcription = ipa
            address = download_url_from_filename(address)
        elif ipastring.match(link.text) and uploadurl.match(address):
            transcription = link.text[1:-1]
        else:
            continue
        with urllib.request.urlopen(
                "https:" + address) as remoteoggfile:
            name = Path(address).name
            with open(PATH / name, "wb") as localoggfile:
                localoggfile.write(remoteoggfile.read())
            with open(PATH / (name[:-4] + ".txt"),
                      "w") as localtxtfile:
                localtxtfile.write(transcription)

page = "https://en.wikipedia.org/wiki/Special:WhatLinksHere/Template:IPA_audio_link"
articles = set()
while True:
    with urllib.request.urlopen(page) as v:
        soup = BeautifulSoup(v.read())
    try:
        page = "https://en.wikipedia.org" + [
            a for a in soup.find_all("a")
            if "next" in a.text][0]["href"]
    except IndexError:
        break
    articles |= set(a.get("href") for a in soup.find_all("a")
                if a.get("href") and a.get("href").startswith("/wiki/"))

print(articles)

for article in articles:
    with urllib.request.urlopen(
            URL_TEMPLATE.format(page=article[1:])) as v:
        article_soup = BeautifulSoup(v.read())
    try:
        ipa = consonant_soup.find_all("span", class_="IPA")[0].text
    except IndexError:
        continue
    print(article)
    for link in consonant_soup.find_all("a"):
        address = link.get("href")
        if link.text == "source" and fileurl.match(address):
            transcription = ipa
            address = download_url_from_filename(address)
        elif ipastring.match(link.text) and uploadurl.match(address):
            transcription = link.text[1:-1]
        else:
            continue
        with urllib.request.urlopen(
                "https:" + address) as remoteoggfile:
            name = Path(address).name
            with open(PATH / name, "wb") as localoggfile:
                localoggfile.write(remoteoggfile.read())
            with open(PATH / (name[:-4] + ".txt"),
                      "w") as localtxtfile:
                localtxtfile.write(transcription)










