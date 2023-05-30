import requests
from bs4 import BeautifulSoup
from typing import List


def get_next_event_link() -> List[str]:
    url = "http://www.ufcstats.com/statistics/events/upcoming"
    # url = "http://www.ufcstats.com/statistics/events/completed"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    # Find all links on the page with the soup BeautifulSoup('a', class_ 'b-link b-link_style_black'
    link = soup.find_all('a', class_='b-link b-link_style_black')[0]
    href = link['href']
    
    return href


def get_future_matchups():
    event_link = get_next_event_link()
    response = requests.get(event_link)
    soup = BeautifulSoup(response.content, 'html.parser')
    fighters = soup.find_all('a', class_="b-link b-link_style_black")
    matchups = []
    
    for i in range(0, len(fighters), 3): # step by 2 for past fights, 3 for future fights
        matchup = (fighters[i].text.strip(), fighters[i+1].text.strip())
        matchups.append(matchup)
        
    return matchups


if __name__ == "__main__":
    get_future_matchups()