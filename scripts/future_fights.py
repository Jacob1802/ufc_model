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
    fighters = [fighter.text.strip() for fighter in soup.find_all('a', class_="b-link b-link_style_black")]
    list_items = soup.find_all('li', class_='b-list__box-list-item')
    weightclass_column = []
    matchups = []
    
    for row in soup.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) > 0 and cells[6].text != 'Weight class':
            weightclass_column.append(cells[6].text.strip())

    for item in list_items:
        if 'Date:' in item.text:
            date = item.text.replace('Date:', '').strip()
            break
        
    for i in range(0, len(fighters), 3): # 3 for future fights
        matchup = (fighters[i], fighters[i+1])
        matchups.append(matchup)
        
    return (date, matchups, weightclass_column)


if __name__ == "__main__":
    get_future_matchups()