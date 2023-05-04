import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import List
from datetime import datetime
import re


def main():
    extract_fight_details()
    


def get_events() -> List[str]:
    """Extacts event links from ufcstats.com

    Returns:
        List[str]: links to ufc events
    """
    event_links = []
    # url = "http://www.ufcstats.com/statistics/events/completed?page=all"
    url = "http://www.ufcstats.com/statistics/events/completed"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    # Find all links on the page with the class 'b-link b-link_style_black'
    links = soup.find_all('a', class_='b-link b-link_style_black')

    # Extract the href attribute from each link and append it to event_links
    for link in links:
        href = link['href']
        event_links.append(href)
    
    return event_links


def extract_event_details():
    event_links = get_events()
    rows = []
    for event in event_links:
        links = []
        response = requests.get(event)
        soup = BeautifulSoup(response.content, "html.parser")
        row = {}
        for li in soup.find_all("li", class_="b-list__box-list-item"):
            title = li.find("i").text.strip()
            value = li.text.strip().replace(title, "").strip()
            row[title.replace(":", "").lower()] = value
        
        fight_links = soup.find_all("tr", class_="b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click")
        for fight in fight_links:
            fight_details = fight['data-link']
            links.append(fight_details)
        
        row['fight_details'] = links
        rows.append(row)
    return rows


def extract_fight_details():
    data = extract_event_details()
    for event in data:
        for fight in event['fight_details']:
            response = requests.get(fight)
            soup = BeautifulSoup(response.content, "html.parser")
            
        return
        

def get_fighter_links() -> List[str]:
    """Extract fighter profile links from ufcstats.com

    Returns:
        List[str]: links to ufc fighter profile
    """
    fighter_links = set()
    # url = "http://www.ufcstats.com/statistics/fighters?char=a&page=all"
    url = "http://www.ufcstats.com/statistics/fighters"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    # Find all links on the page with the class 'b-link b-link_style_black'
    links = soup.find_all('a', class_='b-link b-link_style_black')

    # Extract the href attribute from each link and append it to fighter_links
    for link in links:
        href = link['href']
        if href not in fighter_links:
            fighter_links.add(href)
    
    return fighter_links


def extract_fighter_stats():
    links = get_fighter_links()
    rows = []
    # if fighter not in fights since last run skip
    for fighter_link in links[:5]:
        response = requests.get(fighter_link)
        soup = BeautifulSoup(response.content, "html.parser")
        name = soup.find("span", class_="b-content__title-highlight").text.strip()
        record = soup.find("span", class_="b-content__title-record").text.strip()

        row = {"fighter" : name, "mma_record" : record}
        # extract the attributes from the list items
        for stat in soup.find_all("li", class_="b-list__box-list-item b-list__box-list-item_type_block"):
            values = [item.strip() for item in stat.text.split(":")]
            if values[0] != '':
                key, value = values
                if value == "--":
                    value = None
                
                if value is not None:
                    if key in ["Height", "Reach"]:
                        value = inch_to_cm(value)
                    elif key == "DOB":
                        # Convert date string to datetime object
                        dob = datetime.strptime(value, '%b %d, %Y').date()
                        # Calculate age
                        value = (datetime.now().date() - dob).days // 365
                        key = "age"
                    elif key == "SLpM":
                        key = "sig_str_per_min"
                    elif key == "Str. Acc.":
                        key = "sig_str_acc"
                    elif key == "SApM":
                        key = "sig_str_absorbed_per_min"
                    elif key == "Str. Def":
                        key = "sig_str_def"
                    elif key == "TD Avg.":
                        key = "td_avg"
                    elif key in ["TD Acc.", "TD Def.", "Sub. Avg."]:
                        key = key.lower().replace(" ", "_").replace(".", "")
                        
                row[key.lower()] = value
                
        if 'dob' in row.keys():
            del row['dob']
            
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv("fighters.csv", index=False)


def inch_to_cm(height) -> float:
    
    if "'" in height:
        feet, inches = height.split("'")
        feet = int(feet)
        inches = int(inches.strip('"'))
        total_inches = (feet * 12) + inches
        cm = total_inches * 2.54
        return round(cm, 1)
    
    elif '"' in height:
        inches = int(height.strip('"'))
        cm = inches * 2.54
        return round(cm, 1)


if __name__ == "__main__":
    main()