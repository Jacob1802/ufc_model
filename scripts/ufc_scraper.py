import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import List
from datetime import datetime


def main():
    try:
        fights_df = pd.read_csv("data/raw_fight_totals.csv")
    except FileNotFoundError:
        extract_fight_details(None)
    # print(fights_df['event'])

    # fighters_df = pd.read_csv("fighters.csv")
    # extract_fighter_stats(fighters_df)
    extract_fight_details(fights_df)


def get_events() -> List[str]:
    """Extacts event links from ufcstats.com

    Returns:
        List[str]: links to ufc events
    """
    event_links = []
    url = "http://www.ufcstats.com/statistics/events/completed?page=all"
    # url = "http://www.ufcstats.com/statistics/events/completed"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    # Find all links on the page with the class 'b-link b-link_style_black'
    link = soup.find('a', class_='b-link b-link_style_black')
    
    # Extract the href attribute from each link and append it to event_links
    with open('data/cards.csv', 'r') as f:
        cards = f.readlines()

        while link:
            event = link.text.strip()
            if event in cards[-1].strip():
                return event_links[::-1]
            href = link['href']
            event_links.append(href)
            link = link.find_next('a', class_='b-link b-link_style_black')
            
    return event_links[::-1]


def extract_event_details():
    event_links = get_events()
    rows = []
    for event in event_links:
        links = []
        response = requests.get(event)
        soup = BeautifulSoup(response.content, "html.parser")
        event = soup.find("span", class_="b-content__title-highlight").text.strip()
        row = {"event" : event}
        
        for li in soup.find_all("li", class_="b-list__box-list-item"):
            title = li.find("i").text.strip()
            value = li.text.strip().replace(title, "").strip()
            row[title.replace(":", "").lower()] = value
        
        fight_links = soup.find_all("tr", class_="b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click")
        for fight in fight_links:
            fight_details = fight['data-link']
            links.append(fight_details)
        
        row['fight_details'] = links[::-1]
        rows.append(row)
        
    return rows


def extract_fight_details(df):
    data = extract_event_details()
    rows = []
    if df is not None:
        fight_num = df.iloc[-1]['fight_num']
    else:
        fight_num = 0
    for card in data:
        date = card['date']
        location = card['location']
        event = card['event']
        print(event)
        for i, fight in enumerate(card['fight_details'], start=1):
            fight_num += 1
            row = {"fight_num" : fight_num, "date": date, "location": location, "event" : event}
            response = requests.get(fight)
            soup = BeautifulSoup(response.content, "html.parser")
            names = [name.text.strip() for name in soup.find_all("h3", class_="b-fight-details__person-name")]
            result = soup.find("i", class_="b-fight-details__person-status").text.strip()
            weight_class = soup.find("i", class_="b-fight-details__fight-title").text.strip()
            if result == "W":
                row["result_1"] = result
                row["result_2"] = "L"
            elif result == "L":
                row["result_1"] = result
                row["result_2"] = "W"
            else:
                row["result_1"] = result
                row["result_2"] = result
                
                
            row['weight_class'] = weight_class
            
            for i, name in enumerate(names, start=1):
                row[f'fighter_{i}'] = name
            
            # Add fight details (method, round, time, format, ref)
            details = [i for detail in soup.find_all("p", class_="b-fight-details__text") for i in detail.stripped_strings]
            for i in range(0, len(details), 2):
                if details[i] != "Details:":
                    if details[i] == "Referee:" and details[i + 1] == "Details:":
                        row[details[i].strip(":").lower()] = None
                        try:
                            row[details[i+1].strip(":").lower()] = details[i+2]
                        except IndexError:
                            row[details[i+1].strip(":").lower()] = None
                        break
                    else:
                        row[details[i].strip(":").lower()] = details[i + 1]
                elif row['method'] == "TKO - Doctor's Stoppage":
                    row[details[i].strip(":").lower()] = None
                    break
                else:
                    try:
                        if "\n" in details[i+1:][0]:
                            row[details[i].strip(":").lower()] = ' '.join([part.strip() for part in details[i+1:][0].split('\n')])
                        else:
                            row[details[i].strip(":").lower()] = " ".join(details[i+1:])
                    except IndexError:
                            row[details[i].strip(":").lower()] = None
                    break
            try:
                stats = [i for i in soup.find("tbody", class_="b-fight-details__table-body").stripped_strings]
                header = [i for i in soup.find("thead", class_="b-fight-details__table-head").stripped_strings]
                row = totals(stats, header, row)
            except AttributeError:
                pass
            # breakdown of fight stats, round by round and strike breakdown
            # stats = soup.find_all("tbody", class_="b-fight-details__table-body")
            # all_stats(stats, row)
            # only total
            
            rows.append(row)
    
    if df is not None:
        temp = pd.DataFrame(rows)
        df = pd.concat([df, temp])
    else:
        df = pd.DataFrame(rows)
    
    # df.to_csv("fights_rbr.csv", index=False)
    df.to_csv("data/raw_fight_totals.csv", index=False)
    unique_events = df["event"].unique()
    pd.Series(unique_events).to_csv("data/cards.csv", index=False)


def totals(stats, header, row):
    # Pair stats with headers
    pairs = zip(stats[::2], stats[1::2], header)

    # Iterate through the pairs
    for pair in pairs:
        item1, item2, key = pair
        
        if "of" in item1 and key != 'Fighter':
            key = key.replace(".", "").strip().replace("Total ", "").replace(" ", "_").lower()
            landed_1, attempts_1 = item1.split("of")
            landed_2, attempts_2 = item2.split("of")
            landed_1, attempts_1, landed_2, attempts_2 = float(landed_1), float(attempts_1), float(landed_2), float(attempts_2)
            avoided_1 = attempts_2 - landed_2
            avoided_2 = attempts_1 - landed_1
                
            row[key + "_landed_1"] = landed_1
            row[key + "_landed_2"] = landed_2
            row[key + "_attempts_1"] = attempts_1
            row[key + "_attempts_2"] = attempts_2
            row[key + "_received_1"] = landed_2
            row[key + "_received_2"] = landed_1
            row[key + "_avoided_1"] = avoided_1
            row[key + "_avoided_2"] = avoided_2
            if key == "str":
                per_1 = 0 if attempts_1 == 0 else landed_1 / attempts_1
                per_2 = 0 if attempts_2 == 0 else landed_2 / attempts_2
                
                row[key + "_percent_1"] = per_1
                row[key + "_percent_2"] = per_2
        elif "%" in key:
            key = key.strip("%").replace(".", "").strip().replace(" ", "_").lower()
            row[key + "_percent_1"] = item1
            row[key + "_percent_2"] = item2
        else:
            key = key.strip("%").replace(".", "").strip().replace(" ", "_").lower()
            row[key + "_1"] = item1
            row[key + "_2"] = item2
            if key == "kd":
                row[key + "_received_1"] = item2
                row[key + "_received_2"] = item1
    
    for i in range(1, 3):
        if row[f"str_attempts_{i}"] == 0:
            row[f"sig_reg_mixture_{i}"] = 0
            row[f"sig_reg_percent_{i}"] = 0
        else:
            row[f"sig_reg_mixture_{i}"] = row[f"sig_str_attempts_{i}"] / row[f"str_attempts_{i}"]
            row[f"sig_reg_percent_{i}"] = (row[f"str_landed_{i}"] + row[f"sig_str_landed_{i}"]) / (row[f"str_attempts_{i}"] + row[f"sig_str_attempts_{i}"])
            
    return row


def all_stats(stats, row):
    for j in range(len(stats)):
        data = [i for i in stats[j].stripped_strings]

        if j < 2:
            CATEGORIES = [    
            ("fighter", [0, 20, 40, 60, 80]),
            ("kd", [2, 22, 42, 62, 82]),
            ("sig_str", [4, 24, 44, 64, 84]),
            ("sig_str_%", [6, 26, 46, 66, 86]),
            ("str", [8, 28, 48, 68, 88]),
            ("td", [10, 30, 50, 70, 90]),
            ("td_%", [12, 32, 52, 72, 92]),
            ("sub_attempt", [14, 34, 54, 74, 94]),
            ("reversal", [16, 36, 56, 76, 96]),
            ("ctrl", [18, 38, 58, 78, 98])
            ]
        else:
            CATEGORIES = [    
            ("fighter", [0, 18, 36, 54, 72]),
            ("sig_str", [2, 20, 38, 56, 74]),
            ("sig_str%", [4, 22, 40, 58, 76]),
            ("head", [6, 24, 42, 60, 78]),
            ("body", [8, 26, 44, 62, 80]),
            ("leg", [10, 28, 46, 64, 82]),
            ("distance", [12, 30, 48, 66, 84]),
            ("clinch", [14, 32, 50, 68, 86]),
            ("ground", [16, 34, 52, 70, 88]),
            ]
        inc = 0
        current_round = None
        for i in range(0, len(data), 2):
            for category_name, category_indices in CATEGORIES:
                if i in category_indices:
                    key = category_name
                    break
            try:
                if data[i+inc].startswith("Round"):
                    current_round = data[i+inc].replace(" ", "_").lower()
                    inc += 1

            except IndexError:
                break
            if key != "fighter":
                if current_round is not None:
                    row[current_round + "_" + key + "_1"] = data[i+inc]
                    row[current_round + "_" + key + "_2"] = data[i+1+inc]
                else:
                    row["total_" + key + "_1"] = data[i+inc]
                    row["total_" + key + "_2"] = data[i+1+inc]
    return row


def get_fighter_links() -> List[str]:
    """Extract fighter profile links from ufcstats.com

    Returns:
        List[str]: links to ufc fighter profile
    """
    alphabet = [chr(i) for i in range(ord('a'), ord('z')+1)]
    fighter_links = set()
    for letter in alphabet:
        # url = f"http://www.ufcstats.com/statistics/fighters?char={letter}&page=all"
        url = "http://www.ufcstats.com/statistics/fighters"
        session = requests.Session()
        response = session.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        # Find all links on the page with the class 'b-link b-link_style_black'
        links = soup.find_all('a', 'b-link b-link_style_black')
        # Extract the href attribute from each link and append it to fighter_links
       
        for link in links:
            fighter_links.add(link['href'])

    return list(fighter_links)


def extract_fighter_stats():
    links = get_fighter_links()
    rows = []
    # if fighter not in fights since last run skip
    for fighter_link in links:
        response = requests.get(fighter_link)
        soup = BeautifulSoup(response.content, "html.parser")
        name = soup.find("span", class_="b-content__title-highlight").text.strip()
        record = soup.find("span", class_="b-content__title-record").text.strip()

        row = {"fighter" : name, "mma_record" : record}
        foo = soup.find("tr", class_="b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click")

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
                    elif key == "Weight":
                        value = get_weightclass(value)
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

        rows.append(row)
    
    df = pd.DataFrame(rows)
    if 'dob' in df.columns:
        df.drop('dob', axis=1, inplace=True)
    
    df = df.sort_values('fighter')
    df.to_csv("fighters.csv", index=False)


def get_weightclass(weight):
    weight = int(weight.rstrip(" lbs."))
    weight_classes = {
        134: "Flyweight",
        144: "Bantamweight",
        154: "Featherweight",
        169: "Lightweight",
        184: "Welterweight",
        204: "Middleweight",
        224: "Lightheavyweight",
        float('inf'): "Heavyweight"
    }
    for upper_limit, weight_class in weight_classes.items():
        if weight < upper_limit:
            return weight_class


def inch_to_cm(height) -> float:
    
    if "'" in height:
        feet, inches = height.split("'")
        total_inches = (int(feet) * 12) + int(inches.strip('"'))
        cm = total_inches * 2.54
        return round(cm, 1)
    
    elif '"' in height:
        inches = int(height.strip('"'))
        return round(inches * 2.54, 1)


if __name__ == "__main__":
    main()