from scripts.utils import get_weightclass, inch_to_cm
from typing import Tuple, List, Dict
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import requests


class UfcScraper:
    # Read the last card from the file
    with open('data/cards.txt', 'r') as f:
        cards = f.readlines()
    last_event = cards[-1].strip() if cards else ""

    def get_event_urls(self) -> List[str]:
        """
        Extracts URLs of UFC events from ufcstats.com.

        This method retrieves the HTML content of the UFC events page, parses it using BeautifulSoup,
        and extracts the URLs of completed UFC events. The URLs are collected into a list and returned.

        Returns:
            List[str]: A list of URLs to UFC event detail pages.
        """
        target_url = "http://www.ufcstats.com/statistics/events/completed?page=all"
        # target_url = "http://www.ufcstats.com/statistics/events/completed"
        response = requests.get(target_url)
        soup = BeautifulSoup(response.content, "html.parser")

        # Find first event url
        url = soup.find('a', class_='b-link b-link_style_black')

        event_urls = []
        while url:
            event = url.text.strip()
            if event == self.last_event:
                break
            href = url['href']
            event_urls.append(href)
            # Find next url
            url = url.find_next('a', class_='b-link b-link_style_black')
                    
        return event_urls[::-1]

    def get_event_details(self, event: List[str]) -> Dict:
        """
        Extracts detailed information about a UFC event from the given event URL.

        This method fetches the HTML content of the event page from the provided URL,
        parses it using BeautifulSoup, and extracts detailed information about the event.
        The extracted details include the event name, metadata (such as date and location),
        and URLs to the individual fight details. The information is compiled into a dictionary.

        Args:
            event (str): The URL of the event details page.

        Returns:
            Dict: A dictionary containing detailed information about the event, including:
                - 'event': The name of the event.
                - Other keys representing event metadata (e.g., date, location).
                - 'fight_urls': A list of URLs to detailed fight information for the event.
        """

        # Fetch the event page
        response = requests.get(event)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract the event name
        event = soup.find("span", class_="b-content__title-highlight").text.strip()
        row = {"event": event}
        
        # Extract other event metadata
        for li in soup.find_all("li", class_="b-list__box-list-item"):
            title = li.find("i").text.strip()
            value = li.text.strip().replace(title, "").strip()
            row[title.replace(":", "").lower()] = value
        
        raw_fight_data = soup.find_all("tr", class_="b-fight-details__table-row b-fight-details__table-row__hover js-fight-details-click")
        # Extract fight urls
        fight_urls = [data['data-link'] for data in raw_fight_data]
        
        # Add the list of fight urls to the event row
        row['fight_urls'] = fight_urls[::-1]
        
        # Return the list of event details
        return row


    def get_fight_details(self, fight_url: str, event: str, date: str, location: str, fight_num: int) -> Dict:
        """
        Extracts detailed information about a UFC fight from the given URL and compiles it into a dictionary.

        This method fetches the HTML content of the fight page from the provided URL, parses it using BeautifulSoup,
        and extracts detailed information such as fight results, weight class, fighter names, and additional fight details
        (e.g., method, round, time, format, referee). It also calculates and includes relevant fight statistics.

        Args:
            fight_url (str): The URL of the fight details page.
            event (str): The name of the event.
            date (str): The date of the event.
            location (str): The location of the event.
            fight_num (int): The sequential number of the fight.

        Returns:
            Dict: A dictionary containing detailed information about the fight.
        """

        row = {"fight_num": fight_num, "date": date, "location": location, "event": event}
        
        # Fetch the fight page
        response = requests.get(fight_url)
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Extract fight result
        result = soup.find("i", class_="b-fight-details__person-status").text.strip()
        
        # Determine results for both fighters
        if result == "W":
            row["result_1"] = result
            row["result_2"] = "L"
        elif result == "L":
            row["result_1"] = result
            row["result_2"] = "W"
        else:
            # In the event of a draw/NC/DQ
            row["result_1"] = result
            row["result_2"] = result
        
        # Extract weight class
        weight_class = soup.find("i", class_="b-fight-details__fight-title").text.strip()
        row['weight_class'] = weight_class
        
        raw_names = soup.find_all("h3", class_="b-fight-details__person-name")
        # Extract fighter names
        names = [name.text.strip() for name in raw_names]
        # Add fighter names to the row
        for i, name in enumerate(names, start=1):
            row[f'fighter_{i}'] = name
        
        # Add fight details (method, round, time, format, ref)
        details = [i for detail in soup.find_all("p", class_="b-fight-details__text") for i in detail.stripped_strings]
        for i in range(0, len(details), 2):
            if details[i] != "Details:":
                if details[i] == "Referee:" and details[i + 1] == "Details:":
                    row[details[i].strip(":").lower()] = None
                    try:
                        row[details[i + 1].strip(":").lower()] = details[i + 2]
                    except IndexError:
                        row[details[i + 1].strip(":").lower()] = None
                    break
                else:
                    row[details[i].strip(":").lower()] = details[i + 1]
            elif row['method'] == "TKO - Doctor's Stoppage":
                row[details[i].strip(":").lower()] = None
                break
            else:
                try:
                    if "\n" in details[i + 1:][0]:
                        row[details[i].strip(":").lower()] = ' '.join([part.strip() for part in details[i + 1:][0].split('\n')])
                    else:
                        row[details[i].strip(":").lower()] = " ".join(details[i + 1:])
                except IndexError:
                    row[details[i].strip(":").lower()] = None
                break
        
        # Add fight stats (if available)
        try:
            stats = [i for i in soup.find("tbody", class_="b-fight-details__table-body").stripped_strings]
            header = [i for i in soup.find("thead", class_="b-fight-details__table-head").stripped_strings]
            row = self.totals(stats, header, row)
        except AttributeError:
            pass
        
        return row

    def totals(self, stats: List[str], header: List[str], row: Dict) -> Dict:
        """
        Processes and pairs fight statistics with their corresponding headers, and updates a given row dictionary
        with these values, including calculations for landed, attempted, received, and avoided strikes, as well as
        percentages.

        Args:
            stats (List[str]): A list of statistics in the format "value of value" or percentage values.
            header (List[str]): A list of headers corresponding to the statistics.
            row (Dict): A dictionary representing a single fight's data to be updated with the processed statistics.

        Returns:
            Dict: The updated row dictionary with the added fight statistics.
        """
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


    def all_stats(self, stats: List[str], row: Dict) -> Dict:
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


    def get_fighter_links(self) -> List[str]:
        """Extract fighter profile links from ufcstats.com

        Returns:
            List[str]: links to ufc fighter profile
        """
        alphabet = [chr(i) for i in range(ord('a'), ord('z')+1)]
        fighter_links = set()
        for letter in alphabet:
            url = f"http://www.ufcstats.com/statistics/fighters?char={letter}&page=all"
            # url = "http://www.ufcstats.com/statistics/fighters"
            session = requests.Session()
            response = session.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            # Find all links on the page with the class 'b-link b-link_style_black'
            links = soup.find_all('a', 'b-link b-link_style_black')
            # Extract the href attribute from each link and append it to fighter_links
        
            for link in links:
                fighter_links.add(link['href'])

        return list(fighter_links)


    def extract_fighter_stats(self):
        links = self.get_fighter_links()
        rows = []
        # if fighter not in fights since last run skip
        for fighter_link in links:
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
                        elif key == "Weight":
                            value = get_weightclass(value)
                        elif key == "DOB":
                            # Convert date string to datetime object
                            value = datetime.strptime(value, '%b %d, %Y').date()
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
        df = df.sort_values('fighter')
        df.to_csv("data/fighter_stats.csv", index=False)

    def get_next_event_link(self) -> List[str]:
        """
        Retrieves the URL of the next upcoming UFC event from ufcstats.com.

        This method fetches the HTML content of the upcoming events page, parses it using BeautifulSoup,
        and extracts the URL of the next UFC event. The URL is returned as a string.

        Returns:
            str: The URL of the next upcoming UFC event.
        """
        url = "http://www.ufcstats.com/statistics/events/upcoming"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        # Find all links on the page with the soup BeautifulSoup('a', class_ 'b-link b-link_style_black'
        link = soup.find_all('a', class_='b-link b-link_style_black')[0]
        href = link['href']
        
        return href

    def get_future_matchups(self) -> Tuple[str, List[Tuple[str, str]], List[str]]:
        """
        Retrieves future matchups, their weight classes, and the event date from the next UFC event.

        This method fetches the URL of the next upcoming UFC event using the `get_next_event_link` method,
        then retrieves and parses the HTML content of the event page to extract the fighters' names,
        weight classes, and the event date. The information is compiled into a tuple containing the event
        date, a list of matchups, and a list of weight classes.

        Returns:
            Tuple[str, List[Tuple[str, str]], List[str]]:
                - str: The date of the event.
                - List[Tuple[str, str]]: A list of matchups, each represented as a tuple of two fighter names.
                - List[str]: A list of weight classes for the matchups.
        """
        event_link = self.get_next_event_link()
        response = requests.get(event_link)
        soup = BeautifulSoup(response.content, 'html.parser')
        fighters = [fighter.text.strip() for fighter in soup.find_all('a', class_="b-link b-link_style_black")]
        list_items = soup.find_all('li', class_='b-list__box-list-item')
        weightclass_column = []
        matchups = []
        
        for row in soup.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) > 0 and cells[6].text != 'Weight class':
                if row.find('img'):
                    weightclass_column.append(cells[6].text.strip() + " Title")
                else:
                    weightclass_column.append(cells[6].text.strip())

        for item in list_items:
            if 'Date:' in item.text:
                date = item.text.replace('Date:', '').strip()
                break
            
        for i in range(0, len(fighters), 3):
            matchup = (fighters[i], fighters[i+1])
            matchups.append(matchup)
            
        return (date, matchups, weightclass_column)

    def process_events_and_write_data(self):
        """
        Processes UFC event data and updates the fight totals.

        This method reads existing fight data from a CSV file and determines the starting fight number.
        It then retrieves URLs for new events and processes each event to extract fight details.
        The collected data is appended to the existing DataFrame and saved back to the CSV file.

        Raises:
            FileNotFoundError: If "data/raw_fight_totals.csv" does not exist.
        """
        try:
            fights_df = pd.read_csv("data/raw_fight_totals.csv")
            # Determine the starting fight number
            fight_num = fights_df.iloc[-1]['fight_num']
        except FileNotFoundError:
            fights_df = None
            fight_num = 0
    
        urls = self.get_event_urls()
        if not urls:
            print("Fight data up to date")
            return False
        
        rows = []
        events = []
        i = 0
        print("Getting latest fight data")
        try:
            for url in urls:
                event_data = self.get_event_details(url)
                # Iterate over each fight URL in the event
                date = event_data['date']
                location = event_data['location']
                event = event_data['event']
                # Log current event
                print(event)
                # Iter over fight urls and extract fight data
                for i, fight_url in enumerate(event_data['fight_urls']):
                    fight_num += 1
                    result = self.get_fight_details(fight_url, event, date, location, fight_num)
                    rows.append(result)
                events.append(event)

        except KeyboardInterrupt:
            print("Interrupted")
        finally:
            # Remove rows if event data was incomplete
            length = len(rows)
            rows = rows[:length-i]

            # Combine new data with existing DataFrame
            if not fights_df.empty:
                temp = pd.DataFrame(rows)
                fights_df = pd.concat([fights_df, temp])
            else:
                fights_df = pd.DataFrame(rows)
                
            # Save the updated DataFrame to CSV
            fights_df.to_csv("data/raw_fight_totals.csv", index=False)
            # Write the last card to a txt
            with open("data/cards.txt", "a") as file:
                for event in events:
                    file.write(f"{event}\n")
        
        return True