from scripts.ufc_scraper import UfcScraper
from scripts.feature_engineering import prep_model_data
from scripts.model import create_predictions_csv

def main():
    scraper = UfcScraper()
    result = scraper.process_events_and_write_data()
    if result == True:
        prep_model_data()
    create_predictions_csv()

if __name__ == "__main__":
    main()